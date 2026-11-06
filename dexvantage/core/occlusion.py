"""Occlusion as a cube map of distances around a point, for articulated scenes (the posed robot, the task objects and
the tables of core/scenegeom.py).

    CubeMap.build(occluders | tris, origin, n=512, backend='auto', near=NEAR_M)
    cube.depth(dirs) -> (N,) distance (m) to the nearest occluder along each world direction (inf: nothing)
    cube.visible(points, eps=EPS_M) -> (N,) bool
    visible(points, origin, occluders, n=512) -> (N,) bool, through CACHE (a CubeCache)
    gl_status() -> what the GPU backend found (renderer, how the context was made, or why it is not used)

Ported from Vantage core/occlusion.py: the cube-map idea (six 90° faces around the camera centre, world-axis aligned,
so the map does not depend on the camera's orientation and every rotation of a camera reuses it), the face basis,
texel_index and face_dirs, and the exact plane depth per texel. What changed: Vantage painted a static vehicle body far
to near with QPainter (painter's order, a triangle index per texel); here the scene is re-posed for every joint vector
q, so the map is built by a z-buffer, either

- 'numpy', the reference rasteriser: for each face, each triangle's candidate texels (the bounding box of its
  projection, clipped against the near plane) are tested exactly: the ray through the texel centre, d, hits the
  triangle (a0, a1, a2) relative to the centre iff the three triple products d · (aᵢ × aᵢ₊₁) all have the sign of
  D = a0 · (a1 × a2) (then d is a positive combination of the corners). The depth is the exact distance along that ray
  to the triangle's plane, |d| · D / (n · d) with n = (a1 − a0) × (a2 − a0), and the z-buffer keeps the nearest per
  texel (np.minimum.at). Texel-centre coverage and an inclusive edge rule.
- 'gl', the GPU: an offscreen EGL OpenGL context (the discrete GPU's EGL device first, then the default display,
  Mesa's surfaceless platform, a software renderer last), the six faces drawn into one n × 6n float framebuffer by a
  depth-only shader that writes length(p − origin) per fragment. Perspective-correct interpolation of p − origin at
  the fragment centre is the exact point of the triangle's plane on the texel-centre ray, so the GPU writes the same
  plane distance (float32) and the depth test keeps the nearest (distance along a fixed ray is proportional to its
  z). OpenGL's top-left rule decides texels whose centre lies exactly on an edge.
- 'auto': 'gl' when a hardware OpenGL context can be made (in this process when PyOpenGL is, or can be, loaded for
  EGL; otherwise in a small worker process, python -m dexvantage.core.occlusion --gl-worker, that is), else 'numpy'
  with the reason logged once (logging, logger dexvantage.core.occlusion) and kept in gl_status().
  DEXVANTAGE_EGL='EGL device 2' (or part of a renderer's name, e.g. 'llvmpipe') picks the context;
  DEXVANTAGE_NO_GL=1 switches the GPU backend off.

Near plane: the part of a triangle less than near (default NEAR_M = 5 mm, a default) from the centre along a face's
axis is clipped (both backends), so a sensor pressed against the surface it is screwed to is not blinded by it.

A point is visible when its distance from the centre is at most the depth of its texel plus eps (default EPS_M =
1 mm, a default): a point on a surface counts as visible when nothing is in front of it. The answer is the texel's,
so a point within about a texel's angle (2/n rad at a face centre, 0.22° at n = 512) of an occluder's silhouette may
be wrong either way; the validation check occlusion.cube_vs_raycast measures how often against exact ray casting
(core/raycast.py), and docs/models/occlusion.md gives the figures and the timings.

Known limits: one centre per map (a stereo pair needs one map per imager); visibility is per texel, not per point;
the occluders are the visual meshes as posed (no deformable cables or cloth; no other sensor's housing); triangles
seen exactly edge-on occlude nothing; float32 depth (about 0.1 µm at 1 m); the GPU backend needs Linux EGL and
OpenGL 3.3, everywhere else 'auto' uses numpy.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_N = 512
NEAR_M = 0.005              # default near clip along a face's axis (m), see the module docstring
EPS_M = 1e-3                # default visibility tolerance (m)
CHUNK = 1_500_000           # candidate texels tested at once by the numpy rasteriser
CACHE_BYTES = 256 * 2 ** 20
BACKENDS = ('auto', 'gl', 'numpy')
NOT_LOADED = 'the geometry is not loaded on the GPU'
SOFTWARE_GL = ('llvmpipe', 'softpipe', 'swrast', 'software')

# (forward, right, up) per face (+x, −x, +y, −y, +z, −z): ported from Vantage core/occlusion.py
FACES = [
    (np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
]


class OcclusionError(RuntimeError):
    """A cube map could not be built with the backend asked for; the message says why."""


# ====================================================================== texels
def texel_index(dirs: np.ndarray, n: int):
    """(face, row, column) of the texel each world direction (K, 3) falls in (ported from Vantage)."""
    d = np.asarray(dirs, float).reshape(-1, 3)
    ad = np.abs(d)
    ax = np.argmax(ad, axis=1)
    sign = np.take_along_axis(d, ax[:, None], 1)[:, 0] >= 0
    face = ax * 2 + (~sign).astype(int)
    u = np.zeros(len(d))
    v = np.zeros(len(d))
    for f in range(6):
        m = face == f
        if not m.any():
            continue
        fw, rt, up = FACES[f]
        dd = d[m]
        z = dd @ fw
        with np.errstate(divide='ignore', invalid='ignore'):
            u[m] = (dd @ rt) / z
            v[m] = (dd @ up) / z
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    px = np.clip(np.floor((u + 1) * 0.5 * n).astype(np.int64), 0, n - 1)
    py = np.clip(np.floor((1 - v) * 0.5 * n).astype(np.int64), 0, n - 1)
    return face, py, px


def texel_dirs(f: int, n: int) -> np.ndarray:
    """(n, n, 3) unit directions through the texel centres of face f (row 0 at the face's top)."""
    fw, rt, up = FACES[f]
    u = (np.arange(n) + 0.5) / n * 2 - 1
    v = 1 - (np.arange(n) + 0.5) / n * 2
    d = fw[None, None, :] + u[None, :, None] * rt[None, None, :] + v[:, None, None] * up[None, None, :]
    return d / np.linalg.norm(d, axis=2, keepdims=True)


_FACE_DIRS: dict = {}


def face_dirs(n: int) -> np.ndarray:
    """(6, n, n, 3) unit directions through every texel centre (cached, read-only)."""
    d = _FACE_DIRS.get(n)
    if d is None:
        d = np.stack([texel_dirs(f, n) for f in range(6)])
        d.setflags(write=False)
        _FACE_DIRS[n] = d
    return d


# ====================================================================== geometry input
def _tris_of(occ) -> np.ndarray:
    t = getattr(occ, 'tris', occ)
    t = np.ascontiguousarray(t, np.float32)
    if t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError(f'occluder triangles must be an (M, 3, 3) array; got shape {t.shape}')
    return t


def geometry_digest(occ) -> str:
    """A content digest of the occluder triangles (an Occluders' own digest when it has one)."""
    fn = getattr(occ, 'digest', None)
    if callable(fn):
        return fn()
    t = _tris_of(occ)
    return hashlib.blake2b(t.tobytes(), digest_size=16).hexdigest()


# ====================================================================== the numpy rasteriser
def raster_numpy(tris, origin, n: int = DEFAULT_N, near: float = NEAR_M) -> np.ndarray:
    """(6, n, n) float32 distances (inf: nothing) of the triangles (M, 3, 3) seen from origin: the reference
    z-buffer (see the module docstring)."""
    rel = np.asarray(tris, np.float64) - np.asarray(origin, np.float64).reshape(1, 1, 3)
    out = np.full((6, n * n), np.inf)
    if not len(rel):
        return out.reshape(6, n, n).astype(np.float32)
    a0, a1, a2 = rel[:, 0], rel[:, 1], rel[:, 2]
    C = np.stack([np.cross(a0, a1), np.cross(a1, a2), np.cross(a2, a0), np.cross(a1 - a0, a2 - a0)], axis=1)
    D = np.einsum('ij,ij->i', a0, C[:, 1])
    area2 = np.linalg.norm(C[:, 3], axis=1)
    scale = np.linalg.norm(rel, axis=2).max(axis=1)
    keep = (area2 > 1e-18) & (np.abs(D) > 1e-12 * area2 * np.maximum(scale, 1e-9))     # not edge-on, not degenerate
    sgn = np.sign(D)
    half = n / 2.0
    edges = ((0, 1), (1, 2), (2, 0))
    for f, (fw, rt, up) in enumerate(FACES):
        z = rel @ fw
        x = rel @ rt
        y = rel @ up
        front = z >= near
        cand = keep & front.any(axis=1)
        if not cand.any():
            continue
        idx = np.nonzero(cand)[0]
        z, x, y, fr = z[idx], x[idx], y[idx], front[idx]
        # the projection's bounding box: the corners in front, and where the edges cross the near plane
        with np.errstate(divide='ignore', invalid='ignore'):
            pu = [np.where(fr[:, k], x[:, k] / z[:, k], np.nan) for k in range(3)]
            pv = [np.where(fr[:, k], y[:, k] / z[:, k], np.nan) for k in range(3)]
            for a, b in edges:
                cross = fr[:, a] != fr[:, b]
                s = (near - z[:, a]) / (z[:, b] - z[:, a])
                pu.append(np.where(cross, (x[:, a] + s * (x[:, b] - x[:, a])) / near, np.nan))
                pv.append(np.where(cross, (y[:, a] + s * (y[:, b] - y[:, a])) / near, np.nan))
        PU, PV = np.stack(pu, axis=1), np.stack(pv, axis=1)
        umin, umax = np.nanmin(PU, axis=1), np.nanmax(PU, axis=1)
        vmin, vmax = np.nanmin(PV, axis=1), np.nanmax(PV, axis=1)
        # texel centres inside the box: column c has u = (c + 0.5) / half − 1, row r has v = 1 − (r + 0.5) / half
        with np.errstate(invalid='ignore'):
            c0 = np.clip(np.ceil((np.clip(umin, -2, 2) + 1) * half - 0.5), 0, n).astype(np.int64)
            c1 = np.clip(np.floor((np.clip(umax, -2, 2) + 1) * half - 0.5), -1, n - 1).astype(np.int64)
            r0 = np.clip(np.ceil((1 - np.clip(vmax, -2, 2)) * half - 0.5), 0, n).astype(np.int64)
            r1 = np.clip(np.floor((1 - np.clip(vmin, -2, 2)) * half - 0.5), -1, n - 1).astype(np.int64)
        w, h = c1 - c0 + 1, r1 - r0 + 1
        ok = (w > 0) & (h > 0)
        if not ok.any():
            continue
        idx, c0, r0, w, h = idx[ok], c0[ok], r0[ok], w[ok], h[ok]
        # per triangle, each triple product and n·d as a linear function of (u, v): A + u·B + v·Cv
        basis = np.stack([fw, rt, up])                                   # (3, 3)
        coef = np.einsum('mkj,bj->mkb', C[idx], basis)                   # (m, 4, 3)
        coef[:, :3] *= sgn[idx][:, None, None]
        Dm = D[idx]
        cnt = w * h
        cum = np.cumsum(cnt)
        a = 0
        zf = out[f]
        while a < len(idx):
            base = int(cum[a - 1]) if a else 0
            b = max(a + 1, int(np.searchsorted(cum, base + CHUNK, 'right')))
            cc = cnt[a:b]
            K = int(cc.sum())
            ti = np.repeat(np.arange(a, b), cc)
            local = np.arange(base, base + K, dtype=np.int64) - np.repeat(cum[a:b] - cc, cc)
            ww = w[ti]
            col = c0[ti] + local % ww
            row = r0[ti] + local // ww
            u = (col + 0.5) / half - 1.0
            v = 1.0 - (row + 0.5) / half
            g = coef[ti]                                                 # (K, 4, 3)
            val = g[:, :, 0] + u[:, None] * g[:, :, 1] + v[:, None] * g[:, :, 2]
            inside = (val[:, 0] >= 0) & (val[:, 1] >= 0) & (val[:, 2] >= 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = Dm[ti] / val[:, 3]                                   # the hit's coordinate along the face axis
            hit = inside & (t >= near) & np.isfinite(t)
            if hit.any():
                dist = t[hit] * np.sqrt(1.0 + u[hit] ** 2 + v[hit] ** 2)
                np.minimum.at(zf, row[hit] * n + col[hit], dist)
            a = b
    return out.reshape(6, n, n).astype(np.float32)


# ====================================================================== the GPU backend
_VS = """#version 330
layout(location = 0) in vec3 pos;
uniform mat4 mvp;
uniform vec3 origin;
out vec3 rel;
void main() { rel = pos - origin; gl_Position = mvp * vec4(rel, 1.0); }
"""
_FS = """#version 330
in vec3 rel;
layout(location = 0) out float dist;
void main() { dist = length(rel); }
"""


def _face_mvp(f: int, near: float, far: float) -> np.ndarray:
    fw, rt, up = FACES[f]
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = rt, up, -fw
    P = np.zeros((4, 4))
    P[0, 0] = P[1, 1] = 1.0                         # 90° field of view, square
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1.0
    return (P @ V).astype(np.float32)


class _GLRaster:
    """An EGL context with the depth program, framebuffers per n and vertex buffers per geometry digest. Not
    thread-safe on its own: _GLBackend serialises the calls and makes the context current around each."""

    VBOS = 6

    def __init__(self):
        import ctypes
        if 'OpenGL' in sys.modules:
            import OpenGL.platform as _plat
            name = type(_plat.PLATFORM).__name__
            if 'EGL' not in name:
                raise OcclusionError(f'PyOpenGL is loaded for {name} in this process; EGL needs it loaded for EGL')
        else:
            if os.environ.get('PYOPENGL_PLATFORM', 'egl') != 'egl':
                raise OcclusionError(f'PYOPENGL_PLATFORM is {os.environ["PYOPENGL_PLATFORM"]}, not egl')
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        from OpenGL import EGL, GL
        from ..selftest import _EGL_PLATFORM_DEVICE, _EGL_PLATFORM_SURFACELESS, _egl_try, _first_line
        self.EGL, self.GL = EGL, GL
        tries = []
        try:
            from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
            from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
            devs, cnt = (EGL.EGLDeviceEXT * 8)(), EGL.EGLint()
            if eglQueryDevicesEXT(8, devs, ctypes.pointer(cnt)):
                tries += [(f'EGL device {i}', lambda d=devs[i]: eglGetPlatformDisplayEXT(_EGL_PLATFORM_DEVICE, d, None))
                          for i in range(cnt.value)]
        except Exception:                       # noqa: BLE001  (no device enumeration in this libEGL)
            pass
        tries += [('default display', lambda: EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)),
                  ('surfaceless', lambda: EGL.eglGetPlatformDisplay(_EGL_PLATFORM_SURFACELESS,
                                                                    EGL.EGL_DEFAULT_DISPLAY, None))]
        want = os.environ.get('DEXVANTAGE_EGL', '').strip().lower()     # e.g. 'EGL device 2' or 'llvmpipe'
        said, soft = [], None
        self.handles = None
        for how, get in tries:
            if want and want.startswith('egl device') and how.lower() != want:
                continue
            try:
                h = _egl_try(get)
            except Exception as e:              # noqa: BLE001
                said.append(f'{how}: {_first_line(e)}')
                continue
            ren = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
            ver = GL.glGetString(GL.GL_VERSION).decode(errors='replace')
            if want and not want.startswith('egl device') and want not in ren.lower() and want not in how.lower():
                self._release(h)
                said.append(f'{how}: {ren} is not the renderer DEXVANTAGE_EGL asks for')
                continue
            if any(s in ren.lower() for s in SOFTWARE_GL) and not want:
                if soft is None:
                    soft = (h, how, ren, ver)
                    EGL.eglMakeCurrent(h[0], EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
                else:
                    self._release(h)
                continue
            self.handles, self.how, self.renderer, self.version = h, how, ren, ver
            break
        if self.handles is None:
            if soft is None:
                raise OcclusionError('no EGL OpenGL context (' + '; '.join(said) + ')')
            self.handles, self.how, self.renderer, self.version = soft
            EGL.eglMakeCurrent(self.handles[0], self.handles[1], self.handles[1], self.handles[2])
        elif soft is not None:
            self._release(soft[0])
            EGL.eglMakeCurrent(self.handles[0], self.handles[1], self.handles[1], self.handles[2])
        self.software = any(s in self.renderer.lower() for s in SOFTWARE_GL)
        try:
            self._setup()
        finally:
            self.done_current()

    def _release(self, h):
        from ..selftest import _egl_release
        try:
            _egl_release(h)
        except Exception:                       # noqa: BLE001
            pass

    def make_current(self):
        dpy, surf, ctx = self.handles
        if not self.EGL.eglMakeCurrent(dpy, surf, surf, ctx):
            raise OcclusionError('the EGL context could not be made current')

    def done_current(self):
        E = self.EGL
        E.eglMakeCurrent(self.handles[0], E.EGL_NO_SURFACE, E.EGL_NO_SURFACE, E.EGL_NO_CONTEXT)

    def _setup(self):
        GL = self.GL
        from OpenGL.GL import shaders
        try:
            self.prog = shaders.compileProgram(shaders.compileShader(_VS, GL.GL_VERTEX_SHADER),
                                               shaders.compileShader(_FS, GL.GL_FRAGMENT_SHADER), validate=False)
        except Exception as e:                  # noqa: BLE001
            raise OcclusionError(f'the depth shader does not compile on {self.renderer}: {str(e)[:200]}') from None
        self.u_mvp = GL.glGetUniformLocation(self.prog, 'mvp')
        self.u_origin = GL.glGetUniformLocation(self.prog, 'origin')
        self.vao = GL.glGenVertexArrays(1)
        self.vbos: OrderedDict = OrderedDict()          # digest -> (vbo, vertex count)
        self.fbos: dict = {}                            # n -> (fbo, colour texture, depth renderbuffer)
        self.max_size = int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE))

    def _fbo(self, n: int):
        GL = self.GL
        if n in self.fbos:
            return self.fbos[n][0]
        if 6 * n > self.max_size:
            raise OcclusionError(f'{self.renderer} allows framebuffers up to {self.max_size} px; n = {n} needs {6 * n}')
        for k in list(self.fbos):                       # keep one size at a time (memory)
            fb, tex, rb = self.fbos.pop(k)
            GL.glDeleteFramebuffers(1, [fb])
            GL.glDeleteTextures([tex])
            GL.glDeleteRenderbuffers(1, [rb])
        fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, n, 6 * n, 0, GL.GL_RED, GL.GL_FLOAT, None)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)
        rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT32F, n, 6 * n)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rb)
        st = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if st != GL.GL_FRAMEBUFFER_COMPLETE:
            raise OcclusionError(f'the float framebuffer is incomplete on {self.renderer} (status {int(st)})')
        self.fbos[n] = (fb, tex, rb)
        return fb

    def _vbo(self, digest: str, tris):
        GL = self.GL
        hit = self.vbos.get(digest)
        if hit is not None:
            self.vbos.move_to_end(digest)
            return hit
        if tris is None:
            return None
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        data = np.ascontiguousarray(tris, np.float32).reshape(-1, 3)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        self.vbos[digest] = (vbo, len(data))
        while len(self.vbos) > self.VBOS:
            _d, (old, _n) = self.vbos.popitem(last=False)
            GL.glDeleteBuffers(1, [old])
        return self.vbos[digest]

    def has(self, digest: str) -> bool:
        return digest in self.vbos

    def render(self, digest: str, tris, origin, n: int, near: float, far: float) -> np.ndarray:
        """(6, n, n) float32 distances, inf where nothing was drawn. tris may be None when digest is loaded."""
        GL = self.GL
        self.make_current()
        try:
            fb = self._fbo(n)
            entry = self._vbo(digest, tris)
            if entry is None:
                raise OcclusionError(NOT_LOADED)
            vbo, nv = entry
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
            GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            try:
                GL.glDisable(GL.GL_MULTISAMPLE)
            except Exception:                   # noqa: BLE001
                pass
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glViewport(0, 0, n, 6 * n)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClearDepth(1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glUseProgram(self.prog)
            GL.glUniform3f(self.u_origin, *[float(v) for v in origin])
            GL.glBindVertexArray(self.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
            for f in range(6):
                GL.glViewport(0, f * n, n, n)
                GL.glUniformMatrix4fv(self.u_mvp, 1, GL.GL_TRUE, _face_mvp(f, near, far))
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, nv)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
            data = GL.glReadPixels(0, 0, n, 6 * n, GL.GL_RED, GL.GL_FLOAT)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            err = GL.glGetError()
            if err:
                raise OcclusionError(f'OpenGL error {int(err)} while drawing the cube map on {self.renderer}')
        finally:
            self.done_current()
        a = np.frombuffer(data, np.float32) if isinstance(data, (bytes, bytearray)) else \
            np.asarray(data, np.float32).ravel()
        a = a.reshape(6, n, n)[:, ::-1, :]
        return np.where(a > 0, a, np.inf).astype(np.float32)


class _GLWorkerClient:
    """The GPU backend in a worker process (python -m dexvantage.core.occlusion --gl-worker), used when this process
    has PyOpenGL loaded for another platform (GLX). Messages on its stdin and stdout: see _worker_main."""

    def __init__(self):
        env = dict(os.environ, PYOPENGL_PLATFORM='egl')
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        self.proc = subprocess.Popen([sys.executable, '-m', 'dexvantage.core.occlusion', '--gl-worker'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=env, cwd=root)
        line = self.proc.stdout.readline().decode('utf-8', 'replace').strip()
        if not line.startswith('ok\t'):
            self.close()
            raise OcclusionError(line[6:] if line.startswith('error\t') else 'the GPU worker did not start')
        _ok, self.how, self.renderer, soft = line.split('\t')[:4]
        self.software = soft == '1'
        self.loaded: OrderedDict = OrderedDict()
        atexit.register(self.close)

    def close(self):
        p = getattr(self, 'proc', None)
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
                p.wait(timeout=2)
            except Exception:                   # noqa: BLE001
                p.kill()

    def has(self, digest: str) -> bool:
        return digest in self.loaded

    def render(self, digest: str, tris, origin, n: int, near: float, far: float) -> np.ndarray:
        send = tris is not None and digest not in self.loaded
        body = np.ascontiguousarray(tris, np.float32).tobytes() if send else b''
        m = len(body) // 36
        hdr = struct.pack('<16s3dIffI', bytes.fromhex(digest)[:16].ljust(16, b'\0'), *[float(v) for v in origin],
                          int(n), float(near), float(far), m)
        try:
            self.proc.stdin.write(hdr + body)
            self.proc.stdin.flush()
            st = struct.unpack('<I', self._read(4))[0]
            msg = self._read(st).decode('utf-8', 'replace') if st else ''
            a = None if st else np.frombuffer(self._read(6 * n * n * 4), np.float32).reshape(6, n, n)
        except (OSError, struct.error, ValueError) as e:
            raise OcclusionError(f'the GPU worker stopped ({e})') from None
        if st:
            if msg == NOT_LOADED and not send and tris is not None:
                self.loaded.pop(digest, None)         # the worker dropped it after all: upload it again, once
                return self.render(digest, tris, origin, n, near, far)
            raise OcclusionError(msg)
        # mirror the worker's least-recently-used order exactly (_GLRaster._vbo moves a geometry it reuses to the end):
        # recording only the uploads would let the two orders drift apart, and the worker would then be asked to draw
        # a geometry it has already dropped
        if digest in self.loaded:
            self.loaded.move_to_end(digest)
        elif send:
            self.loaded[digest] = True
            while len(self.loaded) > _GLRaster.VBOS:
                self.loaded.popitem(last=False)
        return a

    def _read(self, k: int) -> bytes:
        buf = b''
        while len(buf) < k:
            chunk = self.proc.stdout.read(k - len(buf))
            if not chunk:
                raise OSError('end of stream')
            buf += chunk
        return buf


def _worker_main():
    """The GPU worker: prints 'ok\\t<how>\\t<renderer>\\t<software 0|1>' (or 'error\\t<why>'), then answers render
    requests: header <16s digest, 3d origin, I n, f near, f far, I M> + M·36 bytes of float32 triangles (M = 0: the
    geometry with this digest is loaded already) -> <I 0> + 6·n·n float32, or <I len> + an error message."""
    out = sys.stdout.buffer
    try:
        r = _GLRaster()
    except Exception as e:                      # noqa: BLE001
        out.write(f'error\t{str(e)[:300]}\n'.encode())
        out.flush()
        return
    out.write(f'ok\t{r.how}\t{r.renderer}\t{int(r.software)}\n'.encode())
    out.flush()
    inp = sys.stdin.buffer
    size = struct.calcsize('<16s3dIffI')
    while True:
        hdr = inp.read(size)
        if len(hdr) < size:
            return
        dg, ox, oy, oz, n, near, far, m = struct.unpack('<16s3dIffI', hdr)
        tris = None
        if m:
            raw = inp.read(m * 36)
            tris = np.frombuffer(raw, np.float32).reshape(m, 3, 3)
        try:
            a = r.render(dg.hex(), tris, (ox, oy, oz), int(n), float(near), float(far))
            out.write(struct.pack('<I', 0) + np.ascontiguousarray(a, np.float32).tobytes())
        except Exception as e:                  # noqa: BLE001
            msg = str(e).encode('utf-8', 'replace')[:2000] or b'error'
            out.write(struct.pack('<I', len(msg)) + msg)
        out.flush()


class _GLBackend:
    """The process's GPU backend, made on first use: in process (EGL) or through the worker; thread-safe."""

    def __init__(self):
        self.lock = threading.Lock()
        self.impl = None
        self.tried = False
        self.reason = ''
        self.mode = ''

    def get(self):
        with self.lock:
            if not self.tried:
                self.tried = True
                if os.environ.get('DEXVANTAGE_NO_GL'):
                    self.reason = 'DEXVANTAGE_NO_GL is set'
                    log.info('Occlusion: the GPU cube-map backend is switched off (%s), numpy is used', self.reason)
                else:
                    reasons = []
                    for mode, make in (('in process', _GLRaster), ('worker process', _GLWorkerClient)):
                        try:
                            self.impl, self.mode = make(), mode
                            break
                        except Exception as e:      # noqa: BLE001
                            reasons.append(f'{mode}: {str(e)[:300]}')
                    if self.impl is None:
                        self.reason = '; '.join(reasons)
                        log.warning('Occlusion: the GPU cube-map backend is not available, numpy is used (%s)',
                                    self.reason)
            return self.impl

    def render(self, digest, tris, origin, n, near, far):
        impl = self.get()
        if impl is None:
            raise OcclusionError(f'the GPU backend is not available: {self.reason}')
        with self.lock:
            # the triangles go along every time: an implementation uploads them only when it does not hold them
            return impl.render(digest, tris, origin, n, near, far)


GL_BACKEND = _GLBackend()


def gl_status() -> dict:
    """What the GPU backend found: {'available', 'renderer', 'how', 'mode', 'software', 'reason'}."""
    impl = GL_BACKEND.get()
    if impl is None:
        return {'available': False, 'renderer': '', 'how': '', 'mode': '', 'software': False,
                'reason': GL_BACKEND.reason}
    return {'available': True, 'renderer': impl.renderer, 'how': impl.how, 'mode': GL_BACKEND.mode,
            'software': bool(impl.software), 'reason': ''}


def resolve_backend(backend: str = 'auto') -> str:
    """'gl' or 'numpy' for a backend request ('auto' picks 'gl' when a hardware or software context exists)."""
    if backend not in BACKENDS:
        raise ValueError(f'the occlusion backend is one of {", ".join(BACKENDS)}; got {backend!r}')
    if backend == 'numpy':
        return 'numpy'
    if GL_BACKEND.get() is not None:
        return 'gl'
    if backend == 'gl':
        raise OcclusionError(f'the GPU backend is not available: {GL_BACKEND.reason}')
    return 'numpy'


# ====================================================================== the cube map
class CubeMap:
    """Distances to the nearest occluder around origin, per texel of six n × n faces (see the module docstring)."""

    def __init__(self, faces: np.ndarray, origin, *, near: float = NEAR_M, backend: str = 'numpy',
                 seconds: float = 0.0, digest: str = '', renderer: str = ''):
        self.faces = faces                       # (6, n, n) float32 metres, inf = nothing
        self.origin = np.asarray(origin, float).reshape(3)
        self.n = int(faces.shape[1])
        self.near = float(near)
        self.backend = backend
        self.seconds = float(seconds)
        self.digest = digest
        self.renderer = renderer

    @classmethod
    def build(cls, occluders, origin, n: int = DEFAULT_N, backend: str = 'auto', near: float = NEAR_M) -> 'CubeMap':
        """The cube map of occluders (an Occluders or an (M, 3, 3) array, world frame) around origin."""
        n = int(n)
        if n < 2 or n > 4096:
            raise ValueError(f'the cube-map face size must be 2 to 4096 texels; got {n}')
        tris = _tris_of(occluders)
        origin = np.asarray(origin, float).reshape(3)
        digest = geometry_digest(occluders)
        be = resolve_backend(backend)
        t0 = time.perf_counter()
        renderer = 'numpy'
        if be == 'gl' and len(tris):
            far = float(np.linalg.norm(tris.reshape(-1, 3) - origin, axis=1).max()) * 1.01 + 0.01
            try:
                faces = GL_BACKEND.render(digest, tris, origin, n, near, max(far, 2 * near))
                renderer = GL_BACKEND.impl.renderer
            except OcclusionError as e:
                if backend == 'gl':
                    raise
                log.warning('Occlusion: the GPU cube map failed, numpy is used (%s)', e)
                be = 'numpy'
        if be == 'numpy' or not len(tris):
            faces = raster_numpy(tris, origin, n, near)
        return cls(faces, origin, near=near, backend=be, seconds=time.perf_counter() - t0, digest=digest,
                   renderer=renderer)

    @property
    def nbytes(self) -> int:
        return int(self.faces.nbytes)

    @property
    def texel_rad(self) -> float:
        """The angle a texel spans at a face centre (rad): 2 / n (at a face corner it is a third of that)."""
        return 2.0 / self.n

    def depth(self, dirs) -> np.ndarray:
        """(N,) distance (m) to the nearest occluder along each world direction (N, 3) (inf: nothing)."""
        d = np.asarray(dirs, float).reshape(-1, 3)
        f, r, c = texel_index(d, self.n)
        return self.faces[f, r, c].astype(np.float64)

    def visible(self, points, eps: float = EPS_M) -> np.ndarray:
        """(N,) True where nothing in the map is nearer than the point (less eps) in the point's texel."""
        P = np.asarray(points, float).reshape(-1, 3)
        v = P - self.origin
        dist = np.linalg.norm(v, axis=1)
        return dist <= self.depth(v) + eps


class CubeCache:
    """An LRU of cube maps keyed by (geometry digest, origin to 0.1 mm, n, near, backend), bounded in bytes."""

    def __init__(self, max_bytes: int = CACHE_BYTES):
        self.max_bytes = int(max_bytes)
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(digest: str, origin, n: int, near: float, backend: str) -> tuple:
        return digest, tuple(np.round(np.asarray(origin, float).reshape(3) * 1e4).astype(np.int64).tolist()), \
            int(n), round(float(near), 6), backend

    def get(self, occluders, origin, n: int = DEFAULT_N, backend: str = 'auto', near: float = NEAR_M) -> CubeMap:
        be = resolve_backend(backend)
        k = self.key(geometry_digest(occluders), origin, n, near, be)
        with self._lock:
            c = self._d.get(k)
            if c is not None:
                self._d.move_to_end(k)
                self.hits += 1
                return c
            self.misses += 1
        c = CubeMap.build(occluders, origin, n, be, near)
        with self._lock:
            self._d[k] = c
            total = sum(x.nbytes for x in self._d.values())
            while total > self.max_bytes and len(self._d) > 1:
                _k, old = self._d.popitem(last=False)
                total -= old.nbytes
        return c

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)


CACHE = CubeCache()


def visible(points, origin, occluders, n: int = DEFAULT_N, eps: float = EPS_M, backend: str = 'auto',
            near: float = NEAR_M) -> np.ndarray:
    """(N,) bool: the points seen from origin past the occluders, through the shared CACHE."""
    return CACHE.get(occluders, origin, n, backend, near).visible(points, eps)


def decimation_report(occluders, origin, points, cell_m: float, n: int = DEFAULT_N, backend: str = 'auto',
                      eps: float = EPS_M) -> dict:
    """What decimating the occluders by vertex clustering (cell_m, core.scenegeom.decimate) changes: triangle
    counts, build times and the fraction of points whose visibility from origin differs from the full meshes'."""
    from .scenegeom import decimate
    P = np.asarray(points, float).reshape(-1, 3)
    full = CubeMap.build(occluders, origin, n, backend)
    dec = decimate(occluders, cell_m)
    small = CubeMap.build(dec, origin, n, backend)
    a, b = full.visible(P, eps), small.visible(P, eps)
    return {'cell_mm': cell_m * 1000, 'triangles': int(len(_tris_of(occluders))), 'triangles_decimated':
            int(len(_tris_of(dec))), 'points': int(len(P)), 'changed': int((a != b).sum()),
            'changed_fraction': float((a != b).mean()) if len(P) else 0.0,
            'visible_fraction_full': float(a.mean()) if len(P) else 0.0,
            'visible_fraction_decimated': float(b.mean()) if len(P) else 0.0,
            'seconds_full': full.seconds, 'seconds_decimated': small.seconds, 'backend': full.backend}


if __name__ == '__main__':
    if '--gl-worker' in sys.argv:
        _worker_main()
