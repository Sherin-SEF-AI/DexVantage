"""Run the shader renderer in a headless EGL context and report what it drew, as one JSON line (used by
test_render_gl.py in a subprocess, because PyOpenGL picks its platform once per process). Ported from Vantage
tests/gl_harness.py, with a box standing in for the robot until Phase 1 brings link meshes.

    python tests/gl_harness.py      prints {"skip": why} when no EGL OpenGL is available
"""
import json
import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    from dexvantage.selftest import _egl_context, _egl_release
    try:
        handles, how = _egl_context()
    except Exception as e:
        print(json.dumps({'skip': str(e)[:200]}))
        return
    import numpy as np
    from OpenGL import GL
    from PyQt6.QtGui import QGuiApplication
    app = QGuiApplication([])
    from dexvantage.ui.render_gl import Frame, GLRenderer, Style
    from dexvantage.ui.scene import box_tris
    from dexvantage.ui.viewport3d import MeshDraw, OrbitCam
    out = {'renderer': GL.glGetString(GL.GL_RENDERER).decode(errors='replace'), 'how': how}
    r = GLRenderer()
    out['init'] = bool(r.init())
    if not out['init']:
        out['error'] = r.error
        print(json.dumps(out))
        return
    tri = box_tris((0.3, -0.15, 0.0), (0.6, 0.15, 0.3)).astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    md = MeshDraw(tri, n, np.full((len(tri), 3), 0.8), np.ones(len(tri)), version=1)
    W, H = 400, 260
    cam = OrbitCam()
    bg = (0.1, 0.1, 0.1, 1.0)

    def render(**kw):
        return r.render_offscreen(Frame(cam.view(), cam.proj(W / H), W, H, target=cam.target.copy(), clear=bg, **kw))
    lit = render(mesh=md)
    flat = render(mesh=md, style=Style(shadows=False, ao=False))
    empty = render(mesh=None, style=Style(ground_plane=False))
    lum = lambda im: im[..., :3].astype(float).mean(axis=2)
    out.update(shape=list(lit.shape), box_px=float((lum(lit) > 120).mean()), empty_max=float(lum(empty).max()),
               shadow_darkens=float(lum(flat).mean() - lum(lit).mean()), gl_error=int(GL.glGetError()))
    r.release()
    _egl_release(handles)
    del app
    print(json.dumps(out))


if __name__ == '__main__':
    main()
