"""Checks of the VLA token study (core/studies/vla_tokens.py) and the Plücker ray maps (core/plucker.py).

vla.patch_counts_raycast — the study rasterises each object's projected triangles in the policy's input
(subdivided where the lens bends edges). The reference shares nothing with it but the crop mapping (validated by
policies.crop_geometry) and the lens model (validated by the optics checks): every input pixel's centre is mapped back
to the sensor, unprojected through the lens, turned into the world and ray-cast against the object's triangles
(core/raycast.py, Möller–Trumbore through a uniform grid); a pixel is covered when its ray hits. Tokens are counted
from those pixels with a plain loop. Scripted cameras (a pinhole, a Brown-Conrady lens with strong barrel distortion,
a Kannala-Brandt fisheye) and four crop rules, with boxes, cylinders and spheres at random poses (some crossing the
crop's edge). Tolerance: no token may differ (0 tokens); the pixels that differ are reported, and are expected only
where a pixel centre lies within the subdivision tolerance (0.01 input px, a default) of a silhouette edge.

vla.plucker_moment_cv2 — the ray map's directions and moments against rays cast independently: the input pixel
centre is mapped to the sensor with the crop's own rectangles written out again, undistorted by OpenCV
(cv2.undistortPoints with its termination criteria for Brown-Conrady and pinhole, cv2.fisheye.undistortPoints for
Kannala-Brandt, from CameraModel.to_opencv), normalised, turned into the robot base frame by a pose composed here from
the sensor's world pose and the robot's base, and m = o × d. Tolerance 1e-6 (unitless for d, metres for m): OpenCV's
iterative undistortion is run to 1e-14 and agrees with the model's own inverse well below it; also m · d = 0 and the
moment of o + t·d. OpenCV's undistortion returns x/z, y/z, which has no ray at or beyond 90° off the axis and is
ill-conditioned close to it (a 0.018 difference at 89° on the scripted fisheye), so pixels whose ray lies more than
85° off the axis are not compared; how many is reported per case.

vla.safetensors_roundtrip — a file written by core/plucker.py write_safetensors read back by a parser written here
from the public format (8-byte little-endian header length, JSON header, offsets into the buffer), and by the
safetensors package when it is installed; every dtype the export uses; the header must start with '{', be padded to
8 bytes and index the buffer without holes. Tolerance 0 (bit-exact).
"""
from __future__ import annotations

import json
import math
import os
import struct

import numpy as np

from .suite import Outcome, check

TOL_PX = 0.01
MAX_CMP_DEG = 85.0           # rays compared within this of the axis (OpenCV's normalised x/z, y/z)


def _cameras():
    from ..core.optics import CameraModel
    return [
        ('pinhole', CameraModel(1280, 720, 900.0, 900.0, 639.5, 359.5)),
        ('brown', CameraModel(1280, 800, 700.0, 700.0, 645.2, 396.8, 'brown', (-0.28, 0.09, 0.0012, -0.0007, -0.012))),
        ('kb', CameraModel(1280, 960, 420.0, 420.0, 641.0, 478.5, 'kb', (0.021, -0.008, 0.0021, -0.0003))),
        ('pinhole_vga', CameraModel(640, 480, 520.0, 525.0, 318.2, 241.7)),
    ]


# (camera index, rule, input, cell, crop fraction, pre-resize)
CASES = (
    (0, 'resize_pad', (224, 224), 14, None, None),
    (1, 'center_crop_resize', (256, 256), 16, 0.9487, None),
    (2, 'resize', (224, 224), 14, None, None),
    (3, 'center_crop', (288, 216), 8, None, (320, 240)),
    (0, 'resize_pad_top_left', (512, 512), 16, None, None),
)


def _objects(rng, n):
    """n scripted task objects in front of a camera at the origin looking along +z (world = optical here, turned
    later): (name, triangles)."""
    from ..core.frames import Pose
    from ..core.project import TaskObject
    from ..core.scenegeom import object_triangles
    out = []
    shapes = ('box', 'cylinder', 'sphere')
    for k in range(n):
        shape = shapes[k % 3]
        z = rng.uniform(0.3, 1.2)
        x, y = rng.uniform(-0.6, 0.6) * z, rng.uniform(-0.4, 0.4) * z
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        big = 4.0 if k == 0 else 1.0               # one large object per case: long, visibly curved edges
        size = {'box': tuple(big * rng.uniform(0.01, 0.12, 3)), 'cylinder': (rng.uniform(0.005, 0.05),
                                                                        rng.uniform(0.02, 0.15), 0.0),
                'sphere': (rng.uniform(0.005, 0.06), 0.0, 0.0)}[shape]
        o = TaskObject(name=f'{shape}{k}', shape=shape, size=size, pose=Pose(np.array([x, y, z]), q))
        out.append((o.name, object_triangles(o)))
    return out


def reference_mask(view, tris) -> np.ndarray:
    """(h, w) input pixels whose centre ray hits the triangles (the independent reference)."""
    from ..core import raycast as RC
    g = view.geom
    w, h = g.input_size
    W, H = g.src_size
    xi, yi = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dx0, dy0, dx1, dy1 = g.dst_rect
    content = (xi > dx0) & (xi < dx1) & (yi > dy0) & (yi < dy1)
    xs, ys = g.to_source(xi, yi)
    inside = content & (xs > 0) & (xs < W) & (ys > 0) & (ys < H)
    uv = np.stack([xs - 0.5, ys - 0.5], axis=-1)[inside]
    rays = view.camera.unproject(uv)
    ok = np.isfinite(rays).all(axis=1)
    d = np.where(ok[:, None], rays, np.array([0.0, 0.0, 1.0])) @ view.R.T
    t = RC.first_hit(view.t, d, np.asarray(tris, float))
    hit = np.zeros((h, w), bool)
    hit[inside] = ok & np.isfinite(t)
    return hit


def reference_tokens(mask, cell) -> int:
    h, w = mask.shape
    rows, cols = h // cell, w // cell
    cells = set()
    for j, i in zip(*np.nonzero(mask)):
        if j < rows * cell and i < cols * cell:
            cells.add((j // cell, i // cell))
    return len(cells)


def patch_case_rows(seed: int, quick: bool) -> list:
    from ..core.frames import quat_to_matrix
    from ..core.policies import crop_geometry
    from ..core.studies import vla_tokens as VT
    rng = np.random.default_rng(seed)
    cams = _cameras()
    rows = []
    for ci, rule, inp, cell, frac, pre in CASES:
        name, cam = cams[ci]
        g = crop_geometry(rule, (cam.width, cam.height), inp, frac, pre)
        # the camera somewhere in the world, looking at the objects (they are placed in its optical frame)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_matrix(q)
        t = rng.uniform(-1, 1, 3)
        view = VT.View(None, None, cam, g, R, t)
        for oname, tri in _objects(rng, 4 if quick else 9):
            tw = tri @ R.T + t
            sil = VT.silhouette(view, tw, TOL_PX, VT.DEFAULTS['max_subdiv'])
            ref = reference_mask(view, tw)
            ours = VT.count_mask(sil.input_mask, cell)['tokens']
            theirs = reference_tokens(ref, cell)
            rows.append({'camera': name, 'rule': rule, 'input': list(inp), 'cell_px': cell, 'object': oname,
                         'tokens': ours, 'tokens_reference': theirs, 'token_error': abs(ours - theirs),
                         'pixels': int(sil.input_mask.sum()), 'pixels_reference': int(ref.sum()),
                         'pixels_differ': int(np.count_nonzero(sil.input_mask != ref))})
    return rows


@check('vla.patch_counts_raycast', 'vla', 'ViT token counts of an object against ray casting each input pixel',
       'largest difference in the number of tokens an object touches between the study\'s rasterised silhouette and '
       'per-pixel ray casting, over pinhole, Brown-Conrady and Kannala-Brandt lenses, four crop rules and boxes, '
       'cylinders and spheres at random poses', 'each input pixel\'s centre mapped back through the crop, '
       'unprojected through the lens and ray-cast against the object\'s triangles (core/raycast.py), tokens counted '
       'with a plain loop', 0.0, 'tokens', refs=('moller1997', 'amanatides_woo1987', 'opencv_calib3d',
                                                  'opencv_fisheye'))
def _patch_counts(ctx):
    seed = int(ctx.settings.seed) % (2 ** 31)
    rows = patch_case_rows(seed, ctx.quick)
    worst = max(r['token_error'] for r in rows)
    px = sum(r['pixels_reference'] for r in rows)
    diff = sum(r['pixels_differ'] for r in rows)
    return Outcome(float(worst), {'cases': rows, 'objects': len(rows), 'pixels_reference': px,
                                  'pixels_differ': diff, 'pixels_differ_share': diff / max(px, 1),
                                  'subdivision_tolerance_px': TOL_PX},
                   note=f'{diff} of {px} covered pixels differ (silhouette edges)')


# ====================================================================== Plücker rays
def cv2_rays(cam, geom) -> tuple[np.ndarray, np.ndarray]:
    """(h, w, 3) optical rays of every input pixel by OpenCV, and (h, w) where they exist (content, in the image)."""
    import cv2
    w, h = geom.input_size
    W, H = geom.src_size
    sx0, sy0, sx1, sy1 = geom.src_rect
    dx0, dy0, dx1, dy1 = geom.dst_rect
    xi, yi = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    # the crop written out again: the content rectangle is the source rectangle, scaled
    xs = sx0 + (xi - dx0) * (sx1 - sx0) / (dx1 - dx0)
    ys = sy0 + (yi - dy0) * (sy1 - sy0) / (dy1 - dy0)
    ok = (xi > dx0) & (xi < dx1) & (yi > dy0) & (yi < dy1) & (xs >= 0) & (xs <= W) & (ys >= 0) & (ys <= H)
    pts = np.stack([xs - 0.5, ys - 0.5], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    K, D, kind = cam.to_opencv()
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-14)
    if kind == 'kb':
        n = cv2.fisheye.undistortPoints(pts, K, D.reshape(4, 1), criteria=crit).reshape(-1, 2)
    else:
        it = getattr(cv2, 'undistortPointsIter', None)      # OpenCV 4 names the variant with criteria so
        n = (it(pts, K, D, None, None, crit) if it is not None else
             cv2.undistortPoints(pts, K, D, None, None, None, crit)).reshape(-1, 2)
    r = np.c_[n, np.ones(len(n))]
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return r.reshape(h, w, 3), ok


def plucker_rows(project) -> list:
    """Every policy-shaped case on the project's cameras plus scripted lenses."""
    from ..core import library as LB
    from ..core import plucker as PL
    from ..core import scenegeom as SG
    from ..core.policies import crop_geometry
    rows = []
    sensors = [s for s in project.variant.sensors if s.kind in ('camera', 'depth')]
    cams = _cameras()
    for k, s in enumerate(sensors):
        im = PL.pick_imager(s, PL.sensor_stream(s))
        if im is None:
            continue
        variants = [('lens of the library', im.camera)] + [(n, c) for n, c in cams[1:3]]
        for vname, cam in variants:
            ci, rule, inp, cell, frac, pre = CASES[k % len(CASES)]
            g = crop_geometry(rule, (cam.width, cam.height), inp, frac, pre)
            # the sensor's imager with this lens
            im2 = LB.Imager(im.name, im.role, im.pose_in_body, cam, label=im.label)
            f = PL.ray_field(project, s, im2, geom=g, frame='base')
            # the pose composed here: robot base^-1 · sensor world pose · imager pose in the body
            Tw = SG.sensor_world_pose(project, s, None).matrix() @ im.pose_in_body.matrix()
            Tb = project.robots[0].base.matrix() if project.robots else np.eye(4)
            Tbo = np.linalg.solve(Tb, Tw)
            r_opt, ok = cv2_rays(cam, g)
            d = r_opt @ Tbo[:3, :3].T
            o = Tbo[:3, 3]
            m = np.cross(np.broadcast_to(o, d.shape), d)
            # OpenCV returns x/z, y/z: no ray at or beyond 90°, ill-conditioned close to it; compared within 85°
            near = r_opt[..., 2] > math.cos(math.radians(MAX_CMP_DEG))
            beyond = int(np.count_nonzero(ok & f.valid & ~near))
            both = ok & f.valid & near
            ed = float(np.abs(f.d[both] - d[both]).max()) if both.any() else 0.0
            em = float(np.abs(f.m[both] - m[both]).max()) if both.any() else 0.0
            dm = float(np.abs(np.einsum('ij,ij->i', f.d[both], f.m[both])).max()) if both.any() else 0.0
            P = f.origin + 2.7 * f.d[both]
            el = float(np.abs(np.cross(P, f.d[both]) - f.m[both]).max()) if both.any() else 0.0
            mask_diff = int(np.count_nonzero((ok & np.isfinite(r_opt).all(-1) & near) != (f.valid & near)))
            rows.append({'sensor': s.name, 'lens': vname, 'model': cam.model, 'rule': rule, 'input': list(inp),
                         'pixels': int(both.sum()), 'd_error': ed, 'm_error_m': em, 'm_dot_d': dm,
                         'moment_along_line_m': el, 'valid_mask_differs': mask_diff,
                         'beyond_85deg_not_compared': beyond,
                         'origin_vs_pose_m': float(np.abs(f.origin - o).max()),
                         'rotation_vs_pose': float(np.abs(f.rotation - Tbo[:3, :3]).max())})
    return rows


@check('vla.plucker_moment_cv2', 'vla', 'Plücker ray maps against rays cast by OpenCV',
       'largest difference of a ray map\'s direction d and moment m = o × d from rays computed independently, over '
       'the project\'s cameras with their library lens and with scripted Brown-Conrady and Kannala-Brandt lenses, '
       'each through a policy crop', 'the crop rectangle written out again, cv2.undistortPoints / '
       'cv2.fisheye.undistortPoints from CameraModel.to_opencv, the base-frame pose composed here, m = o × d',
       1e-6, '(unitless d, m for m)', refs=('opencv_calib3d', 'opencv_fisheye', 'plucker1865'))
def _plucker(ctx):
    rows = plucker_rows(ctx.project)
    if not rows:
        return Outcome(0.0, {'cases': []}, ok=False, note='the project has no camera to make rays for')
    worst = max(max(r['d_error'], r['m_error_m'], r['m_dot_d'], r['moment_along_line_m'], r['origin_vs_pose_m'],
                    r['rotation_vs_pose']) for r in rows)
    masks = sum(r['valid_mask_differs'] for r in rows)
    return Outcome(worst, {'cases': rows, 'pixels': sum(r['pixels'] for r in rows)}, ok=masks == 0,
                   note='' if masks == 0 else f'{masks} pixels valid in one and not the other')


# ====================================================================== safetensors
def parse_safetensors(path) -> tuple[dict, dict, dict]:
    """(tensors, metadata, facts) read with the public format only (module docstring)."""
    with open(path, 'rb') as fh:
        data = fh.read()
    n = struct.unpack('<Q', data[:8])[0]
    head = data[8:8 + n]
    facts = {'header_bytes': n, 'header_aligned_8': n % 8 == 0, 'starts_with_brace': head[:1] == b'{',
             'file_bytes': len(data)}
    hdr = json.loads(head.decode('utf-8'))
    meta = hdr.pop('__metadata__', {})
    names = {'F16': '<f2', 'F32': '<f4', 'F64': '<f8', 'BOOL': '|b1', 'U8': '|u1', 'I32': '<i4', 'I64': '<i8',
             'I8': '|i1', 'I16': '<i2', 'U16': '<u2', 'U32': '<u4', 'U64': '<u8'}
    body = data[8 + n:]
    out, spans = {}, []
    for k, t in hdr.items():
        b, e = t['data_offsets']
        out[k] = np.frombuffer(body[b:e], dtype=np.dtype(names[t['dtype']])).reshape(t['shape'])
        spans.append((b, e))
    spans.sort()
    holes = spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])) or spans[-1][1] != len(body)
    facts['no_holes'] = not holes
    return out, meta, facts


@check('vla.safetensors_roundtrip', 'vla', 'safetensors files written without the safetensors package',
       'largest difference between the tensors written and those read back by an independent parser (and the '
       'safetensors package when installed); header rules checked', 'a parser written from the public format '
       '(header length, JSON header, data offsets)', 0.0, 'tensor value', refs=('safetensors',))
def _safetensors(ctx):
    from ..core import plucker as PL
    rng = np.random.default_rng(int(ctx.settings.seed) % (2 ** 31))
    tens = {'plucker': rng.normal(size=(7, 5, 6)).astype(np.float16), 'f32': rng.normal(size=(3, 4)).astype(
        np.float32), 'f64': rng.normal(size=(3, 3)), 'valid': rng.random((7, 5)) > 0.5,
        'i32': rng.integers(-9, 9, 5).astype(np.int32), 'i64': rng.integers(-9, 9, (2, 2)).astype(np.int64),
        'u8': rng.integers(0, 255, 3).astype(np.uint8), 'scalar': np.array(1.5, np.float32),
        'empty': np.zeros((0, 3), np.float32)}
    path = os.path.join(ctx.tmp(), 'roundtrip.safetensors')
    PL.write_safetensors(path, tens, {'schema': PL.SCHEMA, 'note': 'Plücker ✓'})
    got, meta, facts = parse_safetensors(path)
    own, meta2 = PL.read_safetensors(path)
    err = 0.0
    same = True
    for k, a in tens.items():
        for b in (got[k], own[k]):
            same &= b.shape == a.shape and b.dtype == a.dtype and a.tobytes() == np.ascontiguousarray(b).tobytes()
            if a.size and a.dtype != bool:
                err = max(err, float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))))
    pkg = None
    try:
        from safetensors.numpy import load_file
        pk = load_file(path)
        pkg = all(np.array_equal(pk[k], tens[k]) for k in tens)
    except ImportError:
        pass
    ok = same and facts['header_aligned_8'] and facts['starts_with_brace'] and facts['no_holes'] and \
        meta.get('note') == 'Plücker ✓' and meta2 == meta and pkg is not False
    return Outcome(err, dict(facts, tensors=len(tens), package_checked=pkg is not None,
                             package_agrees=pkg), ok=ok,
                   note='' if pkg is not None else 'the safetensors package is not installed: the parser here only')
