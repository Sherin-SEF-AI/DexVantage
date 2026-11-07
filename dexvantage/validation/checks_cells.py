"""Checks of the multi-cell comparison (core/cells.py, core/cells_pool.py) against references written out here.

- cells.gap_identical: a cell against an exact copy of itself (the SO-100 sample: a wrist camera in the contact frame
  and a scene camera; a synthetic three-camera rig without a robot) gives a gap index of 0 and every component 0.
- cells.gap_rotation: one camera turned by a known angle θ about its own optical centre (a SciPy rotation vector,
  independent of core/frames; on a wrist camera through its link mount and on a scene camera in the world): the
  reported rotation between the matched optical frames is θ, the viewpoint angle is θ for an axis perpendicular to
  the optical axis (and 0 for a turn about the optical axis itself), the distance 0.
- cells.gap_shift: one camera moved by a known vector d (no turn): the matched distance is |d|, in the world for a
  scene camera and in the contact frame for a wrist camera (rigid with the link, so the same |d|).
- cells.coverage_polygon: the share of a pinhole image a box fills (rays through a pixel grid, slab-tested) against
  the exact share: the convex hull of the eight projected corners (u = f_x x/z + c_x, written out; Andrew's monotone
  chain), clipped to the image rectangle (Sutherland and Hodgman 1974) and measured by the shoelace formula, for
  random boxes wholly in front of the camera.
- cells.pooling_lerobot: two synthetic LeRobot datasets with known differences written by core/datasets/lerobot's
  writer (v3.0 and v2.1: another key and size for the wrist camera, another frame rate, the joints in another
  order, the gripper over half the range) and a third with another joint count, read back through the LeRobot
  reader: the pooling rules find exactly the expected mapping steps and the unmappable joint count, and the schema's
  camera sizes and rates equal the frames the reader decodes and the spacing of their timestamps.
- cells.drift_known: copies of a design with known mount offsets, as hand-eye results: the offsets are recovered
  exactly, the median and robust spread (1.4826 · MAD) and the outlier's robust z equal the hand computation, and a
  build offset shared by every copy is called systematic.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check

LR_KEY_A = 'observation.images.wrist'
LR_KEY_B = 'observation.images.hand'
LR_TOP = 'observation.images.top'


# ====================================================================== synthetic rigs
def synthetic_rig(name='Synthetic rig'):
    """A project with no robot: a task volume and three static D435 cameras looking at it."""
    from ..core.project import Mount, Sensor, Volume, blank_project
    from ..core.robot.samples import look_at
    p = blank_project(name)
    p.workspace.volumes = [Volume(name='Table', center=(0.5, 0.0, 0.1), size=(0.4, 0.5, 0.2))]
    for k, (pos, tgt) in enumerate((((1.2, 0.0, 0.8), (0.5, 0.0, 0.1)), ((0.5, 0.9, 0.6), (0.5, 0.0, 0.1)),
                                    ((0.5, -0.8, 0.9), (0.45, 0.05, 0.1)))):
        p.variant.sensors.append(Sensor(name=f'Cam {k + 1}', kind='depth', model_id='intel-realsense-d435',
                                        mount=Mount('cage', '', look_at(pos, tgt))))
    return p


def turn_camera(project, sensor_name: str, rotvec, shift=(0.0, 0.0, 0.0)):
    """A copy of project with the named sensor's optical frame turned by rotvec (rad, about its own optical axes, at
    its centre) and moved by shift (m, in its mount frame)."""
    from scipy.spatial.transform import Rotation

    from ..core.calib import records as RC
    from ..core.frames import Pose
    q = project.snapshot()
    s = next(s for s in q.variant.sensors if s.name == sensor_name)
    im = RC.sensor_imager(s)
    Tbo = im.pose_in_body.matrix()
    X = s.mount.pose.matrix() @ Tbo
    R = np.eye(4)
    R[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    X2 = X @ R
    X2[:3, 3] += np.asarray(shift, float)
    s.mount.pose = Pose.from_matrix(X2 @ np.linalg.inv(Tbo))
    return q


def _gap(a, b):
    from ..core import cells as CE
    A, _ = CE.cameras(a)
    B, _ = CE.cameras(b)
    return CE.gap(A, B, 'a', 'b')


@check('cells.gap_identical', 'cells', 'Cells: the gap of a cell against an exact copy of itself',
       'largest gap index or component of a cell against its own copy (a sample with a wrist and a scene camera, a '
       'synthetic three-camera rig)', 'zero by construction: the same cameras in the same frames', 1e-12, 'index',
       refs=('vantage_rig_gap',))
def _identical(ctx):
    from ..core.robot.samples import sample_project
    worst, rows = 0.0, {}
    for name, p in (('so100', sample_project('so100')), ('synthetic', synthetic_rig())):
        g = _gap(p, p.snapshot())
        e = max([g['total']] + list(g['components'].values()))
        rows[name] = {'total': g['total'], 'matches': len(g['matches'])}
        worst = max(worst, e)
    return Outcome(worst, rows, ok=all(r['matches'] >= 2 for r in rows.values()))


@check('cells.gap_rotation', 'cells', 'Cells: a camera turned by a known angle about its centre',
       'largest |reported − true| of the rotation between the matched optical frames, of the viewpoint angle (θ for '
       'an axis perpendicular to the optical axis, 0 about it) and of the distance (0, in mm)',
       'the rotation vector\'s magnitude θ (SciPy Rotation, independent of core/frames)', 1e-9, '°',
       refs=('scipy_rotation', 'vantage_rig_gap'))
def _rotation(ctx):
    from ..core.robot.samples import sample_project
    rng = ctx.rng('cells.gap_rotation')
    base = {'so100': sample_project('so100'), 'synthetic': synthetic_rig()}
    cams = (('so100', 'Wrist camera (sample placement)'), ('so100', 'Third-person camera (sample placement)'),
            ('synthetic', 'Cam 2'))
    worst, rows = 0.0, []
    n = 3 if ctx.quick else 12
    for k in range(n):
        which, cam = cams[k % len(cams)]
        th = float(rng.uniform(math.radians(1.0), math.radians(60.0)))
        kind = ('perpendicular', 'about the axis', 'general')[k % 3]
        if kind == 'perpendicular':
            a = rng.normal(size=2)
            axis = np.r_[a / np.linalg.norm(a), 0.0]
        elif kind == 'about the axis':
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
        p = base[which]
        g = _gap(p, turn_camera(p, cam, axis * th))
        m = next(x for x in g['matches'] if x['a'] == cam)
        true = math.degrees(th)
        e_rot = abs(m['rotation_deg'] - true)
        e_view = abs(m['viewpoint_deg'] - (true if kind == 'perpendicular' else 0.0)) if kind != 'general' else 0.0
        e_d = m['distance_m'] * 1000.0
        rows.append({'cell': which, 'camera': cam, 'kind': kind, 'theta_deg': true, 'rotation_deg': m['rotation_deg'],
                     'viewpoint_deg': m['viewpoint_deg'], 'distance_mm': e_d})
        worst = max(worst, e_rot, e_view, e_d)
    return Outcome(worst, {'cases': rows})


@check('cells.gap_shift', 'cells', 'Cells: a camera moved by a known vector',
       'largest |reported − true| distance between the matched camera centres (in the world for a scene camera, in '
       'the contact frame for a wrist camera)', 'the length of the shift |d| (the wrist camera is rigid with its link, '
       'so its distance in the contact frame is |d| too)', 1e-9, 'mm', refs=('vantage_rig_gap',))
def _shift(ctx):
    from ..core.robot.samples import sample_project
    rng = ctx.rng('cells.gap_shift')
    p = sample_project('so100')
    worst, rows = 0.0, []
    for k in range(2 if ctx.quick else 8):
        cam = ('Wrist camera (sample placement)', 'Third-person camera (sample placement)')[k % 2]
        d = rng.normal(size=3)
        d *= float(rng.uniform(0.005, 0.2)) / np.linalg.norm(d)
        g = _gap(p, turn_camera(p, cam, np.zeros(3), d))
        m = next(x for x in g['matches'] if x['a'] == cam)
        e = abs(m['distance_m'] - float(np.linalg.norm(d))) * 1000.0
        rows.append({'camera': cam, 'shift_mm': float(np.linalg.norm(d)) * 1000, 'reported_mm': m['distance_m'] * 1000})
        worst = max(worst, e)
    return Outcome(worst, {'cases': rows})


# ====================================================================== the image share of a box
def _hull(P):
    """Andrew's monotone chain: the convex hull of 2-D points, counter-clockwise."""
    P = sorted(map(tuple, np.asarray(P, float)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _clip(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clipping of a polygon to the rectangle [x0, x1] × [y0, y1]."""
    def clip(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            if inside(b):
                if not inside(a):
                    out.append(inter(a, b))
                out.append(b)
            elif inside(a):
                out.append(inter(a, b))
        return out

    def ix(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def iy(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)
    pts = list(poly)
    for inside, inter in ((lambda p: p[0] >= x0, ix(x0)), (lambda p: p[0] <= x1, ix(x1)),
                          (lambda p: p[1] >= y0, iy(y0)), (lambda p: p[1] <= y1, iy(y1))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return pts


def _area(pts):
    if len(pts) < 3:
        return 0.0
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@check('cells.coverage_polygon', 'cells', 'Cells: the image share of a task volume against its projected polygon',
       'largest |grid share − exact share| of a pinhole image a box fills, over random boxes wholly in front of the '
       'camera', 'the convex hull of the projected corners (pinhole projection written out, monotone chain) clipped '
       'to the image (Sutherland-Hodgman) and its shoelace area over the image area; the grid samples at most 128 '
       'columns, so the boundary cells allow about 1-2 % of a share', 0.02, 'share',
       refs=('sutherland_hodgman1974', 'andrew1979', 'kay_kajiya1986'))
def _coverage(ctx):
    from ..core import cells as CE
    from ..core.optics import CameraModel
    rng = ctx.rng('cells.coverage_polygon')
    worst, rows = 0.0, []
    for k in range(6 if ctx.quick else 40):
        W, H = int(rng.choice([320, 640, 160])), int(rng.choice([240, 480, 120]))
        f = float(rng.uniform(0.5, 1.5)) * W
        cam = CameraModel(W, H, f, f, (W - 1) / 2 + rng.uniform(-10, 10), (H - 1) / 2 + rng.uniform(-10, 10))
        c = np.r_[rng.uniform(-0.3, 0.3, 2), rng.uniform(0.8, 2.0)]
        half = rng.uniform(0.05, 0.4, 3)
        half[2] = min(half[2], c[2] - 0.3)                 # wholly in front of the camera
        lo, hi = c - half, c + half
        share = CE.task_coverage(cam, np.eye(4), [(lo, hi)])['image_pct'] / 100.0
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uv = np.c_[f * corners[:, 0] / corners[:, 2] + cam.cx, f * corners[:, 1] / corners[:, 2] + cam.cy]
        exact = _area(_clip(_hull(uv), -0.5, -0.5, W - 0.5, H - 0.5)) / (W * H)
        e = abs(share - exact)
        rows.append({'size': [W, H], 'exact': exact, 'grid': share})
        worst = max(worst, e)
    return Outcome(worst, {'cases': rows[:12], 'n': len(rows)})


# ====================================================================== pooling against the LeRobot readers
def lerobot_episode(key_wrist, size_wrist, fps, names, n=8, seed=0, gripper=(0.0, 0.08)):
    """A synthetic Episode for core/datasets/lerobot.write_lerobot: a wrist camera (key_wrist, (w, h)), a top camera
    64 × 48, the joints names (radians; the joint named gripper runs over exactly the range gripper), the actions the
    same joints."""
    from ..core.datasets import episode as E
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps

    def cam(key, w, h):
        frames = [np.full((h, w, 3), (20 * i) % 255, np.uint8) for i in range(n)]

        def dec(idx):
            for i in (range(n) if idx is None else idx):
                yield i, frames[i]
        return E.CameraStream(key, w, h, float(fps), t.copy(), _decode=dec)
    q = rng.uniform(-1, 1, (n, len(names)))
    if 'gripper' in names:
        q[:, names.index('gripper')] = np.linspace(gripper[0], gripper[1], n)
    return E.Episode('e0', 0, t, q, list(names), action=q.copy(), action_names=list(names), fps=float(fps),
                     task='pick', cameras={key_wrist: cam(key_wrist, *size_wrist), LR_TOP: cam(LR_TOP, 64, 48)})


def write_pooling_datasets(folder: str) -> dict:
    """Three small LeRobot datasets with known differences (module docstring); {'a', 'b', 'c': path}."""
    import os

    from ..core.datasets import lerobot as LR
    j = ['j1', 'j2', 'gripper']
    a = os.path.join(folder, 'pool_a')
    b = os.path.join(folder, 'pool_b')
    c = os.path.join(folder, 'pool_c')
    LR.write_lerobot(a, [lerobot_episode(LR_KEY_A, (64, 48), 10, j)], 'v3.0', 10, 'arm')
    LR.write_lerobot(b, [lerobot_episode(LR_KEY_B, (80, 48), 5, [j[2], j[0], j[1]], seed=1, gripper=(0.0, 0.04))],
                     'v2.1', 5, 'arm')
    LR.write_lerobot(c, [lerobot_episode(LR_KEY_A, (64, 48), 10, j + ['j4'], seed=2)], 'v3.0', 10, 'arm')
    return {'a': a, 'b': b, 'c': c}


EXPECTED_B = {('rename', LR_KEY_B), ('resize', LR_KEY_B), ('resample', LR_KEY_B), ('resample', LR_TOP),
              ('resample', 'rows'), ('reorder', 'observation.state'), ('reorder', 'action'), ('resample', 'actions'),
              ('rescale', 'gripper')}


@check('cells.pooling_lerobot', 'cells', 'Cells: pooling rules on synthetic LeRobot datasets with known differences',
       'mapping steps missing or extra against the known differences, plus schema figures (camera sizes, rates) '
       'differing from the frames the LeRobot reader decodes', 'the differences the datasets were written with '
       '(another wrist key and size, 5 against 10 fps, the joints reordered, a gripper over half the range; a fourth '
       'joint), read back through '
       'core/datasets/lerobot (v3.0 and v2.1)', 0, 'count', refs=('lerobot_dataset_format',), cost='study')
def _pooling(ctx):
    import tempfile

    from ..core import cells_pool as CP
    from ..core.datasets import open_dataset
    try:
        import av  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        return Outcome(float('nan'), {}, ok=False, note='pyarrow and av are needed (the lerobot extra)')
    folder = tempfile.mkdtemp(prefix='dexvantage-cells-check-')
    try:
        paths = write_pooling_datasets(folder)
        sc = {k: CP.recorded_schema(v) for k, v in paths.items()}
        pool = CP.pooling([sc['a'], sc['b'], sc['c']], ['a', 'b', 'c'], 0)
        got = {(o['op'], o['feature']) for o in pool['cells'][1]['operations']}
        errors = len(got ^ EXPECTED_B)
        reorder = next((o for o in pool['cells'][1]['operations'] if o['op'] == 'reorder'), None)
        errors += 0 if reorder and reorder['index'] == [1, 2, 0] else 1
        rs = next((o for o in pool['cells'][1]['operations'] if o['op'] == 'rescale'), None)
        errors += 0 if rs and np.allclose(rs['from_range'], [0, 0.04]) and np.allclose(rs['to_range'], [0, 0.08]) \
            else 1
        bad_c = {u['what'] for u in pool['cells'][2]['unmappable']}
        errors += 0 if bad_c == {'observation.state', 'action'} else 1
        errors += 0 if pool['cells'][1]['verdict'] == 'poolable after mapping' else 1
        decoded = {}
        for k, p in paths.items():
            with open_dataset(p) as ds:
                e = ds.load(ds.episodes()[0].index)
                for key, cs in e.cameras.items():
                    i, img = next(iter(cs.frames([0])))
                    rate = 1.0 / float(np.median(np.diff(cs.timestamps)))
                    want = next(c for c in sc[k]['cameras'] if c['key'] == key)
                    decoded[f'{k}:{key}'] = {'frame': [img.shape[1], img.shape[0]], 'rate': rate,
                                             'schema': [want['width'], want['height'], want['fps']]}
                    errors += int([img.shape[1], img.shape[0]] != [want['width'], want['height']])
                    errors += int(abs(rate - float(want['fps'])) > 1e-6)
        return Outcome(float(errors), {'operations_b': sorted(map(list, got)), 'expected_b': sorted(map(list,
                                                                                                        EXPECTED_B)),
                                       'unmappable_c': sorted(bad_c), 'decoded': decoded})
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)


# ====================================================================== drift
def calibrated_copy(project, sensor_name: str, offset_mm=(0, 0, 0), rot_deg=(0, 0, 0), name='copy',
                    date='2026-11-01T00:00:00Z'):
    """A copy of project whose sensor has a solved hand-eye session placing its optical frame at the plan's plus a
    known offset (mm, in the mount frame) and turn (a rotation vector, degrees, about the mount frame's axes)."""
    from scipy.spatial.transform import Rotation

    from ..core import cells as CE
    q = project.snapshot()
    q.name = name
    s = next(s for s in q.variant.sensors if s.name == sensor_name)
    (_r, _l, X), _im = CE._mount_X(q, s)
    Xm = X.copy()
    Xm[:3, :3] = Rotation.from_rotvec(np.radians(rot_deg)).as_matrix() @ X[:3, :3]
    Xm[:3, 3] = X[:3, 3] + np.asarray(offset_mm, float) / 1000.0
    q.calibration['sessions'].append({'uid': f'{name}-cal', 'name': f'{name} calibration', 'sensor_uid': s.uid,
                                      'created': date, 'mode': 'eye_in_hand',
                                      'result': {'ok': True, 'cameras': {s.name: {'matrix': Xm.tolist(),
                                                                                  'sigma': None}}}})
    return q


@check('cells.drift_known', 'cells', 'Cells: drift of built copies with known offsets',
       'largest error of the recovered offsets (mm and degrees), the median and robust spread and the outlier\'s '
       'robust z against the hand computation; a missed systematic offset counts 1',
       'the offsets the copies were built with; median, 1.4826 · median absolute deviation (floored at a quarter of '
       'the tolerance) and z = (x − median) / spread written out (Hampel 1974, as Vantage core/fleet.py)', 1e-9,
       'mm or °', refs=('hampel1974', 'scipy_rotation'))
def _drift(ctx):
    from ..core import cells as CE
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    cam = 'Wrist D405'
    xs = [0.10, -0.20, 0.05, 2.30, 0.00]                 # mm along x: the fourth copy is the outlier
    copies = [calibrated_copy(p, cam, (x, 0, 0), (0, 0.01 * k, 0), name=f'copy {k + 1}') for k, x in enumerate(xs)]
    cells = CE.make_cells(copies, open_project=p)
    dr = CE.drift(cells, 0)
    row = next(r for r in dr['cameras'] if r['sensor'] == cam)
    err = 0.0
    for k, x in enumerate(xs):
        e = next(e for e in row['copies'] if e['name'] == f'copy {k + 1}')
        err = max(err, abs(e['offset'][0] - x), abs(e['offset'][4] - 0.01 * k), *map(abs, e['offset'][1:4]),
                  abs(e['offset'][5]))
    c = next(c for c in dr['consistency'] if c['sensor'] == cam)
    med = float(np.median(xs))
    sig = 1.4826 * float(np.median(np.abs(np.array(xs) - med)))
    spread = max(sig, CE.SIGMA_FLOOR * 1.0)
    err = max(err, abs(c['median'][0] - med), abs(c['sigma'][0] - sig))
    out = c['outliers'][0] if c['outliers'] else None
    err = max(err, abs(out['z'] - (2.30 - med) / spread) if out and out['name'] == 'copy 4' else 1.0)
    # every copy built 3 mm high: systematic
    sys_cells = CE.make_cells([calibrated_copy(p, cam, (0, 0, 3.0 + 0.1 * k), name=f'high {k}') for k in range(4)],
                              open_project=p)
    sysd = CE.drift(sys_cells, 0)
    missed = 0 if cam in sysd['systematic'] else 1
    return Outcome(max(err, float(missed)), {'median_mm': c['median'][0], 'sigma_mm': c['sigma'][0],
                                             'outlier': out, 'systematic': sysd['systematic']})
