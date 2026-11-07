"""Checks of the workspace check (core/workspace_check.py, the Workspace page) against closed forms written out here.

- workspace.object_extent: the world bounding box of a turned box, cylinder and sphere (from core/scenegeom's
  triangles) against the exact box of the true shape: a box's half-extents |R| h (Arvo 1990), a cylinder's
  r sqrt(1 − a_i²) + (h/2) |a_i| per world axis i (a the cylinder's axis in the world: its disc caps' extent plus
  half its length along that axis), a sphere's r. Rotations are drawn by SciPy (not core/frames).
- workspace.inside_classification: random objects placed clearly inside or clearly outside random task volumes (at
  least 3 mm, beyond the tessellation bound and the 1 mm tolerance, from every face) against the exact
  classification of the true shapes' boxes.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check

TESS = 1.0 - math.cos(math.pi / 24) ** 2     # relative: a 24-gon around and 15° rings pole to pole inside the circle
SHAPES = ('box', 'cylinder', 'sphere')


def _objects(rng, n):
    from scipy.spatial.transform import Rotation

    from ..core.frames import Pose
    from ..core.project import TaskObject
    out = []
    for k in range(n):
        shape = SHAPES[k % 3]
        q = Rotation.random(random_state=int(rng.integers(2 ** 31))).as_quat()    # (x, y, z, w)
        if shape == 'box':
            size = tuple(rng.uniform(0.01, 0.12, 3))
        elif shape == 'cylinder':
            size = (float(rng.uniform(0.005, 0.06)), float(rng.uniform(0.01, 0.2)), 0.0)
        else:
            size = (float(rng.uniform(0.005, 0.06)), 0.0, 0.0)
        out.append(TaskObject(name=f'{shape} {k}', shape=shape, size=size,
                              pose=Pose(rng.uniform(-0.5, 0.5, 3), q)))
    return out


def _exact_half(obj) -> np.ndarray:
    """The true shape's world half-extents (closed forms in the module docstring)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(obj.pose.quaternion).as_matrix()
    s = np.asarray(obj.size, float)
    if obj.shape == 'box':
        return np.abs(R) @ (s / 2)
    if obj.shape == 'cylinder':
        a = R[:, 2]
        return s[0] * np.sqrt(np.clip(1.0 - a ** 2, 0.0, 1.0)) + s[1] / 2 * np.abs(a)
    return np.full(3, s[0])


def _bound(obj) -> float:
    """How far (m) the tessellated shape's box may fall short of the true box: 0 for a box (its corners are exact),
    r (1 − cos²(π/24)) for a cylinder and a sphere."""
    return 1e-12 if obj.shape == 'box' else float(obj.size[0]) * TESS + 1e-12


@check('workspace.object_extent', 'workspace', 'Workspace: the bounding box of a turned object against closed forms',
       'largest shortfall or overshoot of an object\'s world bounding box (from its triangles) against the true '
       'shape\'s, as a share of the bound the tessellation allows (0 for a box, r (1 − cos²(π/24)) for a cylinder or '
       'a sphere)', 'the box of a turned box |R| h (Arvo 1990); of a cylinder r sqrt(1 − a_i²) + (h/2)|a_i|; of a '
       'sphere r (written out here); rotations from SciPy', 1.0, 'of the bound', refs=('arvo1990', 'scipy_rotation'))
def _object_extent(ctx):
    from ..core.workspace_check import object_extent
    rng = ctx.rng('workspace.object_extent')
    objs = _objects(rng, 30 if ctx.quick else 300)
    worst, worst_mm, by = 0.0, 0.0, {}
    for o in objs:
        lo, hi = object_extent(o)
        c, h = np.asarray(o.pose.position, float), _exact_half(o)
        short = np.r_[(c - h) - lo, hi - (c + h)]          # > 0: the triangles stick out of the true box
        under = np.r_[lo - (c - h), (c + h) - hi]          # > 0: the triangles fall short of it
        over = float(max(0.0, short.max()))
        gap = float(max(0.0, under.max()))
        e = max(over / 1e-12 if over > 1e-12 else 0.0, gap / _bound(o))
        by.setdefault(o.shape, 0.0)
        by[o.shape] = max(by[o.shape], gap * 1000.0)
        worst = max(worst, e)
        worst_mm = max(worst_mm, max(over, gap) * 1000.0)
    return Outcome(worst, {'objects': len(objs), 'largest_difference_mm': worst_mm,
                           'largest_shortfall_mm_by_shape': by, 'tessellation_bound_relative': TESS},
                   note='The tessellated cylinder and sphere lie inside the true shapes, so their boxes may fall short '
                        f'by up to {TESS * 100:.2f} % of the radius and never stick out; a box is exact to rounding. '
                        'A wrong rotation or a size read as a diameter shows as many times the bound.')


def _within(lo, hi, vol, m) -> bool:
    """The box lo..hi inside the volume grown by m (shrunk for m < 0)."""
    vlo, vhi = vol.bounds()
    return bool((lo >= vlo - m).all() and (hi <= vhi + m).all())


@check('workspace.inside_classification', 'workspace', 'Workspace: objects inside or outside task volumes',
       'objects classified inside or outside every task volume differently from the exact classification of the true '
       'shape\'s box (objects at least 3 mm from every face)', 'exact boxes of the true shapes (closed forms above) '
       'against the volumes\' boxes', 0.0, 'objects', refs=('arvo1990',))
def _inside(ctx):
    from ..core.project import Volume, blank_project
    from ..core.workspace_check import check as wcheck
    rng = ctx.rng('workspace.inside_classification')
    p = blank_project('Workspace check')
    p.workspace.volumes = [Volume(name=f'V{k}', center=tuple(rng.uniform(-0.6, 0.6, 3)),
                                  size=tuple(rng.uniform(0.15, 0.6, 3))) for k in range(3)]
    margin = 0.003
    want, objs, tries = {}, [], 0
    n = 30 if ctx.quick else 240
    while len(objs) < n and tries < 200 * n:
        tries += 1
        o = _objects(rng, 1)[0]
        o.name = f'{o.shape} {len(objs)}'
        h = _exact_half(o)
        if len(objs) % 2 == 0:                           # every other one aimed inside a volume (where it fits)
            v = p.workspace.volumes[int(rng.integers(len(p.workspace.volumes)))]
            vlo, vhi = v.bounds()
            room = (vhi - vlo) / 2 - h
            if (room > 0).all():
                o.pose.position = np.asarray(v.center, float) + rng.uniform(-room, room)
        c = np.asarray(o.pose.position, float)
        lo, hi = c - h, c + h
        strict = [_within(lo, hi, v, -margin) for v in p.workspace.volumes]
        loose = [_within(lo, hi, v, margin) for v in p.workspace.volumes]
        if any(strict):
            want[o.uid] = True
        elif not any(loose):
            want[o.uid] = False
        else:
            continue                                     # within 3 mm of a face: ambiguous by construction
        objs.append(o)
    p.workspace.objects = objs
    doc = wcheck(p)
    got_out = {x['uid'] for x in doc['outside']}
    wrong = sum((uid in got_out) == inside for uid, inside in want.items())
    return Outcome(float(wrong), {'objects': len(objs), 'inside': sum(want.values()),
                                  'outside': len(want) - sum(want.values()), 'tries': tries},
                   ok=0 < sum(want.values()) < len(want),
                   note='Every object is clear of each face by 3 mm (more than the 1 mm tolerance plus the 1.7 % '
                        'tessellation shortfall of a 60 mm radius), so the classification is unambiguous: any '
                        'disagreement is a fault.')
