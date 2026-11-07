"""The workspace check (spec §13, the Workspace page): does every task object lie inside a task volume, and the
task volume that would hold the ones that do not. No Qt; the page is ui/pages/workspace.py.

Rule. An object counts as inside a volume when its axis-aligned world bounding box (of the triangles
core/scenegeom.object_triangles makes of it: a box's 12, a cylinder's and a sphere's tessellation, a mesh object's
mesh) lies inside the volume's box within INSIDE_TOL_M (1 mm). The studies measure inside the task volumes (coverage,
depth validity, grasp sampling on support surfaces, VLA object tokens), so an object outside every volume is not
measured. The suggestion: with objects outside, the volume nearest to them (centre to centre) grows to the union of
its box and theirs with MARGIN_M (20 mm, a DexVantage default) of clearance; with no volume, a new one around every
object, or the default tabletop volume when there are no objects. The margin stops at the floor (z = 0) when the
objects stand on or above it.

Known limits: volumes are axis-aligned boxes (core/project.py Volume), so a suggested volume around a turned object
is larger than the object; the tessellated cylinder and sphere lie inside the true shapes, so their boxes are smaller
than the true boxes by at most r (1 − cos² (π/24)) (1.7 % of the radius: validation workspace.object_extent); an
object whose geometry cannot be read (an empty mesh) is not checked, and the advice says so.
"""
from __future__ import annotations

import traceback

import numpy as np

from . import project as P
from .frames import quat_to_matrix

MARGIN_M = 0.02          # a DexVantage default: the clearance the suggested volume leaves around the objects
INSIDE_TOL_M = 0.001     # an object's box may stick out of a volume by this much and still count as inside


def box_edges(lo, hi, R=None, c=None) -> np.ndarray:
    """(12, 2, 3) segments of a box: axis-aligned lo..hi, or the box of half-extents (hi - lo) / 2 turned by R about
    its centre c."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    if R is not None:
        mid = (lo + hi) / 2
        corners = (corners - mid) @ np.asarray(R, float).T + np.asarray(c if c is not None else mid, float)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.array([[corners[a], corners[b]] for a, b in idx])


def object_extent(obj, project=None):
    """(lo, hi) of a task object's axis-aligned world bounding box (of its triangles), or None (an empty mesh)."""
    from .scenegeom import object_triangles
    tri = np.asarray(object_triangles(obj, project), float).reshape(-1, 3)
    if not len(tri) or not np.isfinite(tri).all():
        return None
    return tri.min(axis=0), tri.max(axis=0)


def object_outline(obj, project=None) -> np.ndarray | None:
    """(12, 2, 3) segments of an object's oriented bounding box in the world (its own frame's box)."""
    from .scenegeom import object_triangles
    R = quat_to_matrix(obj.pose.quaternion)
    tri = np.asarray(object_triangles(obj, project), float).reshape(-1, 3)
    if not len(tri):
        return None
    local = (tri - np.asarray(obj.pose.position, float)) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    return box_edges(lo, hi, R, R @ ((lo + hi) / 2) + np.asarray(obj.pose.position, float))


def surface_outline(s: dict) -> np.ndarray | None:
    """(4, 2, 3) segments of a surface's top rectangle (at its z)."""
    sz = np.asarray(s.get('size') or (0.0, 0.0), float).reshape(-1)[:2]
    if not (sz > 0).all():
        return None
    c = np.asarray(s.get('center') or (0.0, 0.0), float).reshape(-1)[:2]
    z = float(s.get('z', 0.0))
    x0, y0 = c - sz / 2
    x1, y1 = c + sz / 2
    pts = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    return np.array([[pts[i], pts[(i + 1) % 4]] for i in range(4)])


def inside(lo, hi, vol, tol: float = INSIDE_TOL_M) -> bool:
    vlo, vhi = vol.bounds()
    return bool((np.asarray(lo) >= vlo - tol).all() and (np.asarray(hi) <= vhi + tol).all())


def check(project) -> dict:
    """The workspace check (module docstring): {'workspace' (Workspace.to_dict()), 'volumes', 'objects',
    'surfaces', 'lights', 'outside': [{'uid', 'name'}], 'no_extent': [names], 'status', 'word', 'advice',
    'suggestion': None | {'kind': 'grow' | 'add', 'volume_uid'?, 'name', 'center', 'size', 'text'}} (JSON-able)."""
    w = project.workspace
    outside, no_ext, boxes = [], [], []
    for o in w.objects:
        try:
            ext = object_extent(o, project)
        except Exception:                               # noqa: BLE001 — a broken mesh: said, not fatal
            traceback.print_exc()
            ext = None
        if ext is None:
            no_ext.append(o.name)
            continue
        if not any(inside(*ext, v) for v in w.volumes):
            outside.append({'uid': o.uid, 'name': o.name})
            boxes.append(ext)
    doc = {'workspace': w.to_dict(), 'volumes': len(w.volumes), 'objects': len(w.objects),
           'surfaces': len(w.surfaces), 'lights': len(w.lights), 'outside': outside, 'no_extent': no_ext,
           'margin_m': MARGIN_M, 'margin_default': True, 'suggestion': None}
    if not w.volumes:
        doc.update(status='warning', word='no task volume',
                   advice='Add a task volume: the coverage, depth, grasp and VLA studies measure inside it.')
        if w.objects:
            allb = [object_extent(o, project) for o in w.objects]
            allb = [b for b in allb if b is not None]
        else:
            allb = []
        if allb:
            lo = np.min([b[0] for b in allb], axis=0) - MARGIN_M
            hi = np.max([b[1] for b in allb], axis=0) + MARGIN_M
            if lo[2] + MARGIN_M >= 0:                   # objects on or above the floor: the volume stops at it
                lo[2] = max(0.0, lo[2])
            doc['suggestion'] = {'kind': 'add', 'name': 'Task volume', 'center': ((lo + hi) / 2).tolist(),
                                 'size': (hi - lo).tolist(),
                                 'text': f'add a task volume around the {len(allb)} object'
                                         f'{"s" if len(allb) != 1 else ""} with a {MARGIN_M * 1000:.0f} mm margin '
                                         '(a default)'}
        else:
            v = P.Volume()
            doc['suggestion'] = {'kind': 'add', 'name': v.name, 'center': list(v.center), 'size': list(v.size),
                                 'text': 'add the default tabletop volume (500 × 600 × 300 mm, 0.45 m in front of the '
                                         'world origin: a DexVantage default; move it to your table)'}
    elif outside:
        lo = np.min([b[0] for b in boxes], axis=0) - MARGIN_M
        hi = np.max([b[1] for b in boxes], axis=0) + MARGIN_M
        if lo[2] + MARGIN_M >= 0:                       # objects on or above the floor: the margin stops at it
            lo[2] = max(0.0, lo[2])
        mid = (lo + hi) / 2
        vol = min(w.volumes, key=lambda v: float(np.linalg.norm(np.asarray(v.center, float) - mid)))
        vlo, vhi = vol.bounds()
        nlo, nhi = np.minimum(vlo, lo), np.maximum(vhi, hi)
        names = ', '.join(o['name'] for o in outside[:3]) + (' …' if len(outside) > 3 else '')
        doc.update(status='warning', word='objects outside',
                   advice=f'{len(outside)} of {len(w.objects)} objects lie outside every task volume ({names}): the '
                          'studies do not measure them. Apply grows the nearest volume to hold them.')
        doc['suggestion'] = {'kind': 'grow', 'volume_uid': vol.uid, 'name': vol.name,
                             'center': ((nlo + nhi) / 2).tolist(), 'size': (nhi - nlo).tolist(),
                             'text': f'grow {vol.name} to hold {names} with a {MARGIN_M * 1000:.0f} mm margin '
                                     '(a default)'}
    else:
        doc.update(status='good', word='ready',
                   advice=('Every task object lies inside a task volume.' if w.objects else
                           'No task objects: the grasp and VLA studies sample the volumes\' support surfaces.'))
    if no_ext:
        doc['advice'] += f' Not checked (no geometry): {", ".join(no_ext)}.'
    return doc


def apply_suggestion(project, sug: dict):
    """Make the check's suggestion on project (the caller makes it one undo step); returns the volume."""
    c, s = tuple(float(x) for x in sug['center']), tuple(max(0.001, float(x)) for x in sug['size'])
    if sug['kind'] == 'grow':
        v = next(x for x in project.workspace.volumes if x.uid == sug['volume_uid'])
        v.center, v.size = c, s
        return v
    v = P.Volume(name=sug.get('name') or 'Task volume', kind='tabletop', center=c, size=s)
    project.workspace.volumes.append(v)
    return v
