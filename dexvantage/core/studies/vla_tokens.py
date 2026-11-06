"""VLA token resolution (spec §7.12): how many of a policy's ViT tokens each task object and each fingertip covers,
per camera the policy takes, at the pre-grasp and contact poses of a grasp set.

    @jobs.study('vla_tokens', 'VLA token resolution', quick=False, batch='vla-tokens')   → --vla-tokens OUT.json
    run(project, ctx) -> dict
    settings_of(project) -> dict                    the settings with the defaults filled in (and which are defaults)
    token_grid(resolved) -> dict | None             the grid the tokens are counted on, and where it comes from
    View(sensor, imager, camera, geom, R, t)        one camera of one policy at one pose
        .to_input(P) -> (xy input edge coordinates, valid)
    silhouette(view, tris, tol_px, max_depth) -> Silhouette   the object's projected triangles rasterised
    count_mask(mask, cell) -> dict                  tokens touched, tokens at least half covered, the extent in tokens
    count_points(view, P, cell) -> dict             the same for points (a fingertip's contact pad)
    grid_polylines(geom, cell) -> [(K, 2)]          the token grid in sensor pixels (the Camera view)
    crop_polyline(geom) -> (5, 2)                   the crop boundary in sensor pixels
    footprints(view, tris, cell, cells) -> (S, 2, 3)   token boundaries cast onto an object (the 3D view)
    live_views(project, policy_uid, q) -> [View]    the policy's cameras at q (the page's overlays)

The image -> input mapping. Each camera is modelled at its stream (Sensor.stream, else the imager's largest listed
stream) through its imager (the RGB imager when it has one, as the Camera view; else its first), with its lens model
(core/optics CameraModel). The policy's crop rule and input resolution (core/policies.py resolve: the library entry,
cited, or the user's measured figure) give a CropGeometry: the crop box in sensor pixels, src_rect = (x0, y0, x1, y1)
in pixel-edge coordinates (the OpenCV pixel u spans x = u to u + 1, its centre at x = u + ½), the rectangle it fills in
the input, dst_rect, and the scales sx, sy (input pixels per sensor pixel):

    x_in = dst_x0 + (u + ½ − src_x0) · sx,   y_in = dst_y0 + (v + ½ − src_y0) · sy

Each camera record carries the crop box in sensor pixels and the scales ('mapping').

The token grid. The input is sliced into square cells of patch_px (the ViT patch, stated by the entry: 14 px for π0,
OpenVLA; 16 px for Octo, SmolVLA): cols = w // patch, rows = h // patch (a remainder is not in any patch, as a strided
patch embedding drops it). A policy whose entry states no patch but states tokens_per_image T gets the cell
c = √(w · h / T) when that is a whole number of pixels dividing both sides (ACT: 640 × 480 and 300 tokens, 32 px
cells; GR00T N1: 224 × 224 and 64 tokens, 28 px): labelled "derived from tokens per image" (it assumes the tokens
tile the input on a square grid). When both are stated and differ (SmolVLA: 16 px patches, 64 tokens of 64 px after
its pixel shuffle) the counts are on the patches and the merged cell is reported beside them. A policy with neither
(Diffusion Policy's CNN) gets pixel counts and no tokens, said so.

The silhouette. A task object's world triangles (core/scenegeom.py object_triangles) are projected through the lens
and the crop mapping and rasterised at input resolution: an input pixel is covered when its centre lies inside a
projected triangle (edge functions, both windings). A lens that bends straight lines (Brown-Conrady, the fisheyes) is
handled by subdividing each triangle (midpoint split, 4 children) until every projected edge midpoint lies within
raster_tol_px (default 0.01 input px) of the midpoint of its projected ends, at most max_subdiv levels (default 8);
triangles with a corner behind a pinhole camera or beyond the lens's valid angle are split the same way and the parts
still invalid at the last level are dropped (said in the notes). The raster covers the whole camera image mapped into
input coordinates, so the part of an object that the camera sees but the crop cuts away is measured ('outside the
crop'), as is padding (never a token of the image).

Counts per object, camera and pose: tokens touched (cells with at least one covered pixel), tokens at least half
covered, covered input pixels, the extent in tokens (the silhouette's bounding box in input pixels divided by the
cell: width × height), the share of the object's image outside the crop, and in view or not. A critical object
(settings critical_objects, default every task object, a default) "spans fewer than 2 × 2 tokens" when its extent is
below min_tokens_side (default 2, a default) along either side; it "falls outside the crop" when more than
crop_warn_fraction (default 1 %, a default) of what the camera sees of it is cut away, and entirely when none of it is
left. Fingertips: the contact pad's sample points (core/grasps.py contact_points, PAD_POINTS area-weighted points on
each pad): tokens touched (cells holding a point) and the points' extent in tokens.

The poses. With a grasp set (settings grasp_set, else the project's first), up to max_grasps grasps (default 12,
evenly through the set) are taken at two poses each: contact (the grasp's gripper pose) and pre-grasp (the gripper
backed off pre_grasp_m along its approach, default 100 mm, a default). The fingertip pads are carried rigidly from the
home pose with the gripper frame (core/grasps.py measures the pads at home: the finger opening is the home opening). A
camera mounted on that robot moves with it: the arm is posed by IK of the grasp centre (core/robot/ik.py solve_batch
with grasps.ik_targets' tool offset); a pose IK does not reach is left out for that camera and counted. Without a grasp
set the robots stand at their home pose (or the joint state given in the settings: the pose on screen when run from
the page), said in the notes.

Headline: the fewest tokens a critical object touches in a camera that sees it (over every policy, camera and pose),
with its extent; status 'good' when no warning, 'warning' when part of a critical object is cropped away, 'serious'
when one spans fewer than min_tokens_side × min_tokens_side tokens or the crop cuts it out entirely, 'critical' when
a policy sees a critical object in none of its cameras. The suggestion: the camera stream mode (of those its device
lists) under which the worst warned object spans the most tokens (a 16:9 stream padded into a square input wastes a
quarter of the tokens on black), applied as one undo step on the page.

Known limits: silhouettes are not occlusion-tested (an object hidden behind the gripper still counts its tokens; the
grasp self-occlusion study treats visibility); an object is its triangle model (cylinders and spheres tessellated as
scenegeom does); tokens are counted on the inference-time crop of core/policies.py (training augmentation is not
modelled) and on a square cell grid (merging schemes other than a square pixel shuffle are not modelled); Octo's
128 × 128 wrist input is listed by the Policies page but not applied per camera role; the fingertip pads are carried
rigidly from home, so a grasp that closes the fingers moves them by up to half the opening; IK picks one arm
configuration per grasp (another may move a wrist camera elsewhere); interpolation filters are not modelled (a token
"touched" by a sliver of an object may carry little of it).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import jobs
from ..jobs import StudyError

DEFAULTS = {
    'min_tokens_side': 2.0,        # default: a critical object should span at least 2 × 2 tokens (labelled)
    'crop_warn_fraction': 0.01,    # default: warn when more than 1 % of what a camera sees of an object is cropped
    'pre_grasp_m': 0.10,           # default: the pre-grasp pose, the gripper 100 mm back along its approach
    'max_grasps': 12,              # default: grasps taken evenly through the set
    'raster_tol_px': 0.01,         # default: subdivide until projected edge midpoints are within this (input px)
    'max_subdiv': 8,               # default: subdivision levels at most
}
SEVERITY = ('good', 'warning', 'serious', 'critical')
WORDS = {'good': 'resolved', 'warning': 'partly cropped', 'serious': 'too few tokens', 'critical': 'not in the input'}
CHUNK = 2_000_000                  # candidate pixels tested at once
MAX_TRIS = 400_000                 # subdivided triangles per object at most (the rest are not split further)


# ====================================================================== settings
def settings_of(project) -> dict:
    """The study's settings, the defaults filled in; 'defaults' lists the keys left at their default."""
    given = dict(project.analysis.studies.get('vla_tokens') or {})
    out, used = dict(DEFAULTS), []
    for k in DEFAULTS:
        if given.get(k) is not None:
            out[k] = given[k]
        else:
            used.append(k)
    try:
        for k in ('min_tokens_side', 'crop_warn_fraction', 'pre_grasp_m', 'raster_tol_px'):
            out[k] = float(out[k])
        out['max_grasps'] = int(out['max_grasps'])
        out['max_subdiv'] = int(out['max_subdiv'])
    except (TypeError, ValueError):
        raise StudyError('The VLA token settings must be numbers.') from None
    if not (math.isfinite(out['min_tokens_side']) and out['min_tokens_side'] > 0):
        raise StudyError('The smallest object span must be more than 0 tokens.')
    if not 0 <= out['crop_warn_fraction'] < 1:
        raise StudyError('The cropped share that warns must be from 0 to 1.')
    if not (math.isfinite(out['pre_grasp_m']) and 0 <= out['pre_grasp_m'] <= 2):
        raise StudyError('The pre-grasp distance must be from 0 to 2 m.')
    if not 1 <= out['max_grasps'] <= 1000:
        raise StudyError('The number of grasps must be from 1 to 1000.')
    if not (math.isfinite(out['raster_tol_px']) and out['raster_tol_px'] > 0):
        raise StudyError('The raster tolerance must be more than 0 px.')
    if not 0 <= out['max_subdiv'] <= 10:
        raise StudyError('The subdivision levels must be from 0 to 10.')
    out['policies'] = [str(u) for u in given.get('policies') or []]
    out['critical_objects'] = None if given.get('critical_objects') is None else \
        [str(u) for u in given.get('critical_objects') or []]
    out['grasp_set'] = str(given.get('grasp_set') or '')
    out['q'] = dict(given['q']) if isinstance(given.get('q'), dict) else {}
    out['suggest'] = bool(given.get('suggest', True))
    out['defaults'] = used
    return out


# ====================================================================== the token grid
def token_grid(resolved: dict) -> dict | None:
    """{'cell_px', 'cols', 'rows', 'kind': 'patch' | 'token', 'source', 'merged_cell_px', 'merged_source'} for a
    resolved policy (policies.resolve), or None when neither the patch nor the token count gives a grid."""
    size = resolved['input_resolution'][0]
    if size is None:
        return None
    w, h = int(size[0]), int(size[1])
    patch = resolved['patch_px'][0]
    tpi = resolved['tokens_per_image'][0]
    derived = None
    if tpi:
        c = math.sqrt(w * h / float(tpi))
        ci = int(round(c))
        if abs(c - ci) < 1e-9 and ci >= 1 and w % ci == 0 and h % ci == 0:
            derived = ci
    if patch:
        p = int(patch)
        g = {'cell_px': p, 'cols': w // p, 'rows': h // p, 'kind': 'patch', 'source': resolved['patch_px'][1],
             'merged_cell_px': None, 'merged_source': ''}
        if derived and derived != p:
            g['merged_cell_px'] = derived
            g['merged_source'] = (f'derived from tokens per image ({tpi}: {w // derived} × {h // derived} cells of '
                                  f'{derived} px; it assumes a square grid)')
        return g
    if derived:
        return {'cell_px': derived, 'cols': w // derived, 'rows': h // derived, 'kind': 'token',
                'source': (f'derived from tokens per image ({tpi} tokens on a {w} × {h} input: {w // derived} × '
                           f'{h // derived} cells of {derived} px; it assumes the tokens tile the input on a square '
                           f'grid). Tokens per image: {resolved["tokens_per_image"][1]}'),
                'merged_cell_px': None, 'merged_source': ''}
    return None


# ====================================================================== one camera at one pose
@dataclass
class View:
    sensor: object
    imager: object
    camera: object                 # optics.CameraModel at the stream
    geom: object                   # policies.CropGeometry
    R: np.ndarray                  # optical -> world rotation
    t: np.ndarray                  # optical centre in the world

    def to_input(self, P):
        """World points (N, 3) -> (input edge coordinates (N, 2), valid (N,): in front within the lens's valid
        angle; a valid point may still fall outside the image)."""
        P = np.asarray(P, float).reshape(-1, 3)
        uv, _ok = self.camera.project((P - self.t) @ self.R)
        valid = np.isfinite(uv).all(axis=1)
        x, y = self.geom.to_input(uv[:, 0] + 0.5, uv[:, 1] + 0.5)
        return np.stack([x, y], axis=1), valid

    def canvas(self):
        """(x0, y0, cw, ch): the integer input-pixel window covering the input and the whole camera image."""
        W, H = self.geom.src_size
        w, h = self.geom.input_size
        xa, ya = self.geom.to_input(0.0, 0.0)
        xb, yb = self.geom.to_input(float(W), float(H))
        x0 = int(math.floor(min(xa, 0.0) + 1e-9))
        y0 = int(math.floor(min(ya, 0.0) + 1e-9))
        x1 = int(math.ceil(max(xb, float(w)) - 1e-9))
        y1 = int(math.ceil(max(yb, float(h)) - 1e-9))
        return x0, y0, x1 - x0, y1 - y0


def view_of(project, sensor, imager, geom, q) -> View:
    from .. import camrender as CR
    from ..frames import quat_to_matrix
    pose = CR.optical_pose(project, sensor, imager, q)
    return View(sensor, imager, imager.camera, geom, quat_to_matrix(pose.quaternion),
                np.asarray(pose.position, float))


def _masks(view: View, x0, y0, cw, ch):
    """(in_image (ch, cw), in_input (ch, cw)) of the canvas pixels: the centre maps inside the camera image; it is
    an input pixel showing the image (inside the input and its content rectangle)."""
    g = view.geom
    W, H = g.src_size
    w, h = g.input_size
    xc = x0 + np.arange(cw) + 0.5
    yc = y0 + np.arange(ch) + 0.5
    xs, _ = g.to_source(xc, np.zeros_like(xc))
    _, ys = g.to_source(np.zeros_like(yc), yc)
    img = ((ys > 0) & (ys < H))[:, None] & ((xs > 0) & (xs < W))[None, :]
    dx0, dy0, dx1, dy1 = g.dst_rect
    inp = ((yc > max(0.0, dy0)) & (yc < min(float(h), dy1)))[:, None] & \
          ((xc > max(0.0, dx0)) & (xc < min(float(w), dx1)))[None, :]
    return img, inp & img


# ====================================================================== rasterising
def raster_triangles(tri2: np.ndarray, x0: int, y0: int, cw: int, ch: int) -> np.ndarray:
    """(ch, cw) bool: canvas pixels whose centre (x0 + i + ½, y0 + j + ½) lies inside one of the 2-D triangles
    (M, 3, 2) (input edge coordinates; either winding; edges inclusive)."""
    out = np.zeros(ch * cw, bool)
    T = np.asarray(tri2, float).reshape(-1, 3, 2)
    if not len(T) or cw <= 0 or ch <= 0:
        return out.reshape(ch, cw)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    good = np.isfinite(area) & (np.abs(area) > 1e-12)
    T, area = T[good], area[good]
    if not len(T):
        return out.reshape(ch, cw)
    lo, hi = T.min(axis=1), T.max(axis=1)
    c0 = np.clip(np.ceil(lo[:, 0] - x0 - 0.5), 0, cw).astype(np.int64)
    c1 = np.clip(np.floor(hi[:, 0] - x0 - 0.5), -1, cw - 1).astype(np.int64)
    r0 = np.clip(np.ceil(lo[:, 1] - y0 - 0.5), 0, ch).astype(np.int64)
    r1 = np.clip(np.floor(hi[:, 1] - y0 - 0.5), -1, ch - 1).astype(np.int64)
    w, h = c1 - c0 + 1, r1 - r0 + 1
    ok = (w > 0) & (h > 0)
    if not ok.any():
        return out.reshape(ch, cw)
    T, area, c0, r0, w, h = T[ok], area[ok], c0[ok], r0[ok], w[ok], h[ok]
    sgn = np.sign(area)
    cnt = w * h
    cum = np.cumsum(cnt)
    s = 0
    while s < len(T):
        base = int(cum[s - 1]) if s else 0
        e = max(s + 1, int(np.searchsorted(cum, base + CHUNK, 'right')))
        cc = cnt[s:e]
        ti = np.repeat(np.arange(s, e), cc)
        local = np.arange(int(cc.sum()), dtype=np.int64) - np.repeat(np.r_[0, np.cumsum(cc)[:-1]], cc)
        col = c0[ti] + local % w[ti]
        row = r0[ti] + local // w[ti]
        px = x0 + col + 0.5
        py = y0 + row + 0.5
        inside = np.ones(len(ti), bool)
        for k in range(3):
            p, q = T[ti, k], T[ti, (k + 1) % 3]
            ef = (q[:, 0] - p[:, 0]) * (py - p[:, 1]) - (q[:, 1] - p[:, 1]) * (px - p[:, 0])
            inside &= ef * sgn[ti] >= 0
        out[row[inside] * cw + col[inside]] = True
        s = e
    return out.reshape(ch, cw)


def project_triangles(view: View, tris, tol_px: float, max_depth: int):
    """(M', 3, 2) input-coordinate triangles of world triangles (M, 3, 3), subdivided where the lens bends their edges
    or a corner has no projection (module docstring); and the number of triangle parts dropped at the last level."""
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    keep, dropped = [], 0
    x0, y0, cw, ch = view.canvas()
    straight = view.camera.model == 'pinhole'
    for level in range(max_depth + 1):
        if not len(T):
            break
        xy, val = view.to_input(T.reshape(-1, 3))
        xy, val = xy.reshape(-1, 3, 2), val.reshape(-1, 3)
        allv = val.all(axis=1)
        if straight:
            need = ~allv
        else:
            M = 0.5 * (T + T[:, [1, 2, 0]])
            mxy, mval = view.to_input(M.reshape(-1, 3))
            mxy, mval = mxy.reshape(-1, 3, 2), mval.reshape(-1, 3)
            with np.errstate(invalid='ignore'):
                dev = np.linalg.norm(mxy - 0.5 * (xy + xy[:, [1, 2, 0]]), axis=2).max(axis=1)
            need = ~allv | ~mval.all(axis=1) | ~(dev <= tol_px)
        # cull triangles wholly off the canvas (all corners valid and the bounding box outside it)
        with np.errstate(invalid='ignore'):
            lo = np.where(val[..., None], xy, np.inf).min(axis=1)
            hi = np.where(val[..., None], xy, -np.inf).max(axis=1)
            off = allv & ((hi[:, 0] < x0) | (lo[:, 0] > x0 + cw) | (hi[:, 1] < y0) | (lo[:, 1] > y0 + ch))
        done = ~need & ~off
        keep.append(xy[done])
        split = need & ~off
        if level == max_depth or len(T) * 4 > MAX_TRIS:
            ok = split & allv
            keep.append(xy[ok])
            dropped += int(np.count_nonzero(split & ~allv))
            break
        S = T[split]
        if not len(S):
            break
        a, b, c = S[:, 0], S[:, 1], S[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                            np.stack([ab, bc, ca], 1)])
    tri2 = np.concatenate(keep) if keep else np.zeros((0, 3, 2))
    return tri2, dropped


@dataclass
class Silhouette:
    canvas: tuple                  # (x0, y0, cw, ch)
    mask: np.ndarray               # (ch, cw) covered canvas pixels (inside the camera image)
    input_mask: np.ndarray         # (h, w) covered input pixels showing the image
    pixels_image: int              # covered pixels the camera sees (input-pixel units)
    pixels_input: int              # of which inside the crop
    dropped: int = 0               # triangle parts with no projection (behind the camera, beyond the valid angle)
    bbox: tuple | None = None      # (x0, y0, x1, y1) of the covered input pixels (edges), None when none
    extra: dict = field(default_factory=dict)

    @property
    def outside_fraction(self) -> float | None:
        if not self.pixels_image:
            return None
        return 1.0 - self.pixels_input / float(self.pixels_image)


def silhouette(view: View, tris, tol_px: float = DEFAULTS['raster_tol_px'],
               max_depth: int = DEFAULTS['max_subdiv']) -> Silhouette:
    """The object's silhouette at input resolution (module docstring)."""
    x0, y0, cw, ch = view.canvas()
    tri2, dropped = project_triangles(view, tris, tol_px, max_depth)
    m = raster_triangles(tri2, x0, y0, cw, ch)
    img, inp = _masks(view, x0, y0, cw, ch)
    m &= img
    w, h = view.geom.input_size
    im = np.zeros((h, w), bool)
    sub = (m & inp)[-y0:-y0 + h, -x0:-x0 + w]
    im[:sub.shape[0], :sub.shape[1]] = sub
    bbox = None
    if im.any():
        rows, cols = np.nonzero(im.any(axis=1))[0], np.nonzero(im.any(axis=0))[0]
        bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return Silhouette((x0, y0, cw, ch), m, im, int(m.sum()), int(im.sum()), dropped, bbox)


def count_mask(mask: np.ndarray, cell: int | None, bbox=None) -> dict:
    """Tokens of an input mask (h, w) on a cell grid: {'tokens' touched, 'tokens_half' at least half covered,
    'cells' [[row, col], ...] touched, 'extent' [width, height] of the bounding box in tokens, 'pixels'}."""
    px = int(mask.sum())
    if bbox is None and px:
        rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    out = {'pixels': px, 'tokens': None, 'tokens_half': None, 'cells': [], 'extent': None,
           'extent_px': None if bbox is None else [bbox[2] - bbox[0], bbox[3] - bbox[1]]}
    if not cell:
        return out
    h, w = mask.shape
    c = int(cell)
    rows, cols = h // c, w // c
    if rows < 1 or cols < 1:
        return out
    blk = mask[:rows * c, :cols * c].reshape(rows, c, cols, c)
    frac = blk.mean(axis=(1, 3))
    touched = frac > 0
    out['tokens'] = int(touched.sum())
    out['tokens_half'] = int((frac >= 0.5).sum())
    out['cells'] = [[int(r), int(k)] for r, k in zip(*np.nonzero(touched))]
    if bbox is not None:
        out['extent'] = [(bbox[2] - bbox[0]) / c, (bbox[3] - bbox[1]) / c]
    return out


def count_points(view: View, P, cell: int | None) -> dict:
    """Tokens holding at least one of the world points P (a fingertip's pad samples) inside the input: {'tokens',
    'cells', 'extent' [w, h] of the points' bounding box in tokens, 'points_in_input', 'points_in_image',
    'points'}."""
    xy, val = view.to_input(P)
    g = view.geom
    W, H = g.src_size
    w, h = g.input_size
    xs, ys = g.to_source(xy[:, 0], xy[:, 1])
    with np.errstate(invalid='ignore'):
        img = val & (xs >= 0) & (xs <= W) & (ys >= 0) & (ys <= H)
        dx0, dy0, dx1, dy1 = g.dst_rect
        inp = img & (xy[:, 0] >= max(0.0, dx0)) & (xy[:, 0] < min(float(w), dx1)) & \
            (xy[:, 1] >= max(0.0, dy0)) & (xy[:, 1] < min(float(h), dy1))
    out = {'points': int(len(xy)), 'points_in_image': int(img.sum()), 'points_in_input': int(inp.sum()),
           'tokens': None, 'cells': [], 'extent': None}
    if not cell or not inp.any():
        if cell and inp.size:
            out['tokens'] = 0
        return out
    c = int(cell)
    q = xy[inp]
    cols, rows = (q[:, 0] // c).astype(int), (q[:, 1] // c).astype(int)
    good = (cols < w // c) & (rows < h // c)
    cells = sorted({(int(r), int(k)) for r, k in zip(rows[good], cols[good])})
    out['tokens'] = len(cells)
    out['cells'] = [list(x) for x in cells]
    lo, hi = q.min(axis=0), q.max(axis=0)
    out['extent'] = [float(hi[0] - lo[0]) / c, float(hi[1] - lo[1]) / c]
    return out


# ====================================================================== overlay geometry
def crop_polyline(geom) -> np.ndarray:
    """(5, 2) the crop boundary (the sensor region the input shows) in sensor pixel coordinates (OpenCV)."""
    x0, y0, x1, y1 = (v - 0.5 for v in geom.src_rect)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)


def grid_polylines(geom, cell: int) -> list:
    """The token grid's lines over the input's image content, in sensor pixel coordinates (OpenCV): [(2, 2)]."""
    w, h = geom.input_size
    c = int(cell)
    cols, rows = w // c, h // c
    dx0, dy0, dx1, dy1 = geom.dst_rect
    ya, yb = max(0.0, dy0), min(float(rows * c), dy1)
    xa, xb = max(0.0, dx0), min(float(cols * c), dx1)
    out = []
    if xb <= xa or yb <= ya:
        return out
    for k in range(cols + 1):
        x = float(k * c)
        if xa - 1e-9 <= x <= xb + 1e-9:
            out.append(np.array([[x, ya], [x, yb]]))
    for k in range(rows + 1):
        y = float(k * c)
        if ya - 1e-9 <= y <= yb + 1e-9:
            out.append(np.array([[xa, y], [xb, y]]))
    res = []
    for seg in out:
        sx, sy = geom.to_source(seg[:, 0], seg[:, 1])
        res.append(np.stack([sx - 0.5, sy - 0.5], axis=1))
    return res


def cell_polylines(geom, cell: int, cells) -> list:
    """The outlines of token cells [[row, col], ...] in sensor pixel coordinates (OpenCV): [(5, 2)]."""
    out = []
    c = float(cell)
    for r, k in cells:
        xs = np.array([k, k + 1, k + 1, k, k]) * c
        ys = np.array([r, r, r + 1, r + 1, r]) * c
        sx, sy = geom.to_source(xs, ys)
        out.append(np.stack([sx - 0.5, sy - 0.5], axis=1))
    return out


def footprints(view: View, tris, cell: int, cells, step_px: float = 1.0) -> np.ndarray:
    """(S, 2, 3) world segments: the boundaries of the token cells [[row, col], ...] cast from the camera onto the
    triangles (core/raycast.py first_hit; a boundary point whose ray misses them is left out)."""
    from .. import raycast as RC
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(T) or not cells:
        return np.zeros((0, 2, 3))
    c = float(cell)
    n = max(2, int(math.ceil(c / step_px)) + 1)
    s = np.linspace(0.0, 1.0, n)
    segs = set()
    for r, k in cells:
        segs.add(('v', k, r))
        segs.add(('v', k + 1, r))
        segs.add(('h', r, k))
        segs.add(('h', r + 1, k))
    pts = []
    for kind, a, b in sorted(segs):
        if kind == 'v':
            pts.append(np.stack([np.full(n, a * c), (b + s) * c], axis=1))
        else:
            pts.append(np.stack([(b + s) * c, np.full(n, a * c)], axis=1))
    xy = np.concatenate(pts)
    sx, sy = view.geom.to_source(xy[:, 0], xy[:, 1])
    rays = view.camera.unproject(np.stack([sx - 0.5, sy - 0.5], axis=1))
    ok = np.isfinite(rays).all(axis=1)
    d = np.where(ok[:, None], rays, np.array([0.0, 0.0, 1.0])) @ view.R.T
    t = RC.first_hit(view.t, d, T)
    hit = ok & np.isfinite(t)
    P = view.t + d * np.where(hit, t, 0.0)[:, None]
    P, hit = P.reshape(len(pts), n, 3), hit.reshape(len(pts), n)
    both = hit[:, :-1] & hit[:, 1:]
    return np.stack([P[:, :-1][both], P[:, 1:][both]], axis=1)


# ====================================================================== cameras and poses
def _policy_views(project, pol, notes: list) -> list:
    """[(sensor, imager)] the policy takes (in use, with an imager)."""
    from .. import policies as PO
    from ..plucker import pick_imager, sensor_stream
    out = []
    for s in PO.policy_cameras(project, pol):
        if not s.enabled:
            notes.append(f'{pol.name}: {s.name} is switched off and is left out.')
            continue
        try:
            im = pick_imager(s, sensor_stream(s))
        except Exception as e:                     # noqa: BLE001 — a device the optics cannot model: said why
            notes.append(f'{pol.name}: {s.name} is left out: {e}')
            continue
        if im is None:
            notes.append(f'{pol.name}: {s.name} has no imager and is left out.')
            continue
        out.append((s, im))
    return out


def live_views(project, policy_uid: str, q: dict | None = None) -> list:
    """The policy's cameras posed at q: [(View, grid or None)] (for the page's overlays; [] when its input is not
    stated)."""
    from .. import policies as PO
    pol = PO.policy(project, policy_uid)
    if pol is None:
        return []
    try:
        grid = token_grid(PO.resolve(project, pol.uid))
    except PO.PolicyError:
        return []
    out = []
    for s, im in _policy_views(project, pol, []):
        try:
            geom = PO.input_geometry(project, pol.uid, (im.camera.width, im.camera.height))
            out.append((view_of(project, s, im, geom, q), grid))
        except Exception:                          # noqa: BLE001 — a camera that cannot be posed: no overlay
            continue
    return out


def _base_q(project, st: dict, notes: list) -> tuple[dict, dict]:
    from .coverage import joint_state
    return joint_state(project, st, notes)


def _grasp_states(project, st: dict, qd: dict, notes: list) -> tuple[list, dict | None]:
    """The poses: [{'id', 'phase', 'grasp_id', 'label', 'T' gripper 4 × 4 or None}] and the grasp set used."""
    from .. import grasps as GR
    sets = GR.sets(project)
    gs = None
    if st['grasp_set']:
        gs = GR.get_set(project, st['grasp_set'])
        if gs is None:
            raise StudyError('The grasp set named in the VLA token settings is not in the project (it may have been '
                             'deleted): choose another on the page.')
    elif sets:
        gs = next((g for g in sets if project.robot(g.get('robot_uid', '')) is not None and g.get('poses')), None)
    if gs is None or not gs.get('poses'):
        how = 'the joint state given (the pose on screen)' if st['q'] else 'their home pose'
        notes.append(f'There is no grasp set, so the robots stand at {how} and the fingertips are counted there. '
                     'Make a grasp set on the Tasks page to count them at pre-grasp and contact poses.')
        return [{'id': 'home', 'phase': 'home', 'grasp_id': '', 'label': 'home' if not st['q'] else 'pose given',
                 'T': None}], None
    poses = gs['poses']
    n = min(len(poses), st['max_grasps'])
    idx = sorted({int(round(x)) for x in np.linspace(0, len(poses) - 1, n)})
    if len(poses) > n:
        notes.append(f'{n} of the {len(poses)} grasps of "{gs.get("name", "the grasp set")}" are counted, evenly '
                     f'through the set (max_grasps, default {DEFAULTS["max_grasps"]}).')
    out = []
    for i in idx:
        g = poses[i]
        Tc = GR.grasp_matrix(g)
        Tp = GR.pre_contact_poses(g, [st['pre_grasp_m']])[0]
        gid = str(g.get('id', i))
        out.append({'id': f'{gid}:pre', 'phase': 'pre_grasp', 'grasp_id': gid, 'label': f'{gid} pre-grasp', 'T': Tp,
                    'index': i})
        out.append({'id': f'{gid}:contact', 'phase': 'contact', 'grasp_id': gid, 'label': f'{gid} contact',
                    'T': Tc, 'index': i})
    return out, gs


def _solve_states(project, gs, states, qd) -> dict:
    """{state id: q (nq,)} for the states IK reaches (the grasp set's robot), the others absent."""
    from .. import grasps as GR
    from ..robot import ik as IK
    from ..robot.model import get_model
    ruid = gs.get('robot_uid', '')
    m = get_model(project, ruid)
    link, tcp, _ = GR.ik_targets(project, {**gs, 'poses': []})
    targets = np.array([s['T'] @ tcp for s in states]).reshape(-1, 4, 4)
    q0 = qd.get(ruid, m.home)
    res = IK.solve_batch(m, link, targets, q0=q0, seeds=4, rng=0, tcp=tcp, collision=False)
    return {s['id']: res.q[k].copy() for k, s in enumerate(states) if res.ok[k]}


def _pads_local(project, ruid: str, frame: str):
    """{fingertip: (K, 3) points in the gripper frame at home} (carried rigidly with the gripper)."""
    from .. import grasps as GR
    from ..robot.model import get_model
    m = get_model(project, ruid)
    pts = GR.contact_points(project, ruid, None)
    if not pts:
        return {}
    Ti = np.linalg.inv(m.fk(m.home)[frame])
    return {k: P @ Ti[:3, :3].T + Ti[:3, 3] for k, P in pts.items()}


# ====================================================================== the study
def _sev_max(a: str, b: str) -> str:
    return a if SEVERITY.index(a) >= SEVERITY.index(b) else b


def _num(x, nd=4):
    return None if x is None else round(float(x), nd)


def mapping_record(geom) -> dict:
    """The image -> input mapping of a crop geometry, written out (JSON-able)."""
    x0, y0, x1, y1 = geom.src_rect
    dx0, dy0, dx1, dy1 = geom.dst_rect
    return {'rule': geom.rule, 'sensor_size': list(geom.src_size), 'input_size': list(geom.input_size),
            'crop_box_sensor_px': {'u0': x0 - 0.5, 'v0': y0 - 0.5, 'u1': x1 - 0.5, 'v1': y1 - 0.5,
                                   'width': x1 - x0, 'height': y1 - y0},
            'content_rect_input_px': [dx0, dy0, dx1, dy1], 'scale_input_per_sensor_px': [geom.sx, geom.sy],
            'aspect_kept': geom.aspect_kept, 'used_fraction': geom.used_fraction,
            'padded_fraction': geom.padded_fraction, 'note': geom.note,
            'equation': (f'x_in = {dx0:g} + (u + 0.5 − {x0:g}) × {geom.sx:.6g}; '
                         f'y_in = {dy0:g} + (v + 0.5 − {y0:g}) × {geom.sy:.6g} (u, v sensor pixels, OpenCV '
                         'centres; x_in, y_in input edge coordinates)')}


def _eval_objects(view, objects, cell, st) -> dict:
    out = {}
    for o in objects:
        sil = silhouette(view, o['tris'], st['raster_tol_px'], st['max_subdiv'])
        c = count_mask(sil.input_mask, cell, sil.bbox)
        out[o['uid']] = {'pixels_image': sil.pixels_image, 'pixels_input': sil.pixels_input,
                         'outside_crop': _num(sil.outside_fraction), 'in_view': sil.pixels_image > 0,
                         'in_input': sil.pixels_input > 0, 'tokens': c['tokens'], 'tokens_half': c['tokens_half'],
                         'extent': None if c['extent'] is None else [_num(v, 3) for v in c['extent']],
                         'extent_px': c['extent_px'], 'cells': c['cells'], 'dropped': sil.dropped}
    return out


def _object_summary(rows: list, st: dict, critical: bool) -> dict:
    """Aggregate an object's per-pose rows for one camera: the worst pose."""
    seen = [r for r in rows if r['in_view']]
    inp = [r for r in seen if r['in_input']]
    rec = {'poses': len(rows), 'poses_in_view': len(seen), 'poses_in_input': len(inp), 'critical': critical,
           'tokens_min': None, 'tokens_max': None, 'tokens_half_min': None, 'extent_min': None, 'worst_pose': None,
           'outside_crop_max': None, 'status': 'good', 'warnings': []}
    if seen:
        oc = [r['outside_crop'] or 0.0 for r in seen]
        rec['outside_crop_max'] = _num(max(oc))
    if inp:
        if inp[0]['tokens'] is not None:
            k = int(np.argmin([r['tokens'] for r in inp]))
            rec['tokens_min'] = inp[k]['tokens']
            rec['tokens_max'] = max(r['tokens'] for r in inp)
            rec['tokens_half_min'] = min(r['tokens_half'] for r in inp)
            ext = [min(r['extent']) for r in inp if r['extent']]
            rec['extent_min'] = _num(min(ext), 3) if ext else None
            j = int(np.argmin([min(r['extent']) if r['extent'] else 0.0 for r in inp]))
            rec['worst_pose'] = inp[j]['pose']
            rec['worst_extent'] = inp[j]['extent']
        else:
            rec['pixels_min'] = min(r['pixels_input'] for r in inp)
            rec['worst_pose'] = min(inp, key=lambda r: r['pixels_input'])['pose']
    return rec


def _counts_for(project, st, pol, s, im, geom, grid, states, cam_q, objects, pads, base_q):
    """Per-pose object and fingertip counts for one camera: (object rows {uid: [row]}, fingertip rows [row],
    poses left out)."""
    from ..robot.model import RobotModelError
    cell = grid['cell_px'] if grid else None
    obj_rows = {o['uid']: [] for o in objects}
    tip_rows = []
    skipped = []
    cache = {}
    on_link = s.mount.on_link
    for stt in states:
        if on_link and stt['T'] is not None and s.mount.robot_uid == stt.get('robot_uid'):
            q = cam_q.get(stt['id'])
            if q is None:
                skipped.append(stt['id'])
                continue
            qd = dict(base_q)
            qd[stt['robot_uid']] = q
        else:
            qd = base_q
        try:
            view = view_of(project, s, im, geom, qd)
        except (RobotModelError, KeyError) as e:
            raise StudyError(f'{s.name} could not be posed: {e}') from None
        key = (tuple(np.round(view.t, 12)), tuple(np.round(view.R.ravel(), 12)))
        if key not in cache:
            cache[key] = _eval_objects(view, objects, cell, st)
        for uid, r in cache[key].items():
            obj_rows[uid].append(dict(r, pose=stt['id'], phase=stt['phase']))
        for name, P in pads(stt).items():
            c = count_points(view, P, cell)
            tip_rows.append({'fingertip': name, 'pose': stt['id'], 'phase': stt['phase'], 'tokens': c['tokens'],
                             'extent': None if c['extent'] is None else [_num(v, 3) for v in c['extent']],
                             'points_in_input': c['points_in_input'], 'points_in_image': c['points_in_image'],
                             'points': c['points'], 'cells': c['cells']})
    return obj_rows, tip_rows, skipped


def _tip_summary(rows: list) -> list:
    out = []
    names = sorted({r['fingertip'] for r in rows})
    for n in names:
        for ph in ('pre_grasp', 'contact', 'home'):
            rr = [r for r in rows if r['fingertip'] == n and r['phase'] == ph]
            if not rr:
                continue
            tk = [r['tokens'] for r in rr if r['tokens'] is not None]
            seen = [r for r in rr if r['points_in_input'] > 0]
            out.append({'fingertip': n, 'phase': ph, 'poses': len(rr), 'poses_in_input': len(seen),
                        'tokens_min': min(tk) if tk else None, 'tokens_median': float(np.median(tk)) if tk else None,
                        'tokens_max': max(tk) if tk else None,
                        'points_in_input_share': _num(np.mean([r['points_in_input'] / max(r['points'], 1)
                                                               for r in rr]))})
    return out


def _warnings_for(cam_rec, pol, s, st) -> list:
    out = []
    side = st['min_tokens_side']
    for o in cam_rec['objects']:
        if not o['critical']:
            continue
        if o['poses_in_view'] and not o['poses_in_input']:
            out.append({'severity': 'serious', 'kind': 'outside_crop', 'policy': pol.uid, 'sensor': s.uid,
                        'object': o['uid'], 'text': f'{o["name"]} falls outside {pol.name}\'s crop of {s.name}: '
                                                    'the camera sees it but the input does not.'})
            continue
        oc = o.get('outside_crop_max')
        if oc is not None and oc > st['crop_warn_fraction'] and o['poses_in_input']:
            out.append({'severity': 'warning', 'kind': 'cropped', 'policy': pol.uid, 'sensor': s.uid,
                        'object': o['uid'], 'text': f'{oc * 100:.0f} % of {o["name"]} as {s.name} sees it is outside '
                                                    f'{pol.name}\'s crop at the worst pose.'})
        e = o.get('extent_min')
        if e is not None and e < side:
            we = o.get('worst_extent') or [e, e]
            out.append({'severity': 'serious', 'kind': 'small', 'policy': pol.uid, 'sensor': s.uid,
                        'object': o['uid'],
                        'text': f'{o["name"]} spans {we[0]:.1f} × {we[1]:.1f} tokens in {s.name} for {pol.name} '
                                f'(fewer than {side:g} × {side:g}, a default) at its worst pose.'})
    return out


def _suggest(project, st, policies_out, ctx_data) -> dict | None:
    """The stream mode that most improves the worst warned object (module docstring)."""
    from .. import library as LB
    from .. import policies as PO
    from ..plucker import pick_imager
    best = None
    for pr in policies_out:
        for cr in pr['cameras']:
            warned = {w['object'] for w in pr['warnings'] if w['sensor'] == cr['uid'] and w['kind'] in
                      ('small', 'cropped', 'outside_crop')}
            if not warned:
                continue
            s = project.variant.sensor(cr['uid'])
            entry = LB.get(s.model_id) if s is not None and s.model_id else None
            if entry is None:
                continue
            modes = sorted({(int(m['width']), int(m['height'])) for m in entry.streams(cr['imager'])},
                           key=lambda x: -x[0] * x[1])
            cur = tuple(cr['stream'])
            objs = [o for o in ctx_data['objects'] if o['uid'] in warned]
            grid = pr['grid']
            cell = grid['cell_px'] if grid else None

            def score(mode):
                im = pick_imager(s, mode)
                if im is None or im.name != cr['imager']:
                    return None
                g = PO.input_geometry(project, pr['uid'], (im.camera.width, im.camera.height))
                rows = []
                for stt in ctx_data['eval_states'].get(cr['uid'], []):
                    v = view_of(project, s, im, g, stt)
                    rows.append(_eval_objects(v, objs, cell, st))
                if not rows:
                    return None
                worst = math.inf
                for r in rows:
                    for o in objs:
                        x = r[o['uid']]
                        if not x['in_input']:
                            worst = min(worst, -1.0)
                        elif x['extent'] is not None:
                            worst = min(worst, min(x['extent']) - (x['outside_crop'] or 0.0))
                        else:
                            worst = min(worst, x['pixels_input'] / 1e6)
                return worst
            base = score(cur)
            if base is None:
                continue
            for mode in modes:
                if mode == cur:
                    continue
                try:
                    sc = score(mode)
                except Exception:                  # noqa: BLE001 — a mode the lens cannot be fitted for: skipped
                    continue
                if sc is not None and sc > base + 0.05 and (best is None or sc - base > best['gain']):
                    best = {'sensor_uid': cr['uid'], 'sensor': cr['name'], 'policy_uid': pr['uid'],
                            'policy': pr['name'], 'mode': list(mode), 'from': list(cur), 'gain': sc - base,
                            'worst_before': _num(base, 3), 'worst_after': _num(sc, 3)}
    if best is not None:
        best['text'] = (f'Set {best["sensor"]}\'s stream to {best["mode"][0]} × {best["mode"][1]} (now '
                        f'{best["from"][0]} × {best["from"][1]}): for {best["policy"]} the worst warned object then '
                        f'spans {max(best["worst_after"], 0):.1f} tokens on its narrower side instead of '
                        f'{max(best["worst_before"], 0):.1f}.')
    return best


@jobs.study('vla_tokens', 'VLA token resolution', quick=False, batch='vla-tokens', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """VLA token resolution of the task objects and fingertips, per policy and camera (module docstring)."""
    from .. import policies as PO
    from .. import scenegeom as SG
    t_all = time.perf_counter()
    st = settings_of(project)
    notes: list = []
    pols = [p for p in getattr(project, 'policies', []) if not st['policies'] or p.uid in st['policies']]
    if not pols:
        raise StudyError('There is no policy in the project: add one on the Policies page (its input resolution, '
                         'crop rule and ViT patch come from its model card).')
    objects = []
    for o in project.workspace.objects:
        tris = SG.object_triangles(o, project)
        if not len(tris):
            notes.append(f'{o.name} has no geometry to project (its shape or mesh cannot be read) and is left out.')
            continue
        crit = st['critical_objects'] is None or o.uid in st['critical_objects']
        objects.append({'uid': o.uid, 'name': o.name, 'tris': tris, 'critical': crit})
    if not project.workspace.objects:
        notes.append('There is no task object: only the fingertips are counted. Add the objects the policy must see '
                     '(Workspace ▸ Add task object).')
    elif st['critical_objects'] is None:
        notes.append('Every task object counts as critical (the default); choose the critical ones on the page.')
    base_q, _qsrc = _base_q(project, st, notes)
    states, gs = _grasp_states(project, st, base_q, notes)
    cam_q: dict = {}
    pad_robots = []
    if gs is not None:
        ruid = gs.get('robot_uid', '')
        for s_ in states:
            s_['robot_uid'] = ruid
        from .. import grasps as GR
        try:
            frame = gs.get('frame') or GR.gripper_geometry(project, ruid)['frame']
            local = _pads_local(project, ruid, frame)
        except Exception as e:                     # noqa: BLE001 — a gripper without pads: said why
            notes.append(f'The fingertips of the grasp set\'s robot could not be found: {e}')
            local = {}
        if not local:
            notes.append('The grasp set\'s robot has no fingertip or contact pad tagged: only the objects are '
                         'counted (tag them on the Robot page).')

        def pads(stt, _local=local):
            if stt['T'] is None:
                return {}
            T = stt['T']
            return {k: P @ T[:3, :3].T + T[:3, 3] for k, P in _local.items()}
        needs_ik = any(s.mount.on_link and s.mount.robot_uid == ruid for s in project.variant.sensors)
        if needs_ik:
            ctx.report(0.02, 'Posing the arm for each grasp (IK)')
            try:
                cam_q = _solve_states(project, gs, states, base_q)
            except Exception as e:                 # noqa: BLE001 — IK that cannot run: wrist cameras left out
                notes.append(f'The arm could not be posed for the grasps ({e}): cameras on it are left out.')
                cam_q = {}
            miss = len(states) - len(cam_q)
            if miss:
                notes.append(f'IK did not reach {miss} of the {len(states)} grasp poses: cameras mounted on the arm '
                             'are left out at those poses (fixed cameras still count them).')
    else:
        from .. import grasps as GR
        tips = {}
        for r in project.robots:
            try:
                for k, P in GR.contact_points(project, r.uid, base_q.get(r.uid)).items():
                    tips[k if len(project.robots) == 1 else f'{r.name}: {k}'] = P
                pad_robots.append(r.uid)
            except Exception:                      # noqa: BLE001 — a robot without fingertips
                continue

        def pads(stt, _tips=tips):
            return _tips
        if project.robots and not tips:
            notes.append('No robot has a fingertip or contact pad tagged: only the objects are counted.')
    if not objects and not any(pads(s_) for s_ in states):
        raise StudyError('There is nothing to count tokens on: add a task object (Workspace ▸ Add task object) or '
                         'tag the gripper\'s fingertips on the Robot page.')

    out_pols, all_warn = [], []
    total = sum(len(PO.policy_cameras(project, p)) for p in pols) or 1
    done = 0
    worst_head = None
    eval_states = {}
    for pol in pols:
        if ctx.cancelled():
            break
        try:
            res = PO.resolve(project, pol.uid)
        except PO.PolicyError as e:
            notes.append(f'{pol.name} is left out: {e}')
            continue
        size, rule = res['input_resolution'][0], res['crop_rule'][0]
        if size is None or rule is None:
            what = 'input resolution' if size is None else 'crop rule'
            notes.append(f'{pol.name} is left out: its {what} is not stated by its source. Enter it on the Policies '
                         'page (a figure you measured or read in its code).')
            continue
        grid = token_grid(res)
        if grid is not None and (grid['cols'] < 1 or grid['rows'] < 1):
            notes.append(f'{pol.name} is left out: its token cell ({grid["cell_px"]} px) is larger than its input '
                         f'({size[0]} × {size[1]} px), so no token fits. Check the ViT patch on the Policies page.')
            continue
        if grid is None:
            notes.append(f'{pol.name}: neither a ViT patch nor a token count is stated (a CNN encoder?), so only '
                         'pixels are counted, not tokens.')
        if res.get('wrist_input_resolution', (None,))[0] is not None:
            notes.append(f'{pol.name} takes wrist images at {res["wrist_input_resolution"][0][0]} × '
                         f'{res["wrist_input_resolution"][0][1]}; every camera is modelled at the main input here '
                         '(per-role inputs are not applied yet).')
        cams_out, pol_warn = [], []
        for s, im in _policy_views(project, pol, notes):
            if ctx.cancelled():
                break
            ctx.report(0.05 + 0.85 * done / total, f'{pol.name}: {s.name}')
            done += 1
            cam = im.camera
            try:
                geom = PO.input_geometry(project, pol.uid, (cam.width, cam.height))
            except PO.PolicyError as e:
                notes.append(f'{pol.name}: {s.name} is left out: {e}')
                continue
            obj_rows, tip_rows, skipped = _counts_for(project, st, pol, s, im, geom, grid, states, cam_q, objects,
                                                      pads, base_q)
            if skipped:
                notes.append(f'{pol.name}: {s.name} rides on the arm and is left out at {len(skipped)} pose'
                             f'{"s" if len(skipped) != 1 else ""} IK did not reach.')
            orecs = []
            for o in objects:
                rec = _object_summary(obj_rows[o['uid']], st, o['critical'])
                rec.update(uid=o['uid'], name=o['name'], rows=[{k: v for k, v in r.items() if k != 'cells'}
                                                               for r in obj_rows[o['uid']]])
                wp = rec.get('worst_pose')
                rec['worst_cells'] = next((r['cells'] for r in obj_rows[o['uid']] if r['pose'] == wp), [])
                drop = sum(r['dropped'] for r in obj_rows[o['uid']])
                if drop:
                    notes.append(f'{s.name}: parts of {o["name"]} lie behind the camera or beyond its lens\'s valid '
                                 'angle and are not drawn.')
                orecs.append(rec)
            # the poses the suggestion re-evaluates: the worst pose of each warned object
            qmap = {}
            for stt in states:
                if s.mount.on_link and stt['T'] is not None and s.mount.robot_uid == stt.get('robot_uid'):
                    if stt['id'] in cam_q:
                        qmap[stt['id']] = dict(base_q, **{stt['robot_uid']: cam_q[stt['id']]})
                else:
                    qmap[stt['id']] = base_q
            worst_ids = {r.get('worst_pose') for r in orecs if r.get('worst_pose')}
            eval_states[s.uid] = [qmap[k] for k in worst_ids if k in qmap][:4] or list(qmap.values())[:1]
            cam_rec = {'uid': s.uid, 'name': s.name, 'slot': s.slot, 'imager': im.name, 'imager_label': im.label,
                       'is_default': im.is_default, 'on_link': s.mount.on_link,
                       'stream': [cam.width, cam.height], 'lens': cam.model, 'mapping': mapping_record(geom),
                       'objects': orecs, 'fingertips': _tip_summary(tip_rows),
                       'fingertip_rows': [{k: v for k, v in r.items() if k != 'cells'} for r in tip_rows],
                       'poses_left_out': skipped}
            w = _warnings_for(cam_rec, pol, s, st)
            pol_warn += w
            for o in orecs:
                if o['critical'] and o.get('tokens_min') is not None and o['poses_in_input']:
                    cand = (o['tokens_min'], o.get('extent_min') or 0.0)
                    if worst_head is None or cand < worst_head['key']:
                        worst_head = {'key': cand, 'tokens': o['tokens_min'], 'extent': o.get('worst_extent'),
                                      'object': o['name'], 'object_uid': o['uid'], 'camera': s.name,
                                      'sensor_uid': s.uid, 'policy': pol.name, 'policy_uid': pol.uid,
                                      'pose': o.get('worst_pose')}
            cams_out.append(cam_rec)
        # a critical object no camera of the policy puts in its input
        for o in objects:
            if not o['critical'] or not cams_out:
                continue
            if not any(r['poses_in_input'] for c in cams_out for r in c['objects'] if r['uid'] == o['uid']):
                pol_warn.append({'severity': 'critical', 'kind': 'not_in_input', 'policy': pol.uid, 'sensor': '',
                                 'object': o['uid'], 'text': f'{pol.name} sees {o["name"]} in none of its cameras '
                                                             'at any pose counted.'})
        entry = PO.title_of(pol)
        out_pols.append({'uid': pol.uid, 'name': pol.name, 'entry': entry, 'library_id': pol.library_id,
                         'input_size': list(size), 'crop_rule': rule, 'crop_label': PO.CROP_LABEL.get(rule, rule),
                         'fields': {k: {'value': v[0], 'source': v[1]} for k, v in res.items()
                                    if k in ('input_resolution', 'crop_rule', 'crop_side_fraction', 'pre_resize',
                                             'patch_px', 'tokens_per_image')},
                         'grid': grid, 'cameras': cams_out, 'warnings': pol_warn})
        all_warn += pol_warn
    if not out_pols:
        raise StudyError('No policy could be counted: ' + ' '.join(notes[-2:]))
    sev = 'good'
    for w in all_warn:
        sev = _sev_max(w['severity'], sev)
    head = None
    if worst_head is not None:
        head = {k: v for k, v in worst_head.items() if k != 'key'}
    tips_min = None
    for pr in out_pols:
        for cr in pr['cameras']:
            for t in cr['fingertips']:
                if t['phase'] in ('contact', 'home') and t['tokens_min'] is not None and t['poses_in_input']:
                    if tips_min is None or t['tokens_min'] < tips_min['tokens']:
                        tips_min = {'tokens': t['tokens_min'], 'fingertip': t['fingertip'], 'camera': cr['name'],
                                    'policy': pr['name'], 'phase': t['phase']}
    word = WORDS[sev]
    if head is None and sev == 'good':
        # nothing critical was counted: "resolved" in green would claim more than the study checked
        sev, word = 'warning', 'no task object'
    summary = {'status': sev, 'word': word, 'warnings': len(all_warn), 'fewest': head,
               'fingertips_fewest': tips_min,
               'advice': _advice(sev, all_warn, head)}
    sugg = None
    if st['suggest'] and all_warn and not ctx.cancelled():
        ctx.report(0.92, 'Trying the stream modes')
        try:
            sugg = _suggest(project, st, out_pols, {'objects': objects, 'eval_states': eval_states})
        except Exception as e:                     # noqa: BLE001 — the suggestion is an extra: say why it is missing
            notes.append(f'No stream suggestion: {e}')
    ctx.report(1.0, 'Done')
    return {
        'title': 'VLA token resolution',
        'settings': {k: v for k, v in st.items() if k != 'q'},
        'q': {k: [float(x) for x in np.asarray(v).reshape(-1)] for k, v in base_q.items()},
        'poses': [{k: v for k, v in s_.items() if k not in ('T',)} | {'ik_reached': (s_['id'] in cam_q)
                                                                      if gs is not None else None} for s_ in states],
        'pose_source': 'grasp_set' if gs is not None else ('settings' if st['q'] else 'home'),
        'grasp_set': None if gs is None else {'uid': gs.get('uid'), 'name': gs.get('name'),
                                               'robot_uid': gs.get('robot_uid'), 'poses': len(gs.get('poses') or [])},
        'objects': [{'uid': o['uid'], 'name': o['name'], 'critical': o['critical'], 'triangles': int(len(o['tris']))}
                    for o in objects],
        'policies': out_pols,
        'warnings': all_warn,
        'summary': summary,
        'suggestion': sugg,
        'notes': notes,
        'timing_s': round(time.perf_counter() - t_all, 4),
    }


def _advice(sev: str, warns: list, head: dict | None) -> str:
    if head is None and not warns:
        return 'No critical object is in a policy input to count; the fingertips are counted below.'
    if sev == 'good':
        return 'Every critical object spans enough tokens and stays inside the crops.'
    kinds = {w['kind'] for w in warns}
    if 'not_in_input' in kinds:
        return ('A policy sees a critical object in none of its cameras: aim a camera at it, or assign another camera '
                'to the policy on the Policies page.')
    if 'outside_crop' in kinds or 'cropped' in kinds:
        return ('The crop cuts away part of a critical object: re-aim the camera so the object sits in the part the '
                'input keeps, or choose a stream whose aspect ratio matches the input.')
    return ('A critical object covers too few tokens: move the camera closer, use a narrower lens, or a stream whose '
            'aspect ratio matches the input (less padding).')
