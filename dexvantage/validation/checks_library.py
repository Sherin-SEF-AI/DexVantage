"""Data-consistency checks of the hardware library (core/library): the figures each built-in entry copies from its
maker are cross-checked against each other with closed forms written out again here. They test the entries, not a
model: a failure means a figure was mistyped, or the maker's own figures disagree and the entry does not say so.

library.sensor_geometry — the stated pixel pitch times the native resolution against the stated active-area width,
height and diagonal (sensor_size_mm, sensor_diagonal_mm): w = p W, h = p H, d = p sqrt(W² + H²). Tolerance 2 %: the
makers round the sizes to 0.1 mm (0.05 mm of a 3.8 mm side is 1.3 %).

library.fov_shape — the stated horizontal, vertical and diagonal angles against the native image shape (and each
fov_modes row against its aspect ratio). For a centred, radially symmetric lens whose field angle θ(r) is a concave
function of the image radius r with θ(0) = 0 — the pinhole θ = atan(r/f), barrel distortion and mild pincushion on a
pinhole, the stereographic fisheye θ = 2 atan(r/2f) and, as the limiting case, the equidistant θ = r/f — θ(r)/r
does not increase with r, so with
half-extents r_h = W/2, r_v = H/2, r_d = sqrt(W² + H²)/2 (square pixels):
    θ_v ≤ θ_h ≤ θ_v W/H  (W ≥ H),   max(θ_h, θ_v) ≤ θ_d ≤ min(θ_h r_d/r_h, θ_v r_d/r_v).
The error is the largest violation of these bounds in degrees (full angles). Tolerance 3°: the makers state FOVs to
about ±3° (Intel D400 datasheet Table 3-49 note 3; Orbbec "± 3°"), and rounding to whole degrees adds up to about
1°. Imagers whose entry declares figures_conflict (the maker's own angles cannot all hold) are reported, not counted;
so are fisheye imagers (a fisheye lens model, or an angle of 180° or more): the equisolid θ = 2 asin(r/2f) and
orthographic θ = asin(r/f) projections grow faster than linearly and may exceed the bounds legitimately.

library.fov_focal — where a focal length is stated: no lens without pincushion distortion shows less than the pinhole
of that focal length, θ ≥ atan(r/f) with r = p N/2 (N = W, H or the diagonal in pixels), so the stated full angle is at
least 2 atan(p N / 2f). The error is the largest shortfall in degrees; tolerance 3° (the same FOV tolerance). The
equivalent barrel distortion r / (f tan θ) − 1 the stated angle implies is reported beside it.

Known limits: the bounds hold for centred, radially symmetric lenses without pincushion distortion and square pixels;
a decentred or anamorphic lens, or a FOV measured on a crop other than the stated native image, can break them
without a typing error. Optical formats in inches are nominal types, not sizes, and are not checked.
"""
from __future__ import annotations

import math

from .suite import Outcome, check

FOV_TOL_DEG = 3.0


def _imagers():
    """(entry, imager) for every imager of every built-in camera and depth entry."""
    from ..core import library as L
    for e in L.entries():
        if e.is_custom or e.kind not in ('camera', 'depth'):
            continue
        for im in e.imagers:
            yield e, im


def fov_violations(W: float, H: float, h, v, d) -> list:
    """[(bound, violation in degrees)] of the stated full angles h, v, d (degrees, None when not stated) against the
    bounds a centred, radially symmetric, non-pincushion lens obeys on a W × H image (module docstring)."""
    out = []
    Dg = math.hypot(W, H)
    if h is not None and v is not None:
        if W >= H:
            out += [('vertical wider than horizontal', v - h), ('horizontal / vertical above W / H', h - v * W / H)]
        else:
            out += [('horizontal wider than vertical', h - v), ('vertical / horizontal above H / W', v - h * H / W)]
    if d is not None:
        sides = [x for x in (h, v) if x is not None]
        if sides:
            out.append(('diagonal narrower than a side', max(sides) - d))
        if h is not None:
            out.append(('diagonal above horizontal x diag / W', d - h * Dg / W))
        if v is not None:
            out.append(('diagonal above vertical x diag / H', d - v * Dg / H))
    return out


def _fisheye(im, *angles) -> bool:
    """A fisheye imager: a fisheye lens model, or a stated angle of 180° or more."""
    return (im.get('lens_model') not in (None, 'pinhole', 'brown')) or any(a is not None and a >= 180.0 for a in angles)


def _cases(im):
    res = im.get('resolution')
    if res:
        yield 'native', res[0], res[1], im.get('hfov_deg'), im.get('vfov_deg'), im.get('dfov_deg')
    for fm in im.get('fov_modes') or []:
        a = fm.get('aspect')
        if a:
            yield f'{a[0]}:{a[1]} mode', a[0], a[1], fm.get('hfov_deg'), fm.get('vfov_deg'), fm.get('dfov_deg')


@check('library.sensor_geometry', 'library', 'Pixel pitch × resolution against the stated sensor size',
       'largest relative difference between pixel pitch × native resolution and the stated active-area width, height '
       'and diagonal, over every built-in entry that states them (data consistency)',
       'the geometry of a pixel grid, p·W, p·H and p·sqrt(W² + H²), written out again here', 0.02, 'fraction',
       refs=('hartley2004',))
def _sensor_geometry(ctx):
    rows, worst = [], 0.0
    for e, im in _imagers():
        p, res = im.get('pixel_um'), im.get('resolution')
        size, diag = im.get('sensor_size_mm'), im.get('sensor_diagonal_mm')
        if not p or not res or not (size or diag):
            continue
        q = p / 1000.0
        W, H = res
        pairs = []
        if size:
            pairs += [('width', q * W, size[0]), ('height', q * H, size[1])]
        if diag:
            pairs.append(('diagonal', q * math.hypot(W, H), diag))
        for what, got, stated in pairs:
            err = abs(got - stated) / stated
            worst = max(worst, err)
            rows.append({'entry': e.id, 'imager': im['name'], 'what': what, 'pitch_x_pixels_mm': round(got, 4),
                         'stated_mm': stated, 'relative': round(err, 5)})
    if not rows:
        return Outcome(float('inf'), {}, ok=False, note='No built-in entry states a sensor size: nothing to check.')
    return Outcome(worst, {'rows': len(rows), 'entries': len({r['entry'] for r in rows}), 'checked': rows},
                   note='Measured on the shipped entries. A failure is a mistyped pixel pitch, resolution or size.')


@check('library.fov_shape', 'library', 'Stated FOVs against the image shape',
       'largest violation (degrees) of the bounds a centred, radially symmetric lens without pincushion distortion '
       'puts on the stated horizontal, vertical and diagonal FOVs for the native image shape, over every built-in '
       'camera and depth imager (data consistency); imagers declaring figures_conflict, and fisheye imagers, are '
       'reported, not counted',
       'the concave field-angle bounds θ_v ≤ θ_h ≤ θ_v W/H and max(θ_h, θ_v) ≤ θ_d ≤ min(θ_h r_d/r_h, θ_v r_d/r_v), '
       'derived and written out here (pinhole and fisheye projections per Hartley-Zisserman and Kannala-Brandt)',
       FOV_TOL_DEG, 'deg', refs=('hartley2004', 'kannala_brandt2006', 'intel_d400'))
def _fov_shape(ctx):
    worst, counted, flagged, failing, conflicts, fisheyes = 0.0, 0, [], [], [], []
    for e, im in _imagers():
        for tag, W, H, h, v, d in _cases(im):
            vs = fov_violations(W, H, h, v, d)
            if not vs:
                continue
            bound, viol = max(vs, key=lambda x: x[1])
            row = {'entry': e.id, 'imager': im['name'], 'case': tag, 'fov_deg': [h, v, d], 'bound': bound,
                   'violation_deg': round(max(viol, 0.0), 3)}
            if im.get('figures_conflict'):
                conflicts.append(row)
                continue
            if _fisheye(im, h, v, d):
                fisheyes.append(row)
                continue
            counted += 1
            worst = max(worst, viol)
            if viol > 0:
                flagged.append(row)
            if viol > FOV_TOL_DEG:
                failing.append(row)
    flagged.sort(key=lambda r: -r['violation_deg'])
    return Outcome(max(worst, 0.0), {'cases_checked': counted, 'within_rounding_but_positive': flagged[:12],
                                     'over_tolerance': failing, 'declared_conflicts': conflicts,
                                     'fisheye_reported': fisheyes},
                   note=f'{counted} FOV statements checked; {len(conflicts)} imagers declare that the maker\'s own '
                        f'angles disagree (figures_conflict) and {len(fisheyes)} FOV statements are of fisheye '
                        'lenses (the bounds need not hold): both are listed, not counted.')


@check('library.fov_focal', 'library', 'Stated FOVs against the stated focal length',
       'largest shortfall (degrees) of a stated FOV below the pinhole FOV 2 atan(p N / 2f) of the stated focal length, '
       'pixel pitch and native resolution, over every built-in imager that states all three (data consistency)',
       'the pinhole projection r = f tan θ (Hartley-Zisserman), a lower bound on the FOV of any lens without '
       'pincushion distortion, written out again here', FOV_TOL_DEG, 'deg', refs=('hartley2004',))
def _fov_focal(ctx):
    rows, worst = [], -math.inf
    for e, im in _imagers():
        f, p, res = im.get('focal_length_mm'), im.get('pixel_um'), im.get('resolution')
        if not f or not p or not res:
            continue
        q = p / 1000.0
        for key, n in (('hfov_deg', res[0]), ('vfov_deg', res[1]), ('dfov_deg', math.hypot(*res))):
            stated = im.get(key)
            if stated is None:
                continue
            r = q * n / 2.0
            pin = 2.0 * math.degrees(math.atan(r / f))
            implied = (r / (f * math.tan(math.radians(stated) / 2.0)) - 1.0) if stated < 180.0 else None
            short = pin - stated
            if not im.get('figures_conflict'):
                worst = max(worst, short)
            rows.append({'entry': e.id, 'imager': im['name'], 'fov': key[:4], 'stated_deg': stated,
                         'pinhole_deg': round(pin, 3), 'shortfall_deg': round(short, 3),
                         'implied_barrel_distortion': None if implied is None else round(implied, 4),
                         'declared_conflict': bool(im.get('figures_conflict'))})
    if not rows:
        return Outcome(float('inf'), {}, ok=False, note='No built-in entry states a focal length: nothing to check.')
    return Outcome(max(worst, 0.0), {'rows': len(rows), 'checked': rows},
                   note='A negative shortfall means the lens shows more than the pinhole of its focal length (barrel '
                        'distortion, as expected for wide lenses); the implied distortion is reported, not tested.')
