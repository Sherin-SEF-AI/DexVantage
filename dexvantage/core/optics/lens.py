"""Camera models: projection, unprojection, fields of view, pixel density and defocus.

Ported from Vantage core/lens.py: the radial functions r(θ) of the projections (r_of_theta, dr_dtheta, theta_of_r),
the Kannala-Brandt polynomial, fitting Kannala-Brandt terms to a stated field of view (best_base, focal_from_fov,
fit_kb_to_fov, fit_sheet) and the projection / unprojection of Vantage's Intrinsics. Stripped: the automotive camera
library, sensor sizes in millimetres (DexVantage works in pixels throughout: the radial functions are normalised,
r / f) and Vantage's pixel convention. Added: the OpenCV Brown-Conrady model, OpenCV's parameterisation of every model,
an exact (constrained) Kannala-Brandt fit through the stated points, and the thin-lens defocus.

Conventions
-----------
- Points are in the camera's **optical frame** (REP-103 `_optical_frame`): z forward along the optical axis, x right,
  y down, metres.
- Pixels follow **OpenCV's convention: the centre of pixel (i, j) is the point u = i, v = j**, so the image spans
  u ∈ [−0.5, W − 0.5] and v ∈ [−0.5, H − 0.5], and a centred principal point is cx = (W − 1)/2. Vantage put pixel
  centres at i + ½ (image [0, W], centre W/2); DexVantage does not. `CameraModel.scaled` keeps the convention:
  cx' = s (cx + ½) − ½.
- u = fx·xd + cx, v = fy·yd + cy, with (xd, yd) the model's normalised image coordinates.

Models (`CameraModel.model`)
---------------------------
- ``pinhole``: xd = X/Z, yd = Y/Z (no distortion; `dist` is empty).
- ``brown``: OpenCV's Brown-Conrady model [Brown 1966; OpenCV calib3d], `dist` = (k1, k2, p1, p2, k3) or the rational
  (k1, k2, p1, p2, k3, k4, k5, k6): with x = X/Z, y = Y/Z, r² = x² + y²,
  xd = x·c + 2 p1 x y + p2 (r² + 2x²), yd = y·c + p1 (r² + 2y²) + 2 p2 x y,
  c = (1 + k1 r² + k2 r⁴ + k3 r⁶) / (1 + k4 r² + k5 r⁴ + k6 r⁶). Thin-prism and tilt terms are not supported.
- ``kb``: OpenCV's fisheye model, Kannala-Brandt [Kannala and Brandt 2006; OpenCV fisheye], `dist` = (k1, k2, k3, k4):
  θ = atan2(√(X² + Y²), Z), θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸), (xd, yd) = θd · (X, Y)/√(X² + Y²).
  OpenCV computes θ as atan(r) of X/Z, Y/Z, which is the same in front of the camera; here θ may exceed 90°.
- ``equidistant`` (r = f θ), ``equisolid`` (r = 2 f sin(θ/2)), ``stereographic`` (r = 2 f tan(θ/2)) [Miyamoto 1964]:
  the classical fisheye projections, (xd, yd) = r(θ)/f · (X, Y)/√(X² + Y²); `dist` is empty. ``equidistant`` is
  ``kb`` with k1 = … = k4 = 0 exactly.

**The valid angle.** `max_theta` is the largest angle off the optical axis (rad) the model maps: a point beyond it, or
behind the camera for ``pinhole`` and ``brown``, is never projected (uv is NaN, ok False) and a pixel beyond it has no
ray (NaN). It is never wrapped onto the image. When it is not given it is the angle where the model stops being
one-to-one: 90° for ``pinhole``; for ``brown`` the first angle where, in some direction around the axis, the image
radius stops rising or the distortion map's Jacobian determinant (or c's denominator) reaches zero; for ``kb`` the
first angle where θd(θ) stops rising, at most 180°; 180° for the classical fisheye projections. A given `max_theta`
beyond that angle is refused, since two rays would then share a pixel.

**Pixel density.** `pixels_per_metre(P)` is the local linear pixel density at P on a small flat patch facing the
camera (its normal along the line of sight): √|det J|, J the 2 × 2 Jacobian of (u, v) with respect to the two lateral
coordinates of the patch (metres), i.e. the square root of the pixels per square metre, the geometric mean of the
densities along the patch's two principal directions. Closed forms: fx fy / (R² cos³θ) for ``pinhole``;
fx fy |det D| / (R² cos³θ) for ``brown`` (D the Jacobian of the distortion map at X/Z, Y/Z); fx fy g'(θ) g(θ) /
(R² sin θ) for the radial models (g = r/f), with R = |P|. A tilted surface (`normals`) scales the area by |cos α|
(α between the normal and the line of sight; the line of sight is in J's null space), so the density by √|cos α|.
Pixels per millimetre are pixels_per_metre / 1000.

**Defocus.** `depth_of_field` and `blur_circle_px` use the thin lens [Ray 2002]: with focal length f, aperture
diameter A = f/N and focus distance s (object distances from the lens), a point at distance z images a blur disc of
diameter b = A · f |s − z| / (z (s − f)) on the sensor. The near and far limits solve b = c (the circle of confusion).
docs/models/optics.md derives this and lists the checks.

Known limits
------------
Thin lens only (no pupil magnification, no diffraction, no aberrations; the defocus distances are measured from the
lens's principal plane, not the camera's mounting face). No skew (OpenCV's fisheye alpha, K[0, 1]), no thin-prism or
tilt terms, no rolling-shutter or time dimension. A model fitted from a stated field of view (`from_fov`) reproduces
the stated angles; its distortion curve between them is a model, not a measurement. The Brown-Conrady valid angle is
found on 72 directions around the axis and 4000 angles (then refined), so a fold narrower than that could slip through;
a Jacobian that stays positive on a disc makes the map locally, not provably globally, one-to-one.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

MODELS = ('pinhole', 'brown', 'kb', 'equidistant', 'equisolid', 'stereographic')
RADIAL = ('kb', 'equidistant', 'equisolid', 'stereographic')      # the models that are a function of θ alone
MODEL_LABEL = {'pinhole': 'Pinhole', 'brown': 'Brown-Conrady (OpenCV)', 'kb': 'Kannala-Brandt (OpenCV fisheye)',
               'equidistant': 'Equidistant fisheye', 'equisolid': 'Equisolid fisheye',
               'stereographic': 'Stereographic fisheye'}
# OpenCV and ROS names accepted by CameraModel.from_opencv
ALIASES = {'fisheye': 'kb', 'plumb_bob': 'brown', 'rational_polynomial': 'brown', 'opencv': 'brown'}
FIT_BASES = ('pinhole', 'stereographic', 'equidistant', 'equisolid')
SHEET_WEIGHTS = (50.0, 20.0, 8.0)   # fit_sheet's fallback weights (Vantage's SHEET_WEIGHT and its gentler ones)
FIT_MARGIN_DEG = 2.0                # from_fov('kb'): the valid angle reaches this far past the image corner
EDGE_TOL_PX = 1e-9                  # the image edges are widened by this in project's ok (rounding of an edge ray)
# a default of DexVantage: from_fov refuses a fit that misses a stated angle by more than this (fit_fov reports it)
FOV_FIT_TOL_DEG = 0.5
# a default of DexVantage (not a standard): a blur spot up to two pixels across counts as sharp; a user setting
DEFAULT_COC_PX = 2.0
_PINHOLE_CAP = math.radians(89.9)   # tan() is evaluated below this in the fitting helpers (Vantage's cap)


# ---------------------------------------------------------------------------- radial functions (normalised, r / f)
def r_of_theta(theta, model: str, dist=()) -> np.ndarray:
    """g(θ) = r/f, the normalised image radius of a ray θ rad off the axis (ported from Vantage r_of_theta)."""
    t = np.asarray(theta, float)
    if model in ('pinhole', 'brown'):
        return np.tan(np.minimum(t, _PINHOLE_CAP))
    if model == 'equidistant':
        return t.copy()
    if model == 'equisolid':
        return 2.0 * np.sin(t / 2.0)
    if model == 'stereographic':
        return 2.0 * np.tan(t / 2.0)
    if model == 'kb':
        k1, k2, k3, k4 = _kb4(dist)
        t2 = t * t
        return t * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(f'unknown lens model {model!r}')


def dr_dtheta(theta, model: str, dist=()) -> np.ndarray:
    """g'(θ), the derivative of r_of_theta (ported from Vantage dr_dtheta)."""
    t = np.asarray(theta, float)
    if model in ('pinhole', 'brown'):
        c = np.cos(np.minimum(t, _PINHOLE_CAP))
        return 1.0 / (c * c)
    if model == 'equidistant':
        return np.ones_like(t)
    if model == 'equisolid':
        return np.cos(t / 2.0)
    if model == 'stereographic':
        c = np.cos(t / 2.0)
        return 1.0 / (c * c)
    if model == 'kb':
        k1, k2, k3, k4 = _kb4(dist)
        t2 = t * t
        return 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
    raise ValueError(f'unknown lens model {model!r}')


def theta_of_r(r, model: str, dist=(), max_theta: float = math.pi) -> np.ndarray:
    """The inverse of r_of_theta on [0, max_theta]: θ for a normalised radius r, NaN where no ray in that range has
    it. Closed forms for the classical projections; for Kannala-Brandt, Newton's method on θ safeguarded by
    bisection on [0, min(max_theta, the angle where θd(θ) stops rising)], where the inverse is unique."""
    x = np.asarray(r, float)
    with np.errstate(invalid='ignore'):
        if model in ('pinhole', 'brown'):
            t = np.arctan(x)
        elif model == 'equidistant':
            t = x.copy()
        elif model == 'equisolid':
            t = np.where(x <= 2.0, 2.0 * np.arcsin(np.clip(x / 2.0, -1.0, 1.0)), np.nan)
        elif model == 'stereographic':
            t = 2.0 * np.arctan(x / 2.0)
        elif model == 'kb':
            t = _kb_theta(x, _kb4(dist), min(float(max_theta), kb_fold_angle(dist)))
        else:
            raise ValueError(f'unknown lens model {model!r}')
        return np.where((x >= 0) & (t <= max_theta + 1e-12), t, np.nan)


def _kb4(dist) -> tuple:
    d = tuple(float(v) for v in dist) if dist is not None else ()
    return (d + (0.0, 0.0, 0.0, 0.0))[:4]


def _kb_theta(rd: np.ndarray, k: tuple, tmax: float) -> np.ndarray:
    """θ with θ(1 + k1θ² + …) = rd on [0, tmax], safeguarded Newton (a bisection step whenever Newton leaves the
    bracket). NaN where rd is beyond the curve's value at tmax or the iteration does not converge."""
    rd = np.asarray(rd, float)
    shape = rd.shape
    rd = rd.ravel()
    gmax = float(r_of_theta(tmax, 'kb', k))
    reach = np.isfinite(rd) & (rd >= 0) & (rd <= gmax * (1 + 1e-12) + 1e-15)
    target = np.where(reach, np.minimum(rd, gmax), 0.0)
    lo = np.zeros_like(target)
    hi = np.full_like(target, tmax)
    t = np.clip(target, 0.0, tmax)            # θ ≈ θd: the equidistant first guess
    act = np.flatnonzero(reach)                # iterate only the entries still moving
    for _ in range(100):
        if not act.size:
            break
        ta, ra = t[act], target[act]
        g = r_of_theta(ta, 'kb', k) - ra
        la = np.where(g <= 0, ta, lo[act])
        ha = np.where(g >= 0, ta, hi[act])
        d = dr_dtheta(ta, 'kb', k)
        with np.errstate(divide='ignore', invalid='ignore'):
            tn = ta - g / d
        bad = ~np.isfinite(tn) | (tn < la) | (tn > ha)
        tn = np.where(bad, 0.5 * (la + ha), tn)
        t[act], lo[act], hi[act] = tn, la, ha
        moving = np.abs(tn - ta) > 1e-15 * np.maximum(tn, 1.0)
        act = act[moving]
    res = np.abs(r_of_theta(t, 'kb', k) - target)
    conv = res <= 1e-12 * np.maximum(target, 1.0)
    return np.where(reach & conv, t, np.nan).reshape(shape)


def _first_fold(deriv, lo: float, hi: float, n: int = 4000) -> float:
    """The first angle in (lo, hi] where deriv(θ) > 0 stops holding (refined by bisection), or hi when it holds all the
    way. deriv returns a boolean array: True where the curve still rises."""
    t = np.linspace(lo, hi, n)
    good = np.asarray(deriv(t), bool)
    if good.all():
        return hi
    k = int(np.argmin(good))
    if k == 0:
        return lo
    a, b = float(t[k - 1]), float(t[k])
    for _ in range(60):
        m = 0.5 * (a + b)
        if bool(np.asarray(deriv(np.array([m])), bool)[0]):
            a = m
        else:
            b = m
    return a


def kb_monotonic(dist, max_theta: float, n: int = 720) -> bool:
    """True when the Kannala-Brandt curve θd(θ) rises all the way to max_theta (a curve that folds back maps two rays to
    one pixel). Ported from Vantage monotonic."""
    t = np.linspace(1e-4, max(float(max_theta), 1e-3), n)
    return bool(np.all(dr_dtheta(t, 'kb', dist) > 0))


def kb_fold_angle(dist) -> float:
    """The first angle (rad, at most π) where θd(θ) stops rising. Cached by the terms."""
    return _kb_fold(_kb4(dist))


@functools.lru_cache(maxsize=256)
def _kb_fold(dist: tuple) -> float:
    return _first_fold(lambda t: (dr_dtheta(t, 'kb', dist) > 0) & (r_of_theta(t, 'kb', dist) > 0) | (t == 0),
                       0.0, math.pi)


# ---------------------------------------------------------------------------- Brown-Conrady
def _brown8(dist) -> tuple:
    d = tuple(float(v) for v in dist) if dist is not None else ()
    return (d + (0.0,) * 8)[:8]


def _brown(x, y, dist):
    """OpenCV's Brown-Conrady distortion of normalised coordinates (x, y) = (X/Z, Y/Z)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = _brown8(dist)
    r2 = x * x + y * y
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    c = num / den
    xy = x * y
    return x * c + 2 * p1 * xy + p2 * (r2 + 2 * x * x), y * c + p1 * (r2 + 2 * y * y) + 2 * p2 * xy


def _brown_jac(x, y, dist):
    """The Jacobian ∂(xd, yd)/∂(x, y) of _brown, as its four entries (j00, j01, j10, j11)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = _brown8(dist)
    r2 = x * x + y * y
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    dnum = k1 + r2 * (2 * k2 + 3 * k3 * r2)
    dden = k4 + r2 * (2 * k5 + 3 * k6 * r2)
    c = num / den
    dc = (dnum * den - num * dden) / (den * den)      # ∂c/∂(r²)
    j00 = c + 2 * x * x * dc + 2 * p1 * y + 6 * p2 * x
    j01 = 2 * x * y * dc + 2 * p1 * x + 2 * p2 * y
    j10 = 2 * x * y * dc + 2 * p1 * x + 2 * p2 * y
    j11 = c + 2 * y * y * dc + 6 * p1 * y + 2 * p2 * x
    return j00, j01, j10, j11


def brown_fold_angle(dist, directions: int = 72) -> float:
    """The first angle off the axis (rad, below 90°) where the Brown-Conrady map stops being one-to-one: in some
    direction (of `directions` around the axis) the image radius stops rising, the Jacobian determinant of the
    distortion map reaches zero, or c's denominator does. Tangential terms grow as r² against the radial r·c, so a
    model with p1 or p2 always folds somewhere. Cached by the terms."""
    return _brown_fold(_brown8(dist), int(directions))


@functools.lru_cache(maxsize=256)
def _brown_fold(dist: tuple, directions: int) -> float:
    if not any(dist):
        return math.pi / 2
    phi = np.linspace(0.0, 2 * math.pi, directions, endpoint=False)
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]
    k1, k2, p1, p2, k3, k4, k5, k6 = dist

    def rising(t):
        r = np.tan(np.asarray(t, float))[:, None]
        x, y = r * cp, r * sp
        with np.errstate(all='ignore'):
            r2 = x * x + y * y
            den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
            num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            j00, j01, j10, j11 = _brown_jac(x, y, dist)
            radial = cp * (j00 * cp + j01 * sp) + sp * (j10 * cp + j11 * sp)
            good = (den > 0) & (num / den > 0) & (j00 * j11 - j01 * j10 > 0) & (radial > 0)
        return good.all(axis=1)
    return _first_fold(rising, 0.0, math.pi / 2 - 1e-9)


def _undistort(xd, yd, dist, rmax: float):
    """(x, y, converged) with _brown(x, y) = (xd, yd) and |(x, y)| ≤ rmax: Newton's method with the analytic Jacobian
    (OpenCV iterates the fixed point instead), kept inside the one-to-one disc. converged: the distortion of the
    result is within 1e-12 of the target (normalised units; about 1e-9 px at f = 1000 px)."""
    x, y = xd.copy(), yd.copy()
    act = np.flatnonzero(np.isfinite(xd) & np.isfinite(yd))   # iterate only the entries still moving
    clamped = np.zeros(xd.shape, int)
    for _ in range(60):
        if not act.size:
            break
        xa, ya = x[act], y[act]
        ex, ey = _brown(xa, ya, dist)
        ex, ey = ex - xd[act], ey - yd[act]
        j00, j01, j10, j11 = _brown_jac(xa, ya, dist)
        det = j00 * j11 - j01 * j10
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = (j11 * ex - j01 * ey) / det
            dy = (-j10 * ex + j00 * ey) / det
        dx = np.where(np.isfinite(dx), dx, 0.0)
        dy = np.where(np.isfinite(dy), dy, 0.0)
        xa, ya = xa - dx, ya - dy
        r = np.hypot(xa, ya)
        out = r > rmax
        if out.any():                          # back onto the one-to-one disc; twice out there: no root inside
            s = np.where(out, rmax / np.maximum(r, 1e-300), 1.0)
            xa, ya = xa * s, ya * s
        clamped[act] = np.where(out, clamped[act] + 1, 0)
        x[act], y[act] = xa, ya
        moving = (np.maximum(np.abs(dx), np.abs(dy)) > 1e-16 * np.maximum(r, 1.0)) & (clamped[act] < 2)
        act = act[moving]
    ex, ey = _brown(x, y, dist)
    res = np.hypot(ex - xd, ey - yd)
    conv = (res <= 1e-12 * np.maximum(np.hypot(xd, yd), 1.0)) & (np.hypot(x, y) <= rmax * (1 + 1e-12))
    return x, y, conv


# ---------------------------------------------------------------------------- the camera model
def _rows(P, n: int):
    a = np.asarray(P, dtype=np.float64)
    if a.shape[-1:] != (n,):
        raise ValueError(f'expected an array of shape (N, {n}), got {a.shape}')
    return a.reshape(-1, n), a.shape[:-1]


@dataclass(frozen=True)
class CameraModel:
    """One imager's intrinsics: image size (px), focal lengths and principal point (px, OpenCV convention: pixel
    centres at integers), the projection model, its distortion terms and the valid half angle (rad). See the module
    docstring for the equations and conventions. Immutable and hashable (a cache key)."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = 'pinhole'
    dist: tuple = ()
    max_theta: float | None = None

    def __post_init__(self):
        s = object.__setattr__
        try:
            w, h = int(self.width), int(self.height)
        except (TypeError, ValueError):
            raise ValueError(f'the image size must be whole pixels, got {self.width!r} × {self.height!r}') from None
        if w < 1 or h < 1 or w != self.width or h != self.height:
            raise ValueError(f'the image size must be at least 1 × 1 whole pixels, got {self.width} × {self.height}')
        s(self, 'width', w)
        s(self, 'height', h)
        for k in ('fx', 'fy', 'cx', 'cy'):
            v = float(getattr(self, k))
            if not math.isfinite(v):
                raise ValueError(f'{k} must be a finite number of pixels, got {getattr(self, k)!r}')
            s(self, k, v)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f'the focal lengths must be positive, got fx = {self.fx:g} px, fy = {self.fy:g} px')
        m = str(self.model)
        if m not in MODELS:
            raise ValueError(f'unknown lens model {m!r}: use one of {", ".join(MODELS)}')
        s(self, 'model', m)
        d = tuple(float(v) for v in (self.dist or ()))
        if not all(math.isfinite(v) for v in d):
            raise ValueError('the distortion terms must be finite numbers')
        if m == 'brown':
            if len(d) in (0, 4):
                d = (d + (0.0,) * 5)[:5]
            if len(d) not in (5, 8):
                raise ValueError(f'a Brown-Conrady model takes 4, 5 or 8 distortion terms (k1, k2, p1, p2[, k3[, k4, '
                                 f'k5, k6]]), got {len(d)}; thin-prism and tilt terms are not supported')
        elif m == 'kb':
            if len(d) == 0:
                d = (0.0,) * 4
            if len(d) != 4:
                raise ValueError(f'a Kannala-Brandt (OpenCV fisheye) model takes 4 distortion terms (k1..k4), got '
                                 f'{len(d)}')
        else:
            if any(v != 0.0 for v in d):
                raise ValueError(f'the {m} model has no distortion terms; use "brown" or "kb" for a distorted lens')
            d = ()
        s(self, 'dist', d)
        fold = self._fold()
        if self.max_theta is None:
            mt = fold
        else:
            mt = float(self.max_theta)
            if not math.isfinite(mt) or mt <= 0:
                raise ValueError(f'the valid angle must be a positive number of radians, got {self.max_theta!r}')
            if mt > fold + 1e-9:
                why = 'a pinhole camera sees only in front of itself' if m in ('pinhole', 'brown') and \
                    fold == math.pi / 2 else 'the model folds back there, so two rays would share a pixel'
                raise ValueError(f'the valid angle {math.degrees(mt):.2f}° is beyond {math.degrees(fold):.2f}° off the '
                                 f'axis, where the {MODEL_LABEL[m]} model stops being one-to-one ({why})')
            mt = min(mt, fold)
        s(self, 'max_theta', mt)

    def _fold(self) -> float:
        if self.model == 'pinhole':
            return math.pi / 2
        if self.model == 'brown':
            return brown_fold_angle(self.dist)
        if self.model == 'kb':
            return kb_fold_angle(self.dist)
        return math.pi

    # ------------------------------------------------------------------ projection
    def _normalised(self, P: np.ndarray):
        """(xd, yd, valid) of optical-frame points (N, 3): the model's normalised image coordinates."""
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        valid = np.isfinite(P).all(axis=1) & ((rho > 0) | (z > 0)) & (theta <= self.max_theta)
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.model in ('pinhole', 'brown'):
                valid &= z > 0
                zz = np.where(valid, z, 1.0)
                xn, yn = x / zz, y / zz
                if self.model == 'brown':
                    xn, yn = _brown(xn, yn, self.dist)
            else:
                g = r_of_theta(theta, self.model, self.dist)
                sc = np.where(rho > 0, g / np.where(rho > 0, rho, 1.0), 0.0)
                xn, yn = x * sc, y * sc
        xn = np.where(valid, xn, np.nan)
        yn = np.where(valid, yn, np.nan)
        return xn, yn, valid

    def project(self, P):
        """Optical-frame points (N, 3) (any leading shape) → (uv (N, 2) float64, ok (N,) bool). ok: in front of the
        camera within the valid angle and inside the image (u in [−0.5, W − 0.5], v in [−0.5, H − 0.5], each edge
        widened by EDGE_TOL_PX so a ray unprojected from the edge itself counts as inside). uv is NaN
        where the point is outside the valid angle (or behind a pinhole / Brown camera, or at the origin), and finite
        but ok False where it is valid but lands off the image."""
        A, shape = _rows(P, 3)
        xn, yn, valid = self._normalised(A)
        u = self.fx * xn + self.cx
        v = self.fy * yn + self.cy
        e = 0.5 + EDGE_TOL_PX
        with np.errstate(invalid='ignore'):
            inside = (u >= -e) & (u <= self.width - 1 + e) & (v >= -e) & (v <= self.height - 1 + e)
        ok = valid & inside
        return np.stack([u, v], axis=-1).reshape(shape + (2,)), ok.reshape(shape)

    def unproject(self, uv, with_ok: bool = False):
        """Pixel coordinates (N, 2) (any leading shape) → unit rays (N, 3) in the optical frame. A pixel with no ray
        within the valid angle (beyond a fisheye's image circle, beyond the Brown-Conrady one-to-one disc, or where
        the iterative inverse does not converge) gives NaN. with_ok=True also returns the (N,) validity."""
        A, shape = _rows(uv, 2)
        xn = (A[:, 0] - self.cx) / self.fx
        yn = (A[:, 1] - self.cy) / self.fy
        if self.model in ('pinhole', 'brown'):
            if self.model == 'brown':
                xn, yn, conv = _undistort(xn, yn, self.dist, math.tan(min(self.max_theta, math.pi / 2 - 1e-12)))
            else:
                conv = np.ones(xn.shape, bool)
            n = np.sqrt(xn * xn + yn * yn + 1.0)
            rays = np.stack([xn / n, yn / n, 1.0 / n], axis=-1)
            theta = np.arctan2(np.hypot(xn, yn), 1.0)
            valid = conv & np.isfinite(n) & (theta <= self.max_theta + 1e-12)
        else:
            rn = np.hypot(xn, yn)
            theta = theta_of_r(rn, self.model, self.dist, self.max_theta)
            valid = np.isfinite(theta)
            st = np.sin(np.where(valid, theta, 0.0))
            with np.errstate(divide='ignore', invalid='ignore'):
                sc = np.where(rn > 0, st / np.where(rn > 0, rn, 1.0), 0.0)
            rays = np.stack([xn * sc, yn * sc, np.cos(np.where(valid, theta, 0.0))], axis=-1)
        rays[~valid] = np.nan
        rays = rays.reshape(shape + (3,))
        return (rays, valid.reshape(shape)) if with_ok else rays

    def pixel_rays(self, stride: int = 1) -> np.ndarray:
        """(H', W', 3) unit rays through the pixel centres u = 0, stride, 2·stride … < W (likewise v); NaN for pixels
        with no ray within the valid angle."""
        stride = int(stride)
        if stride < 1:
            raise ValueError(f'the pixel stride must be at least 1, got {stride}')
        us = np.arange(0, self.width, stride, dtype=float)
        vs = np.arange(0, self.height, stride, dtype=float)
        U, V = np.meshgrid(us, vs)
        return self.unproject(np.stack([U, V], axis=-1))

    def angle_to_axis(self, P) -> np.ndarray:
        """(N,) angle (rad) between the optical axis and the line of sight to each point."""
        A, shape = _rows(P, 3)
        return np.arctan2(np.hypot(A[:, 0], A[:, 1]), A[:, 2]).reshape(shape)

    def pixels_per_metre(self, P, normals=None) -> np.ndarray:
        """(N,) local linear pixel density (px/m) at optical-frame points P: √|det ∂(u, v)/∂(lateral)| on a small
        patch facing the camera (module docstring), NaN outside the valid angle. normals (N, 3) or (3,), in the
        optical frame: the density on a surface with those normals instead (× √|cos α|, α the angle between the
        normal and the line of sight). Pixels per millimetre = pixels_per_metre / 1000."""
        A, shape = _rows(P, 3)
        x, y, z = A[:, 0], A[:, 1], A[:, 2]
        R = np.linalg.norm(A, axis=1)
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        _, _, valid = self._normalised(A)
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.model in ('pinhole', 'brown'):
                c = np.cos(theta)
                det = self.fx * self.fy / (R * R * c * c * c)
                if self.model == 'brown':
                    zz = np.where(valid, z, 1.0)
                    j00, j01, j10, j11 = _brown_jac(x / zz, y / zz, self.dist)
                    det = det * np.abs(j00 * j11 - j01 * j10)
            else:
                g = r_of_theta(theta, self.model, self.dist)
                dg = dr_dtheta(theta, self.model, self.dist)
                st = np.sin(theta)
                tang = np.where(st > 1e-12, g / np.where(st > 1e-12, st, 1.0), dr_dtheta(0.0, self.model, self.dist))
                det = self.fx * self.fy * dg * tang / (R * R)
            ppm = np.sqrt(np.abs(det))
            if normals is not None:
                N = np.broadcast_to(np.asarray(normals, float), A.shape)
                nn = np.linalg.norm(N, axis=1)
                cosa = np.abs(np.einsum('ij,ij->i', N, A)) / (nn * R)
                ppm = ppm * np.sqrt(cosa)
        return np.where(valid, ppm, np.nan).reshape(shape)

    def pixels_per_mm(self, P, normals=None) -> np.ndarray:
        """pixels_per_metre / 1000."""
        return self.pixels_per_metre(P, normals) / 1000.0

    # ------------------------------------------------------------------ fields of view
    def _edge_theta(self, u: float, v: float) -> float:
        ray = self.unproject(np.array([[u, v]], float))[0]
        if not np.isfinite(ray).all():
            return self.max_theta           # the image ends at the valid angle before this edge
        return float(np.arctan2(math.hypot(ray[0], ray[1]), ray[2]))

    def fov_deg(self) -> dict:
        """{'h', 'v', 'd'} degrees, from the model: the angle from the left edge (u = −0.5) to the right edge
        (u = W − 0.5) through the principal point's row, likewise top to bottom, and corner to corner (the mean of the
        two diagonals). An edge beyond the valid angle counts at the valid angle (a circular fisheye image)."""
        W, H = self.width - 0.5, self.height - 0.5
        e = self._edge_theta
        h = e(-0.5, self.cy) + e(W, self.cy)
        v = e(self.cx, -0.5) + e(self.cx, H)
        d = 0.5 * ((e(-0.5, -0.5) + e(W, H)) + (e(W, -0.5) + e(-0.5, H)))
        return {'h': math.degrees(h), 'v': math.degrees(v), 'd': math.degrees(d)}

    # ------------------------------------------------------------------ constructors and conversions
    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, vfov_deg: float | None = None,
                 model: str = 'pinhole', *, dfov_deg: float | None = None,
                 max_theta: float | None = None) -> 'CameraModel':
        """A model with a centred principal point (cx = (W − 1)/2, cy = (H − 1)/2) that reproduces the stated field
        of view edge to edge (fov_deg). See fit_fov for the rules and the fit's report. Raises ValueError, saying
        why, when no one-to-one model reproduces the stated angles within FOV_FIT_TOL_DEG (a default; fit_fov
        returns such a best effort with its error instead)."""
        cam, rep = fit_fov(width, height, hfov_deg, vfov_deg, model, dfov_deg=dfov_deg, max_theta=max_theta)
        if rep['max_error_deg'] > FOV_FIT_TOL_DEG:
            got = rep['model_fov']
            said = ', '.join(f'{k} {v:g}°' for k, v in rep['stated'].items() if v)
            shows = ', '.join(f'{k} {got[k]:.1f}°' for k, v in rep['stated'].items() if v)
            raise ValueError(f'the stated field of view ({said}) cannot be reproduced on a {width} × {height} image by '
                             f'a one-to-one {MODEL_LABEL[rep["model"]]} lens with square pixels: the closest shows '
                             f'{shows}. Check that the angles belong to this resolution (a datasheet may state them '
                             f'for another crop), or leave out the vertical or diagonal angle')
        return cam

    @classmethod
    def from_opencv(cls, K, D, width: int, height: int, model: str = 'brown',
                    max_theta: float | None = None) -> 'CameraModel':
        """From OpenCV's camera matrix K (3 × 3) and distortion D: model 'pinhole', 'brown' (cv2.projectPoints;
        aliases 'plumb_bob', 'rational_polynomial'), 'kb' (cv2.fisheye; alias 'fisheye') or one of the classical
        fisheye projections (D empty). K must have no skew. 'pinhole' with non-zero D is read as 'brown' (that is
        what cv2.projectPoints does with it)."""
        m = ALIASES.get(str(model), str(model))
        if m not in MODELS:
            raise ValueError(f'unknown lens model {model!r}: use one of {", ".join(MODELS + tuple(ALIASES))}')
        K = np.asarray(K, float)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError('the camera matrix K must be a 3 × 3 array of finite numbers')
        if abs(K[0, 1]) > 1e-9 * max(abs(K[0, 0]), 1.0):
            raise ValueError('the camera matrix has a skew term K[0, 1]; DexVantage models no skew')
        if abs(K[1, 0]) > 1e-12 or abs(K[2, 0]) > 1e-12 or abs(K[2, 1]) > 1e-12 or abs(K[2, 2] - 1) > 1e-12:
            raise ValueError('the camera matrix must have the form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]')
        d = () if D is None else tuple(float(v) for v in np.asarray(D, float).ravel())
        if m == 'pinhole' and any(d):
            m = 'brown'
        return cls(int(width), int(height), float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), m,
                   () if m == 'pinhole' else d, max_theta)

    def to_opencv(self):
        """(K 3 × 3, D, model_str) for OpenCV: 'pinhole' (D five zeros) and 'brown' for cv2.projectPoints, 'kb' for
        cv2.fisheye. 'equidistant' is exactly 'kb' with zero terms; 'equisolid' and 'stereographic' have no OpenCV
        form and are returned as a least-squares Kannala-Brandt approximation (kb_approximation gives its error)."""
        K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        if self.model == 'pinhole':
            return K, np.zeros(5), 'pinhole'
        if self.model == 'brown':
            return K, np.array(self.dist, float), 'brown'
        if self.model == 'kb':
            return K, np.array(self.dist, float), 'kb'
        if self.model == 'equidistant':
            return K, np.zeros(4), 'kb'
        return K, np.array(self.kb_approximation()[0], float), 'kb'

    def kb_approximation(self, n: int = 400) -> tuple:
        """((k1, k2, k3, k4), largest error px) of the Kannala-Brandt curve closest (least squares in θd) to this
        radial model, over the angles the image shows (up to the farther image corner, at most the valid angle).
        Exact for 'kb' and 'equidistant'."""
        if self.model in ('kb', 'equidistant'):
            return (tuple(self.dist) if self.model == 'kb' else (0.0,) * 4), 0.0
        if self.model not in RADIAL:
            raise ValueError(f'the {self.model} model is not a fisheye; it has no Kannala-Brandt form')
        corners = [self._edge_theta(u, v) for u in (-0.5, self.width - 0.5) for v in (-0.5, self.height - 0.5)]
        tmax = min(max(corners), self.max_theta, math.radians(179.0))
        t = np.linspace(0.0, tmax, n)
        A = np.stack([t ** 3, t ** 5, t ** 7, t ** 9], axis=1)
        b = r_of_theta(t, self.model) - t
        k = np.linalg.lstsq(A, b, rcond=None)[0]
        err = float(np.max(np.abs(A @ k - b)) * max(self.fx, self.fy))
        return tuple(float(v) for v in k), err

    def scaled(self, width: int, height: int) -> 'CameraModel':
        """The same lens on the image resampled to width × height (a binned or scaled stream): focal lengths scale
        with the size, the principal point as cx' = s (cx + ½) − ½ (OpenCV's integer pixel centres)."""
        sx, sy = float(width) / self.width, float(height) / self.height
        return CameraModel(int(width), int(height), self.fx * sx, self.fy * sy, sx * (self.cx + 0.5) - 0.5,
                           sy * (self.cy + 0.5) - 0.5, self.model, self.dist, self.max_theta)

    def to_dict(self) -> dict:
        return {'width': self.width, 'height': self.height, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx,
                'cy': self.cy, 'model': self.model, 'dist': list(self.dist), 'max_theta': self.max_theta,
                'pixel_centres': 'opencv'}

    @classmethod
    def from_dict(cls, d: dict) -> 'CameraModel':
        """From to_dict's output. A missing figure raises ValueError naming it."""
        if not isinstance(d, dict):
            raise ValueError('a camera model must be a dictionary of its figures')
        missing = [k for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy') if d.get(k) is None]
        if missing:
            raise ValueError(f'the camera model lacks {", ".join(missing)}')
        if d.get('pixel_centres', 'opencv') != 'opencv':
            raise ValueError(f'pixel centres {d.get("pixel_centres")!r} are not OpenCV\'s (integer centres)')
        return cls(d['width'], d['height'], d['fx'], d['fy'], d['cx'], d['cy'], str(d.get('model', 'pinhole')),
                   tuple(d.get('dist') or ()), d.get('max_theta'))

    def summary(self) -> str:
        f = self.fov_deg()
        return (f'{MODEL_LABEL[self.model]}, {self.width} × {self.height} px, fx {self.fx:.1f} px, fy {self.fy:.1f} '
                f'px, FOV {f["h"]:.1f}° × {f["v"]:.1f}° ({f["d"]:.1f}° diagonal)')


def incidence_angle(P, normals) -> np.ndarray:
    """(N,) angle (rad) between a surface's normal and the line of sight from the camera centre to the point (0 = the
    surface faces the camera; either side of the surface counts alike). P and normals in the same frame, the camera
    at its origin."""
    A, shape = _rows(P, 3)
    N = np.broadcast_to(np.asarray(normals, float), A.shape)
    c = np.abs(np.einsum('ij,ij->i', N, A)) / (np.linalg.norm(N, axis=1) * np.linalg.norm(A, axis=1))
    return np.arccos(np.clip(c, 0.0, 1.0)).reshape(shape)


# ---------------------------------------------------------------------------- fitting to a stated field of view
def _sheet_points(fov_deg, half_px):
    th = [math.radians(f / 2) for f in tuple(fov_deg)[:3] if f and f > 0]
    r = [h for f, h in zip(tuple(fov_deg)[:3], half_px) if f and f > 0]
    return np.array(th, float), np.array(r, float)


def focal_from_fov(fov_deg, half_px, base: str) -> float:
    """The focal length (px) of a `base` projection that best reproduces the stated FOV points (h, v, d; None or 0 =
    not stated): least squares of r = f·g(θ) relative to each half size, so the diagonal does not dominate. Ported
    from Vantage focal_from_fov (millimetres there, pixels here)."""
    th, r = _sheet_points(fov_deg, half_px)
    if not len(th):
        raise ValueError('no field of view to fit')
    g = np.asarray(r_of_theta(th, base), float) / r
    return float(g.sum() / (g @ g))


def best_base(fov_deg, half_px, bases=FIT_BASES) -> tuple:
    """(base projection, focal px, relative RMS) of the base that alone best reproduces the stated FOV points: the
    starting curve for the distortion fit. Pinhole is a candidate only below 160° (a wider lens cannot be one).
    Ported from Vantage best_base."""
    th, r = _sheet_points(fov_deg, half_px)
    best = None
    for b in bases:
        if b == 'pinhole' and th.max() >= math.radians(80):
            continue
        f = focal_from_fov(fov_deg, half_px, b)
        rel = float(np.sqrt(np.mean((np.asarray(r_of_theta(th, b), float) * f / r - 1.0) ** 2)))
        if best is None or rel < best[2] - 1e-12:
            best = (b, f, rel)
    return best


def _kb_design(f, fov_deg, half_px, base, upto=0.0):
    th_s, r_s = _sheet_points(fov_deg, half_px)
    th_b = np.linspace(0.02, max(th_s.max() if len(th_s) else 1.2, float(upto or 0.0)), 24)
    r_b = r_of_theta(th_b, base) * f
    return th_s, r_s, th_b, r_b


def fit_kb_to_fov(f, fov_deg, half_px, base='equisolid', ridge=1e-3, weight=8.0, upto=0.0) -> tuple:
    """Kannala-Brandt k1..k4 so f·θd(θ_stated) matches the image half sizes (px): least squares through the stated
    points (weighted `weight` against 1) plus the base projection sampled densely, so the curve stays well behaved.
    upto: sample the base curve to at least this angle (rad). Ported from Vantage fit_kb_to_fov."""
    th_s, r_s, th_b, r_b = _kb_design(f, fov_deg, half_px, base, upto)
    th = np.concatenate([th_s, th_b])
    r = np.concatenate([r_s, r_b])
    w = np.concatenate([np.full(len(th_s), float(weight)), np.full(len(th_b), 1.0)])
    # r/f − θ = k1 θ³ + k2 θ⁵ + k3 θ⁷ + k4 θ⁹
    A = np.stack([th ** 3, th ** 5, th ** 7, th ** 9], axis=1) * w[:, None]
    b = (r / f - th) * w
    k = np.linalg.solve(A.T @ A + ridge * np.eye(4), A.T @ b)
    return tuple(float(v) for v in k)


def fit_kb_exact(f, fov_deg, half_px, base='equisolid', ridge=1e-3, upto=0.0) -> tuple:
    """Kannala-Brandt k1..k4 whose curve passes exactly through the stated points (up to three: four terms leave
    room) and is otherwise closest to the base projection: least squares on the base samples with the stated points
    as equality constraints (the KKT system). New in DexVantage; Vantage only weighted the points."""
    th_s, r_s, th_b, r_b = _kb_design(f, fov_deg, half_px, base, upto)
    A = np.stack([th_b ** 3, th_b ** 5, th_b ** 7, th_b ** 9], axis=1)
    b = r_b / f - th_b
    C = np.stack([th_s ** 3, th_s ** 5, th_s ** 7, th_s ** 9], axis=1)
    d = r_s / f - th_s
    n, m = 4, len(th_s)
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = 2 * (A.T @ A + ridge * np.eye(n))
    KKT[:n, n:] = C.T
    KKT[n:, :n] = C
    rhs = np.concatenate([2 * A.T @ b, d])
    k = np.linalg.lstsq(KKT, rhs, rcond=None)[0][:n]
    return tuple(float(v) for v in k)


def fit_sheet(f, fov_deg, half_px, base='equisolid', reach_deg=0.0) -> tuple:
    """(k1..k4, how): the exact fit when its curve still rises to reach_deg (default: the widest stated half angle
    + 2°), else Vantage's weighted fit at the strongest of SHEET_WEIGHTS whose curve rises (inconsistent figures would
    fold the exact curve back). how is 'exact' or 'weighted <w>'; ported from Vantage fit_sheet."""
    reach = math.radians(reach_deg or max(v / 2 for v in tuple(fov_deg)[:3] if v) + 2.0)
    upto = reach - math.radians(2.0)
    kb = fit_kb_exact(f, fov_deg, half_px, base=base, upto=upto)
    if kb_monotonic(kb, reach):
        return kb, 'exact'
    for w in SHEET_WEIGHTS:
        kb = fit_kb_to_fov(f, fov_deg, half_px, base=base, weight=w, upto=upto)
        if kb_monotonic(kb, reach):
            return kb, f'weighted {w:g}'
    return kb, f'weighted {SHEET_WEIGHTS[-1]:g} (folds back before {math.degrees(reach):.1f}°)'


def _fov_value(name, v, limit):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'the {name} field of view must be a number of degrees, got {v!r}') from None
    if not math.isfinite(v) or v <= 0 or v >= limit:
        raise ValueError(f'the {name} field of view must be between 0° and {limit:g}°, got {v:g}°')
    return v


def fit_fov(width: int, height: int, hfov_deg: float, vfov_deg: float | None = None, model: str = 'pinhole', *,
            dfov_deg: float | None = None, max_theta: float | None = None) -> tuple:
    """(CameraModel, report) reproducing a stated field of view on a width × height image, principal point centred.

    - 'pinhole', 'equidistant', 'equisolid', 'stereographic' and 'brown' (zero distortion): fx from the horizontal
      FOV with the model's own r(θ) (W/2 = fx·g(h/2)); fy from the vertical FOV when it is given, else fy = fx (square
      pixels). A diagonal FOV cannot be matched as well and is refused for these.
    - 'kb': square pixels. Horizontal only: the equidistant curve (k = 0). With a vertical and/or diagonal FOV: the
      base projection and focal length that alone fit the stated points best (best_base), then Kannala-Brandt terms
      through the points (fit_sheet: exact when the curve stays one-to-one, else weighted). The valid angle reaches
      FIT_MARGIN_DEG past the image corner unless max_theta is given.

    report: {'model', 'base', 'f_px', 'fit', 'stated': {h, v, d}, 'model_fov': {h, v, d}, 'max_error_deg', 'note'};
    max_error_deg is the largest miss of a stated angle; the note says which figures were stated and what is a model.
    """
    m = ALIASES.get(str(model), str(model))
    if m not in MODELS:
        raise ValueError(f'unknown lens model {model!r}: use one of {", ".join(MODELS)}')
    W, H = int(width), int(height)
    if W < 1 or H < 1:
        raise ValueError(f'the image size must be at least 1 × 1 pixels, got {width} × {height}')
    lim = 180.0 if m in ('pinhole', 'brown') else 360.0
    h = _fov_value('horizontal', hfov_deg, lim)
    if h is None:
        raise ValueError('a field of view needs at least the horizontal angle')
    v = _fov_value('vertical', vfov_deg, lim)
    dd = _fov_value('diagonal', dfov_deg, lim)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    half = (W / 2.0, H / 2.0, math.hypot(W, H) / 2.0)
    stated = {'h': h, 'v': v, 'd': dd}
    if m != 'kb':
        if dd is not None:
            raise ValueError(f'a {MODEL_LABEL[m]} model cannot match a stated diagonal FOV as well as the horizontal '
                             f'and vertical ones; leave the diagonal out or use model "kb", which fits all three')
        base = 'pinhole' if m in ('pinhole', 'brown') else m
        fx = half[0] / float(r_of_theta(math.radians(h / 2), base))
        fy = fx if v is None else half[1] / float(r_of_theta(math.radians(v / 2), base))
        cam = CameraModel(W, H, fx, fy, cx, cy, m, (), max_theta)
        fit, note_base, note = 'closed form', base, (
            f'{MODEL_LABEL[m]} with fx from the stated horizontal FOV' +
            (' and fy from the stated vertical FOV' if v is not None else ' and square pixels (fy = fx)') +
            ('; no distortion is stated, so the terms are zero' if m == 'brown' else ''))
        f_px = fx
    elif v is None and dd is None:
        fx = half[0] / math.radians(h / 2)
        cam = CameraModel(W, H, fx, fx, cx, cy, 'kb', (0.0,) * 4, max_theta)
        fit, note_base, f_px = 'closed form', 'equidistant', fx
        note = 'Only the horizontal FOV is stated: the equidistant curve (Kannala-Brandt terms zero), square pixels'
    else:
        sheet = (h, v or 0.0, dd or 0.0)
        note_base, f_px, rel = best_base(sheet, half)
        # the curve must stay one-to-one out to the image corner (as the base projection places it) + the margin
        corner_base = float(theta_of_r(half[2] / f_px, note_base))
        widest = max(x / 2 for x in sheet if x)
        reach_deg = max(widest, math.degrees(corner_base) if math.isfinite(corner_base) else widest) + FIT_MARGIN_DEG
        kb, fit = fit_sheet(f_px, sheet, half, base=note_base, reach_deg=min(reach_deg, 180.0))
        cam0 = CameraModel(W, H, f_px, f_px, cx, cy, 'kb', kb)
        if max_theta is None:
            corner = float(theta_of_r(half[2] / f_px, 'kb', kb, cam0.max_theta))
            mt = cam0.max_theta if not math.isfinite(corner) else min(cam0.max_theta,
                                                                      corner + math.radians(FIT_MARGIN_DEG))
        else:
            mt = max_theta
        cam = CameraModel(W, H, f_px, f_px, cx, cy, 'kb', kb, mt)
        note = (f'Kannala-Brandt fitted to the stated FOV ({", ".join(k for k in "hvd" if stated[k])}) with square '
                f'pixels, starting from the {note_base} projection (focal length {f_px:.1f} px from the stated FOV, '
                f'relative RMS {rel:.3f} alone); fit {fit}. The stated angles are reproduced; the curve between them '
                f'is a model, not a measurement.')
    got = cam.fov_deg()
    err = max(abs(got[k] - v) for k, v in stated.items() if v)
    if err > FOV_FIT_TOL_DEG:
        note += f' The model misses a stated angle by {err:.2f}°: the figures do not fit one such lens.'
    return cam, {'model': cam.model, 'base': note_base, 'f_px': float(f_px), 'fit': fit, 'stated': stated,
                 'model_fov': got, 'max_error_deg': float(err), 'note': note}


# ---------------------------------------------------------------------------- defocus (thin lens)
def _pos(name, v, unit):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'the {name} must be a number ({unit}), got {v!r}') from None
    if math.isnan(v) or v <= 0:
        raise ValueError(f'the {name} must be positive ({unit}), got {v:g}')
    return v


def depth_of_field(f_mm, f_number, focus_m, coc_mm) -> tuple:
    """(near_m, far_m): the nearest and farthest object distances whose blur disc is at most coc_mm on the sensor,
    thin lens focused at focus_m (inf: focused at infinity). far_m is inf from the hyperfocal distance on. Solved from
    the blur-disc equation b(z) = (f/N) f |s − z| / (z (s − f)) = c:

        near = s A f / (A f + c (s − f)),   far = s A f / (A f − c (s − f))  (inf when A f ≤ c (s − f)),  A = f/N.

    Any figure None (not stated) gives (None, None). Distances are from the lens's principal plane."""
    if f_mm is None or f_number is None or focus_m is None or coc_mm is None:
        return None, None
    f = _pos('focal length', f_mm, 'mm')
    N = _pos('f-number', f_number, 'f/N')
    c = _pos('circle of confusion', coc_mm, 'mm')
    s = _pos('focus distance', focus_m, 'm') * 1000.0
    if s <= f:
        raise ValueError(f'the focus distance {s / 1000:g} m is not beyond the focal length {f:g} mm; a thin lens '
                         f'cannot focus there')
    Af = (f / N) * f
    if math.isinf(s):
        return Af / c / 1000.0, math.inf
    near = s * Af / (Af + c * (s - f))
    den = Af - c * (s - f)
    far = s * Af / den if den > 0 else math.inf
    return near / 1000.0, far / 1000.0


def blur_circle_px(f_mm, f_number, focus_m, z_m, pixel_um):
    """Defocus blur-disc diameter in pixels of a point at object distance z_m (scalar or array, m) for a thin lens of
    focal length f_mm at f/N focused at focus_m (inf allowed), on pixels of pixel_um: b = (f/N) f |s − z| / (z (s − f))
    (for s = inf: (f/N) f / z). A point no farther than the focal length gives inf. Any figure None (not stated)
    gives None."""
    if f_mm is None or f_number is None or focus_m is None or z_m is None or pixel_um is None:
        return None
    f = _pos('focal length', f_mm, 'mm')
    N = _pos('f-number', f_number, 'f/N')
    p = _pos('pixel pitch', pixel_um, 'µm') / 1000.0
    s = _pos('focus distance', focus_m, 'm') * 1000.0
    if s <= f:
        raise ValueError(f'the focus distance {s / 1000:g} m is not beyond the focal length {f:g} mm')
    z = np.asarray(z_m, float) * 1000.0
    A = f / N
    with np.errstate(divide='ignore', invalid='ignore'):
        if math.isinf(s):
            b = A * f / z
        else:
            b = A * f * np.abs(s - z) / (z * (s - f))
        b = np.where(z > f, b, np.inf) / p
    return float(b) if b.ndim == 0 else b
