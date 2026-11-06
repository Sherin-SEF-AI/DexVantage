# Optics: lens models, pixel density and defocus {#optics}

Module: `dexvantage/core/optics/lens.py` (exported by `dexvantage/core/optics/__init__.py`). Checks:
`dexvantage/validation/checks_optics.py` (model `optics`). Tests: `tests/test_optics.py`.

The projection mathematics is ported from Vantage `core/lens.py` (the radial functions r(θ), the Kannala-Brandt
polynomial, fitting Kannala-Brandt terms to a stated field of view, and Intrinsics' projection and unprojection).
DexVantage works in pixels throughout, uses OpenCV's parameterisation and pixel convention, adds the Brown-Conrady
model, an exact constrained fit through the stated angles, and thin-lens defocus.

## Conventions

- **Points** are in the camera's optical frame (REP-103 `_optical_frame`): z forward along the optical axis, x right,
  y down, metres. θ = atan2(√(X² + Y²), Z) is the angle off the axis.
- **Pixels** follow OpenCV: the centre of pixel (i, j) is the point (u, v) = (i, j). The image spans
  u ∈ [−0.5, W − 0.5], v ∈ [−0.5, H − 0.5]; a centred principal point is cx = (W − 1)/2, cy = (H − 1)/2.
  *Vantage put pixel centres at i + ½ (image [0, W], centre W/2); DexVantage does not.* Resampling the image by s maps
  u to s (u + ½) − ½ (`CameraModel.scaled`).
- u = fx · xd + cx, v = fy · yd + cy, with (xd, yd) the model's normalised image coordinates. No skew.

## The models

`CameraModel(width, height, fx, fy, cx, cy, model, dist, max_theta)` is a frozen (hashable) dataclass.

| `model` | normalised image point (xd, yd) | `dist` |
|---|---|---|
| `pinhole` | (X/Z, Y/Z) | empty |
| `brown` | x = X/Z, y = Y/Z, r² = x² + y², c = (1 + k1 r² + k2 r⁴ + k3 r⁶)/(1 + k4 r² + k5 r⁴ + k6 r⁶); xd = x c + 2 p1 x y + p2 (r² + 2x²), yd = y c + p1 (r² + 2y²) + 2 p2 x y [Brown 1966; OpenCV calib3d] | (k1, k2, p1, p2, k3) or (k1, k2, p1, p2, k3, k4, k5, k6) |
| `kb` | θd = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸); (xd, yd) = θd (X, Y)/√(X² + Y²) [Kannala and Brandt 2006; OpenCV fisheye] | (k1, k2, k3, k4) |
| `equidistant` | g(θ) = θ | empty |
| `equisolid` | g(θ) = 2 sin(θ/2) | empty |
| `stereographic` | g(θ) = 2 tan(θ/2) [Miyamoto 1964] | empty |

For the radial models (xd, yd) = g(θ) (X, Y)/√(X² + Y²) with g = r/f. `equidistant` is `kb` with k = 0 exactly.
OpenCV's fisheye computes θ as atan(r) of (X/Z, Y/Z), which equals atan2 in front of the camera; DexVantage's θ may
pass 90° (a fisheye sees behind its own plane).

**The valid angle `max_theta`** (rad) is the largest angle off the axis the model maps. A point beyond it, or behind
a `pinhole` or `brown` camera, is never projected: its uv is NaN and ok is False; it is never wrapped onto the image.
A pixel beyond it has no ray (NaN). Left out, it is the angle where the model stops being one-to-one:

- `pinhole`: 90°.
- `brown`: the first angle where, in one of 72 directions around the axis, the image radius stops rising, the
  Jacobian determinant of the distortion map reaches zero, or c's denominator does (4000 angles, refined by
  bisection). Tangential terms grow as r² against the radial r·c, so a model with p1 or p2 always folds somewhere.
- `kb`: the first angle where θd(θ) stops rising, at most 180°.
- `equidistant`, `equisolid`, `stereographic`: 180°.

A `max_theta` beyond that angle is refused with the reason (two rays would share a pixel). A smaller one models a
circular fisheye image: the corners then have no rays and `fov_deg` counts them at the valid angle.

### Projection and unprojection

`project(P)` returns `(uv (N, 2), ok (N,))`; ok means in front within the valid angle and inside the image (each edge
widened by `EDGE_TOL_PX` = 10⁻⁹ px so a ray unprojected from the edge itself counts as inside). uv is finite but ok
False when the point is valid but off the image. `unproject(uv)` returns unit rays (N, 3); `with_ok=True` also
returns the validity. `pixel_rays(stride)` gives the rays of the pixel centres u = 0, stride, … as (H′, W′, 3).
Any leading shape is accepted.

- **Brown-Conrady inverse**: Newton's method on (x, y) with the analytic 2 × 2 Jacobian (OpenCV's `undistortPoints`
  iterates the fixed point instead), started at the distorted point, kept inside the one-to-one disc (an iterate
  outside is pulled back to its edge; twice outside means no root inside), converged when the re-distorted point is
  within 10⁻¹² (normalised units, about 10⁻⁹ px at f = 1000 px). A pixel that does not converge has no ray.
- **Kannala-Brandt inverse**: Newton's method on θ, safeguarded by bisection inside a bracket on
  [0, min(max_theta, fold angle)], where θd(θ) rises, so the root is unique. Radii beyond θd(max_theta) have no ray.
- **Classical projections**: closed forms (atan, θ = r, 2 asin(r/2), 2 atan(r/2)).

### Fields of view

`fov_deg()` returns `{'h', 'v', 'd'}` from the model itself: the angle off the axis of the left edge (u = −0.5) plus
that of the right edge (u = W − 0.5) along the principal point's row; likewise top and bottom; and corner to corner,
the mean of the two diagonals. An edge beyond the valid angle counts at the valid angle.

`CameraModel.from_fov(width, height, hfov_deg, vfov_deg=None, model='pinhole', *, dfov_deg=None, max_theta=None)`
centres the principal point and reproduces the stated angles; `fit_fov(...)` returns the model with a report
(`base`, `f_px`, `fit`, `stated`, `model_fov`, `max_error_deg`, `note`).

- `pinhole`, `brown` (zero distortion: none is stated), `equidistant`, `equisolid`, `stereographic`:
  W/2 = fx · g(h/2) in closed form; fy from the vertical angle when one is given (non-square pixels), else fy = fx.
  A diagonal angle cannot be matched as well and is refused.
- `kb`: square pixels. With the horizontal angle only, the equidistant curve. With a vertical and/or diagonal angle,
  ported from Vantage: the base projection and focal length that alone fit the stated points best
  (`best_base`, least squares of r = f g(θ) relative to each half size; pinhole a candidate only below 160°), then
  Kannala-Brandt terms. New in DexVantage, `fit_kb_exact` passes the curve **through** the stated points (up to three
  equality constraints on four terms, the remaining freedom kept closest to the base curve sampled out to the image
  corner: a KKT system). If that curve folds back before the image corner plus 2°, Vantage's weighted fit
  (`fit_kb_to_fov`, weights 50, 20, 8) is used at the strongest weight whose curve still rises. The valid angle
  reaches `FIT_MARGIN_DEG` = 2° past the image corner. The stated angles are reproduced; **the curve between them is
  a model, not a measurement**, and the report's note says so.
- `from_fov` refuses (ValueError, saying why and what to check) a fit that misses a stated angle by more than
  `FOV_FIT_TOL_DEG` = 0.5°, a default of DexVantage: such figures fit no one-to-one lens with square pixels on that
  image, typically because they were stated for another crop or resolution. `fit_fov` returns the best effort.

### Conversions

`from_opencv(K, D, width, height, model)` takes OpenCV's K and D (aliases: `fisheye` → `kb`; `plumb_bob`,
`rational_polynomial` → `brown`; `pinhole` with non-zero D is read as `brown`, as `cv2.projectPoints` reads it). Skew,
thin-prism and tilt terms are refused. `to_opencv()` returns `(K, D, model_str)`: `pinhole` (D five zeros) and `brown`
for `cv2.projectPoints`, `kb` for `cv2.fisheye`; `equidistant` is exactly `kb` with zero terms; `equisolid` and
`stereographic` have no OpenCV form and are returned as the least-squares Kannala-Brandt curve over the image's angles
(`kb_approximation()` gives the terms and the largest error in pixels: 6 × 10⁻⁸ px for a 180° equisolid lens on
1920 × 1080, 0.016 px for a 180° stereographic one). `to_dict` / `from_dict` store the figures with
`"pixel_centres": "opencv"`; `scaled(w, h)` resamples.

## Pixel density and viewing angle (§7.1)

`pixels_per_metre(P, normals=None)` is the **local linear pixel density** at P: √|det J|, with J the 2 × 2 Jacobian
of (u, v) with respect to the two lateral coordinates (metres) of a small flat patch at P facing the camera (normal
along the line of sight). It is the square root of the pixels per square metre: the geometric mean of the densities
along the patch's two principal directions (the densities along single directions lie between J's singular values).
Pixels per millimetre are `pixels_per_metre / 1000` (`pixels_per_mm`). With R = |P| and g = r/f:

    pinhole:         det J = fx fy / (R² cos³θ)
    brown:           det J = fx fy |det D(x, y)| / (R² cos³θ)          D = ∂(xd, yd)/∂(x, y) at x = X/Z, y = Y/Z
    radial models:   det J = fx fy g′(θ) g(θ) / (R² sin θ)            → fx fy g′(0)² / R² on the axis

(the image area per solid angle times the solid angle per area at range R, 1/R²). On the axis a pinhole gives f/Z.
For a surface with normal n, α the angle between n and the line of sight: the line of sight is in the null space of
the projection's 3-D Jacobian, so the tilted patch's area scales by |cos α| and the density by √|cos α|
(`normals=`). `angle_to_axis(P)` is θ; `incidence_angle(P, normals)` is α (0 = facing the camera).

## Defocus (§7.1): thin lens

For focal length f, f-number N (aperture diameter A = f/N), focus distance s and an object at z (distances from the
lens's principal plane), the Gaussian lens equation 1/f = 1/s + 1/v gives the image distances v_s = f s/(s − f) and
v_z = f z/(z − f); by similar triangles the blur disc on the sensor at v_s has diameter
b = A |v_z − v_s| / v_z, which simplifies to

    b(z) = A f |s − z| / (z (s − f))            (s = ∞:  b = A f / z)   [Ray 2002; Hecht 2017]

`blur_circle_px(f_mm, f_number, focus_m, z_m, pixel_um)` returns b / pixel pitch (z may be an array; z ≤ f gives
inf). `depth_of_field(f_mm, f_number, focus_m, coc_mm)` solves b(z) = c:

    near = s A f / (A f + c (s − f)),    far = s A f / (A f − c (s − f))   (∞ when A f ≤ c (s − f))

which equals the photographic hyperfocal form (H = f²/(N c) + f; near = s (H − f)/(H + s − 2f), far = s (H − f)/(H − s))
that the check writes out separately. Any figure `None` (not stated by the maker) gives `None`; an f-number is often
not stated, so the coverage study then reports defocus as not given. `DEFAULT_COC_PX` = 2 px is a **default** of
DexVantage (a blur spot up to two pixels across counts as sharp), a user setting, not a standard.

## Validation {#optics-validation}

Measured with the default seed (20261027), full run, 0.6 s:

| Check | Reference | Measured | Tolerance and why |
|---|---|---|---|
| `optics.opencv_pinhole_brown` | `cv2.projectPoints` (16 cameras, none / 5-term / 8-term rational, 32 000 points) | 1.8 × 10⁻¹² px | 10⁻⁶ px: the same polynomial in double precision, differing only in operation order; a half-pixel shift or swapped p1/p2 shows as ≥ 10⁻³ px |
| `optics.opencv_fisheye` | `cv2.fisheye.projectPoints` (16 cameras, up to 85°, points on the axis) | 5.7 × 10⁻¹³ px | 10⁻⁶ px: as above |
| `optics.round_trip` | the pixels and rays themselves, every model | 4.7 × 10⁻¹³ px; ray angle 1.1 × 10⁻¹⁴ rad (bound 10⁻⁹) | 10⁻⁶ px: the iterative inverses converge to 10⁻¹² normalised |
| `optics.from_fov` | the stated angles, read back through `fov_deg` and, for the undistorted pinhole fits, also through the pinhole closed form from the fitted fx, fy, cx, cy (independent of `fov_deg`; worst 1.4 × 10⁻¹⁴ °); for `kb`, consistent H/V/D generated from pinhole and fisheye lenses written out in the check (210 fits) | 2.3 × 10⁻¹² ° | 10⁻⁶ °: closed forms and an exact constrained fit; half a pixel at the edge is about 0.05° |
| `optics.pixel_density` | central finite differences of `project` on a patch (facing and tilted), every model | 8.2 × 10⁻⁹ relative | 10⁻⁶: truncation ~10⁻¹⁰, rounding ~10⁻⁸; a missing cos θ factor is ≥ 10⁻² off axis |
| `optics.depth_of_field` | the hyperfocal-distance formulae, both sides of H, and focus at infinity | 2.9 × 10⁻¹³ relative | 10⁻¹⁰: two algebraic forms of one result |
| `optics.blur_circle` | the thin-lens equation and similar triangles in exact rational arithmetic; blur at the DOF limits = c | 1.1 × 10⁻¹³ relative | 10⁻¹¹: DexVantage's own rounding, amplified by s/|s − z| through the metre-to-millimetre step (points within 0.1 % of focus moved to 0.1 %) |

The round-trip check counts the pixels that have no ray (1025 of 120 000 with the default seed: random Brown-Conrady
models whose curve folds back before the image corner); they are refused, not compared.

## Known limits

- Thin lens only: no pupil magnification, diffraction, aberrations or focus breathing; defocus distances are from
  the principal plane, not the camera's front face.
- No skew, no thin-prism or tilt terms, no rolling shutter (§7.9 adds time).
- A model fitted from a stated FOV reproduces the stated angles; its distortion between and beyond them is a model.
  A calibration (`Sensor.calibration`, OpenCV K and D) replaces it when the unit has one.
- The Brown-Conrady valid angle is found on 72 directions and 4000 angles: a fold narrower than that could slip
  through, and a positive Jacobian on a disc makes the map locally, not provably globally, one-to-one. A strongly
  distorted calibration can fold back inside its own image; those pixels get no ray rather than a wrong one.
- `pixels_per_metre` is the geometric mean density; a study that needs the worst direction should use J's smaller
  singular value (not provided yet).
- MuJoCo's fiducial check (below) covers MuJoCo's pinhole cameras only; MuJoCo renders no distortion, so the
  distorted models are checked against OpenCV alone.

## References

Brown 1966 (`brown1966`); Kannala and Brandt 2006 (`kannala_brandt2006`); Miyamoto 1964 (`miyamoto1964`); OpenCV calib3d
and fisheye (`opencv_calib3d`, `opencv_fisheye`); Ray 2002 (`ray2002`); Hecht 2017 (`hecht2017`); Hartley and Zisserman
2004 (`hartley2004`). Full entries in `dexvantage/validation/references.py`.

## Against MuJoCo-rendered fiducials {#optics-mujoco}

Spec §10 asks for MuJoCo's camera projection of rendered fiducials within a stated, measured pixel tolerance. Two
checks (`dexvantage/validation/checks_optics_mujoco.py`) do it. A scripted MJCF scene holds a 9 × 7 grid of spheres
spread over the image, at 0.4 to 1.5 m, each about 14 px across (150–220 px masks), drawn with 128 slices so their
outline is round; MuJoCo renders a segmentation image through EGL with multi-sampling off (`validation/mjrender.py`).
Three cameras: `fovy="60"` at 640 × 480 (our `CameraModel.from_fov`); `focal` in metres with a `sensorsize` and
fx ≠ fy at 640 × 480; `focalpixel` with a `principalpixel` offset at 800 × 600.

Frames, written out: MuJoCo's camera looks along its −z axis, +x right, +y up (XML reference, body/camera). Our optical
frame looks along +z, +x right, +y down. A world point X is P = R_optᵀ (X − p) in the optical frame, with
R_opt = R_cam · diag(1, −1, −1), R_cam = `data.cam_xmat` and p = `data.cam_xpos`. Row 0 of a `mujoco.Renderer` image is
the top, and pixel centres are at integer (column, row), as OpenCV counts them. A fovy camera has square pixels with
fy = (H/2) / tan(fovy/2) and the principal point at ((W − 1)/2, (H − 1)/2); with a sensor size, fx = focal_x · W /
sensor width (MuJoCo stores focal and principal in length units in `cam_intrinsic`).

**The principal offset's direction is not stated in MuJoCo's reference.** Rendered on MuJoCo 3.17.0, `principalpixel`
(px, py) moves the image of the optical axis to ((W − 1)/2 − px, (H − 1)/2 − py), row 0 at the top: an x offset to the
left, a y offset upwards. `mjrender.camera_model` uses that; the third camera then checks that the offset is a pure
shift and combines correctly with the focal lengths, not the sign itself (which was read off a render). An MJCF
exporter (§9.2) must write the principal point with these signs.

| Check | Compared | Measured (default seed) | Tolerance |
|---|---|---|---|
| `optics.mujoco_fiducial_masks` | centroid of each sphere's MuJoCo mask against the centroid of the mask CameraModel predicts (pixel centres whose ray falls within the sphere's angular radius asin(r/d)) | worst 0.075 px over 189 fiducials (fovy 0.046, focal 0.075, principal 0.045); median 0 (identical masks); smallest mask overlap 98.9 % | 0.1 px: one boundary pixel decided the other way moves a 180 px centroid by about 0.04 px |
| `optics.mujoco_fiducial_centres` | the MuJoCo centroid against the image centre CameraModel.project predicts: midway between the projections of the two tangent rays at θ ± α in the plane of the axis and the sphere centre (a sphere's image is an ellipse whose centre is not the projected sphere centre) | worst 0.167 px, median 0.068 px; without the ellipse correction the offset is up to 0.254 px (the perspective bias) | 0.3 px: the centroid sampling error of these masks alone, measured on CameraModel's own predicted masks against the predicted centres, is 0.167 px |

So the sub-pixel bias of a fiducial centroid has two parts, both measured here: the perspective bias of a sphere's
image (0.25 px at 14 px across, growing with the angle from the axis and the size) and pixel sampling (0.17 px).
Neither is an error of the projection model; with both removed, MuJoCo and CameraModel agree to 0.075 px. The render
takes about 0.4 s. Without EGL the checks fail with the reason, they are not skipped.
