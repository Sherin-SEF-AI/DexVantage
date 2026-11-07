# Cross-embodiment canonical view warp

`dexvantage/core/exporters/canonical_warp.py` (the model and the exporter `canonical_warp`),
`dexvantage/core/studies/canonical_view.py` (study `canonical_view`, batch `--canonical-view OUT.json`), the page
`dexvantage/ui/pages/canonical_view.py` (Export ▸ Canonical view), the checks `dexvantage/validation/checks_warp.py`
(model `vla`, plus `formats.canonical_warp_files`). Spec §7.16.

When wrist-camera datasets are pooled across embodiments (an ALOHA, a Franka, an SO-101), every wrist camera sits
somewhere else relative to its gripper. The warp maps robot B's wrist image to robot A's canonical viewpoint. It is
exact for one plane, the fingertip contact plane, where the object is held, and wrong elsewhere by the residual
parallax, which the study reports over the task volume's depth range. The output is a homography (for undistorted
images), a full remap LUT that includes both lenses, and the parallax map.

## The contact frame

Each robot gets its own contact frame, measured from its contact pads at its home pose by
`core/grasps.py gripper_geometry`:

- origin: the tool centre point, the mean of the pads' centroids;
- z: the approach axis;
- x: the closing axis, times `closing_sign` (±1, a user setting: which pad comes first is arbitrary, and −1 turns the
  frame half a turn about the approach);
- y = z × x.

The frame is posed at q with the robot's gripper frame. The warp **identifies the two robots' contact frames**: that
identification is what "canonical" means here. A robot with no tagged pads gets its gripper frame's axes, and the
result says so.

The contact plane is z = `plane_offset_m` in the contact frame (default 0, the plane through the TCP, normal along
the approach axis).

## Conventions

- Optical frames follow OpenCV (x right, y down, z forward), and pixel centres sit at integers. These are
  DexVantage's conventions (docs/models/optics.md).
- (R, t) maps source to canonical: X_A = R X_B + t. From the cameras' poses in their contact frames,
  T_AB = T_cA⁻¹ T_cB.
- In camera B the plane is n_Bᵀ X_B + d_B = 0. n_B is the unit normal pointing from the plane towards camera B, and
  d_B > 0 is B's distance to the plane. In camera A the same plane is n_A = R n_B, d_A = d_B − n_Aᵀ t. A negative
  d_A means A sees the plane from behind; the result says so, and the warp is still exact on the plane.

## The homography

For every point X of the plane, −n_Bᵀ X / d_B = 1, so R X + t = (R − t n_Bᵀ / d_B) X. Projecting through each
camera's matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] gives

    x_A ~ H x_B,      H = K_A (R − t n_Bᵀ / d_B) K_B⁻¹        [Hartley and Zisserman 2004, ch. 13]

with x the homogeneous pixels of the pinhole (undistorted) images. H is scaled so that H[2][2] = 1. It maps source
pixels to canonical pixels: `cv2.warpPerspective(img_B, H, (W_A, H_A))` applies it, because OpenCV computes
dst(x) = src(H⁻¹ x) when WARP_INVERSE_MAP is not set (the installed OpenCV 5.0.0's docstring, read 2026-11-07).

## The full remap LUT

A lens that is not a pinhole is not described by H. For every canonical pixel p the LUT:

1. unprojects p through A's real lens (`CameraModel.unproject`, iterative for Brown-Conrady);
2. intersects the ray with the plane, s = −d_A / (n_Aᵀ r), keeping s > 0;
3. moves the point into B's frame, X_B = Rᵀ (X_A − t);
4. projects it through B's real lens.

The result is `map_x, map_y` for `cv2.remap(img_B, map_x, map_y, cv2.INTER_LINEAR)`. A pixel whose ray misses the
plane, meets it behind either camera, or lands outside B's image holds −1, and OpenCV's remap gives it the border
value. The GPU form is the same LUT as a `torch.nn.functional.grid_sample` grid (align_corners=False):
x_n = (2u + 1) / W_B − 1, y_n = (2v + 1) / H_B − 1, so ±1 are the image's outer edges and integer pixel u is a
pixel centre. Invalid pixels hold −2, which falls outside the grid and samples as 0 with padding 'zeros'.

## The residual parallax

At canonical pixel p the warp shows what B sees along its ray through the plane point X(p). Suppose the surface
there is not on the plane but at signed distance h from it, measured along that ray (h > 0 towards camera B, h < 0
beyond the plane). Then the true canonical image shows the point at π_A(X_h), and the warp is off by

    Δ(p, h) = π_A(X_h) − p,   X_h = s r_B,  s = (h − d_B) / (n_Bᵀ r_B)

in canonical pixels, through A's real lens (`parallax_at`).

For pinhole cameras this is the plane-plus-parallax term [Irani and Anandan 1996]. Write K_A(R X + t) as
K_A(R − t nᵀ/d) X + K_A t (h / d), with h = nᵀX + d. The first term dehomogenises to p, and the third component of
the sum is Z_A, the point's depth in A. So:

    Δ = h / (d_B Z_A) · ((K_A t)_xy − p (K_A t)_z)

Δ vanishes on the plane (h = 0), and it vanishes at every depth when the two cameras share a centre (t = 0). When
t_z ≠ 0, Δ points along the epipolar line, towards or away from the epipole e = K_A t.

**The map.** As a point moves along one source ray, its canonical image moves monotonically along the epipolar line.
Over a depth range [h_lo, h_hi] the largest |Δ| is therefore at one end. The map is max(|Δ(p, h_lo)|, |Δ(p, h_hi)|)
over the canonical pixels the LUT fills; `tests/test_canonical_warp.py` checks this monotonicity on 16 depths. The
study samples the map on a grid of at most 64 cells across and reports max, P95 and median at each extreme. The
export computes it at every pixel.

**The depth range (defaults, labelled).**

- Beyond the plane: at a grasp the TCP can be anywhere in the task volume. A volume point can therefore lie up to
  D = Σᵢ |aᵢ| sᵢ beyond the plane, where a is the approach axis in the world (at the source robot's q) and s the
  volume's edge lengths. With several volumes, D is the largest.
- Towards the camera: an object held between the pads comes no nearer the camera than the pads' far end, because the
  gripper fills the rest. The near end is the largest distance of a contact-pad point from the plane on the camera's
  side (`core/grasps.py contact_points`), capped at D and at d_B − 5 mm (the point must stay in front of B).

The user can set the range instead (`depth_range_m`).

**The headline.** The P95 at the worse extreme, with a status word (DexVantage defaults, not standards):

| Word | P95 of the residual parallax |
|---|---|
| aligned | ≤ 1 px |
| mild parallax | ≤ 4 px |
| parallax | ≤ 16 px |
| large parallax | above 16 px |

**The suggestion (Apply).** The page offers to mount the source camera at the canonical camera's pose relative to
its own fingertips. That pose is the source contact frame composed with A's pose in the contact frame, then the
imager's pose in the body removed, written in the mount's parent link at q. The cameras then share a centre (t = 0),
the warp is the pure homography K_A K_B⁻¹ (R = I and t = 0: the position and the orientation are copied), and the
parallax is 0 px at every depth. A locked camera is
not moved.

## What is exported

The exporter `canonical_warp` writes `--export OUT --formats canonical_warp`, or Export warp maps… on the page.
Every file goes under `canonical_warp/` with the stem `<source>_to_<canonical>`:

| File | Content |
|---|---|
| `_map1.npy`, `_map2.npy` | float32 map_x, map_y (H_A × W_A), for `cv2.remap` |
| `_remap.yml` | the same maps as `map1`, `map2` in OpenCV FileStorage YAML, with `homography`, the sizes, the invalid value and the convention |
| `_grid.npy`, `_valid.npy` | (H_A, W_A, 2) float32 grid_sample grid, uint8 mask |
| `_parallax.npy` | float32 \|Δ\| px at every canonical pixel (the worse extreme; NaN where not covered) |
| `_homography.json` | H, H⁻¹, the convention, both views (lens, pose in the contact frame), the plane, the depth range, the parallax summary at full resolution, the files and usage lines; stamped with provenance |
| `_canonical_view.json` | the canonical camera and its pose in the contact frame (`dexvantage.canonical_view/1`) |

Robot B's project can load `_canonical_view.json` as its canonical camera (page: Load canonical view…; settings
`canonical: {...}`), so robot A and robot B need not be in the same project. `export_manifest.json` lists every
file's SHA-256.

The YAML is text. A 1280 × 720 pair writes about 20 MB of `.yml` against 3.7 MB per `.npy` map. Leave `yml` out of
`formats` (`{"canonical_warp": {"formats": ["npy", "grid", "json"]}}`) when it is not needed.

## Validation

Every case is synthetic and seeded. The two cameras are posed 0.10 to 0.20 m above the plane with random tilts of up
to ±0.25 rad: a pinhole pair, and a Brown-Conrady canonical camera with a Kannala-Brandt fisheye source. Measured on
2026-11-07 (OpenCV 5.0.0):

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `vla.warp_plane_exact` | plane points projected by K (R X + t), written out in the check; 4 pairs, 1 102 points | 1e-9 px (closed form: double rounding only) | 2.3 × 10⁻¹³ px |
| `vla.warp_lut_lens_cv2` | cv2.projectPoints and cv2.fisheye.projectPoints of plane points; 4 pairs, 1 519 points | 1e-6 px (the LUT runs the lens's iterative inverse) | 2.3 × 10⁻¹³ px |
| `vla.warp_parallax_bruteforce` | off-plane points projected by OpenCV directly and through the plane point on their source ray; the closed form on pinhole pairs; 2 279 points from −0.2 to +0.03 m, parallax up to 119 px | 1e-6 px | closed form 1.5 × 10⁻¹³ px, model 2.8 × 10⁻¹³ px |
| `vla.warp_lut_vs_warpperspective` | cv2.warpPerspective(img, H) against cv2.remap through the maps, undistorted pair, smooth image (gradient ≤ 9 grey levels/px), 183 131 pixels; the float64 LUT against H⁻¹ | 1 grey level (both quantise sample positions to 1/32 px; float32 maps can round to the neighbouring step, ≤ 9/32 level, then both round to integers) | 1 grey level on 16 pixels (mean 8.7 × 10⁻⁵); LUT against H⁻¹ 5.7 × 10⁻¹³ px |
| `formats.canonical_warp_files` | the UR5e sample's export read back: .yml through cv2.FileStorage against the .npy maps, the grid against its formula, H from JSON and YAML against the model, the canonical view file loaded back | 1e-12 | 0 (arrays bit-exact), 1.3 × 10⁻¹⁵ (JSON decimal text) |

The self-test line `warp` runs one homography case and one lens case (both 1 × 10⁻¹³ px, 0.1 s).

## Known limits

- The warp is exact only on the contact plane. Elsewhere the parallax report bounds the error over the depth range;
  nothing corrects it (no depth-based reprojection).
- Occlusion is not modelled. What B sees behind the fingers is warped as if it lay on the plane, and parts of A's
  view that B cannot see stay empty (−1).
- The two grippers' contact frames are identified by their measured TCP, approach and closing axes, and the closing
  sign is a user choice. A gripper without tagged pads uses its gripper frame's axes.
- A camera that is not fixed in its gripper frame (a static or forearm camera) is related to the contact frame at
  the q given, home by default. The result says when this applies.
- The default depth range is a bound from the task volume and the pads, not a measurement of the scene.
- The image is resampled once (OpenCV bilinear by default), with no anti-aliasing when the warp minifies. Rolling
  shutter, exposure and colour differences between the cameras are not modelled.

## References

- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press,
  2004, chapter 13 (homographies induced by a plane). The equation number was not re-read; the checks prove the
  convention used.
- M. Irani and P. Anandan, "Parallax geometry of pairs of points for 3D scene analysis", ECCV 1996, LNCS 1064,
  17–30, doi:10.1007/BFb0015520 (bibliographic details read at dblp.org, 2026-11-07).
- OpenCV imgproc, `cv2.warpPerspective` and `cv2.remap` (the docstrings of the installed 5.0.0, read 2026-11-07;
  docs.opencv.org returned HTTP 403 that day); `cv::FileStorage`.
