# Canonical view: pooling wrist cameras across robots

You want an SO-100's wrist images to look as if they were taken by the UR5e's wrist D405, so that one policy can be
trained on both datasets. The Canonical view page (Export ▸ Canonical view) computes the warp. The warp is exact for
the fingertip contact plane. The page also shows how far off it is for anything nearer or farther, and it writes the
warp in the formats a data pipeline reads. The model is in [docs/models/canonical_warp.md](../models/canonical_warp.md).

## A worked example: an SO-100 warped to the UR5e's wrist D405

1. File ▸ Open sample ▸ UR5e + Robotiq 2F-85.
2. Add the SO-100 arm and its wrist camera to the project.

   In Python, the test file does this in `tests/test_canonical_warp.py _merged()`. It puts the SO-100 0.8 m along −y,
   as a second, non-primary robot, together with its sample wrist camera (a 1280 × 720 default camera: no maker in
   the library states a camera for the SO-100).

   Alternatively, keep the two robots in separate projects. Export the warp once in the UR5e project; that export
   writes `<stem>_canonical_view.json`. Then, in the SO-100 project, use **Load canonical view…** on the page.
3. Open Export ▸ Canonical view. The page picks **Source (robot B)** = the SO-100's *Wrist camera (sample placement)*
   and **Canonical (robot A)** = *Wrist D405*. With two robots it always picks a camera on the second robot's link
   and one on the primary robot's link. Press **Run**.

What the page shows (measured 2026-11-07, the study takes 0.07 s):

- **Headline: 163.9 px, "large parallax".** This is the P95 of the residual parallax at the worse depth extreme, in
  the D405's pixels; the maximum is 215.3 px.
- **Geometry.** The two cameras are 30.7 mm apart relative to their fingertips. The contact plane is 83.4 mm from the
  SO-100's camera and 110.7 mm from the D405. The warp fills 70.8 % of the D405's image.
- **Depth range.** The defaults give −305 mm to +35 mm. Beyond the plane, the range is the task volume's height along
  the approach axis, because the TCP can be anywhere in the volume. Towards the camera, it is the SO-100 pads' far
  end.

  | Depth extreme | Distance | Max | P95 | Median |
  |---|---|---|---|---|
  | Beyond the plane | −305 mm | 215.3 px | 163.9 px | 73.6 px |
  | Towards the camera | +35 mm | 127.9 px | 102.1 px | 47.3 px |

- **Homography.** H ≈ [[−0.854, −0.025, 1142.3], [0.0001, −0.897, 594.4], [0, 2 × 10⁻⁵, 1]]. The negative diagonal
  is a half turn: the SO-100 sample's wrist camera is mounted rolled 180° relative to the D405's view of the
  fingertips.
- **Parallax map.** It shows where in the D405's image the warp is good (light) and poor (dark).
- **3D view.** It draws the contact plane at the SO-100's TCP (green), the SO-100 camera's frustum (blue) and, dashed,
  where the D405 sits relative to those fingertips.
- **Camera view.** Choose the D405 and you see the SO-100's image grid warped into it. Choose the SO-100 camera and
  you see which of its pixels the warp reads.

The figure says that a 30 mm difference in mounting, with things up to 30 cm beyond the fingertips, moves image
content by more than 100 px. The warp is fine for the object between the fingers; it is not fine for the table
behind it.

**Make it smaller.**

- **Apply** mounts the SO-100's camera at the D405's pose relative to the fingertips. It is one undo step, and the
  camera moves 31 mm. Run again: the headline reads 5 × 10⁻¹³ px, "aligned". The warp is then the pure homography
  K_A K_B⁻¹ and holds at every depth. **Preview** draws the pose first. A locked camera is not moved.
- If the scene really is shallow, set the depth range in the project's settings, e.g.
  `{"canonical_view": {"depth_range_m": [-0.05, 0.02]}}`. The headline then reads 69.0 px.
- **Plane offset** moves the plane along the approach axis. **Flip the source's closing axis** turns the source's
  contact frame half a turn about the approach axis. On this pair the flip is wrong: it puts the cameras on opposite
  sides of the gripper, 133 mm apart and 1 899 px.

## Using the exported warp

Export warp maps… (or `dexvantage project.dexv --export out/ --formats canonical_warp`) writes into
`out/canonical_warp/`. On the example the files come to about 40 MB, written in 1.0 s; the `.yml` is 20 MB of that.

```python
import cv2, numpy as np
stem = 'out/canonical_warp/Wrist_camera_sample_placement_to_Wrist_D405'
map1, map2 = np.load(stem + '_map1.npy'), np.load(stem + '_map2.npy')
canonical = cv2.remap(img_so100, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

import torch
grid = torch.from_numpy(np.load(stem + '_grid.npy'))[None]          # (1, 720, 1280, 2)
canonical_t = torch.nn.functional.grid_sample(img_t, grid, mode='bilinear', padding_mode='zeros',
                                              align_corners=False)
```

`_homography.json` carries H for undistorted images, the conventions, both cameras and the parallax summary at full
resolution, with provenance. `_parallax.npy` is the per-pixel error map, which can mask or weight the loss where the
warp is poor.

## Batch

    dexvantage project.dexv --canonical-view result.json
    dexvantage project.dexv --export out/ --formats canonical_warp --export-options '{"canonical_warp": {"formats": ["npy", "grid", "json"]}}'

The study's settings live in `project.analysis.studies["canonical_view"]`: `source`, `canonical` (a sensor uid or a
canonical view document), `q`, `plane_offset_m`, `closing_sign_source`, `closing_sign_canonical`, `depth_range_m`
and `thresholds_px`.
