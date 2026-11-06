# The Camera view: a sensor's simulated image {#camview}

Modules: `dexvantage/core/camrender.py` (the image), `dexvantage/ui/camview.py` (the panel), with the lens models of
`core/optics` and the imagers of `core/library` (`library.sensor_imagers`). Check:
`dexvantage/validation/checks_camview.py` (model `occlusion`). Tests: `tests/test_sensors_ui.py`.

## What it shows

The selected sensor's image through one of its imagers (a combo picks RGB, IR left, IR right…): the robots posed at
the current joint state in their model colours, the task objects, the surfaces (tables) and the floor (z = 0, with a
0.1 m grid as a view aid: the floor occludes nothing in the studies), shaded by `0.30 + 0.70 |n · r|` (a head light at
the camera: n the triangle's normal, r the pixel's ray). The task volumes are drawn on it as dashed edges where the
image sees them, and the current analysis page adds its own overlays (`StudyPage.camview_overlays(cam_model, pose)`,
polylines in the imager's pixel coordinates). The line under the image says which imager, its stream size and FOV,
the size rendered and the time; its tooltip gives where the camera model comes from (a datasheet FOV, a crop
assumption, a calibration or the labelled default camera) and the share of pixels on the robot, objects, surfaces
and floor. A sensor without a library device is drawn through the default camera (90° × 60°, 1280 × 720, pinhole),
and the view says so first.

Ported from Vantage `ui/camview.py`: the idea of shading each pixel from its own ray (unproject every pixel through
the real lens model, find what it hits, shade it), the fit of the image to the panel and the overlays as projected
polylines. What changed: Vantage looked rays up in the cube map of a static car body; here the scene is re-posed for
every joint vector and every pixel needs the triangle it sees (for its colour), so each image is rasterised again.

## How an image is made

Each image pixel (u, v) (OpenCV: pixel centres at integers) has a ray r = `CameraModel.unproject(u, v)` in the
imager's optical frame (z forward, x right, y down), and the imager's optical frame in the world is the sensor body
pose (`scenegeom.sensor_world_pose`: the mount pose, composed with the link pose at q for a link mount) composed with
the imager's pose in the body (from the library entry or the unit's calibration).

**The rasteriser** (`camrender.raster`) is a pinhole z-buffer. For a pixel centre the ray is d = ((u − c_x)/f_x,
(v − c_y)/f_y, 1); with a triangle's corners a₀, a₁, a₂ relative to the camera centre, d passes through the triangle
iff the three triple products d · (aᵢ × aᵢ₊₁) all have the sign of D = a₀ · (a₁ × a₂), and the hit's depth is
z = D / (n · d) with n = (a₁ − a₀) × (a₂ − a₀). This is the test of `core/occlusion.raster_numpy` (ported to an
arbitrary pinhole), so a triangle crossing the near plane (5 mm, a default) is drawn correctly: its candidate pixels
are the bounding box of its corners in front of the plane and of its edges' crossings of it. The nearest hit per
pixel wins (`np.minimum.at`), and the triangle index is kept with it.

**The lens decides the path:**

| Lens | Path | How |
|---|---|---|
| pinhole | `pinhole` | rasterised straight into the image |
| rays within 60° of the axis (a Brown-Conrady lens, a mild Kannala-Brandt) | `base` | a pinhole base image that covers every pixel's ray (its focal length the lens's, at most twice the image along each axis) is rasterised; each image pixel takes the triangle of the base pixel nearest its own ray |
| a fisheye beyond 60° | `cube` | the world-axis cube faces (`core/occlusion.FACES`, ported from Vantage) that the rays need are rasterised at n = 2 f (64…1024 texels); each pixel takes the triangle of its ray's texel (`occlusion.texel_index`) |

In every path the pixel's **distance** is then recomputed exactly along its own ray to the plane of the triangle it
sees: t = (n̂ · a₀) / (n̂ · r̂). The look-up only decides which triangle a pixel sees.

**The sensor's seat:** the rule of `scenegeom.sensor_occluders` — the link a sensor is mounted on stays in the image
(a wrist camera sees its gripper), except its triangles with every corner within 25 mm (a default) of an imager's
centre. Sensor housings are not drawn (as in the occlusion model).

## Frame budget

Images are rendered on a worker thread; the UI thread only snapshots the project and the joint state (0.4 ms on the
UR5e sample). Changes are throttled (one render at a time, at most one queued). While the scene changes quickly — the
store's drag flag, IK drag, a mouse button held, or changes less than 150 ms apart — the image is rendered at half
size (`DRAG_SCALE`), and again at full size 180 ms after the changes stop. The full size is the panel's size, capped at
640 × 400 pixels and at the imager's own stream size. Measured on the UR5e sample (193 113 triangles, numpy, one core
of the development machine): 0.11 s at 320 × 180 (pinhole), 0.32 s at 640 × 360, 0.18 s for a Brown lens and 0.28 s
for a 190° fisheye at 320 × 240.

## The frusta in the 3D view

Each imager's outline in the 3D view comes from its lens model too (`camrender.frustum_body`): the image's outer edge
(u = −0.5 … W − 0.5, the edges the FOV is measured to) is unprojected, so a distorted lens bends the outline and a
fisheye draws a curved cone (the outline then lies on a sphere instead of a plane once a ray is more than 75° off the
axis). The reference imager is drawn with the lines from its centre to the corners; the device's other imagers by
their outline only. The outline is 0.12 m long (0.30 m for the selected sensor): a symbol of the field of view, not a
range.

## Validation

`occlusion.camview_vs_raycast`: pixels sampled at random in images of the UR5e sample (every imager of the wrist
D405 and the overhead D435 through its own lens model, plus a Brown-Conrady lens (k₁ = −0.25) and a 190° equidistant
fisheye at the wrist, so all three paths run) at random arm poses, against the exact first hit along the same ray
from `core/raycast.py` (Möller–Trumbore through a uniform grid, sharing no code with the rasteriser). A pixel is wrong
when only one finds a surface or the distances differ by more than 1 mm; disagreements within two pixels' angle of a
silhouette (`raycast.near_silhouette`) are counted apart, since a pixel centre or the nearest base / cube pixel may
fall on the other side of an edge. Tolerance: 10⁻³ of the pixels away from silhouettes; at most 3 % in all.

Measured with the default seed (full run, 24 images at 192 px wide, 70 061 pixels): 1 pixel wrong away from a
silhouette (1.4 × 10⁻⁵), 142 near a silhouette (0.20 % in all); 7.8 s. Quick run: 4 images, 0 wrong, 0.6 % near a
silhouette.

## Known limits

- One triangle per pixel, no anti-aliasing: edges are as sharp as the pixel grid. The base and cube paths take the
  nearest base pixel, so an edge can move by up to half a base pixel.
- A geometric view, not a photometric one: no shadows, textures, exposure, noise or IR pattern (lighting is Phase 5);
  materials are drawn in their model colours.
- No sensor housings (the sensor's own or other sensors') in the image.
- The floor and its grid are a view aid; tables are the project's surfaces.
- The time per image grows with the triangles in view; the GPU path the occlusion model uses is not used here yet.
