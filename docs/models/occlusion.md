# Occlusion: articulated cube maps and exact ray casting {#occlusion}

Modules: `dexvantage/core/scenegeom.py` (the scene's triangles), `dexvantage/core/occlusion.py` (the cube map, its
numpy and GPU backends), `dexvantage/core/raycast.py` (exact ray casting, the independent reference). Checks:
`dexvantage/validation/checks_occlusion.py` (model `occlusion`). Tests: `tests/test_occlusion.py`. Measurements:
`scripts/measure_occlusion.py` (writes `out/occlusion-measure.json`).

## What occludes: the scene at q

`scenegeom.scene_occluders(project, q, *, robots=True, objects=True, surfaces=True, exclude_links=(), decimate=None)`
returns `Occluders(tris (M,3,3) float32 world, owner (M,) int32, owners)`:

| Owner | Geometry |
|---|---|
| `robot:<uid>:<link>` | the robot's visual meshes posed at `q[uid]` (`RobotModel.posed_triangles`); a robot missing from `q` is at its home pose |
| `object:<uid>` | a task object: `box` (size = its edges), `cylinder` (size = radius, full height along its z axis, centred on its pose), `sphere` (radius), `mesh` (`project.meshes[mesh_key]`: an (M,3,3) array, (V,3) vertices with `<key>/faces`, or a (3K,3) triangle soup, in the object frame) |
| `surface:<uid>` | a table: a rectangle at its `z` of `size` [x, y] around `center` [x, y] (default the world origin); a slab down to `z − thickness` when a `thickness` is given |

Cylinders and spheres are tessellated as `core/robot/model.py` tessellates MuJoCo primitives (24 segments, 12 rings).
`exclude_links` takes owner strings, `(uid, link)` pairs or bare link names (that link on every robot).

**A sensor's own surroundings** (`scenegeom.sensor_occluders(project, sensor, q, own_radius=0.025,
exclude_mount_link=False, origins=None)`): no sensor housing is an occluder in Phase 2. The link a sensor is mounted
on stays an occluder, so a wrist camera sees its own gripper and flange, except its triangles with every corner within
`own_radius` (25 mm, a default) of the sensor's origin (or of each of `origins`, the imagers' centres): the bracket
face or link surface the sensor sits on. `exclude_mount_link=True` drops the whole mount link. On the UR5e sample's
wrist camera the rule drops nothing (the flange is farther than 25 mm), so the gripper, its pads and the flange all
occlude. The cube map's near clip (below) also ignores anything closer than 5 mm to an imager along a face axis.

`scenegeom.sensor_world_pose(project, sensor, q)` is the sensor body frame in the world: the mount pose for static
mounts, `link_pose(q) @ mount.pose` for link mounts.

## The cube map

Ported from Vantage `core/occlusion.py`: six 90° faces of n × n texels around the camera centre, aligned with the
world axes (so the map does not depend on the camera's orientation: every aim change and every imager rotation
reuses it; only the centre or the scene changes it), the same face basis, `texel_index` and `face_dirs`, and the
**exact plane depth per texel**: each texel stores the distance along the ray through its centre to the plane of the
nearest triangle there. Vantage painted a static vehicle far to near with QPainter; here the scene is re-posed for
every joint vector, so the map is built with a z-buffer by one of two backends.

**numpy (the reference rasteriser).** For each face, a triangle's candidate texels are the texel centres inside the
bounding box of its projection (its corners in front of the near plane, and where its edges cross it). Each candidate
is tested exactly: with the corners a₀, a₁, a₂ relative to the centre and d the texel-centre ray,

    D  = a₀ · (a₁ × a₂)
    hit ⇔ d · (a₀ × a₁), d · (a₁ × a₂), d · (a₂ × a₀) all have the sign of D      (d is a positive mix of the corners)
    t  = D / (n · d),  n = (a₁ − a₀) × (a₂ − a₀)                                  (the hit's coordinate on the face axis)
    distance = t · |d|,  kept when t ≥ near; the z-buffer keeps the smallest per texel (np.minimum.at)

The products are linear in the texel's (u, v), so each triangle carries twelve coefficients per face and each
candidate costs a few multiply-adds. Candidates are processed 1.5 million at a time.

**gl (the GPU).** An offscreen EGL OpenGL 3.3 context; the six faces are drawn into one n × 6n `GL_R32F` framebuffer
with a 32-bit float depth buffer, one viewport per face, by a depth-only shader that writes `length(p − origin)` per
fragment. Perspective-correct interpolation of p − origin at the fragment centre is exactly the point of the
triangle's plane on the texel-centre ray, so the GPU stores the same plane distance (float32), and the depth test
keeps the nearest (along one ray the distance is proportional to the face-axis depth). The triangles are uploaded
once per geometry digest (the last six are kept on the GPU), and the faces are read back with one `glReadPixels`.

The context is found in this order: each EGL device (`EGL_EXT_device_enumeration`, a discrete GPU without a display
server), the default display, Mesa's surfaceless platform; a software renderer (llvmpipe) is used only when nothing
else works. `DEXVANTAGE_EGL='EGL device 2'` (or a renderer name such as `radeonsi`, `llvmpipe`) picks one;
`DEXVANTAGE_NO_GL=1` switches the GPU off. PyOpenGL chooses its platform once per process: when this process already
has it loaded for GLX (Qt on an X11 desktop), the GPU backend runs in a worker process (`python -m
dexvantage.core.occlusion --gl-worker`, triangles and maps over a pipe), which costs about 5 ms more per map.

**auto** uses `gl` when a context can be made in either way, else `numpy`, with the reason logged once (logger
`dexvantage.core.occlusion`) and returned by `occlusion.gl_status()`. A GPU failure during a build falls back to
numpy the same way; `backend='gl'` raises `OcclusionError` instead.

**Near clip.** The part of a triangle less than `near` (5 mm, a default) from the centre along a face's axis is
clipped by both backends, so a camera pressed against the surface it is screwed to is not blinded by it.

**Visibility.** `cube.visible(points, eps=1e-3)` is true where the point's distance from the centre is at most its
texel's distance plus `eps` (1 mm, a default), so a point on a surface is visible when nothing else is in front of it.
`occlusion.visible(points, origin, occluders, n=512)` goes through `occlusion.CACHE`, an LRU of maps (256 MB) keyed
by (geometry digest, the exact centre, n, near, backend); the centre was rounded to 0.1 mm before research tools
part 2, which let a nearby origin reuse a map.

## Exact ray casting (the reference)

`raycast.first_hit(origins, dirs, tris, tmax=None)` returns the distance to the nearest triangle along each ray:
Möller–Trumbore intersection [Möller 1997] (two-sided; a triangle seen exactly edge-on is not hit) through a uniform
grid of about two cells per triangle (each triangle listed in every cell its bounding box overlaps), traversed with
the 3D DDA of Amanatides and Woo [1987] for all rays at once. A ray stops at the first cell whose own interval holds
the nearest hit found so far, so the answer is exact, not approximate; the grid only skips triangles that cannot be
hit first. `raycast.visible_exact(points, origin, tris, eps)` asks whether anything lies on the segment closer than
the point less eps. `raycast.near_silhouette(points, origin, tris, angle)` says whether a point's exact visibility
changes within `angle` of its line of sight (eight directions around it): used to sort the cube map's disagreements.
It shares no code with either rasteriser and was itself checked against brute force (every ray against every
triangle: identical to 10⁻¹² m, `tests/test_occlusion.py`).

## Decimation (occlusion only)

`scenegeom.decimate(occluders, cell_m)` (or `scene_occluders(..., decimate=cell_m)`) clusters vertices [Rossignac and
Borrel 1993]: each vertex moves to the mean of the vertices in its cubic cell of edge `cell_m`; collapsed and duplicate
triangles are dropped; owners are kept. `occlusion.decimation_report(occluders, origin, points, cell_m)` measures what
it changes. On the UR5e sample (posed at home, 20 000 points in the task volume, n = 512, GPU):

| Cell | Triangles | Wrist D405: visibility changed | Overhead D435: visibility changed |
|---|---|---|---|
| none | 193 115 | — | — |
| 2 mm | 59 800 | 0.08 % | 0.08 % |
| 5 mm | 22 188 | 0.62 % | 0.22 % |
| 10 mm | 8 375 | 1.7 % | 0.36 % |

The visible fraction grows with the cell (clustering shrinks thin parts and opens slits): 85.3 % → 86.5 % for the
wrist camera at 10 mm. On the GPU, decimation saves 5–11 ms per map, so it is off by default; it matters for the
numpy backend and for exact ray casting.

## Validation

Measured on the UR5e sample with the suite's default seed, the arm's six joints drawn around home (σ = 0.15 rad),
both cameras, the GPU backend (NVIDIA GeForce RTX 5080, EGL device 0):

| Check | Measured | Tolerance |
|---|---|---|
| `occlusion.texel_depth_exact`: texels whose stored distance differs from the exact first hit along the texel-centre ray by more than 1 mm | 4.2 × 10⁻⁶ of 240 000 texels (8 maps, n = 512); median difference 0.25 µm, 99th percentile 5.5 µm | 10⁻³ |
| `occlusion.cube_vs_raycast`: points whose visibility differs from exact ray casting and that are not within two texels of a silhouette | 2.5 × 10⁻⁵ of 80 000 points (2 points); all disagreements 0.22 %, 170 of them within two texels of a silhouette | 10⁻³, and all disagreements under 2 % |

The texel check differs only where a texel centre lies on an edge shared by triangles at different depths (the
rasteriser's edge rule against the ray test's) and by float32 rounding. In `tests/test_occlusion.py`, on a posed
UR5e (6 000 points per camera, n = 512), every disagreement between the cube map and ray casting lies within two
texels of a silhouette.

**The two backends** on the UR5e sample at home (`scripts/measure_occlusion.py`): no texel where one finds a surface
and the other does not (n = 256 and 512, both cameras); distances differ by a median of 0.15–2.6 µm (99th
percentile 3–14 µm); at n = 512, 2 × 10⁻⁶ (wrist) and 2 × 10⁻⁵ (overhead) of the texels differ by more than 1 mm,
texels whose centre lies on a silhouette edge, decided by OpenGL's top-left rule on one side and the inclusive test
on the other.

## Timings

The UR5e + Robotiq 2F-85 sample, 193 115 triangles (arm, gripper and the table), measured 2026-11-06 on Linux x86_64,
AMD Ryzen 9 9900X, Python 3.14.4, NumPy 2.5.4, PyOpenGL 3.1.10, MuJoCo 3.17.0. **EGL picked EGL device 0, the NVIDIA
GeForce RTX 5080 (driver 595.99.02), in process.** Medians of repeated builds with the geometry already on the GPU:

| Backend | Wrist D405, n = 256 | n = 512 | Overhead D435, n = 256 | n = 512 |
|---|---|---|---|---|
| gl, NVIDIA RTX 5080 (EGL device 0) | 12.6 ms | 12.5 ms | 13.8 ms | 15.5 ms |
| gl, AMD Radeon integrated (radeonsi, EGL device 2) | 58 ms | 18 ms | 12 ms | 16 ms |
| gl, llvmpipe (software, EGL device 3) | 39 ms | 43 ms | 32 ms | 35 ms |
| numpy | 0.40 s | 0.81 s | 0.23 s | 0.28 s |

(The integrated GPU's 58 ms at n = 256 is the median of ten builds in one quick run and was not investigated.) The
numpy time grows with how much of the map the scene fills: the wrist camera sees the gripper fill a quarter of its
sphere.

**Per joint vector, as Phase 3 will run it** (pose the arm, build the occluders and their digest, then one map per
camera at n = 512, 20 random arm poses, the geometry uploaded anew for each pose): 84 ms for the two cameras on the
RTX 5080, of which 49 ms is posing the arm's 193 115 triangles (`RobotModel.posed_triangles`, 35 ms) and the digest;
1.75 s with numpy. A 2 000-grasp study with three cameras is therefore about 2000 × (49 + 3 × 18) ms ≈ 3.5 min on
the GPU before any caching. What would cut it further, for Phase 3: upload each link's local mesh once and pass the
link transforms as uniforms (no posing or upload per q), and decimate for the numpy backend. *Since measured: the Phase 2 integration cut `posed_triangles` on the UR5e from 35 ms to about 12 ms (one matrix product per link instead of one per triangle; identical numbers), see [performance.md](performance.md).*

**Exact ray casting**: the grid over 193 115 triangles takes 57 ms to build; 20 000 visibility rays take 0.45 s
(wrist) and 0.28 s (overhead).

## Known limits

- One centre per map: a stereo pair needs a map per imager (the depth study builds both).
- Visibility is decided per texel (2/n rad at a face centre, 0.22° at n = 512), so a point within about a texel of a
  silhouette may be judged either way; the checks above measure how often.
- The occluders are the visual meshes as posed, with MuJoCo's equality constraints not applied (the 2F-85 linkage,
  `docs/BUILD_STATE.md`); no cables, cloth or other sensors' housings; a triangle seen exactly edge-on occludes
  nothing.
- Distances are float32 (about 0.1 µm at 1 m); `eps` (1 mm) decides points on surfaces.
- The GPU backend needs Linux EGL and OpenGL 3.3; elsewhere `auto` uses numpy. With PyOpenGL loaded for GLX it runs
  in a worker process. Making the context current is serialised by a lock, so maps are built one at a time.
- Vertex clustering moves vertices by up to half a cell diagonal and can open or close thin gaps (measured above).

## Against MuJoCo renders and on both samples {#occlusion-mujoco}

Three more checks (`dexvantage/validation/checks_occlusion_mujoco.py`) compare the cube map with references that share
none of its code. Measured with the default seed (20261027) on MuJoCo 3.17.0, EGL on an NVIDIA RTX 5080, n = 512:

| Check | What is compared | Measured | Tolerance |
|---|---|---|---|
| `occlusion.samples_vs_raycast` | cube map (gl and numpy) against exact ray casting, 20 000 random task-volume points per map; SO-100 and UR5e samples, every camera, home and two random poses (24 maps) | 0.19 % of 480 000 answers differ (worst map 0.45 %); none farther than a texel from a silhouette | the measured share of points within half a texel diagonal (√2/n rad) of a silhouette: 1.44 % |
| `occlusion.mujoco_box_spheres` | per-pixel visibility of 20 spheres and a wall behind a tilted box, against MuJoCo segmentation (640 × 480, fovy 60°) | 173 504 target pixels, 84.3 % visible: exact ray casting agrees on 99.9994 %, the cube map (both backends) on 99.82 %; 0 disagreements away from silhouettes | 1e-3 of the pixels away from silhouettes; overall agreement at least 99 % |
| `occlusion.mujoco_ur5e` | the UR5e + 2F-85 at a random pose seen from above, occluding the table top and 12 spheres in the task volume, against MuJoCo segmentation of the same MJCF file | 163 262 target pixels, 79.9 % visible: exact 99.993 %, cube map 99.81 % (both backends); 4 pixels (2.5 × 10⁻⁵) away from silhouettes | as above |

How the MuJoCo comparison works: the scene is rendered twice from one MuJoCo camera, with the targets only and with
the occluders too (`validation/mjrender.py`: `mujoco.Renderer` in a worker process with `MUJOCO_GL=egl` set before
mujoco is imported, multi-sampling off so segmentation ids do not blend at edges). The targets-only depth image gives
the target point behind each pixel centre (back-projected with the camera's `CameraModel`: P = ray · depth / ray_z in
the optical frame; MuJoCo's depth is the distance from the camera plane). MuJoCo calls a pixel visible when the full
render shows the same target there; we call it visible when `CubeMap.visible` (or `raycast.visible_exact`) finds
none of our occluder triangles in front of that point. Frames: MuJoCo's camera looks along its −z axis with +y up, so
the optical frame is R_cam · diag(1, −1, −1).

The cube map's agreement (99.8 %) is below exact ray casting's (99.99 %) for the reason given above: it answers per
texel, 0.22° at a face centre, coarser than the camera's 0.125° pixels, so a band about a texel wide along every
silhouette may go either way. The four UR5e pixels that disagree away from silhouettes do so for exact ray casting
and both cube maps alike: three are single pixels inside a robot mesh where MuJoCo's rasteriser leaves a crack
between two triangles and shows the table through it (our ray hits the mesh there), and one lies on the table's own
edge, which the two renders (with different depth ranges) round differently. They are properties of the reference.

If MuJoCo cannot render through EGL on a machine, the two MuJoCo checks fail with the reason
(`mjrender.RenderUnavailable`: "MuJoCo cannot render offscreen through EGL on this machine …"); they are not skipped.

`occlusion.samples_vs_raycast` takes about 20 s (most of it the numpy rasteriser, which runs on every map even when
the GPU is there); the two MuJoCo checks about 0.5 s and 2–3 s.
