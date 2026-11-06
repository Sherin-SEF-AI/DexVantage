# Performance: the UI-thread budget {#performance}

Spec §16: no UI-thread work over 100 ms. Measured with `scripts/measure_ui.py` (`QT_QPA_PLATFORM=offscreen python
scripts/measure_ui.py --json out/ui-timings.json`) on 2026-11-06: Linux x86_64, AMD Ryzen 9 9900X, Python 3.14.4,
PyQt6 6.12.0, NumPy 2.5.4, MuJoCo 3.17.0. The SO-100 sample has 60 682 visual triangles, the UR5e + 2F-85 sample
193 113. Figures are the median of five runs with the largest in brackets; a single figure is one run.

**How it is measured.** Each handler a user triggers is timed on the UI thread directly. An event-loop watchdog (a
5 ms timer) also records the longest stall while the events that follow each step are processed, so a slow handler
the list does not name still shows. Studies and Camera view images run on worker threads and are listed as
"worker". Offscreen, the 3D view falls back to the QPainter renderer; its frame is excluded from the UI handlers'
figures and listed on its own. The shader renderer's frame (what a desktop pays) is measured headless through EGL on
the integrated AMD GPU (radeonsi), the same context the self-test makes.

## Result

| Step | Thread | SO-100 | UR5e |
|---|---|---|---|
| build and show the main window (startup, once) | start | 252 ms | 306 ms |
| show the Sensors and mounts page | UI | 0.8 ms [4.6] | 1.5 ms [6.2] |
| show the Coverage page | UI | 0.6 ms [2.6] | 0.6 ms [3.7] |
| show the Depth validity page | UI | 0.5 ms [2.6] | 0.8 ms [3.2] |
| coverage study, whole run | worker | 252 ms | 599 ms |
| Coverage page: the result arrives (`show_result`) | UI | 3.8 ms [4.4] | 26.7 ms [27.2] |
| Coverage page: 3D slice overlay for one frame | UI | < 0.1 ms | < 0.1 ms |
| depth-validity study, whole run | worker | 4.8 ms (no depth sensor) | 339 ms |
| Depth page: the result arrives | UI | < 0.1 ms | 0.3 ms [0.6] |
| Depth page: 3D overlays (min-Z zones, failing voxels) for one frame | UI | < 0.1 ms | < 0.1 ms [1.1] |
| a joint change: posed meshes, frusta, volumes, page overlays | UI | 5.0 ms [6.9] | 16.4 ms [19.9] |
| select each of the two sensors (Inspector, Camera view header, frusta) | UI | 26.1 ms [27.6] | 38.9 ms [44.7] |
| Camera view: snapshot of the project for a render | UI | 0.1 ms [0.4] | 0.2 ms [0.6] |
| Camera view: render one full-size image | worker | 27 ms | 78 ms |
| Camera view: take the image and paint it with the page's overlays | UI | 0.6 ms [3.7] | 0.7 ms [5.4] |
| Library panel: refill (40 entries) | UI | 0.8 ms [1.2] | 0.8 ms [1.1] |
| New custom device dialog, started from a library device | UI | 16.8 ms [18.5] | 16.4 ms [16.8] |
| largest watchdog stall after any step | UI | 8.6 ms | 9.9 ms |
| 3D view, shader renderer: first frame (mesh upload) | UI | 28 ms | 57 ms |
| 3D view, shader renderer: steady frame | UI | 7.8 ms [8.6] | 11.9 ms [13.0] |
| 3D view, shader renderer: frame after a joint change (pose, upload, draw) | UI | 26.5 ms [28.1] | 67.6 ms [95.5] |
| 3D view, QPainter fallback: one frame (no OpenGL) | UI | 160 ms [184] | 586 ms [744] |

No UI-thread step is over 100 ms on the samples with the shader renderer. Window start-up (0.25–0.31 s) happens once.

## What was changed to meet it (Phase 2 integration)

Before these changes the shader renderer's frame after a joint change took 108 ms (SO-100) and 385 ms (UR5e):

- **Smooth normals once per model.** The renderer merged every corner by position (`np.unique` over 580 000
  corners of the UR5e, 256 ms) at every pose. The scene now computes them once per robot model in each link's own
  frame (`render_gl.vertex_normals(..., group=link)`) and rotates them with the links at each q
  (`RobotModel.pose_vectors`), handing them to the renderer as `MeshDraw.vnormals`. Merging in link frames is also
  steadier: coinciding vertices of two links are no longer smoothed together, and the shading no longer changes as the
  arm moves.
- **Posing by link.** `RobotModel.posed_triangles` used a 3 × 3 product per triangle (einsum, 33 ms on the UR5e); it
  now does one matrix product per run of a link's triangles (12 ms, the same numbers to the last bit).
- **Upload.** The mesh bounds were an axis-0 reduction of an (N, 3) array (20 ms); per column they take 1 ms. An
  all-opaque mesh is no longer copied to reorder it.

## Known limits

- The QPainter fallback, used only where OpenGL is missing, draws every triangle from Python: 0.16 s per frame for the
  SO-100 and 0.59 s for the UR5e. A decimated mesh for that path (`core/scenegeom.decimate`) is the fix; not done.
- The shader renderer still poses and uploads the whole mesh at each joint change (68 ms median on the UR5e on an
  integrated GPU, 96 ms at worst). Uploading each link's mesh once and passing the link transforms as uniforms would
  remove both; that is also what Phase 3's batched occlusion wants (see [occlusion.md](occlusion.md#timings)).
- Measured on one machine; the GL figures are an integrated GPU's. The on-screen `QOpenGLWidget` path has not been
  timed on a desktop session.
