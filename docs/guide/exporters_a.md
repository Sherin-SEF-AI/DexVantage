# Exporting the rig: MuJoCo, ROS 2, calibration files and masks

A worked example on the UR5e sample (a UR5e with a Robotiq 2F-85, a RealSense D405 on the wrist and a D435 on the
cage). The model behind each file is in [docs/models/exporters_a.md](../models/exporters_a.md).

## In the window

1. File ▸ Open sample ▸ UR5e. Pose the arm if you want the cameras on the wrist exported somewhere other than home.
2. File ▸ Export… (Ctrl+E). Tick the formats: each row says what it writes. The common options are controls (the
   MJCF mode, the ROS package for mesh paths, which imagers, the mask atlas and depth masks, STEP for brackets, the
   dataset for FiftyOne); anything else goes in the row's Options field as JSON, for example `{"scale": 0.5}` for
   half-size masks or `{"atlas_grasps": 16}`.
3. Pose: "The home pose of every robot" or "The pose on screen". Choose an output folder, then Export. The export
   runs in the background (progress in the status bar); the dialog then lists each format with its file count and
   notes, or the reason it was refused.

## In batch

    dexvantage --sample ur5e --export out/ur5e --formats mjcf,urdf,opencv_yaml,masks,plucker

On the UR5e sample this writes 82 files in 9.3 s (offscreen, measured 2026-11-07; 1.6 GB peak), and
prints each format's notes, for example:

    urdf: 24 files
      · UR5e + Robotiq 2F-85: the effort and velocity limits of shoulder_pan_joint, … are not stated by its file and
        are written as 0 (urdfdom needs the attributes): set them before giving the file to a controller or a planner.
    masks: 13 files
      · Mask atlas: 8 grasps of "Tabletop grasps (sample)" (21 of the 32 tried of its 200 are reachable by inverse
        kinematics); tiles at 0.25 of the stream size (by default 8 grasps at 0.25).
    5 of 5 formats written; manifest → out/ur5e/export_manifest.json

`--export-options '{"mjcf": {"mode": "snippets"}, "masks": {"atlas": false}}'` passes options per format (JSON text
or a .json file). A format name DexVantage does not have is refused in one line before anything is written (exit
2); a format that cannot run on the project (no bracket type, no policy, no dataset) is refused with its reason while
the others are written (exit 1).

## What you get, and what to do with it

- **mjcf/cameras_snippet.xml**: paste each block inside the named `<body>` of `ur5e_robotiq_2f85.xml` (the wrist
  D405's three cameras and its mount site go in `wrist_3_link`, the D435's in `<worldbody>`). **mjcf/scene/scene.xml**
  loads on its own: `python -m mujoco.viewer --mjcf out/ur5e/mjcf/scene/scene.xml`.
- **ros2/rig.urdf** with `ros2/meshes/`: `world` → the UR5e's links → `wrist_d405_link` →
  `wrist_d405_rgb_optical_frame` (and the IR frames), `world` → `overhead_d435_link` → its optical frames. Load it
  with robot_state_publisher, or run `ros2 launch static_transforms.launch.py` beside your own UR description to
  publish only the sensor frames. Set the URDF exporter's package option (for example `{"package": "my_cell"}`) so
  RViz finds the meshes as `package://my_cell/meshes/…`.
- **calibration/opencv/*.yaml**: read with `cv2.FileStorage(path, cv2.FILE_STORAGE_READ)`; `distortion_model` says
  which OpenCV functions take the coefficients (plumb_bob and rational_polynomial: `cv2.projectPoints`, fisheye:
  `cv2.fisheye`). The sample's RealSense imagers are pinhole, so every mapping is exact; a calibrated equisolid or
  stereographic lens is written as the closest Kannala-Brandt curve and the file states its error in pixels.
- **calibration/kalibr/camchain.yaml**: a starting camchain for `kalibr_calibrate_cameras` (replace the suggested
  rostopic names with your driver's). The wrist D405 moves with the arm, so the chain from it to the D435 holds at the
  export's pose only (the notes say so).
- **masks/self/wrist_d405__rgb.png**: the robot covers 34.4 % of the wrist D405's colour image at home (the gripper)
  and 25.4 % of the overhead D435's. `_owner.png` (16-bit) says which link each pixel sees: index into
  `masks.json` `owners`. **masks/atlas/** tiles the masks over 8 reachable grasps and the frequency image shows how
  often each pixel is covered.
- **masks/depth/wrist_d405__ir_left.png**: 61.2 % of the D405's depth pixels valid at home, 26.7 % closer than its
  min-Z (the gripper fingers), 6.3 % occluded for one imager, 5.1 % outside the stereo overlap; the overhead D435 49 %
  valid and 48.7 % "no surface" (it looks past the table at the floor, which DexVantage does not model).
- **plucker/**: π0's per-pixel rays for each of its cameras at the export's pose.
- **export_manifest.json**: every file's SHA-256 and the project's provenance; `dexvantage --reproduce` and a
  reviewer can tell which project and settings made the files.

## Brackets and STEP

Choose a bracket type on the Brackets page (the UR5e wrist D405 takes a flange adapter, ISO 9409-1-50-4-M6), then
export `brackets`. STEP needs the optional gmsh package (`pip install "dexvantage[step]"`); it is a faceted solid
(each STL triangle a face). The wrist adapter's mesh at the default 0.5 mm grid has 119 092 triangles, more than the
20 000 the STEP writer takes by default, so pass `{"step": true, "step_res_mm": 2.0}` for a coarser STEP (16.7 MB).

## Checking the files

Help ▸ Validate the models… runs the `formats` checks: the snippets and the URDF are loaded by MuJoCo and every
camera frame is compared with DexVantage's kinematics, the OpenCV YAML and Kalibr files are read back and compared
number for number, and the masks are compared per pixel with exact ray casting and the occlusion study's cube map
(the figures are in the models document).
