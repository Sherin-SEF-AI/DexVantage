# Robots: import, the MuJoCo model and the samples {#robots}

Modules: `dexvantage/core/robot/importer.py`, `model.py`, `filecheck.py`, `trajectory.py`, `samples.py`,
`menagerie.py`. Tests: `tests/test_robot_core.py`, `test_samples.py`, `test_menagerie.py`, `test_robot_ui.py`,
`test_phase1_integration.py`. Forward kinematics is checked in [kinematics.md](kinematics.md); IK, manipulability and
collision distances in [ik.md](ik.md).

## Import

`import_robot(path)` reads a URDF or MJCF file and every file it names (meshes, included XML, textures) into memory.
The project keeps them byte for byte (the `.dexv` archive's `sources/`), so a robot opens on any machine.

- **Where the files go.** Under `<robot uid>/<path relative to the robot's folder>`; a file outside that folder under
  `<uid>/_ext/<name>`. A manifest `<uid>/.dexvantage/refs.json` records which stored file each reference means, so the
  model compiles again from memory exactly as imported. It holds no uid, so the content hash is stable.
- **How references resolve.** URDF `package://pkg/…` is looked for by walking up from the URDF to a folder named
  `pkg`; `file://` and absolute paths are used as written; anything else is relative to the URDF. MJCF includes and
  `meshdir`, `texturedir` and `assetdir` are resolved as MuJoCo does: the list of files comes from MuJoCo's own parser
  (`mujoco.MjSpec`), not a hand-written scan.
- **Figures.** Joint limits, velocities and efforts are the file's own (radians, metres, rad/s or m/s, N·m or N); a
  figure the file does not state is `None` and shows as "—". The home pose is the MJCF keyframe `home`, else zero
  clipped to the limits.
- **Units check.** A mesh more than 10 m across (`MESH_MAX_EXTENT`, a plausibility default of this build) is almost
  certainly in millimetres: the import stops and says to set `scale="0.001 0.001 0.001"`. The import wizard also
  flags a robot under 20 mm or over 5 m across (a plausibility default, labelled as one).
- **Errors** are sentences that name the file and what to pick; a missing MJCF texture is not fatal and is recorded in
  `Robot.source.note`.

## The kinematics model

`RobotModel(robot, sources)` compiles the stored files with MuJoCo from an in-memory asset dictionary (nothing is read
from disk). URDF is compiled with `discardvisual="false"` and `fusestatic="false"` so visual meshes and links joined by
fixed joints keep their own bodies and names. Visual and collision geoms are told apart by MuJoCo's collision masks
(`contype = conaffinity = 0` is visual only).

- `fk(q)`, `link_pose(q, link)`, `fk_batch(Q)`: 4 × 4 world transforms, world = `Robot.base` applied to MuJoCo's frame.
- `jacobian(q, link, point)`: the (6, nq) world-frame [linear; angular] Jacobian from `mj_jac`.
- `posed_triangles(q)`: the visual meshes at q, for drawing and (Phase 3) occlusion.
- q holds the one-degree-of-freedom joints (hinge, slide) in MuJoCo's tree order. Fixed joints are not in q; the
  project summary, the Outliner and the 3D view's caption count the joints in q.

When MuJoCo refuses a model only because a moving link has no mass (common in kinematics-only URDFs), it is compiled
again with mass and inertia bounds of 10⁻⁶ kg and 10⁻⁹ kg·m², and the model's notes say so. Kinematics does not
depend on mass.

`get_model(project, uid)` is a thread-safe cache shared by the UI and the studies, keyed by the SHA-256 of the
robot's files, its records and its base pose.

## Robot records the UI writes

- A gripper joint has `"gripper": true` on its record in `Robot.joints` (the wizard's joint tagger; the samples tag
  theirs). IK leaves joints that do not move the target link alone either way (`manipulability.chain_mask`).
- `Robot.gripper_frame` and `Robot.fingertip_frames` name links. The wizard guesses the gripper frame by name (tcp,
  tool0, ee, gripper, hand, flange, …) and never guesses a link that a gripper joint moves.
- A contact pad is `{"link", "name", "box": {"center": [m], "size": [m]}}`, an axis-aligned box in the link frame.
- A trajectory is `{"uid", "name", "robot_uid", "joints": [names], "t": [s], "q": [[per joint] per sample]}` in
  `project.tasks.trajectories`, interpolated linearly by the pose scrubber.

## Bundled models and samples

Three MuJoCo Menagerie models are bundled after their own licences were read (`THIRD_PARTY.md`): `trs_so_arm100`
(Apache-2.0), `universal_robots_ur5e` (BSD-3-Clause) and `robotiq_2f85` (BSD-2-Clause), from Menagerie commit
`98a70c569c4c`. The UR5e's OBJ meshes are stored as binary STL (the same triangles as MuJoCo compiles them, a
difference of 0 m; 31.0 MB of text became 6.1 MB). `ur5e_robotiq_2f85.xml` is the arm with the gripper attached at
its `attachment_site`, written by MuJoCo's `MjSpec.attach` (`scripts/bundle_menagerie.py`).

| Sample | Robot | Joints in q | Gripper frame | Sensors (sample placements) |
|---|---|---|---|---|
| `so100` | SO-ARM100 | 6 (5 arm, 1 jaw) | `Fixed_Jaw` | wrist camera on `Fixed_Jaw`, third-person camera on a tripod |
| `ur5e` | UR5e + Robotiq 2F-85 | 14 (6 arm, 8 linkage hinges) | `gripper_base` | "Wrist D405" on `wrist_3_link`, "Overhead D435" 0.85 m above the volume |

The camera placements are chosen for the samples, not published figures, and say so in their names or notes. The
sensors get their datasheet models in Phase 2. Robot ▸ Import from MuJoCo Menagerie downloads any other Menagerie
model, shows its own LICENSE and imports it only after the user accepts it ([guide](../user-guide.md#robots)).

## Validation

The kinematics checks run on the bundled samples as well as on the robots written in the checks: with no project
given, the validation suite's built-in project holds both sample robots. Measured with the default seed (full run):

| Check | Robots | Error | Tolerance |
|---|---|---|---|
| `kinematics.fk_poe_mujoco` | both samples and five robots written in the check, 200 random q each, every link | 8.7 × 10⁻¹⁵ m | 10⁻⁹ m |
| `kinematics.jacobian_mujoco` | the same robots, 40 random q each, at each link's origin and a random point fixed in the link | 8.8 × 10⁻¹⁵ | 10⁻⁹ |

`dexvantage --self-test` repeats the comparison quickly ("robots": both samples import, MuJoCo and the
product-of-exponentials chain agree within 10⁻⁹ m at five random poses each).

## Known limits

xacro must be expanded to plain URDF first; USD is not read yet (the wizard says so).
URDF mimic joints and the 2F-85's linkage move independently: MuJoCo's equality constraints are not applied by
kinematics, so moving one finger joint alone poses the linkage in a way the real gripper cannot. Free and ball
joints are held at the home keyframe and are not part of q. Planes and height fields are not drawn. MJCF `<model>`
(attach) assets and plugins are not collected, and the independent reader stops with a plain error on `replicate`,
`composite` and `flexcomp`. Frames are links: MJCF sites
cannot be the gripper frame yet. The SO-101 is not bundled (18 MB of meshes); import it from Menagerie.

What the bundled files are, by Menagerie's own READMEs (read 2026-11-06): "simplified" MJCF descriptions derived
from public URDFs. The SO-ARM100's joint limits are ones Menagerie added as "approximate joint limits" (its derivation
step 7), not The Robot Studio's figures; the robot's source note says so. The UR5e's link offsets are rounded to the
millimetre (d1 = 0.163 m against Universal Robots' 0.1625 m, d5 = d6 = 0.100 m against 0.0997 m and 0.0996 m): after
the best rigid alignment its flange is up to 1.3 mm (0.9 mm RMS over 500 random poses) from the flange of Universal
Robots' DH table (`tests/test_samples.py`). Plan with the maker's figures where a millimetre matters.
