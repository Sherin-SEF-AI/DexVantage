# The .dexv project file

A `.dexv` file is a zip archive (written atomically: a `.part` file renamed over the old one).

| Member | Content |
|---|---|
| `project.json` | the project; JSON Schema: [project.schema.json](project.schema.json) (draft 2020-12) |
| `meshes.npz` | mesh arrays by key (NumPy `savez_compressed`, read with `allow_pickle=False`) |
| `sources/<name>` | the original model files robots and meshes were read from, byte for byte |
| `thumb.png` | a picture of the 3D view at save time (optional) |

## project.json

```jsonc
{
  "format": "dexvantage.project", "version": 1,
  "name": "Tabletop cell", "notes": "", "created": "2026-11-06T10:00:00Z", "saved_at": "…",
  "robots": [ { "uid": "…", "name": "…", "source": {"kind": "urdf|mjcf|usd|", "file": "…", "sha256": "…",
                "license": "…", "note": "…"},
                "base": {"position": [x, y, z], "quaternion": [x, y, z, w]}, "primary": true,
                "links": [], "joints": [], "home": {}, "gripper_frame": "", "fingertip_frames": [],
                "contact_pads": [], "locked": false } ],
  "workspace": { "volumes": [ {"uid", "name", "kind": "tabletop|shelf|bin|handover|custom",
                               "center": [m], "size": [m], "refine": false} ],
                 "objects": [ {"uid", "name", "shape": "box|cylinder|sphere|mesh", "size": [m], "pose": {…},
                               "mesh_key": "", "materials": ["matte|glossy|metallic|transparent|black|ir_absorbing"]} ],
                 "lights": [], "surfaces": [], "background": "studio" },
  "tasks": { "grasp_sets": [], "trajectories": [], "teleop_devices": [] },
  "variants": [ { "uid", "name", "note", "created",
                  "sensors": [ { "uid", "name", "kind": "camera|depth|tactile", "model_id": "", "calibration": {},
                                 "mount": { "parent_frame": "world|cage|tripod|table|<link>", "robot_uid": "",
                                            "pose": {…}, "tolerance": {"position_mm": 1.0, "rotation_deg": 0.5},
                                            "bracket": "" },
                                 "slot": 0, "locked": false, "enabled": true, "stream": [w, h] } ] } ],
  "active_uid": "…",
  "requirements": [ {"uid", "name", "metric", "op": ">=|<=|<|>|==", "threshold", "unit", "study",
                     "is_default": true, "note"} ],
  "analysis": { "voxel_mm": 5.0, "refine_mm": 2.0, "live": true, "studies": {} },
  "rig": { "hosts": [], "hubs": [], "links": [] },           // Phase 6, absent or empty when there is no topology
  "cables": [],                                              // Phase 6
  "policies": [],                                            // Phase 3
  "calibration": { "boards": [], "sessions": [] },           // Phase 7
  "log": [ {"t", "kind", "text"} ],
  "meshes": ["keys in meshes.npz"], "sources": ["names under sources/"]
}
```

Units are SI (metres; joint values in radians); quaternions are (x, y, z, w); frames follow REP-103. A uid is ten
lowercase hex digits. Mount tolerances default to ±1 mm and ±0.5° (labelled defaults, set per mount).
A sensor's `stream` is the stream mode chosen for it (width, height in pixels, one of its library device's stream
modes); `[]` (the default) uses each imager's largest listed stream.

## Robots (Phase 1)

- `sources/<robot uid>/<path in the robot's folder>`: the robot's model file and every file it names; a file outside
  the robot's folder is under `sources/<uid>/_ext/<name>`. `sources/<uid>/.dexvantage/refs.json` (format
  `dexvantage.robot-refs`) maps each reference in the model files to its stored file. A Menagerie import also stores
  the model's `sources/<uid>/LICENSE`.
- `links`: `[{"name", "parent", "parent_joint", "mesh_keys", "collision_keys"}]`.
- `joints`: `[{"name", "type": "revolute|continuous|prismatic|fixed|ball|free", "parent", "child", "axis",
  "origin": {"position", "quaternion"}, "lower", "upper", "velocity", "effort"}]`, the file's own figures (`null`
  where it states none); a gripper joint carries `"gripper": true`.
- `home`: `{joint name: value}`; `gripper_frame` and `fingertip_frames` name links.
- `contact_pads`: `[{"link", "name", "box": {"center": [m], "size": [m]}}]`, an axis-aligned box in the link frame.
- `tasks.trajectories`: `[{"uid", "name", "robot_uid", "joints": [names], "t": [s], "q": [[value per joint] per
  sample]}]` in radians and metres.

## Policies (Phase 3)

- `policies`: `[{"uid", "name", "library_id", "cameras": [sensor uids], "overrides": {field: {"value", "source"}}}]`:
  the policies the data is for (`core/policies.py`, docs/models/policies.md). `library_id` names a library entry of
  kind `policy`; `cameras` `[]` means every camera and depth sensor of the active rig; an override is the user's own
  figure for a field with where it comes from (`"measured by the user: …"`). A file without `policies` reads as none.

## Grasp sets (Phase 3)

- `tasks.grasp_sets`: `[{"uid", "name", "source": "sampled|imported|trajectory|dataset", "robot_uid", "frame",
  "poses": [{"id", "pose": {"position", "quaternion"}, "approach": [x, y, z], "kind", "object_uid", "width_m", "t"}],
  "settings": {…}}]`. `pose` is the gripper frame in the world at contact, `approach` a world unit vector; `settings`
  records how the set was made (sampler options and which are defaults, the seed, the file or trajectory, the gripper
  geometry, notes). Written and read only through `core/grasps.py` ([models/grasps.md](models/grasps.md)).

## Host topology, cables and calibration (Phases 6 and 7)

- `rig`: `{"hosts": [{"uid", "name", "library_id", "controllers": [{"uid", "name", "kind": "usb3|usb2|pcie|gmsl",
  "practical_mbps", "source"}]}], "hubs": [{"uid", "name", "powered", "upstream"}], "links": [{"sensor_uid",
  "stream_ids", "via", "cable_uid"}]}`. `practical_mbps` is `null` when not stated; a link's `stream_ids` name stream
  modes as `imager|WxH|fps|format`, empty for the default streams ([models/bandwidth.md](models/bandwidth.md)).
- `cables`: `[{"uid", "name", "sensor_uid", "type", "library_id", "min_bend_radius_m": {"value", "source", "kind"} or
  null, "diameter_m", "length_m", "wrap_radius_m", "attachments": [{"robot_uid", "link", "point", "length_to_next_m"}],
  "connectors", "strain_relief", "note"}]` ([models/cables.md](models/cables.md)).
- `calibration`: `{"boards": [{"uid", "name", "board": {kind and figures}, "mount": {"parent_frame", "robot_uid",
  "pose"}, "default"}], "sessions": [{"uid", "name", "sensor_uid", "imager", "mode": "eye_in_hand|eye_to_hand",
  "board_uid", "robot_uid", "link", "created", "source": {"kind": "plan|folder", …}, "settings", "poses": [{"name",
  "q", "joint_names", "image", "t_image", "t_joints", "dt_s", "ids", "uv", "note"}], "result"}]}`. Sessions keep image
  file names without folders and the detected corner pixels, about 25 kB per 25 images
  ([models/handeye.md](models/handeye.md#handeye-planner)).
- Each of these sections, like `policies`, enters the content hash (`core/provenance.py`) only when it holds
  something, so files written before them keep their hashes; a file without a section reads it as empty.

## Versions and migrations

`version` is 1. A file of an older version is migrated one version at a time (`core/project.py`, `MIGRATIONS`); a
newer version is refused with a message naming both versions. A file that is not a zip, has no `project.json` or has
damaged JSON is refused with a message that says which.
