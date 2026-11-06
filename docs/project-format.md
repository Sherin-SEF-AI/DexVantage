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
                                 "slot": 0, "locked": false, "enabled": true } ] } ],
  "active_uid": "…",
  "requirements": [ {"uid", "name", "metric", "op": ">=|<=|<|>|==", "threshold", "unit", "study",
                     "is_default": true, "note"} ],
  "analysis": { "voxel_mm": 5.0, "refine_mm": 2.0, "live": true, "studies": {} },
  "log": [ {"t", "kind", "text"} ],
  "meshes": ["keys in meshes.npz"], "sources": ["names under sources/"]
}
```

Units are SI (metres; joint values in radians); quaternions are (x, y, z, w); frames follow REP-103. A uid is ten
lowercase hex digits. Mount tolerances default to ±1 mm and ±0.5° (labelled defaults, set per mount).

## Versions and migrations

`version` is 1. A file of an older version is migrated one version at a time (`core/project.py`, `MIGRATIONS`); a
newer version is refused with a message naming both versions. A file that is not a zip, has no `project.json` or has
damaged JSON is refused with a message that says which.
