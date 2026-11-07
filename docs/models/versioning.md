# Project diff, merge and the Git folder form

Modules: `dexvantage/core/projdiff.py` (diff, metric deltas, three-way merge), `dexvantage/core/projfolder.py`
(the unpacked folder and the `git diff` text), `dexvantage/batch_research.py` (batch options),
`dexvantage/ui/versioning.py` (File ▸ Compare with…, Save as folder (for Git)…, Open folder…). Validation:
`dexvantage/validation/checks_versioning.py` (model `versioning`).

## The diff

`diff(a, b)` compares two projects (Project objects, `.dexv` files or unpacked folders) in the project's own terms
and returns JSON with the schema `dexvantage.diff/1`:

    {'schema', 'a': {'name', 'file'}, 'b': {...}, 'identical', 'counts': {section: n},
     'changes': [{'section', 'op', 'kind', 'item', 'field', 'path', 'before', 'after', 'delta'?, 'text'}],
     'metrics': [{'id', 'title', 'unit', 'better', 'study', 'a', 'b', 'delta', 'verdict', 'status_a', 'status_b'}]}

Sections, in this order: meta (name, notes), robots (and their joints and links by name), variants, sensors (every
variant's), workspace (volumes, objects, lights, surfaces, background), tasks (grasp sets, trajectories, teleop
devices), requirements, policies, rig (hosts and their controllers, hubs, links, clocks, stream triggers), cables,
calibration (boards, sessions), settings (voxel sizes and each study's settings), data (datasets, audit
requirements) and files (mesh arrays and stored model files, by SHA-256). The design log and the creation and save
times are not compared; nor is the "live analysis" view toggle.

**Matching.** Items with a uid are matched by uid; those left are matched by name when both projects hold that name
once. A matched pair is compared field by field: a dict is walked, a list or a pose is one value. Numbers are equal
within one part in 10¹² (a save and load can change the last bit of a normalised quaternion).

**Pose deltas.** For a mount or base pose that changed, the diff reports

    shift (mm) = 1000 · (p_b − p_a)                in the parent frame's axes
    distance (mm) = |shift|
    turn (°) = 2 · atan2(|v|, |w|) of q_a⁻¹ q_b    (frames.quat_angle; full precision near zero)

**Metric deltas.** With study results for both projects ({study id: result doc}, for example the window's analysis
cache), `metric_deltas` reads every metric of core/metric_defs.py whose study both hold, with `core/metrics.read`,
and gives `delta = b − a` and a verdict (better, worse or same by the metric's direction). `evaluate_deltas` runs the
metrics' studies on both projects (`core/metrics.evaluate`, each study once per project). A figure one side does not
state gives no delta ("not stated on both sides"), never a guess.

## The three-way merge

`merge3(base, ours, theirs)` merges two edits of one project on the project's JSON:

- a value only one side changed takes that side; the same change on both sides is taken once;
- lists of items with uids (robots, sensors, variants, volumes, objects, requirements, policies, hosts, cables,
  boards, datasets…) and the rig's links (keyed by sensor) merge item by item; items the other side added are
  appended in its order;
- other lists (joint tables, grasp poses, trajectory samples) and poses (position and orientation together) are single
  values;
- mesh arrays and stored model files merge by key; the design log keeps ours and appends their new entries.

A conflict is a value both sides changed differently, an item one side removed while the other changed it, or a
reference the merge would leave dangling that neither side had (a sensor mounted on a robot the other side removed, a
policy camera, cable or link naming a removed sensor, an active variant that is gone). Any conflict refuses the whole
merge with `MergeConflict`, a sentence that lists each place ("variants ▸ Rig A ▸ sensors ▸ Wrist D405 ▸ mount ▸ pose
(base …, yours …, theirs …)"); nothing half-merged is returned.

## The unpacked folder

`save_unpacked(project, 'cell.dexv.d')` writes one pretty-printed JSON file per section (sorted keys, two-space
indent, UTF-8, a final newline), mesh arrays as `.npy` files (`allow_pickle` off) and the stored model files under
their own names; `project.json` lists them. `Project.load` and `Project.save` use this form for a folder or a path
ending in `.dexv.d`, so the window, batch mode and the API open it like a `.dexv` file.

Byte stability: the same project gives the same bytes in every file; a save writes only the files whose bytes changed
and removes only the files it wrote before (a `.gitattributes` or a README in the folder is left alone). The save time
is not written (it would change every file on every save); git's history says when.

`textconv(path)` prints each section's JSON as the folder holds it, then one line per mesh array (dtype, shape,
SHA-256) and stored file (size, SHA-256), for `git diff` (docs/guide/versioning.md).

## Validation (measured 2026-11-07, quick and full runs agree)

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `versioning.diff_lists_edits` | the 24 seeded edits of nine kinds, recorded as applied | 0 changes (a change is listed or not) | 0 missed, 0 extra; a project against its own copy: identical |
| `versioning.pose_deltas` | the applied shift (m × 1000) and the angle of the applied rotation (SciPy `Rotation.magnitude`) | 10⁻⁶ mm or ° (float64 on metre-sized values carries about 10⁻¹³) | 0 (3 poses) |
| `versioning.merge_cases` | eight cases with the outcome written out | 0 cases | 8 of 8: four merges with the expected number of changes from the base, four refusals |
| `versioning.unpack_round_trip` | the first unpacking's bytes; `provenance.project_digest` | 0 files | 66 files byte-identical; content hash equal through the folder and through `.dexv` |

## Known limits

A renamed item that also got a new uid is reported as removed and added. Lists (joint tables, grasp poses,
trajectory samples) are compared and merged whole: the diff says how many entries a list has and its hash, not which
grasp moved, and two edits of different entries of one list conflict. A merge can leave two items with the same name
(each side added one; the merge result notes it). Metric deltas need results for both projects. Mesh arrays are
binary in the folder (git stores them whole). The folder is written file by file, not as one atomic step (the zip is
atomic). A merge records itself in the merged project's design log (how many changes came from each side), not in a
stamped file.
