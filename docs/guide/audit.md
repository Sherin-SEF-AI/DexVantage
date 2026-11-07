# Auditing a dataset and comparing it with the plan

The dataset audit checks every episode of a recording from its own data: the camera timestamps, the pixels and the
joint states. Then it checks the whole campaign for mount slip. Plan against recording compares the same recording with
the project, camera by camera. The models, the defaults and the measured validation are in `docs/models/audit.md`.

## In the app

1. Open the project the data was recorded with, or a sample. The plan tells the audit how the robot should move in
   each image and where the gripper should be. Without a robot, those checks fall back to the joint speed or are
   skipped.
2. Choose Data ▸ Audit a dataset…, then Add folder… for a LeRobot, RLDS or bag folder, or Add file… for a single
   `.mcap` or `.db3` bag. The dataset is kept in the project (the `data` section), with its path as you chose it.
3. Press Run. The audit runs in the background. The headline shows how many episodes meet every requirement:
   - The requirement table shows each limit, marked as a default or yours, and the episodes that fail it.
   - The episode table gives each episode a verdict: pass; fix (re-stamp the camera); recalibrate (the mount moved);
     or drop (drop it or re-record it).
   - Click an episode to see its cameras.
   - An episode whose files are damaged is listed with the verdict drop and the flag unreadable; the reason is in
     its notes. The audit goes on with the other episodes.
   - A worst figure marked "(not resolved)" is past its limit but within 3 σ of zero (a drift estimate on a short
     episode, for example), so it is not counted as a failure.
   - The chart shows each camera's mount over the campaign, with the steps the classifier found.
4. Change a requirement in the settings below the chart. Each change is one undo step; run the audit again to apply it.
5. Apply records the episodes to drop or re-record in the dataset's entry (`exclude`). Export report… writes
   `audit.json` and `audit.html`. Export to FiftyOne… writes a FiftyOne dataset folder.
6. Data ▸ Plan against recording… compares the same dataset with the plan. For each camera it checks that the camera is
   present and compares its resolution, frame rate, mount, sync and gripper visibility with the plan. If a camera was
   recorded at another size with the same aspect ratio, Apply sets its stream to the recorded size, so the studies
   model what was recorded.

## Worked example: a synthetic SO-100 campaign

Make a campaign of six episodes with every fault on, and a mount slip from episode 3:

    dexvantage --sample so100 --synth-episodes so100-synth \
        --synth-settings '{"episodes": 6, "frames": 240, "max_width": 320, "seed": 7,
                           "faults": {"mount_slip": {"from_episode": 3}}}'

This took 12.8 s on the RTX 5080 (2 820 camera frames). Then audit the campaign:

    dexvantage --sample so100 --audit audit.json --dataset so100-synth

This took 32 s on the CPU. Add `--audit-html audit.html` for the standalone report page and `--fiftyone fiftyone/`
for the FiftyOne folder (a new or empty one), the same files the page's exports write. The summary reads "not usable,
0 of 6 episodes". Its advice:

    Drop or re-record episodes 0, 1, 2, 3, 4 and 5; observation.images.third_person_camera moved at episode 3
    (rotation y +1.41°): calibrate it again from there.

With every fault in every episode, every episode fails. The requirement table shows which requirement each one fails:

| Requirement | Passed | Worst |
|---|---|---|
| Dropped frames | 6 of 12 | 2 |
| USB resets and gaps | 6 of 12 | 1 |
| Timestamp jitter | 6 of 12 | 3.0 ms |
| Exposure and gain steps | 6 of 12 | 2 |
| Gripper hidden | 0 of 6 | 3 frames |
| Mount slip (rotation) | 9 of 12 | 1.41° (episodes 3, 4, 5) |
| Frame rate | 12 of 12 | 0.012 % |
| Action-observation alignment | 6 of 6 | 1.0 row (the generator's action lead) |

The generator places each fault on one camera per episode, so for most faults half of the 12 camera checks fail. The
mount-slip step is found on the third-person camera at episode 3: a 1.41° pitch against the injected 1.5°. The wrist
camera stays steady.

Compare the campaign with the plan:

    dexvantage --sample so100 --plan-vs-recording pvr.json --dataset so100-synth

This took 33 s. The headline reads "70 %, differs": 7 of the 10 comparisons match.
- Both cameras are found and matched to their sensors by name.
- Their resolution is "scaled": 320 × 180 against the plan's 1280 × 720 stream, the same aspect ratio.
- The frame rate is "not stated by the plan": the sample's cameras have no library device.
- The third-person mount is "moved" (1.41° over the campaign).
- The third-person sync is out (offsets 24 to 89 ms).
- The third-person gripper visibility agrees with the grasp study on 75 % of the 32 frames tested, and the wrist
  camera's on 100 %.

The ground truth is in `so100-synth/dexvantage_truth.json`. The audit never reads it; compare it with `audit.json`
yourself, or with `dexvantage.validation.checks_audit.score(result, truth)`.

## FiftyOne

Export to FiftyOne… (or `dexvantage.core.audit.report.export_fiftyone(result, dataset, folder)`) writes one video sample
per camera per episode:
- Sample tags: `audit:<verdict>` and each fault kind found.
- Sample fields: the offset, drift, jitter and frame counts.
- Frame fields: sharpness ratio, gain step, gripper correlation, duplicate and blur flags, and so on.

FiftyOne is not needed to write the folder. To load it:

    import fiftyone as fo
    ds = fo.Dataset.from_dir('so100-fo', dataset_type=fo.types.FiftyOneDataset)
    ds.match_tags('gripper_hidden')

## What the figures mean, and their limits

- A LeRobot recording stores its camera times on the nominal grid. A frame dropped before writing therefore shows as a
  duplicate, not a gap.
- The camera offset is resolved to about ±17 ms on static cameras, and poorly on wrist cameras over plain scenes. A
  wrist offset only fails when it is beyond 3 σ.
- Clock drift within one episode is reported but not resolved below several thousand ppm.
- Mount slip by registration is rotation only, relative to the first episode. It needs at least five episodes to tell
  a step from a drift.
