# The sim-to-real gap: a worked example

How far is the render of your cell from what the cameras recorded, and what would close the gap? This walk-through
uses the UR5e sample and a recording with four known gaps on its overhead D435. It reads the ranked causes and
applies them one at a time. The method, its equations, checks and limits are in `docs/models/sim2real.md`.

The recording here is synthetic, so every gap is known. On a real rig it is your LeRobot, RLDS or ROS 2 bag
recording, made with the rig the project describes.

## A. A recording with known gaps

From Python (no window):

    import dexvantage as dv
    rig = dv.sample('ur5e')
    rig.gap_dataset('out/s2r_gaps', {'camera': 'observation.images.overhead_d435', 'brightness_dn': 20,
                                     'blur_px': 1.5, 'latency_s': 0.04,
                                     'mount': {'position_m': [0.003, 0, 0], 'rotation_deg': [0, 1.0, 0]}},
                    episodes=1, frames=40, size=[320, 180], seed=4)

This renders one 40-row episode of the sample's pick through both cameras (the synthetic episode generator: its mount
slip and camera–joint offset faults). It then brightens the overhead frames by 20 DN and blurs them with σ 1.5 px
before the video codec writes them. It takes 1.6 s. `dexvantage_truth.json` beside the data lists what was injected
under `sim2real_gaps`.

(Simulate ▸ Generate synthetic episodes… gives a recording with the mount slip and offset faults; the photometric
gaps are only in the API and the checks.)

## B. Run it (Data ▸ Sim-to-real gap)

1. Open the sample: File ▸ Open sample ▸ UR5e.
2. On the page, click **Dataset…** and choose `out/s2r_gaps`. The choice is one undo step and is stored with the
   project. Without a choice the page uses the audit's dataset.
3. Click **Run**. The study takes 7–8 s on the RTX 5080: about 330 renders per camera, plus the silhouette
   calibration of the overhead camera.

The headline reads **24.4 DN, "very large gap"**: the overhead camera's mean absolute difference between the recorded
frames and their render, before any correction. The wrist D405, which got no gap, reads 1.1 DN.

## C. Read it

The **Recorded / Rendered / Difference** row shows a frame of the chosen camera. Choose **Rendered with every
correction** and the difference image goes almost black. **Difference over the recording** lays the difference in
red over the recorded frame.

The list **What explains the gap** gives each correction's Shapley share: what it removes, averaged over every order
of applying the corrections.

| # | cause | removes | share | figure | fixed on |
|---|---|---|---|---|---|
| 1 | Exposure and colour | 14.6 DN | 60 % | gain 1.00, 0.99, 1.00, offset +19, +19, +19 DN | Sim-to-real gap |
| 2 | Mount offset | 3.2 DN | 13 % | 6.7 mm, 1.31° | Silhouette calibration |
| 3 | Blur | 2.4 DN | 10 % | σ 1.43 px | Motion blur |
| 4 | Camera latency | 2.1 DN | 9 % | +39 ms ± 1 ms | Time sync |

The four shares add up to what all four corrections remove together. The 2.1 DN left is the scene's own (codec and
noise): a clean recording reads 1.8 DN.

The **Figures of the camera** tree gives more detail:

- histogram EMD of about 18 DN per channel, sharpness ratio 0.01 (the recording is far blurrier than the render);
- noise σ 0.11 DN recorded against 0.42 rendered (the blur removed the noise);
- silhouette IoU from 0.53 to 0.89 (background difference: the dataset stores no masks);
- latency 39.3 ms against the time-sync study's prediction for the stream;
- MMD² 0.142 (p 0.015) as planned and −0.084 with every correction (a kernel MMD on thumbnails, not FID).

The mount reads 6.7 mm against a true 3 mm, because the blur and the brightness weaken the silhouette's edges. On
the same recording without them it recovers 1.08° for the true 1° and is 2.0 mm from the true 3 mm (see the model
doc: a 3 mm translation is not resolved at 320 × 180).

## D. Close it

**Apply** makes the top suggestion (or the row you click) in one undo step. **Preview** says what it would change,
and **Open page** goes to the page that owns the setting. On this recording five passes of Run and Apply gave:

| pass | gap | Apply |
|---|---|---|
| 1 | 24.4 DN | exposure: the render's camera response for the overhead D435 (gain and offset) |
| 2 | 10.0 DN | mount: the recovered mount written to the sensor |
| 3 | 7.6 DN | latency: −39 ms (stamp minus exposure) entered for the camera in the time-sync settings |
| 4 | 4.6 DN | blur: the render's blur σ for the camera |
| 5 | 2.1 DN, "close to real" | nothing left to suggest |

From Python the same loop is:

    res = rig.sim2real('out/s2r_gaps')
    while res['apply']:
        rig = rig.apply_sim2real(res['apply'])
        res = rig.sim2real('out/s2r_gaps')

On a real rig the photometric suggestions say two things. First, what the simulator's camera response must be for
its images to look like yours: export the result for your sim pipeline. Second, what to change on the camera instead:
"lower the exposure or gain", "check the focus". When every camera shows the same gain, the cause is ranked as
**Lighting**: the room's light differs from the project's. Add the room's lights on the Lighting page; with lights in
the project, Apply scales them.

## E. Batch and export

    dexvantage --sample ur5e --sim2real out/s2r.json --dataset out/s2r_gaps

This writes the result with its provenance and exits 0. A missing dataset is refused in one sentence (exit 2).
**Export…** on the page writes the same JSON. The named metrics are `sim2real.image_gap_dn` (the headline),
`sim2real.remaining_dn`, `sim2real.explained`, `sim2real.latency_ms`, `sim2real.mount_offset_mm` and
`sim2real.silhouette_iou`, for requirements, experiments and the API.
