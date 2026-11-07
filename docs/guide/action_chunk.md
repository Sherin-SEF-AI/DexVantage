# Action-chunk latency: a worked example

The Action-chunk latency page (**Analysis ▸ Action-chunk latency**) says how late or early each of your
action-chunked policies closes the gripper on the contact point, and what delay offset and ensemble weight to write
into the training config. The model is in `docs/models/action_chunk.md`.

## 1. Start from the UR5e sample

1. **File ▸ Open sample ▸ UR5e**. It carries the π0 policy (sample choice) and a scripted pick.
2. **Tasks ▸ Policies**: add **ACT** and **Diffusion Policy (CNN, real-world setting)**. Their horizons, executed
   steps, ensemble weights, inference times and control rates come from the library entries (each figure cited; the
   ACT entry's m = 0.01 from the official code).
3. **Analysis ▸ Bandwidth**: add a host with one USB 3 controller and plug both cameras into it. The transfer time of
   each camera's frame then comes from that budget (9.5 ms for the D405's 1280 × 720 YUY2 frame, 15.8 ms for the
   D435's 1920 × 1080).
4. For this example, remove the scripted pick (**Tasks**), so the study uses the default approach speed of 0.10 m/s
   (labelled a default on the page; with the pick in place the hand is still when it closes, and the error at contact
   is 0 mm).

## 2. Run it

Open **Analysis ▸ Action-chunk latency** and press **Run** (the study is quick and also re-runs after edits).

- Headline: **28.3 mm, critical, "closes late"**: Diffusion Policy closes 50 to 283 ms after contact (nominal 167 ms):
  100 ms inference, a 10 Hz control step to land on (±50 ms) and the tick wait. At 0.10 m/s that is 28.3 mm past the
  contact point.
- ACT: 0 to 73 ms late, 7.3 mm. π0: 63 to 136 ms, 13.6 mm.
- The loop bars show each camera's terms. Exposure and readout are listed as **not stated** (no maker in the library
  states them); the loop says "at least". Enter measured figures in the study settings (`figures`) to complete it;
  with the default host timestamps they do not change the timing error, only the reaction.
- For ACT the chart shows the reaction distance against m. With m = 0.01 (ACT's code; i = 0 is the oldest prediction,
  so a positive m weights old predictions more) the executed action rests on predictions 57.7 steps old on average:
  the hand travels 122.6 mm before a change at the contact is acted on.

## 3. Apply

The recommendations:

- Diffusion Policy: delay offset **200 ms** (2 control steps): worst error 28.3 → 15.0 mm (the 10 Hz step limits it).
- ACT: delay offset **40 ms** (2 steps): 7.3 → 4.0 mm; ensemble weight **m = −0.55**: reaction 122.6 → 10.0 mm, the
  ensemble then averages 3.7 predictions in effect instead of 92.4.
- π0: delay offset **100 ms** (5 steps): 13.6 → 3.7 mm.

**Preview** draws the overshoot map as it would be after the offset. **Apply** stores the offsets and m on the three
policies (one undo step; **Edit ▸ Undo** takes it back). The study runs again and reads them as the policies'
configured values: the headline drops to 15.0 mm (Diffusion Policy, now "closes early" by up to 15 mm, "serious").
**Export…** writes the result with provenance.

In batch:

    dexvantage --sample ur5e --action-chunk chunk.json

A policy whose inference time or control rate no source states is left out with a sentence (the SO-100 sample's
SmolVLA: "its inference time is not stated by its source"); with no usable policy the run exits 2 with that sentence.
