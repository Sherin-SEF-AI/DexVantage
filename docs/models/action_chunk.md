# Action-chunk latency and temporal alignment (spec §7.13)

`dexvantage/core/studies/action_chunk_sync.py` · study `action_chunk_sync` · batch `--action-chunk OUT.json` · page
**Analysis ▸ Action-chunk latency** · checks `chunk.loop_vs_simulation`, `chunk.ensemble_closed_form`
(`dexvantage/validation/checks_chunk.py`).

The question: an action-chunked policy (ACT, Diffusion Policy, π0 flow matching) runs on the rig with a
sensor-to-action loop that its training data never saw. How late or early does it close the gripper on the contact
point, how far is the hand from the contact point when the fingers close, and what proprioceptive delay offset and
ensemble weight should go into the training config?

## The loop

Per policy and per camera it takes. Every term carries a value, an origin (library entry, your figure, policy entry,
bandwidth study, default, not stated) and its source.

| Term | Model | Origin |
|---|---|---|
| τ_e/2 | half the exposure: an image shows the scene at mid-exposure | library imager `exposure_s`/`exposure_ms`, or your figure; **no library entry states it today**, so it is "not stated" and left out |
| τ_r | readout (the last row after the exposure) | as above |
| τ_x | one frame of the policy's stream: w·h·bpp·(1 + overhead) / bandwidth left to it | the bandwidth study (§7.6): the lower of the link's capacity less the sensor's other streams and the controller's capacity less everything else; not stated without a link or a stated bits per pixel; refused when the controller is over budget |
| τ_q | the newest frame's age when inference starts: uniform 0 … 1/fps (+ `queue_extra_ms`, default 0) | default model (camera and policy on independent clocks) |
| τ_i | one inference | the policy entry (`core/policies.resolve`): cited or measured. **Not stated → the policy is refused** with a sentence that says where to enter a measurement |
| τ_w | the chunk waits for the next control tick: uniform 0 … Δt, Δt = 1/control_hz | default model |
| τ_c | command latency | `command_ms` with its source, else one control period (a default) |

A missing term makes the loop "at least" the sum of the others; the page and the notes list which figure is missing.

## The close-timing error

The policy is modelled as an **ideal imitator**: it closes on the frame that, in the training data, came with the
close command. While recording, the close command stamped t_c − c_rec closed the gripper at contact t_c (c_rec: the
follower's command latency then; default the same τ_c); the training pairs the image stamped t with the action
stamped t + δ (δ: the proprioceptive delay offset; default 0, as LeRobot's `delta_timestamps` start at 0). An image
stamped on arrival at the host (`host_receive`, the default) carries the age a = τ_e/2 + τ_r + τ_x; one stamped at
mid-exposure (`mid_exposure`; the default when the time-sync settings put the stream on a camera hardware clock)
carries none. At deployment the close reaches the gripper

    e = (τ_e/2 + τ_r + τ_x − a) + τ_q + τ_i + τ_w + ρ + (τ_c − c_rec) − δ − k·Δt

after contact, with ρ uniform over ±Δt/2 (the close lands on a whole control step) and k the actions the executor
skips at the start of a chunk (`skip_steps`, default 0). The interval [low–high] is the sum of the ranges, the nominal
value the sum of the means, and the spread σ = √(Σ wᵢ²/12) of the independent uniform phases. Note that with host
stamps the image's own age cancels: what is left is queueing, inference and the step rounding.

**Spatially** the arm follows the demonstrated motion, so the error is the distance from the contact point when the
fingers close. Along a trajectory that closes the gripper at t_c (tool centre P(t)):

    overshoot = |P(t_c + e_high) − P(t_c)| (e_high > 0),   air-grasp = |P(t_c + e_low) − P(t_c)| (e_low < 0)

over every closing stroke of every trajectory (`core/grasps.py` `closure_events`), the worst kept. A hand that stops
before it closes loses nothing to timing. Without such a trajectory (or with your `approach_speed_m_s`): v·max(0,
e_high) and v·max(0, −e_low) at a constant speed v (default 0.10 m/s, **a default, not a measurement**).

## Chunks and temporal ensembling

They change how old the information behind an executed action is (the reaction age), not the timing above. A chunk
comes every s = max(E, ⌈τ_i/Δt⌉) control steps (E: actions executed per chunk). Open loop (Diffusion Policy's receding
horizon, π0), action k of a chunk rests on an image L + k·Δt old. With temporal ensembling (ACT, E = 1) the executed
action is Σ wᵢ aᵢ / Σ wᵢ with **wᵢ = exp(−m·i), i = 0 the oldest prediction**: the official ACT code
(`imitate_episodes.py`, commit 742c753, read 2026-11-07) stores chunk t in row t of `all_time_actions`, takes the
populated rows of column t in issue order and weights them `np.exp(-k * np.arange(n))`, k = 0.01; LeRobot's
`ACTTemporalEnsembler` (commit d490e47, read 2026-11-07) says "w₀ is the oldest action", a positive coefficient gives
more weight to older actions and a negative one to newer ones. The mean age is

    L + Δt · Σ wᵢ (n − 1 − i) / Σ wᵢ  =  L + Δt · [(n − 1) − S₁/S₀],  r = e^(−m),
    S₀ = (1 − rⁿ)/(1 − r),  S₁ = r (1 − n rⁿ⁻¹ + (n − 1) rⁿ)/(1 − r)²

(n = H for s = 1; for s > 1 averaged over the step's place in the cycle). ACT's m = 0.01 with H = 100 gives 57.7 steps,
1.15 s at 50 Hz. The **reaction distance** (along the trajectory |P(t_c) − P(t_c − age)|, else v × age) is how far the
hand travels before an action that has seen a change at the contact lands.

## Recommendations (Apply)

- **Delay offset**: δ that centres the timing interval of the policy's cameras on contact, rounded to whole control
  steps (the dataset is taken to be recorded at the control rate), with the worst error before and after.
- **Ensemble weight** (ensembling policies only): on m ∈ [−1, 1] in steps of 0.005 (defaults), the m with the most
  averaging, n_eff = (Σw)²/Σw², whose reaction distance stays within `reaction_tolerance_mm` (default 10 mm); when no m
  does, the most averaging within 10 % of the shortest reachable reaction.

Apply stores both on every policy of the result as `Policy.recommendations` (one undo step); the study reads them back
as the policy's configured offset and m, so the headline after Apply shows the predicted change. The result and the
page's export carry them; a LeRobot training-config exporter (§9.2, Phase 10) is not built yet, so enter them in your
training config by hand.

Status: the worst policy's larger of overshoot and air-grasp against `tolerance_mm` (default 5 mm): good up to it,
warning up to 2×, serious up to 4×, critical beyond (defaults, not standards).

## Validation (measured 2026-11-07)

| Check | Reference | Measured | Tolerance |
|---|---|---|---|
| `chunk.loop_vs_simulation` | event-by-event simulation (frames on their own clock, the newest frame ready at inference start, its stamp, the imitator's chunk index rounded to whole steps, the next control tick, skipped actions, command latency), 20 000 trials in each of 4 scenarios | worst mean difference 1.4 × 10⁻⁵ m at 0.1 m/s (0.14 ms), no simulated close outside the model's interval, spread ratio 0.992–1.000 | 5 × 10⁻⁵ m (about five standard errors), spread within 3 % |
| `chunk.ensemble_closed_form` | the geometric-series closed form above; Σ wᵢ aᵢ / Σ wᵢ from the formula | 3.3 × 10⁻¹³ steps over m = −0.1 … 1.5, H = 1 … 100 | 10⁻⁹ steps |

The tests (`tests/test_action_chunk.py`) also replay ACT's `all_time_actions` table as the official code builds it
(including its "populated" row test) against `ActEnsembler`.

## On the samples

- **UR5e** (π0 sample policy): the loop is 120 ms [93–146] without the D405's and D435's exposure and readout (not
  stated); transfer 9.5 ms (D405, 1280 × 720 YUY2) and 15.8 ms (D435, 1920 × 1080) once the cameras are plugged into a
  USB 3.2 Gen 1 controller on the Bandwidth page. The sample's scripted pick stops the hand before the gripper closes,
  so the close (+63 to +136 ms late) costs 0 mm: status good. A 100 ms (5-step) offset centres it anyway.
- **SO-100** (SmolVLA sample policy): refused, "its inference time is not stated by its source".
- **dexhand**: no policy, refused with what to add.
- Run time 0.02–0.05 s per run (no rendering); batch 0.4–0.6 s with start-up.

## Known limits

The policy is an ideal imitator, so its learned errors (and the smoothing ensembling buys against them) are not
modelled, only timing; arm and gripper commands reach the hardware with the same latency and the arm follows the
demonstrated motion exactly (servo lag and the gripper's own closing time are not added: put them in `command_ms`);
clocks are independent with uniform phases (a hardware-triggered rig whose policy loop is locked to the camera has a
narrower interval); with ensembling the close is a weighted mix of the chunks' close steps, which lies inside the
interval given for one chunk; the time-sync study's clock offsets and jitter are not added; transfer assumes the
bandwidth left to a stream is used at its full rate; the trajectory is held at its ends; grasp centres in the map use
one speed; a synchronous loop that pauses the arm during inference is not modelled; no library entry states exposure
or readout today, so those terms need your measured figures.
