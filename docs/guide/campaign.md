# Planning a data-collection campaign: a worked example on the ALOHA 2 sample

This plans the demonstrations for the ALOHA 2 sample's handover under two lighting presets, three operators, enough
demonstrations per preset to tell a 50 % success rate from 70 %, and a budget of 4 TB, ten days and an 8-hour upload
window on a 1 Gbit/s link. The model is in `docs/models/campaign.md`. Figures below are from a run on 2026-11-07.

## 1. The spec

Data ▸ Campaign shows the settings as fields (each change one undo step, "(default)" while a figure is DexVantage's).
The levels of the conditions (lighting presets, object sets, variants, teleop devices, tasks) live in a spec file,
read with **Open spec…** or given to batch mode:

```json
{"schema": "dexvantage.campaign_spec/1", "name": "ALOHA handover, two lighting presets",
 "operators": ["Ana", "Ben", "Chen"],
 "lighting": [{"name": "Bright"}, {"name": "Dim"}],
 "demos": {"mode": "power", "p_base": 0.5, "p_target": 0.7},
 "episode_s": 30, "start_date": "2026-11-16",
 "budget": {"disk_gb": 4000, "days": 10, "upload_mbps": 1000, "upload_hours_per_day": 8}}
```

The sample's two trajectories (left and right arm over the same 4 s) are one bimanual task, "Handover"; its 4 s is the
scripted motion, so `episode_s` sets 30 s for a person teleoperating. The sample has no lights in the project, so the
presets are names only and the checklist says "room light as it is"; add the lights (Workspace ▸ Lights) and name
them in each preset, `{"name": "Dim", "lights": {"Key panel": 0.3}}`, to have the checklist say which to switch.

## 2. Plan

    dexvantage --sample aloha --campaign spec.json

prints `Campaign: 216 episodes over 2 days, 1,304.0 GB (within budget)` and writes `spec.plan.json` (with
provenance), `spec.plan.csv`, `spec.plan.ics` and `spec.plan.html`. On the page the headline reads **216** "episodes
over 2 days, 1,304.0 GB", status **within budget**: storage 33 % of the disk, days 20 %, the busiest day's upload 30 %
of the window (2.4 h).

- **Demonstrations.** The power analysis (two-sided, α 0.05, 80 % power, with the continuity correction: defaults)
  asks for 103 per condition. Three operators recording each condition in two equal blocks makes that 108 (the note
  says so): 36 per operator per condition, blocks of 18.
- **Design.** "Williams balanced Latin square of order 2 (2 rows): 6 sequences (3 operators × 2 rounds) … at every
  position equally often; and after every other condition equally often (carry-over balanced)." With seed 0, Ana
  records Dim then Bright, then Bright then Dim; Ben Dim, Bright twice; Chen Bright, Dim twice: three sequences start
  with each preset.
- **Storage.** The bandwidth study's rates of the four D405 streams, 1 563 Mbit/s raw, plus its 3 % container overhead
  (a default): 6.04 GB per 30 s episode. Recording to a LeRobot dataset with a ratio you measured on one episode
  (`"format": "lerobot", "compression_ratio": {"value": 0.05, "source": "episode 0 of the pilot"}`) scales it; without
  a ratio the page says the raw size is used.
- **Sessions.** Five sessions in 4.5 h on the rig: day 1 (2026-11-16) Ana 09:10–10:37 (72 episodes; her sequences
  repeat Bright at the round boundary, so one changeover fewer lets them fit in the 90-minute limit), Ben, Chen, Ben;
  day 2 Chen 09:10–09:56. The 10-minute daily calibration check moves the first session to 09:10.

## 3. The checklist and the calendar

Click a session for its checklist. The first one starts with the calibration check ("capture 5 poses of the hand-eye
plan … with DexVantage's default board (ChArUco 9 × 7, 30 mm squares, a default) on cam_left_wrist … within 0.5 mm and
0.05° (defaults)"), then the rig variant, the lighting preset of each condition, the teleop devices (both leader
arms), each camera's exposure ("not set: run Analysis ▸ Motion blur and Apply its exposure" until you do; after the
Motion blur page's Apply the line reads the exposure and where it came from), the disk to free (434.68 GB) and the
blocks. `spec.plan.ics` opens in any calendar program: one event per session, the checklist in its description, at
local time. `spec.plan.html` prints a day per page with tick boxes.

## 4. Change and apply

Lower the disk budget to 1 000 GB: the headline turns **over budget** and the advice reads "Storage 1,304.0 GB
against 1,000 GB: record to a compressed format (enter its measured ratio), lower a frame rate on the Bandwidth page,
or add disk." Put it back, then **Apply**: the plan goes into the project (one undo step; the notes say "The project
holds an applied plan (this one)"). **Save spec…** writes the spec for `--campaign` in CI.

From Python:

```python
import dexvantage as dv
rig = dv.sample('aloha')
res = dv.campaign.plan(rig, 'spec.json')
res.headline['text']          # 'Campaign episodes recorded: 216 (within budget)'
dv.campaign.export(res, 'out/plan')       # out/plan.csv, .ics, .html
dv.campaign.apply(rig, res)               # the plan into the project
```
