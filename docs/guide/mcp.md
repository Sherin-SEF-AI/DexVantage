# Claude over MCP: a worked example on the SO-100 sample

DexVantage serves its studies to Claude (Claude Code, Claude Desktop or any MCP client) through an MCP server. Claude
reads the project, runs the studies, tries changes and makes them as undo steps you can take back. Every figure it
quotes comes from a study or a cited datasheet.

## Set it up

Save the sample under a name of your own first (File ▸ Open sample ▸ SO-100, then File ▸ Save as… `so100.dexv`).
Register the server with Claude Code once:

    claude mcp add dexvantage -- python -m dexvantage.mcp --project /path/to/so100.dexv

With the DexVantage window open (the setting `mcp/bridge`, on by default, lets Claude in), Claude works on the project
in the window: each change appears at once as an undo step "Claude · …" in Edit ▸ Undo. With no window, it works on
the file: each change is saved, and the file as it was first opened is kept as `so100.dexv.bak`.

## A session

Ask Claude "how well do the cameras see the task volume?". It calls `get_overview`, then `get_coverage`, which
answers (figures measured on the sample on 2026-11-07):

| | |
|---|---|
| status | serious, "gaps" |
| seen by two or more cameras | 47.302 % |
| seen by at least one | 94.521 % |
| blind | 5.479 % |
| best resolution, median / 5th percentile | 3.6055 / 1.0441 px/mm |

and per camera: the wrist camera sees 53.7 % of the volume, with 35.4 % outside its image; the third-person camera
88.1 %, 40.8 % of the volume by it alone. The study's advice: aim the wrist camera at the volume, move it back or
choose a wider lens.

Ask "would tilting the wrist camera 10° up help?". Claude tries it with `simulate_changes`
(`{"sensor:Wrist camera (sample placement).dpitch": -10}` with the coverage metrics) and nothing changes:

| metric | before | after |
|---|---|---|
| `coverage.covered` (task volume seen) | 94.5208 % | 92.3715 % |
| `coverage.blind` | 5.4792 % | 7.6285 % |
| `coverage.ppmm_p5` | 1.0441 px/mm | 1.0185 px/mm |

It is worse, so Claude says so instead of making it. Had it called `edit_project` with the same change, the reply
would carry the same before and after, the undo step "Claude · Wrist camera (sample placement) turn about its y: by
-10 °", and `undo_last_change` would take it back (only Claude's own last step: a step you made after it is never
undone from here).

Ask "is the teleop leader good enough?": `get_teleop_overlap` answers 96.67 % of the task volume teleoperable without
a dead zone ("clear"). This is long work (about 7 s on the sample); in the app's own assistant it runs from a plan you
approve (`propose_plan`).

Ask about the policy's loop: `get_action_chunk` answers with the study's own sentence, because SmolVLA's inference
time is not stated by its source: enter a measured figure on the Policies page. Claude cannot fill it in.

## What Claude cannot do

- Change a locked sensor or robot (the lock on the Sensors page). Any change to one is refused, whatever route it takes.
- Invent a figure: a figure no study or datasheet states is `None` ("not stated").
- Run the long studies inside the app's assistant without a plan you approve.

The tools and their inputs are listed by `tools/list`; `dexvantage_status` says where Claude is working.
`docs/models/mcp.md` has the protocol, the sessions, the edits and the checks.
