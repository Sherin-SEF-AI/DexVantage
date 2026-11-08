# The assistant's tools and the MCP server {#assistant}

Modules: `dexvantage/assistant/tools.py` (the tool registry), `assistant/figures.py` (the figures each study yields,
as the tools return them), `assistant/session.py` (where the tools work: a project file or the window),
`assistant/page_guide.py` (the page on screen), `dexvantage/mcp/` (`server.py` the stdio server, `service.py` the
tools as MCP lists them, `edits.py` changes as data, `session.py` finding an open window, `batchrun.py` batch jobs),
`dexvantage/ui/mcp_bridge.py` (the window's side). Checks: `dexvantage/validation/checks_assistant.py` (model
`assistant`). Tests: `tests/test_assistant_tools.py`, `tests/test_mcp.py`. Spec §12 (part 1: the tool layer and the
MCP server; the Assistant dock, providers and the Copilot's approval are part 2).

Ported from Vantage `assistant/tools.py` (the registry, the schema check, errors as plain sentences, read tools over
a snapshot, simulate and propose), `assistant/copilot_tools.py` (a plan the user approves), `assistant/page_guide.py`
(the page's purpose, needs and state) and `mcp/` (`server.py`, `service.py`, `session.py`, `edits.py`,
`batchrun.py`) with `ui/mcp_bridge.py`. The vehicle tools (cameras on a car body, seams, CARLA runs) were stripped;
the tools are DexVantage's studies, and changes are DexVantage's factor paths (`core/factors.py`).

## What is computed here: nothing

The tool layer is plumbing. Every figure a tool returns is one of:

- a figure of a study's result document (`core/jobs.run_study`, through the session's result cache, or the result the
  window already holds for the design as it is now: `AnalysisController.results` with the store's revision), read off
  by `figures.brief(study, doc)` and rounded for reading (4 decimals at 1 and above, 4 significant digits below 1);
- a figure of the project (mount poses, joint limits as the model file states them);
- a figure of a library entry, with the entry's cited sources (URL, title, date read);
- a metric (`core/metrics.py`), which is itself a figure of a study's result.

A figure a study does not state stays `None`. A study that refuses (`StudyError`) gives its own sentence as the tool's
`error`. `assistant.figures_from_studies` checks the first point.

## Sessions

| | `FileSession` (headless) | `StoreSession` (the window) |
|---|---|---|
| project | a `.dexv` file (or a sample, unsaved) | the window's `ProjectStore` |
| an edit | an undo step held by the session; the file saved after each one; the file as first opened copied to `<name>.dexv.bak` once | one `QUndoCommand` "Claude · …" in Edit ▸ Undo, like the user's own |
| undo | the session's own steps | only the top step, and only when it is the assistant's ("Claude ·"); a user's step is left alone |
| a change on disk | refused (open it again) | — |
| studies | run in the session (its own cache) | the window's cache; the window's result when it is current |
| long studies | allowed (`allow_long=True`) | per session: the MCP bridge allows them (the MCP client asks the user before each call); the in-app assistant (part 2) sets `allow_long=False` and proposes a plan |
| `show_in_app`, `get_page_context` | refused, plainly | on the UI thread |

A `StoreSession` is safe to call from a worker thread: whatever touches the store or the window runs on the UI thread
through a queued signal, and the worker waits (at most 120 s).

Long studies (`tools.LONG`): `grasp_visibility` (11.5 s on the SO-100 sample, 42.4 s on the UR5e, measured
2026-11-07), `groundtruth`, `audit`, `plan_vs_recording`, `silhouette_calib`, `teleop`, `handeye_plan`,
`handeye_solve`, `dex_occlusion`, and the studies that run other studies many times: `tolerance` (64 draws, 81 s on
the UR5e), `experiments` (one run per row), `optimise` (105 s on the UR5e at its default budget) and `design_agent`
(237 s) (added by the Phase 9/10 product review), and from research tools part 2 `cells` (figures on every cell) and
`sim2real` (renders and a silhouette calibration).

Research tools part 2 added nine reading tools (33 in the registry, 39 in the MCP server's `tools/list` with its own
`dexvantage_status`, `open_project`, `save_project`, `run_batch`, `get_history` and `redo_change`; the counts are
asserted in `tests/test_integration_rt2.py`): `get_runs` (the experiment tracker's runs), `get_statistics` (success
counts and the power calculator computed on the spot, else the page's result), `get_campaign`, `get_sim2real`,
`get_methods_paragraph` (the paper kit's paragraph), `get_cells`, `get_cluster_job` (a Slurm job folder's shards and
gathered headline), `get_rig_cost` (the cited prices) and `check_robot_file` (what a URDF, MJCF or USD file references
and whether each file is found, before an import). Each reads; none changes the project.

Phase 11 added `get_project_diff` (34 in the registry, 40 in `tools/list`): what changed, item by item
(`core/projdiff.py`, as File ▸ Compare with…), from another project file (`path`, read beside this project when
relative) to this project, or from the active variant to another variant (`variant`, its name or uid; both sides are
diffed as one-variant projects under the active variant's uid, so sensors pair up); with `metrics`, each figure on
both sides and its delta, with `delta_unit` (a change of a percentage is in percentage points). The metrics' studies
follow the long-study rule above. It reads only.

## Edits as data (`mcp/edits.py`)

`edit_project` and `simulate_changes` take

- `changes`: `{factor path: value}`, any settable quantity of `core/factors.py`: `sensor:<uid or name>.x|y|z` (m, in
  the parent frame), `.roll|.pitch|.yaw` (degrees, URDF rpy), `.dx|.dy|.dz` (a shift along the parent's axes),
  `.droll|.dpitch|.dyaw` (a turn about the sensor's own axes), `.model` (a library id; the sensor's kind follows the
  entry's), `.parent` (`world`, `cage`, `tripod`, `table` or a link of a robot of the project), `.enabled`,
  `.stream` (`[width, height]`), `setting:<study>.<key>`, `json:/pointer` (not into the log, the format or the
  stored files);
- `ops`: `add_sensor`, `remove_sensor`, `add_variant` (a copy of the active rig by default, keeping its sensors' uids so
  cables, policies and calibration follow), `activate_variant`, `rename_variant`, applied first.

Each value is checked before anything changes; the whole change is made on a copy; the sections it touches are found
by comparing the copy with the project (`changed_keys`), and the undo step restores exactly those. Locked items:
every locked sensor and robot of the project before is compared with the copy after, field for field, and any
difference (moved, re-modelled, removed, unlocked) refuses the whole change, whichever path made it (a factor, an
op or a JSON pointer). Locking is allowed.

`edit_project` returns the lines changed, the undo step's label, and the metrics of the project's requirements (plus
any `metrics` named) before and after, with each requirement's pass before and after (`default_threshold` says when
the threshold is DexVantage's labelled default).

## Plans (`propose_plan`)

A plan is `{'schema': 'dexvantage.plan/1', 'plan': id, 'version': 1, 'title', 'rationale', 'project', 'steps',
'status': 'proposed', 'created_utc'}`; `core/provenance.plan_field` reads its id. A step is a `study` (with settings;
`long` says whether it is long work), an `export` (formats, folder) or an `edit` (changes and ops, with the lines it
would change). Each step is checked when proposed; nothing runs. In the window the plan goes to
`ctx.on_copilot_plan(plan)` when the Assistant dock sets that hook (part 2), else the status bar says a plan waits; a
file session keeps it.

## The MCP server

`python -m dexvantage.mcp [--project FILE.dexv | --sample so100] [--mode auto|live|file]`

The Model Context Protocol, version **2025-11-25** (https://modelcontextprotocol.io/specification/2025-11-25: the
pages basic/lifecycle, basic/transports and server/tools, read 2026-11-07), implemented with the standard library:

- stdio: JSON-RPC 2.0, one UTF-8 message per line with no embedded newline; nothing but protocol messages on stdout
  (the server points `sys.stdout` at stderr before it starts, so a stray print cannot corrupt the stream);
- `initialize`: the client's version when it is 2025-11-25, 2025-06-18, 2025-03-26 or 2024-11-05 (the messages used
  are the same in each), else 2025-11-25; capabilities `{"tools": {"listChanged": false}}`; `serverInfo`; the
  `instructions` text (`mcp/service.py` `INSTRUCTIONS`: what DexVantage is, units and frames, read before you change,
  quote figures and never estimate, locked items are the user's, long studies take time);
- notifications get no reply; `ping`; `tools/list` (name, title, description, `inputSchema`, annotations:
  `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint: false`); `tools/call` answers `content` (the
  whole JSON as text), `structuredContent` (the same object) and `isError`;
- an unknown tool is the protocol error −32602, as the 2025-11-25 tools page says; a tool that cannot do what was
  asked (bad input, a study that refuses, a locked sensor) is a tool execution error: `isError: true` and a plain
  sentence; −32700 for a line that is not JSON, −32601 for an unknown method, −32600 for a message that is not
  JSON-RPC 2.0.

Every input schema is a closed JSON Schema object (`additionalProperties: false`), checked against the 2020-12
meta-schema by `jsonschema` in the tests and the validation check. The server checks each call's input against the
schema itself (`tools._check`, ported from Vantage: type, enum, properties, required, additionalProperties, items,
minItems, maxItems) before the tool runs; `tests/test_mcp.py` shows the two agree on good and bad inputs.

MCP-only tools besides the assistant's: `dexvantage_status`, `redo_change`, `save_project`, `get_history`,
`run_batch` (a batch job on a copy of the project in a run folder of its own: exit code and meaning, output end, the
files and the small JSON ones) and `open_project` (a file or a sample, or back to the window).

### Attaching to the window

A window started from `dexvantage` (app.py) starts its bridge unless the setting `mcp/bridge` is off: it listens on
127.0.0.1 only, on a free port, and writes `<config>/DexVantage/mcp/<pid>.json` (mode 0600) with the port, a random
48-hex-digit token, the project's name and file. The server finds open windows there (dead processes' files are
removed): with `--project`, only a window that has that file open (else it works on the file); without, the newest.
It sends each call as one JSON line with the
token (compared in constant time). The bridge runs the tool on a worker thread through a `StoreSession`, so the
change is an undo step in the window and shows at once. `--mode file` never attaches; `--mode live` only attaches.

## Validation

| Check | Quantity | Reference | Tolerance | Measured (2026-11-07) |
|---|---|---|---|---|
| `assistant.figures_from_studies` | largest relative difference between a tool's figure and the same figure read off the study result here (14 figures of coverage, time sync, bandwidth and cables on the sample) | the result documents, read by an extraction written out again | 5 × 10⁻⁴ (the briefs' rounding: 4 significant digits) | 3.65 × 10⁻⁵ |
| `assistant.mcp_conformance` | nonconformities of the server's replies to the specification's own example requests, and of the 30 tool definitions (names, schemas against the 2020-12 meta-schema by `jsonschema`) | MCP 2025-11-25; JSON-RPC 2.0; jsonschema | 0 | 0 |
| `assistant.edit_undo_identity` | content-hash mismatches after edit then undo (two sensors moved and turned) and after a refused change to a locked sensor | `core/provenance.py` content hash | 0 | 0 |

Self-test line `mcp server`: initialize, tools/list and tools/call `get_overview` on a blank project in process
(about 70 ms).

## Known limits

- The briefs pick the figures a reader needs first; `full: true` (and `run_study`'s) returns the whole result,
  compacted (lists cut with their count, encoded arrays, paths, URLs and secrets left out).
- Studies run on the project as stored. The window's joint sliders and the pose a page runs at are view state, so a
  page run at another pose shows other figures than a tool run at home; `get_page_context` reads what the page shows.
- One request at a time per server; a long study blocks the next call. No progress or list-changed notifications, no
  task-augmented calls, no resources or prompts (their lists are empty). JSON-RPC batches (removed from MCP in
  2025-06-18) are still answered.
- A file session's undo history lasts as long as the session; the `.bak` keeps the file as first opened.
- `add_variant` copies keep the sensors' uids across variants; `json:/` pointers go through `Project.from_dict`, which
  normalises what it does not know.
- `run_batch` waits for the job (600 s by default, 3600 s at most); outputs given as absolute paths land where they
  say.
