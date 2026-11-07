# The Assistant dock: providers, the tool loop, the number check and the Copilot {#copilot}

Modules: `dexvantage/assistant/` — `agent.py` (the tool loop, the system prompt, the context note), `toolbox.py` (the
tools the dock offers and the session they see), `providers.py` (models, usage, prices), `claude.py` (Claude through
Anthropic's SDK), `client.py` (Groq over plain HTTPS, and `ApiError`), `fake.py` (the fake provider every test uses),
`numcheck.py` (the number post-check), `edits.py` (edit proposals), `copilot.py` (plans, approval records, the gate),
`chats.py`, `memory.py`, `store.py` (what is kept per project), `secrets.py` (keys, system keyring only),
`routing.py` (automatic model choice, off by default); the dock is `dexvantage/ui/assistant.py`. Checks:
`dexvantage/validation/checks_copilot.py` (model `copilot`). Tests: `tests/test_assistant_loop.py`,
`tests/test_assistant_dock.py`. Spec §12, part 2 (the tools and the MCP server are part 1: `docs/models/mcp.md`).

Ported from Vantage `assistant/agent.py`, `providers.py`, `claude.py`, `client.py`, `chats.py`, `memory.py`,
`copilot.py`, `routing.py`, `secrets.py` and `ui/assistant.py`. Stripped: the vehicle plan kinds (CARLA drives,
campaigns, detection), agent (goal) mode, pictures, slash commands, Vantage's environment-variable and private-file key
fallbacks. Added: the number post-check, edits from changing tools turned into proposals, Claude Haiku 5.5.

## The loop

A question is one *turn*. On the UI thread the dock reads the page on screen (`page_guide.current_page_state`: its
headline, status word and the study's figures), the selection, the memory notes and the plan on screen, and builds a
`ToolContext`; the turn then runs on a worker thread:

1. The question goes to the model with a context note (on the SO-100 sample: `[DexVantage context: project "SO-100
   sample"; variant "Wrist + third-person (sample)"; open page: Coverage and resolution (Shows 47.3 % of the task
   volume: gaps.)]`) and the memory notes (Claude: as a mid-conversation system message, sent only when they
   changed, so the cached prefix stays the same).
2. The model answers or calls tools. Each call's input is checked against the tool's schema first
   (`tools.validate_args` for the shared tools), then run by `toolbox.run`, and its JSON result goes back. At most
   12 rounds; the last is told to answer without tools.
3. After the answer, `numcheck.unsupported(answer, sources)` lists the figures not found in the sources (every tool
   result of the conversation, the context note, the page context, the user's words, the memory notes). The dock
   shows them under the answer.

A failed or stopped turn is rolled back from the end of the history, so the conversation is as before it. Errors are
`ApiError(kind, sentence)`: `no_key`, `auth`, `rate` (retried once when the provider says to wait 20 s or less),
`overloaded`, `network` ("you may be offline"), `refusal`, `truncated`, `rounds`, ...; the dock shows the sentence,
never a traceback.

## Three rules the loop enforces

**No invented numbers.** The system prompt says to take every number from the tools and to say "not stated" for a
figure the maker does not state; the post-check catches what the prompt does not. A number in the answer is
*supported* when some source value v gives |a − k·v| ≤ ½·10^(−d), with d the decimals the answer writes and
k ∈ {1, 100, 1/100, 1000, 1/1000, 10⁶, 10⁻⁶, 180/π, π/180} (as written, as a percentage, in milli or kilo units, in
degrees from radians), sign ignored. Not checked: integers 0 to 10 written without a decimal point and without a unit, % or ° after them (counts,
steps; `NUMBER_FLOOR`, a default; "5 mm" or "8 ms" is checked), numbers joined to a word (UR5e, SO-100, claude-opus-5-5, §7.3, 3D), versions, ISO dates,
clock times, inline code and URLs, numbered list markers.

**No direct edits.** The tools see a `Proposing` session: its `edit(keys, fn, text)` runs `fn` on a copy and
records the difference as an `edits.Proposal` (the section keys and a before/after row per quantity); `undo` and
`redo` refuse. `propose_changes` (factor paths of `core/factors.py` and the structural ops of `mcp/edits.py`) checks
the change on a copy (a locked sensor or a bad value is a sentence back to the model) and applies it again to the
project as it is when the user clicks Accept. A proposal from a changing tool (`edit_project`) carries its
prediction from `simulate_changes` and is refused at Accept if the project changed since (it would overwrite).
Accept is `ProjectStore.edit(keys, …, 'Assistant · <title>')`: one undo step. Discard writes a `rejected` memory note.

**No long study without an approved plan.** The session has `allow_long=False` (the shared tools refuse the studies
in `tools.LONG`, measured at tens of seconds on the samples), `toolbox.run` refuses `run_study` of such a study
before the tool sees it and refuses `export_files` (exports run from a plan). `propose_plan` builds a
`copilot.Plan`: steps of kind `study` (a study id and the settings for that run), `export` (format ids) or `edit`
(changes and ops); each value records its source (your request, chosen by the Copilot, the page's default); each step
is checked (the study exists, it has its robot, the formats exist, the edit applies to a copy) and carries an
estimate only when this session has measured one (the last run's seconds, labelled as a measurement with other
settings, never a prediction).

Approve re-checks the plan and writes an approval record into `<data>/assistant/<project>/copilot.json`: the steps,
their settings, `settings_sha256` per step and for the plan (SHA-256 of canonical JSON: sorted keys, no spaces), the
request, provider and model, the UTC time. The runner then runs each step only after `gate(records, ticket, kind,
settings)` passes: an approved record of that plan version, `verify` true (every hash recomputes), a step of that kind
with exactly those settings. A study step runs through the window's analysis controller with its settings for that
run only; an export into `<working folder>/<project>/exports/plan-<id>-v<n>`; an edit as one undo step
"Copilot · …". Edit on the card makes a new version (the diff against the previous one is shown; it needs approval
again); an approved plan is never changed.

## Providers

| | Claude (`claude.py`) | Groq (`client.py`) | Fake (`fake.py`) |
|---|---|---|---|
| package | `anthropic==2.6.0` (extra `assistant`) | none (stdlib HTTPS) | none |
| models | `claude-opus-5-5` (default), `claude-sonnet-5-5`, `claude-haiku-5-5` | `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, `llama-3.3-70b-versatile` | `fake-model` |
| request | `beta.messages.stream`, adaptive thinking, `output_config.effort` medium (high on a plan turn), `tool_choice` auto, every tool `eager_input_streaming`, the system prompt cached, compaction `compact-2026-01-12`, `fallbacks: "default"` on Opus and Sonnet | chat completions, streamed | scripted replies and calls |
| key | the system keyring (`DexVantage` / `anthropic-api-key`) | the system keyring (`groq-api-key`) | — |

The Claude request shape follows Anthropic's documentation for these models as the Claude API reference states it
(read 2026-11-07): `budget_tokens`, `temperature`, `top_p`, `top_k` are never sent; forced tool choice is not used;
Haiku 5.5 has no server-side refusal fallback. Prices (`providers.PRICES`, USD per million tokens, the same source):
Opus 5.5 $4 in / $20 out / $0.20 cache read, Sonnet 5.5 $2 / $10 / $0.20, Haiku 5.5 $0.10 / $0.50 (up to 100K-token
prompts); cache writes, and Haiku's cache prices, are not stated there and are `None`, so a conversation that used
them shows tokens without a cost. Groq's prices are not bundled.

Keys are kept only in the system keyring (`secrets.py`, the optional `keyring` package): no environment variable,
file or setting. Without a keyring the settings dialog says why (no package, no Secret Service, or
`DEXVANTAGE_NO_KEYRING` set as the tests set it). The key is passed to the SDK explicitly, so the SDK never looks for
one of its own. What leaves the machine is masked (`agent.mask`: the home folder as `~`, other absolute paths cut to
their last part, anything shaped like an API key removed).

## Kept per project

Under `XDG_DATA_HOME/DexVantage/assistant/<project key>/` (else `~/.local/share/...`), files 0600, folders 0700: the
chats (`chats/<id>.json`, the transcript as the dock showed it, masked), the memory notes (`memory.json`: decision,
constraint, rejected, note; the user adds and deletes them in the Memory view; the assistant only offers one with
`remember`) and the Copilot records (`copilot.json`). The project key is the file's stem and a hash of its path (an
unsaved project: of its name and creation time).

## Validation (measured 2026-11-07, quick and full runs alike)

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| `copilot.number_check` | 25 answers labelled by hand (7 flags) | 0 mismatches (3 ms) | 0: the rule is deterministic |
| `copilot.plan_gate` | the decision each of 12 records must get; SHA-256 by `hashlib` (FIPS 180-4) | 0 wrong, 0 hash mismatches (82 ms) | 0 |
| `copilot.proposal_roundtrip` | `core.factors.apply`; `Project.to_dict` | 0 sections differ after applying, 0 fields after restoring (75 ms) | 0 |

The checks found one bug while being written: a number that opened an answer was never checked (an empty
"character before" counted as a name character); fixed, and the corpus keeps the case. The self-test line `assistant`
runs one scripted turn on the SO-100 sample (three tools, a proposal, a refused long study, one invented figure
flagged) in about 0.2 s.

## Known limits

- The post-check matches values, not meanings: a right number cited for the wrong quantity passes, and a number the
  model derived correctly (a sum, a difference) is flagged. It reads digits only ("twelve" is not read). It says
  "check these figures", not "these are wrong".
- The gate proves what was approved, not who approved it: anyone who can write the user's data folder can write a
  record. `verify` catches a record changed after approval, not a forged one.
- A proposal from a changing tool is refused at Accept when anything in the project changed since it was made, even
  an unrelated section; `propose_changes` proposals are applied again to the current project instead.
- Plan steps carry no time estimate until the study has run in the session; there is no model of a study's run time.
- The Claude request shape (betas, fallbacks, compaction) follows the documentation; no request was sent to the API
  while building this (the tests use the fake provider only), so a change on Anthropic's side shows as a plain error.
- Groq's model list is the one Vantage took from Groq's documentation (October 2026); none of them takes images.
