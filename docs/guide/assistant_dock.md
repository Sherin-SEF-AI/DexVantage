# The Assistant dock

Window ▸ Assistant (Ctrl+J) opens the dock in its own column on the right. It answers questions about the open
project with Claude or Groq, reading DexVantage's own figures through tools; it never changes the project or starts a
long study without you.

## Setting it up

1. Install the optional packages once, in the source folder: `pip install ".[assistant]"` (anthropic 2.6.0 for
   Claude, keyring 25.7.0 for the keys). Groq needs no package.
2. Click the gear in the dock (Assistant settings), pick the provider and the model (Claude Opus 5.5 is the default;
   Claude Sonnet 5.5 and Claude Haiku 5.5 are faster and cheaper), paste the API key and click Save key. The key goes
   into the system keyring only; the dialog shows it masked (first and last four characters). Remove key takes it
   out again.
3. Until a key is stored the dock says so in one sentence at the top ("There is no Claude API key: add one in
   Assistant settings…"), and nothing is sent.

## A worked example on the SO-100 sample

File ▸ Open sample ▸ LeRobot SO-100 arm with a wrist and a third-person camera, then Analysis ▸ Coverage and resolution, and Run. The page's headline reads **47.3 % of
the task volume, gaps**. With the page open, ask the dock *What does this page tell me?*

- The question goes out with the page you are on: `open page: Coverage and resolution (Shows 47.3 % of the task
  volume: gaps.)`. The assistant calls `get_page_context` and cites the headline, its status word and the page's
  explanation (47 % of the volume seen by two or more cameras, 47 % by one only, 5 % by none; the wrist camera sees
  54 %).
- Under the answer, the tools it called are listed ("Tool: Read the page on screen", with its time).
- If the answer carries a number that is not in what the tools returned, a warning says so. In the tests' scripted
  version of this example the answer adds "about 99 % would need a third camera"; the dock shows *Not found in what
  the tools returned or the page shows: 99. … check them before you rely on them.*

Then ask *Raise the wrist camera by 2 cm*. The answer comes with a **proposed change** card: the quantity, its value
before and after (`Wrist camera (sample placement) z −0.0600 m → −0.0400 m`). Nothing has changed yet.

- **Accept** makes it one undo step, "Assistant · Raise the wrist camera" in Edit ▸ Undo; the card then offers
  **Undo**.
- **Discard** leaves the project as it is and writes a memory note ("rejected: …"), so the assistant does not
  propose it again.

Ask *Run the grasp study on every grasp*. Grasp self-occlusion is long work (tens of seconds on the samples), so the
assistant cannot run it from a tool: it puts a **Copilot plan** on screen, each step with its settings and where each
value came from, the checks (an unknown study, a missing robot or export format are errors; the plan cannot be
approved until they are fixed) and, when the study has run in this session, how long it took then.

- **Approve** stores an approval record (the settings and their SHA-256) for the project and runs the steps in
  order; each step shows its progress and its outcome on the card, and the study's page shows the result.
- **Edit…** changes one value (a step's setting, an export's formats): a new version appears with the change listed,
  and it needs approving again.
- **Cancel** drops the plan (or stops it while it runs).

## Chats and memory

- Each answer saves the chat for this project. The clock button lists this project's past chats; opening one shows
  it, and the next question continues with a short summary of it.
- The list button opens the **memory notes**: decisions, constraints, rejected ideas and notes the assistant is told
  with each question. Add one yourself, or keep one the assistant offers (Keep / Discard on its card); delete what no
  longer holds.
- Chats, memory notes and plan records live in your data folder (`XDG_DATA_HOME/DexVantage/assistant/`), per
  project, readable by you only.

## When something goes wrong

The dock says what happened and what to do, never a traceback: no key, the anthropic package missing, the key not
accepted, offline ("Could not reach Anthropic's API: you may be offline…"), rate-limited (tried again once when the
wait is short), the model declined the request, the answer cut off. A failed answer leaves the conversation as it was
before the question.

See `docs/models/assistant_dock.md` for how the loop, the number check and the approval gate work, their validation
and their known limits.
