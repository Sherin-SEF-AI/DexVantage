"""The assistant: a chat model (Claude through Anthropic's SDK, or Groq) that answers questions about the open project
through tools that read DexVantage's own analysis, cites the figures on the page you are on, proposes edits the user
applies as one undo step, and plans long studies that run only after the user approves the plan (the Copilot).

No Qt in this package: the dock (ui/assistant.py) drives it from a worker thread. Ported from Vantage assistant/."""
