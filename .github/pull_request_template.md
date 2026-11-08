**What this changes**


**How it was checked**
The tests you ran and their counts, the validation checks (`--validate out/validation --quick`) and what you
measured, with how.

**Checklist** (CONTRIBUTING.md and docs/DEVELOPING.md)
- [ ] Every new figure has a cited source (and the date read) or a stated measurement; gaps are `None`, not guesses
- [ ] Defaults are labelled as defaults
- [ ] A new model has a validation check against an independent reference, with a stated tolerance and why
- [ ] Known limits are written down (module docstring and docs)
- [ ] New assets are licence-checked and recorded in THIRD_PARTY.md
- [ ] Errors reach the user as plain sentences, not tracebacks
- [ ] No personal paths, user names, host names, e-mail addresses or keys in any file
- [ ] Tests that render MuJoCo scenes are marked `mujoco_render`; no skip without a stated reason
- [ ] `ruff check dexvantage tests scripts` passes; docs and CHANGELOG.md (under [Unreleased]) are updated
