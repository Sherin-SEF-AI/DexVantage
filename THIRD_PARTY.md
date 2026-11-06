# Third-party components

Checked 2026-11-06. Only redistributable assets are bundled; anything else is downloaded on first use with the
licence shown and the user's consent (spec §1.2).

## Bundled

| Component | Source | Licence | Version | Where |
|---|---|---|---|---|
| IBM Plex Sans, IBM Plex Mono (Regular, Medium, SemiBold) | https://github.com/IBM/plex | SIL Open Font License 1.1 (`dexvantage/resources/fonts/OFL.txt`) | as shipped with Vantage | `dexvantage/resources/fonts/` |
| Code ported from Vantage (theme, widgets, charts, icons, renderer, viewport, navigation, palette, provenance, validation machinery) | https://github.com/Sherin-SEF-AI/Vantage | GPL-3.0-or-later | Vantage 0.8.1 | marked in each module's docstring |

## Runtime dependencies (installed by pip, not bundled in the source tree)

| Package | Licence |
|---|---|
| PyQt6 | GPL-3.0 (or commercial) |
| NumPy, SciPy | BSD-3-Clause |
| PyOpenGL | BSD-3-Clause |
| OpenCV (opencv-python) | Apache-2.0 |
| MuJoCo | Apache-2.0 |
| PyYAML | MIT |
| certifi | MPL-2.0 |
| jsonschema (tests) | MIT |

## Robot models

None bundled yet. Phase 1 adds the samples; each MuJoCo Menagerie model is listed here with its own LICENSE file,
version (commit) and the date checked, because Menagerie licences differ per model.
