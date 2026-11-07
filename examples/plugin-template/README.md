# DexVantage plugin template

A pip-installable DexVantage plugin with one study, two metrics, a page and a test. Copy this folder, rename the
package (`dexvantage_plugin_template`), the distribution (`dexvantage-plugin-template`), the entry point
(`aim_template`) and the ids (`aim_check`, `aim.worst_deg`, `aim.share_aimed`), then replace the study with yours.

    pip install -e .                    # into the environment DexVantage runs in
    dexvantage --list-plugins           # aim_template 0.1.0: loaded; added 1 study (aim_check), 2 metrics, 1 page
    dexvantage --sample ur5e --aim-check aim.json
    python -m pytest tests              # the plugin's own test (headless)

What it adds:

- the study `aim_check` (`dexvantage_plugin_template/aim.py`): the angle between each enabled camera's viewing
  direction and its direction to the centre of the first task volume, at the home pose; the batch flag
  `--aim-check OUT.json`;
- the metrics `aim.worst_deg` (the headline, lower is better) and `aim.share_aimed` (higher is better), usable in a
  requirements file, the optimiser and `rig.metrics([...])`;
- the page "Aim at the task volume" (`page.py`, imported only when the window is built): the headline with its
  status word, a row per camera, Apply (aims the worst static camera at the centre, one undo step), Preview, Explain,
  Export, and a 3D overlay.

The threshold `max_angle_deg` (20°) is a template default, not a standard. The study uses no hardware figure; a
plugin that does follows DexVantage's rule: every figure from a cited source with its URL and the date read, else
`None` ("not stated"). The full guide is DexVantage's `docs/guide/plugins.md`.

Licence: GPL-3.0-or-later, like DexVantage.
