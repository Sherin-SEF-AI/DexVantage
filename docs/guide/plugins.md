# Plugins: adding studies, metrics, exporters, readers, library folders and pages

A DexVantage plugin is an installed Python package that declares an entry point in the group `dexvantage.plugins`.
DexVantage finds it with `importlib.metadata`, calls its `register(api)` once at start-up and lists it in
**Help ▸ Plugins**. Through `api` a plugin may add:

| What | Call | Where it shows |
|---|---|---|
| a study | `@api.study(id, title, quick=False, batch='flag', needs_robot=False)` on `fn(project, ctx) -> dict` | `core.jobs`, its batch flag `--flag OUT.json`, `rig.run(id)`, the assistant's study list |
| a metric | `@api.metric(id, title, unit, better, study, note='', tags=())` on `extract(doc) -> float \| None` | `rig.metrics`, requirements files and `--check`, the optimiser, experiments |
| an exporter | `@api.exporter(id, title, needs_robot=False)` on `fn(project, out_dir, options, ctx) -> {'files', 'notes'}` | `--export --formats id`, the Export dialog, `rig.export` |
| a dataset reader | `@api.dataset_reader(format, sniff, title='')` on `open(path) -> core.datasets.Dataset` | `dv.datasets.open`, the audit, plan against recording, `--dataset` |
| a library folder | `api.library_folder(path)` | the Library (read-only, like the built-in entries) |
| a UI page | `api.page(id, title, 'module:Class', menu='Analysis', icon=..., action_id=None, workspace=...)` | a tab of the Analysis dock and a menu command |

The module is `dexvantage/plugin_sdk.py`; the contracts each registry follows are those of the built-in code
(`core/jobs.py`, `core/metrics.py`, `core/exporters/__init__.py`, `core/datasets/episode.py`, `core/library`,
`ui/pages`) and the rules of `docs/DEVELOPING.md` apply to a plugin's figures as to DexVantage's: every hardware or
policy figure from a cited source with its URL and the date read, else `None` ("not stated"); defaults labelled as
defaults; errors raised as `api.StudyError("plain sentence")`.

## A worked example: the template

`examples/plugin-template/` is a pip-installable plugin with one study, two metrics, a page and its own test:

    examples/plugin-template/
      pyproject.toml                          the entry point: aim_template = "dexvantage_plugin_template:register"
      dexvantage_plugin_template/__init__.py  register(api): the study, the metrics and the page
      dexvantage_plugin_template/aim.py       the study (headless: numpy and DexVantage's core only)
      dexvantage_plugin_template/page.py      the page (Qt; imported only when the window is built)
      tests/test_template.py                  the plugin's tests, headless

The study, "Aim at the task volume", is the angle between each enabled camera's viewing direction (its body x axis)
and its direction to the centre of the first task volume, at the home pose; the headline is the worst angle, and a
camera is "aimed" within `max_angle_deg` (20°, a template default). It uses no hardware figure. Its test checks the
angle against an independent closed form (atan2 of the perpendicular and parallel parts) over 200 random cases within
1e-9°.

Install and use it:

    pip install -e examples/plugin-template
    dexvantage --list-plugins
      aim_template 0.1.0: loaded; added 1 study (aim_check), 2 metrics (aim.worst_deg, aim.share_aimed), 1 page (aim_check)
    dexvantage --sample dexhand --aim-check aim.json
      Aim at the task volume → aim.json

On the dexhand sample the worst camera is the palm micro camera, 57.1° off the tabletop's centre ("aimed away"; 2 of 3
cameras aimed); it is on a link, so the advice is to turn it on the Sensors and mounts page. On the UR5e sample the
worst is the wrist D405 at 16.4° ("aimed"). From Python:

    import dexvantage as dv
    rig = dv.sample('ur5e')
    rig.run('aim_check').headline['text']        # 'Worst camera aim off the task volume: 16.4° (aimed)'
    rig.check([{'metric': 'aim.worst_deg', 'op': '<=', 'threshold': 30}])

In the window the page "Aim at the task volume" (Analysis menu) shows the headline with its status word, a row per
camera, a line from each camera to the centre in its status colour, and Apply, which turns the worst static camera to
look at the centre as one undo step (Preview dashes its line first).

To make your own: copy the folder, rename the package, the distribution, the entry point name and every id, replace the
study, and keep the test.

## Writing register(api)

    __version__ = '0.1.0'
    REQUIRES_API = 1                        # the plugin interface version it was written for

    def register(api):
        from . import mystudy
        api.study('my_study', 'My study', batch='my-study')(mystudy.run)
        api.metric('my.figure', 'My figure', 'mm', 'lower', study='my_study', tags=('headline',))(
            lambda doc: doc['summary']['figure_mm'])
        api.page('my_study', 'My study', 'my_plugin.page:MyPage', menu='Analysis')

- **Ids** must be new: a plugin cannot replace a built-in study, metric, exporter or dataset format, or reuse a
  built-in batch flag. A duplicate is skipped and Help ▸ Plugins says so in a sentence ("… tried to add the study
  "coverage", which DexVantage already has; it was not added."). Study ids are lower-case words joined by underscores
  (no dot: factor paths use `setting:<study>.<key>`). Prefix ids with your plugin's name.
- **Pages** are given as `"module:Class"`, so loading a plugin never imports Qt: the class is imported when the window
  builds its pages. A page subclasses `dexvantage.ui.pages.base.StudyPage` (headline with a status word, Apply as one
  undo step through `ctx.store.edit`, Preview, Explain, Export with provenance, `overlays(vp)` for the 3D view).
- **Tag** one metric per study `'headline'`: `Result.headline` and the API's HTML views use it.
- **Licences**: bundle only data you may redistribute; a plugin's library entries follow the library schema and cite
  their sources like the built-in ones.
- **Validation checks**: a plugin's study is a model, and docs/DEVELOPING.md asks every model to be compared with an
  independent reference. `api.check(id, title, quantity, reference, tolerance, unit, references={key: citation})`
  registers `fn(ctx) -> api.Outcome(error, detail)` in the validation suite under the plugin's own model
  (`plugin.<name>`); `--validate`, `--validate --models plugin.<name>`, the self-test's cheap run and Help ▸ Validate
  the models run it with the built-in checks, and its references join the report's list and `--cite`'s BibTeX.
  A check id that is taken, or a reference key that DexVantage has with other text, is skipped with a sentence.

      @api.check('my.vs_closed_form', 'My figure against its closed form', 'absolute error', 'the closed form',
                 1e-9, 'mm', references={'me2026': 'A. Author, "Title", https://..., (read 2026-11-07).'})
      def _check(ctx):
          ours, ref = my_figure(), closed_form()
          return api.Outcome(abs(ours - ref), {'ours': ours, 'reference': ref})

## Loading, errors and --no-plugins

Plugins load once at start-up, before the batch flags are built (a plugin study gets its flag), and on first use of
the Python API. A plugin that raises in `register`, cannot be imported, or needs a newer `REQUIRES_API` is marked
"could not be loaded" with a sentence (`broken could not be loaded: KeyError: 'sensors'.`); what it registered before
the failure is removed, its traceback goes to the log (stderr), and DexVantage carries on. Help ▸ Plugins shows each
plugin's name, version, distribution, status, what it added and its error or skipped items.

`dexvantage --no-plugins` (or `DEXVANTAGE_NO_PLUGINS=1`) finds the installed plugins without loading them: they are
listed as "off". `dexvantage --list-plugins` prints the list and exits.

## Testing a plugin

The template's `tests/conftest.py` gives the tests throw-away config, data and cache folders. To load the plugin
without installing it, give `plugin_sdk.load` a fake entry point and unload it afterwards:

    from importlib.metadata import EntryPoint
    from dexvantage import plugin_sdk as PL
    PL.load(entry_points=[EntryPoint('aim_template', 'dexvantage_plugin_template:register', PL.GROUP)])
    ...
    PL.unload('aim_template')

`plugin_sdk.load(paths=[folder])` finds plugins installed into a folder (`pip install --target folder`), which is how
DexVantage's own `tests/test_plugins.py` installs the template into a temporary folder and finds it as a distribution.

## Known limits

A plugin runs in DexVantage's process with its rights: there is no sandbox, so install only plugins you trust.
Unloading removes what a plugin registered, not modules it imported or other state it changed. Pages are added when
the window is built, so a plugin loaded later from Python adds its page only after a restart, and a page whose id
equals a built-in page's is replaced by the built-in one (Help ▸ Plugins says so). The plugin interface is version 1; a change to a registry's
contract (core/jobs, core/metrics, core/exporters, core/datasets, core/library, ui/pages) is a change for plugins too.
