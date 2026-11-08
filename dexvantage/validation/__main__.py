"""Run the validation suite from a terminal (no window):

    python -m dexvantage.validation [OUT_DIR] [--project FILE.dexv] [--models frames,formats] [--quick] [--seed N]
                                    [--no-plugins]
    python -m dexvantage.validation --list [--no-plugins]

The installed plugins are loaded first (as the app, batch mode and the Python API load them), so their checks
(plugin_sdk PluginAPI.check, model 'plugin.<name>') run with the built-in ones and --models takes their model ids;
--no-plugins (or DEXVANTAGE_NO_PLUGINS=1) runs the built-in checks only.

Ported from Vantage validation/__main__.py.

Writes validation.json, validation.csv, validation.html and figures/*.svg into OUT_DIR (default ./validation), prints
one line per check, and exits 0 when every check passed, 6 when one failed.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m dexvantage.validation',
                                 description='Check DexVantage\'s models against independent references and write a '
                                             'report.')
    ap.add_argument('out', nargs='?', default='validation', help='folder for the report (default: ./validation)')
    ap.add_argument('--project', help='a .dexv project whose robots and exports the project checks look at '
                                      '(default: the built-in sample)')
    ap.add_argument('--models', help='comma-separated model ids (default: all)')
    ap.add_argument('--quick', action='store_true', help='fewer Monte Carlo trials and samples (a smoke run)')
    ap.add_argument('--seed', type=int, help='the seed of every random draw')
    ap.add_argument('--list', action='store_true', help='list the checks and exit')
    ap.add_argument('--no-plugins', action='store_true', help='run the built-in checks only (the installed plugins\' '
                                                              'checks are left out)')
    ap.add_argument('--methods', metavar='HTML', help='also write this run\'s tables into the technical reference '
                                                      '(site/methods.html) between its validation markers')
    a = ap.parse_args(argv)
    from .. import plugin_sdk
    plugin_sdk.load(enabled=False if a.no_plugins else None)        # the plugins' checks join the run
    for r in plugin_sdk.loaded():
        if r.status == 'error':
            print(f'Plugin {r.error}', file=sys.stderr)
    from . import (ValidationSettings, catalogue, export, exit_code, run)
    from .suite import model_ids, qt_app
    known = model_ids()
    d = {'quick': a.quick}
    if a.models:
        unknown = [m for m in a.models.split(',') if m.strip() not in known]
        if unknown:
            print(f'Unknown model id(s): {", ".join(unknown)}. Known: {", ".join(known)}.', file=sys.stderr)
            return 4
        d['models'] = a.models
    if a.seed is not None:
        d['seed'] = a.seed
    st = ValidationSettings.from_dict(d)
    if a.list:
        for c in catalogue(st):
            print(f'{c["id"]:34s} {c["title"]}  (tolerance {c["tolerance"]:g} {c["unit"]})')
        return 0
    qt_app()
    project = None
    if a.project:
        from ..core.project import Project
        try:
            project = Project.load(a.project)
        except Exception as e:                      # noqa: BLE001 — reported, not a traceback
            print(f'Could not read {a.project}: {e}', file=sys.stderr)
            return 4
    res = run(project, settings=st)
    for c in res.checks:
        flag = 'pass' if c.passed else 'FAIL'
        err = f'{c.error:.3g}' if c.error == c.error else 'n/a'
        print(f'{flag}  {c.id:34s} {err} / {c.tolerance:g} {c.unit}' + (f'  ({c.exception})' if c.exception else ''))
    paths = export(res, project, a.out)
    if a.methods:
        from .report import update_methods
        done = update_methods(a.methods, res)
        print(f'{a.methods}: {len(done)} tables written')
    print(f'{res.passed}/{len(res.checks)} checks passed in {res.seconds:.0f} s; report: '
          f'{next(p for p in paths if p.endswith("validation.html"))}')
    return exit_code(res)


if __name__ == '__main__':
    sys.exit(main())
