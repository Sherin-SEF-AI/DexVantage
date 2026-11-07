"""DexVantage: camera, depth and teleop planning for robot-learning data collection.

``import dexvantage as dv`` gives the Python and notebook API (dexvantage/api.py, docs/guide/api.md): dv.open,
dv.sample, dv.new, dv.studies, dv.metrics, dv.exporters, dv.plugins, dv.datasets, dv.runs, dv.track,
dv.publication, dv.stats, dv.Rig, dv.Result. The names are resolved on first use, so importing the package stays cheap
and never imports Qt.
"""
__version__ = "0.1.0.dev0"

_API = ('Rig', 'Result', 'Records', 'Table', 'MetricTable', 'Recording', 'DexVantageError', 'open', 'sample', 'new',
        'samples', 'studies', 'metrics', 'exporters', 'plugins', 'load_plugins', 'datasets', 'runs', 'track',
        'run_store', 'publication', 'campaign', 'cells', 'cluster', 'stats', 'robot_io', 'prices', 'price',
        'diff', 'merge', 'bibtex', 'figures', 'figure', 'export_runs', 'Diff', 'Merge', 'MergeConflict')


def __getattr__(name):
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_API))
