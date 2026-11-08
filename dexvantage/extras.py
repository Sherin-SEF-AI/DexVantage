"""The optional extras (pyproject.toml [project.optional-dependencies]): what each one adds, whether the downloadable
app (the PyInstaller bundle, packaging/dexvantage.spec) carries it, and the sentence the app adds when a feature needs
one the download leaves out.

    EXTRAS[name]          Extra(modules, what, bundled, extra_mb, why)
    bundled(name)         is the extra in the downloadable app (True from a source or pip installation: there the
                          question is only whether it is installed, which each feature checks itself)
    bundle_note(name)     '' outside the downloadable app; inside it, when the extra is left out, one sentence saying so
                          (appended to a note that stays true in the download)
    missing(name, text)   the feature's own "is not installed: pip install ..." sentence; inside the downloadable app,
                          when the extra is left out, one sentence saying so instead (pip cannot add to the download)

extra_mb is the installed size (MB, 10^6 bytes) of the distributions the extra adds on top of the core requirements and
the bundled extras, summed over their RECORD files in the build environment: python packaging/release.py extras
measures it again (docs/packaging.md, measured 2026-11-08 on Linux x86_64, Python 3.14). The bundle's own size for an
extra is close to it (PyInstaller copies the same libraries, leaves out the sources and tests). None: not installed
where it was measured.

Imports nothing but the standard library: packaging/release.py and the PyInstaller spec read this file by path.
"""
from __future__ import annotations

import sys
from typing import NamedTuple


class Extra(NamedTuple):
    modules: tuple          # the top-level modules it provides (the spec includes or excludes them)
    what: str               # what it is for, in a few words
    bundled: bool           # in the downloadable app
    extra_mb: float | None  # measured, see the module docstring
    why: str                # the reason for the bundling decision


EXTRAS: dict[str, Extra] = {
    'lerobot': Extra(('pyarrow', 'av'), 'LeRobot Parquet and AV1/MP4 video', True, 308.0,
                     'the dataset auditor, plan against recording, the sim-to-real gap and the LeRobot exporter read '
                     'and write these; the core of the data workflow'),
    'bags': Extra(('zstandard', 'lz4'), 'compressed ROS 2 bags (MCAP chunks)', True, 28.6,
                  'small; without it a compressed bag cannot be read at all (zstandard\'s 11 MB cffi backend, for '
                  'PyPy, is left out)'),
    'assistant': Extra(('anthropic', 'keyring'), "the Assistant dock's Claude provider and the OS keyring", True, 51.4,
                       'small; the assistant needs a key the user gives, but the dock should work in the download'),
    'usd': Extra(('pxr',), 'OpenUSD import and pxr-written Isaac camera prims', False, 217.6,
                 'large; the Isaac exporter writes cameras.usda as text without it, and a robot can be imported '
                 'as URDF or MJCF instead'),
    'rerun': Extra(('rerun',), 'the Rerun bridge', False, 554.4,
                   'the largest (its viewer is in the wheel); the Rerun viewer is a separate install anyway'),
    'step': Extra(('gmsh',), 'bracket STEP export as a faceted solid', False, 109.4,
                  'large for one export option; STL brackets work without it'),
    'research': Extra(('pandas',), "the Python API's Result.to_frame() DataFrame", False, 75.1,
                      'only the Python API uses it, and the downloadable app has no Python prompt'),
    'tracking': Extra(('mlflow', 'wandb'), "MLflow's client and wandb for the tracker's exports", False, 160.6,
                      "the tracker writes its MLflow file store and W&B replay script without them; they are read "
                      "back by MLflow or W&B in the user's own Python"),
    'pin': Extra(('pinocchio',), 'pinocchio as a second IK backend', False, None,
                 'a validation reference and an alternative IK backend; the built-in backend is the default'),
}

# extras that are not features of the app (build and development tools): never in the downloadable app
TOOL_EXTRAS = ('bundle', 'test', 'dev', 'site')


def frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def bundled(name: str) -> bool:
    e = EXTRAS.get(name)
    return True if e is None or not frozen() else e.bundled


def bundle_note(name: str) -> str:
    """'' unless this is the downloadable app and it leaves the extra out; then one sentence, with a leading space."""
    e = EXTRAS.get(name)
    if e is None or e.bundled or not frozen():
        return ''
    size = f' (it would add about {e.extra_mb:.0f} MB)' if e.extra_mb else ''
    return f' This DexVantage download leaves out {e.what}{size}: run DexVantage from a pip installation to use it.'


def missing(name: str, text: str, alternative: str = '') -> str:
    """The one sentence a feature gives when extra `name` is not installed: text (the feature's own, with the pip
    command) from a source or pip installation; in the downloadable app, when the download leaves the extra out, the
    sentence that says so and how to get it instead, since pip cannot add a package to the download. alternative:
    what to do without it (', or export STL only'), kept in that sentence."""
    e = EXTRAS.get(name)
    if e is None or e.bundled or not frozen():
        return text
    size = f' (it would add about {e.extra_mb:.0f} MB)' if e.extra_mb else ''
    return (f'This DexVantage download leaves out {e.what}{size}: install DexVantage from source with the {name} extra '
            f'(pip install -e ".[{name}]" in its folder) to use it{alternative}.')
