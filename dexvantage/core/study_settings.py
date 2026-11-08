"""The setting names each study reads, so a setting given by name (the Python API's rig.run(study, **settings),
dv.cluster.tolerance(rig, **settings)) can be refused when the study does not know it, instead of being ignored
silently (dv.cluster.tolerance(rig, draws=8) ran the default 64 draws: the key is n).

A study's names are its module's DEFAULTS (each a labelled default) and the names its settings_of() reads besides
them (EXTRA, written out from each study's settings_of; tests/test_study_settings.py checks that every DEFAULTS key
and every key the samples store is known). known(sid) is None for a study whose names are not listed here: its
settings are not checked (the meta studies, the dataset studies, the campaign, the optimiser and the design agent,
whose settings are documented in their modules and specs).

    known(sid) -> tuple | None        the sorted setting names, or None (not checked)
    check(sid, names)                 StudyError (one sentence naming the study's settings) for an unknown name
"""
from __future__ import annotations

import sys

from .jobs import StudyError

# names a study reads besides its module's DEFAULTS (from each study's settings_of)
EXTRA: dict = {
    'tolerance': ('distribution', 'metrics', 'sensors'),
    'coverage': ('q', 'imagers', 'modes', 'optics', 'sensors', 'voxel_mm', 'refine_mm'),
    'brackets': ('q', 'sensors'),
    'cables': (),
    'teleop': (),
    'time_sync': ('pairing', 'joint_rate_hz', 'latency_s'),
    'grasp_visibility': ('grasp_set', 'q', 'imagers', 'modes', 'sensors', 'depth_imager'),
    'groundtruth': (),
    'vla_tokens': ('policies', 'critical_objects', 'grasp_set', 'q', 'suggest'),
    'bandwidth': ('overhead', 'measured', 'extra', 'recorder', 'external_power'),
    'motion_blur': ('exposure_s', 'readout_s', 'shutter', 'trajectories'),
    'splat_views': (),
}


def defaults_of(sid: str) -> dict:
    """The study's module DEFAULTS ({} when it has none)."""
    from . import jobs
    jobs.load_studies()
    info = jobs.STUDIES.get(sid)
    mod = sys.modules.get(info.fn.__module__) if info is not None else None
    d = getattr(mod, 'DEFAULTS', None)
    return dict(d) if isinstance(d, dict) else {}


def known(sid: str) -> tuple | None:
    """The setting names the study reads (sorted), or None when its names are not listed (not checked)."""
    if sid not in EXTRA:
        return None
    return tuple(sorted(set(defaults_of(sid)) | set(EXTRA[sid])))


def _join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def check(sid: str, names) -> None:
    """Refuse (StudyError, one sentence) a setting name the study does not read; a study not listed is not checked."""
    ok = known(sid)
    if ok is None:
        return
    bad = sorted(str(n) for n in names if str(n) not in ok)
    if not bad:
        return
    s = 's' if len(bad) > 1 else ''
    raise StudyError(f'The {sid} study has no setting{s} called {_join(bad)}; its settings are {_join(ok)}.')
