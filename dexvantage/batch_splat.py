"""Batch mode for splat views (core/sim/splat_views.py, spec §9.3):

    dexvantage PROJECT.dexv --splat fit     --splat-out OUT_DIR [--dataset PATH] [--splat-settings JSON]
    dexvantage PROJECT.dexv --splat replay  --splat-out OUT_DIR --dataset PATH [--splat-settings JSON]
    dexvantage PROJECT.dexv --splat augment --splat-out OUT_DIR --dataset PATH [--splat-settings JSON]

fit runs the splat_views study (fit the rig, or reuse the one cached for the same inputs; its quality on held-out
frames; the before/after preview) and writes OUT_DIR/splat_views.json with provenance and the rig as
OUT_DIR/splat_rig.npz with its .json sidecar. replay writes OUT_DIR/splat_replay_synthetic/ (a video and per-frame
images of the episode through the proposed camera); augment writes OUT_DIR/splat_augment_synthetic/ (a LeRobot
dataset of perturbed copies). Every render is marked synthetic. --splat-settings lays settings over the project's
(core/sim/splat_views.DEFAULTS), e.g. {"target_offset_m": [0, 0, 0.02], "augment_n": 8}; without --dataset the
fit uses a MuJoCo render of the project (a demonstration). --splat-views OUT.json (core/jobs) runs the study alone.

Exit 0 when it was written, 2 when it could not be (one sentence on stderr: the splat extra is not installed, there is
no CUDA device, the episode has too few distinct views, a bad setting).
"""
from __future__ import annotations

import json
import os

ACTIONS = ('fit', 'replay', 'augment')


def add_arguments(g):
    g.add_argument('--splat', metavar='ACTION', choices=ACTIONS,
                   help='splat views (core/sim/splat_views.py; needs the splat extra): fit (the rig and its quality on '
                        'held-out frames), replay (the --dataset episode through a proposed camera pose, to a video) '
                        'or augment (perturbed copies of the episode as a LeRobot dataset); with --splat-out OUT_DIR. '
                        'Every render is marked synthetic')
    g.add_argument('--splat-out', metavar='OUT_DIR', help='with --splat: the output folder')
    g.add_argument('--splat-settings', metavar='JSON', help='with --splat: settings as JSON text or a .json file, '
                   'e.g. {"target_offset_m": [0, 0, 0.02], "augment_n": 8}')


def wanted(args) -> bool:
    return bool(getattr(args, 'splat', None) or getattr(args, 'splat_out', None)
                or getattr(args, 'splat_settings', None))


def read_settings(text) -> dict:
    """JSON text or a .json file -> a dict; ValueError with a sentence otherwise."""
    if not text:
        return {}
    try:
        if not str(text).lstrip().startswith('{'):
            with open(text, encoding='utf-8') as fh:
                text = fh.read()
        d = json.loads(text)
    except (OSError, ValueError) as e:
        raise ValueError(f'--splat-settings could not be read as JSON ({e})') from None
    if not isinstance(d, dict):
        raise ValueError('--splat-settings must be a JSON object, e.g. {"augment_n": 8}')
    return d


def run(args, project) -> int:
    from .batch import EXIT_OK, _err, _unexpected, _write
    from .core import jobs
    from .core import provenance as PV
    if not getattr(args, 'splat', None):
        return _err('--splat-out and --splat-settings go with --splat fit|replay|augment')
    if not getattr(args, 'splat_out', None):
        return _err('--splat needs --splat-out OUT_DIR')
    if project is None:
        return _err('--splat needs a project file (or --sample)')
    try:
        given = read_settings(getattr(args, 'splat_settings', None))
    except ValueError as e:
        return _err(str(e))
    snap = project.snapshot()
    from .core.sim import splat_views as SV
    st = {**(snap.analysis.studies.get(SV.SID) or {}), **given}
    if getattr(args, 'dataset', None):
        st['dataset'] = args.dataset
    snap.analysis.studies[SV.SID] = st
    out = args.splat_out
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        return _err(f'could not create the folder {out}: {e.strerror or e}.')
    try:
        if args.splat == 'fit':
            doc = jobs.run_study(SV.SID, snap)
            doc = {k: v for k, v in doc.items() if k != 'cached'}
            dest = os.path.join(out, 'splat_views.json')
            doc = PV.stamp(doc, snap, output=PV.batch_output(doc, dest, option='splat'))
            _write(doc, dest)
            rig, _info = SV.get_rig(snap)
            files = rig.save(os.path.join(out, 'splat_rig.npz'))
            h = doc['headline']
            print(f'Splat views: {h["text"]} → {dest}, {os.path.basename(files[0])}')
        elif args.splat == 'replay':
            doc = SV.replay(snap, out)
            print(f'Splat replay ({doc["frames"]} frames, {doc["video"]["codec"]}, synthetic) → '
                  f'{os.path.join(out, SV.REPLAY_FOLDER)}')
        else:
            doc = SV.augment(snap, out)
            print(f'Splat augmentation ({doc["episodes"]} perturbed copies, {doc["lerobot"]["frames"]} rows, '
                  f'synthetic) → {os.path.join(out, SV.AUGMENT_FOLDER)}')
    except jobs.StudyError as e:
        return _err(str(e))
    except SystemExit as e:
        return int(e.code)
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        return _unexpected('Splat views', e)
    return EXIT_OK
