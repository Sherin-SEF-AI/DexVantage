"""The synthetic-episode round trip (core/sim/episodes.py, spec §9.1 last bullet).

- groundtruth.synthetic_roundtrip: two short synthetic episodes of the SO-100 sample (every fault on, 64 × 48 px) are
  generated and written as LeRobot v2.1 and as v3.0, then read back with core/datasets (open_dataset, the LeRobot
  reader, which formats.lerobot_roundtrip checks independently against pyarrow and the ffmpeg command-line tools) and
  compared with what the generator made, array by array, and with the ground-truth sidecar it wrote. Counted as
  mismatches: a state, action or time column that is not bit-exact after the float32 cast LeRobot stores; a camera
  whose stored frame count differs from the generator's or from the sidecar's timing; a camera timestamp more than
  half a tick of the 1/90000 s time base from the generator's stamp (or the sidecar's); a decoded frame below 30 dB
  PSNR against its generated frame, or closer to a neighbouring generated frame than to its own (only neighbours
  that differ from it by at least twice the codec's squared error, MARGIN_DB = 3 dB below the frame's own PSNR, are
  tested: two frames of a still scene differ by their noise alone, which the codec smooths away; duplicates, equal by
  design, are not tested); a duplicated frame that, decoded, is not within 30 dB of the decoded frame it repeats; a
  drop or USB-reset gap the stored timestamps do not show (the gap before the listed frame at least 1.5 periods,
  every other gap less); a sidecar that is not the document the generator returned, whose
  per-frame visibility list is not one entry per stored frame, or whose fault frames are outside the stored range.
  The tolerances: columns bit for bit (the writer casts once); half a tick, the rounding of a time to the time base;
  30 dB as in formats.lerobot_roundtrip (the codec is lossy, crf 30), for duplicates too: LeRobot's encoder settings
  (g = 2) make every other frame a keyframe, so a repeated picture is coded again rather than skipped and its two
  decoded copies differ by the codec's error (36.2 dB the farthest pair measured at v3.0, 2026-11-07).
"""
from __future__ import annotations

import json
import math

import numpy as np

from .suite import LEROBOT, Outcome, check

PSNR_MIN_DB = 30.0
DUP_PSNR_MIN_DB = 30.0
MARGIN_DB = 3.0
SETTINGS = {'episodes': 2, 'frames': 16, 'size': [64, 48], 'seed': 3}


def psnr(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    mse = float(np.mean(d * d))
    return math.inf if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def roundtrip(summary: dict) -> tuple[list, dict]:
    """(mismatches [str], detail) of a generate(..., keep_frames=True) summary against its dataset read back."""
    from ..core.datasets import open_dataset
    bad, worst, dup_min, n_frames = [], math.inf, math.inf, 0
    arrays = summary['arrays']
    truth = summary['truth']
    with open_dataset(summary['root']) as ds:
        read_truth = ds.truth()
        if json.dumps(read_truth, sort_keys=True) != json.dumps(truth, sort_keys=True):
            bad.append('the sidecar read back is not the document the generator returned')
        refs = ds.episodes()
        if [r.index for r in refs] != list(range(len(arrays))):
            bad.append(f'{len(refs)} episodes read back, {len(arrays)} generated')
        for e, a in enumerate(arrays):
            if e >= len(refs):
                break
            ep = ds.load(e)
            te = truth['episodes'][str(e)]
            for name, got, src in (('state', ep.q, a['q']), ('action', ep.action, a['action']), ('time', ep.t, a['t'])):
                ref = np.asarray(src, np.float32).astype(np.float64)
                if got is None or np.shape(got) != np.shape(ref) or not np.array_equal(got, ref):
                    bad.append(f'episode {e}: the {name} column is not bit-exact after the float32 cast')
            if te.get('rows') != len(a['t']):
                bad.append(f'episode {e}: the sidecar says {te.get("rows")} rows, {len(a["t"])} were generated')
            for key, c in a['cameras'].items():
                cs = ep.cameras.get(key)
                if cs is None:
                    bad.append(f'episode {e}: camera {key} is missing')
                    continue
                gen_ts = np.asarray(c['timestamps'], float)
                tm = te['timing'][key]
                M = len(gen_ts)
                if cs.n_frames != M or len(tm['stamp_s']) != M or len(c['frames']) != M:
                    bad.append(f'episode {e} {key}: {cs.n_frames} frames read, {M} generated, '
                               f'{len(tm["stamp_s"])} in the sidecar')
                    continue
                tick = 0.5 / 90000 + 1e-9
                if np.max(np.abs(cs.timestamps - gen_ts)) > tick or \
                        np.max(np.abs(np.asarray(tm['stamp_s']) - gen_ts)) > tick:
                    bad.append(f'episode {e} {key}: timestamps differ by more than half a tick')
                vis = te['visibility'][key]['gripper_visible']
                if len(vis) != M:
                    bad.append(f'episode {e} {key}: {len(vis)} visibility entries for {M} frames')
                dups = set()
                for f in te['faults']:
                    if f.get('camera') != key:
                        continue
                    fr = f.get('frames') or []
                    if any(not (0 <= i < M) for i in fr):
                        bad.append(f'episode {e} {key}: {f["kind"]} lists a frame outside the {M} stored')
                    if f['kind'] == 'duplicate':
                        dups.update(fr)
                dec = dict(cs.frames())
                n_frames += len(dec)
                src = c['frames']
                for i in range(M):
                    if i not in dec:
                        bad.append(f'episode {e} {key}: frame {i} not decoded')
                        continue
                    p_own = psnr(dec[i], src[i])
                    worst = min(worst, p_own)
                    if p_own < PSNR_MIN_DB:
                        bad.append(f'episode {e} {key} frame {i}: {p_own:.1f} dB against its generated frame')
                    if i not in dups and i - 1 not in dups:
                        for j in (i - 1, i + 1):
                            # only neighbours that differ from the frame by more than the codec's own error: two
                            # frames of a still scene differ by their noise alone, which the codec smooths away
                            if 0 <= j < M and j not in dups and j - 1 not in dups and \
                                    psnr(src[j], src[i]) < p_own - MARGIN_DB and psnr(dec[i], src[j]) > p_own:
                                bad.append(f'episode {e} {key} frame {i}: closer to generated frame {j}')
                for i in sorted(dups):
                    if i < 1 or not np.array_equal(src[i], src[i - 1]):
                        bad.append(f'episode {e} {key}: generated frame {i} does not repeat frame {i - 1}')
                        continue
                    p = psnr(dec[i], dec[i - 1])
                    dup_min = min(dup_min, p)
                    if p < DUP_PSNR_MIN_DB:
                        bad.append(f'episode {e} {key}: duplicate {i} decodes {p:.1f} dB from frame {i - 1}')
                # gaps: exactly where the sidecar's drop and usb_reset faults say
                gap_at = {int(i) for f in te['faults'] if f.get('camera') == key and f['kind'] in ('drop', 'usb_reset')
                          for i in (f.get('frames') or [])}
                period = 1.0 / float(tm['fps'])
                d = np.diff(cs.timestamps)
                seen = {i + 1 for i in np.flatnonzero(d >= 1.5 * period)}
                if seen != gap_at:
                    bad.append(f'episode {e} {key}: gaps before frames {sorted(seen)}, the sidecar lists '
                               f'{sorted(gap_at)}')
    return bad, {'frames_compared': n_frames, 'worst_psnr_db': None if worst == math.inf else round(worst, 2),
                 'duplicate_min_psnr_db': None if dup_min == math.inf else round(dup_min, 2)}


@check('groundtruth.synthetic_roundtrip', 'groundtruth', 'Synthetic LeRobot episodes read back',
       'number of mismatches when two synthetic SO-100 episodes (every fault on, 64 × 48 px) written as LeRobot v2.1 '
       'and v3.0 are read back: columns not bit-exact after the float32 cast, frame counts, camera times off by more '
       'than half a 90 kHz tick, frames below 30 dB PSNR or closer to a neighbour, duplicates not repeating their '
       'frame within 30 dB, gaps not where the sidecar says, a sidecar that differs from the generator\'s',
       'the generator\'s own arrays (joint states, actions, stamps, the frames before encoding) and the ground truth '
       'it returned, against the dataset read back through core/datasets (the LeRobot reader, checked independently '
       'by formats.lerobot_roundtrip)', 0, 'mismatches', refs=('lerobot_dataset_format', 'mujoco_renderer'),
       cost='study', needs=LEROBOT)
def _synthetic_roundtrip(ctx):
    import os

    from ..core.robot.samples import sample_project
    from ..core.sim import episodes as SE
    p = sample_project('so100')
    detail, bad = {}, []
    for version in ('v2.1', 'v3.0'):
        s = SE.generate(p, os.path.join(ctx.tmp(), f'synthetic_{version}'), dict(SETTINGS, version=version),
                        keep_frames=True)
        b, d = roundtrip(s)
        bad += [f'{version}: {x}' for x in b]
        detail[version] = {**d, 'codec': s['codec'], 'camera_frames': s['camera_frames'], 'faults': s['faults'],
                           'seconds': s['seconds'], 'camera_frames_per_s': s['camera_frames_per_s'],
                           'renderer': s['renderer'].get('renderer')}
    detail['mismatches'] = bad
    return Outcome(len(bad), detail, note='The faults are injected by the generator itself and recorded in the '
                   'sidecar; this check confirms the files carry exactly them. Finding them from the data alone is '
                   'the auditor\'s job (spec §8.2).')
