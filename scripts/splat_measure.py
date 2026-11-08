"""Measure the robot-aligned splat rig (core/sim/splat_rig.py) on the bundled samples: reconstruction quality on
held-out views, the per-link segmentation's mislabel rate, and the rasteriser's agreement across back ends. Writes
JSON to --out (the figures docs/models/splat.md quotes).

    python scripts/splat_measure.py --out out/splat/measure.json [--samples so100 ur5e] [--iterations 1500]

The truth for the mislabel rate: every Gaussian starts from back-projected pixels of MuJoCo's per-pixel owner
segmentation of the synthetic capture, grouped per owner, so its starting label is exact by construction. Three
cases per method: (1) the starting rig as it is (the method's own noise), (2) after fitting with segmentation off
(the centres moved up to max_shift_m), (3) after fitting with flip_share of the labels changed to a wrong frame at
random (seeded; each moved into the wrong frame at the reference q, as relabel does), which the method should undo.
Re-posing: the fitted rig at joint states the capture never showed, against MuJoCo's renders (novel_q). Synthetic
captures only: no real images are involved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def measure(name: str, iterations: int, width: int, height: int, n_views: int, n_q: int, max_gaussians: int,
            flip_share: float, device: str, seed: int, montage: str | None = None) -> dict:
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import splat_rig as SRG
    p = sample_project(name)
    t0 = time.perf_counter()
    cap = SRG.synthetic_capture(p, n_views=n_views, qs=SRG.random_qs(p, n_q, seed=seed), width=width,
                                height=height, seed=seed)
    t_cap = time.perf_counter() - t0
    st = SRG.FitSettings(iterations=iterations, device=device, max_gaussians=max_gaussians, segment='none',
                         time_budget_s=3600, seed=seed)
    rig0 = SRG.init_rig(p, cap, st)
    static = SRG.static_frames(rig0, p)
    out = {'sample': name, 'capture': {'views': len(cap.views), 'width': width, 'height': height, 'q_states': n_q,
                                       'seconds': round(t_cap, 2)}, 'gaussians_init': rig0.n, 'segmentation': {}}
    out['static_frames'] = [rig0.links[k][1] for k in static]
    for method in SRG.METHODS:
        lab, _ = SRG.segment_links(rig0, p, cap, method)
        out['segmentation'].setdefault('start', {})[method] = SRG.mislabel_rate(lab, rig0.frame, rig0.links, static)
    rig, rep = SRG.fit_splat_rig(p, cap, st, rig=rig0)
    out['fit'] = {k: rep[k] for k in ('iterations', 'seconds', 'device', 'gaussians', 'pruned', 'loss_first',
                                      'loss_last', 'held_out_before', 'held_out_after', 'train_after', 'stopped_by')}
    out['novel_q'] = novel_q(p, rig, cap, seed, montage)
    truth = rig.frame.copy()
    for method in SRG.METHODS:
        lab, _ = SRG.segment_links(rig, p, cap, method)
        out['segmentation'].setdefault('after_fit', {})[method] = SRG.mislabel_rate(lab, truth, rig.links, static)
    rng = np.random.default_rng(seed + 7)
    flip = rng.random(rig.n) < flip_share
    wrong = truth.copy()
    L = len(rig.links)
    for i in np.flatnonzero(flip):
        choices = [f for f in range(-1, L) if f != truth[i]]
        wrong[i] = choices[int(rng.integers(len(choices)))]
    SRG.relabel(rig, p, wrong)
    out['segmentation']['flipped_share'] = float(flip.mean())
    for method in SRG.METHODS:
        lab, _ = SRG.segment_links(rig, p, cap, method)
        r = SRG.mislabel_rate(lab, truth, rig.links, static)
        r['flipped_recovered'] = float((lab[flip] == truth[flip]).mean()) if flip.any() else None
        out['segmentation'].setdefault('after_flips', {})[method] = r
    return out


def novel_q(p, rig, cap, seed: int, montage: str | None = None, n: int = 6) -> dict:
    """The fitted rig re-posed at joint states the capture never showed (random_qs with another seed), rendered from
    n new poses (synthetic_capture's sphere with another seed) and compared with MuJoCo's render of the same scene
    (PSNR, SSIM), against the same figures at the capture's own first joint state from the same poses. montage: a
    PNG of the renders above MuJoCo's."""
    from dexvantage.core import splat_reproject as RP
    from dexvantage.core.sim import splat_rig as SRG
    qn = SRG.random_qs(p, n + 1, seed=seed + 101)[1:]
    cam = cap.views[0].camera
    novel = SRG.synthetic_capture(p, n_views=n, qs=qn, width=cam.width, height=cam.height, seed=seed + 202)
    same = SRG.synthetic_capture(p, n_views=n, qs=[cap.views[0].q], width=cam.width, height=cam.height,
                                 seed=seed + 202)
    res = {}
    rows = []
    for key, c in (('novel_q', novel), ('capture_q', same)):
        ps, ss = [], []
        for v in c.views:
            r = rig.render(p, v.camera, v.T_world_optical, v.q, backend='torch')
            sc = RP.image_scores(np.clip(r['rgb'], 0, 1), np.asarray(v.rgb, float) / 255.0)
            ps.append(sc['psnr_db'])
            ss.append(sc['ssim'])
            rows.append((np.clip(r['rgb'], 0, 1), np.asarray(v.rgb, float) / 255.0))
        res[key] = {'psnr_db': round(float(np.mean(ps)), 3), 'ssim': round(float(np.mean(ss)), 4), 'views': len(ps)}
    if montage:
        import cv2
        top = np.concatenate([a for a, _ in rows[:n]], 1)
        bot = np.concatenate([b for _, b in rows[:n]], 1)
        cv2.imwrite(montage, (np.concatenate([top, bot], 0)[..., ::-1] * 255).astype(np.uint8))
    return res


def raster_agreement(seed: int = 0) -> dict:
    """numpy reference against the torch path on CPU (float64) and CUDA (float32), 300 fixed random Gaussians, three
    lens models."""
    import torch

    from dexvantage.core import splat_reproject as RP
    from dexvantage.core.optics import CameraModel
    rng = np.random.default_rng(seed)
    n = 300
    gs = RP.GaussianSet(rng.uniform([-0.3, -0.2, 0.8], [0.3, 0.2, 1.6], (n, 3)), rng.uniform(0.005, 0.04, (n, 3)),
                        rng.normal(size=(n, 4)), rng.uniform(0.2, 0.95, n), rng.uniform(0, 1, (n, 3)))
    cams = {'pinhole': CameraModel(64, 48, 60, 60, 31.5, 23.5),
            'kb': CameraModel(64, 48, 40, 40, 31.5, 23.5, 'kb', (0.02, -0.01, 0, 0)),
            'brown': CameraModel(64, 48, 60, 58, 30, 24, 'brown', (-0.1, 0.02, 0.001, -0.001, 0.0))}
    out = {}
    for k, cam in cams.items():
        a = RP.render(gs, cam, np.eye(4), backend='numpy')
        b = RP.render(gs, cam, np.eye(4), backend='torch', device=torch.device('cpu'), dtype='float64')
        row = {'cpu64_max_abs': float(np.abs(a['rgb'] - b['rgb']).max())}
        if torch.cuda.is_available():
            c = RP.render(gs, cam, np.eye(4), backend='torch', device=torch.device('cuda'))
            row['cuda32_max_abs'] = float(np.abs(a['rgb'] - c['rgb']).max())
        out[k] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', nargs='+', default=['so100', 'ur5e'])
    ap.add_argument('--iterations', type=int, default=1500)
    ap.add_argument('--width', type=int, default=160)
    ap.add_argument('--height', type=int, default=120)
    ap.add_argument('--views', type=int, default=48)
    ap.add_argument('--q-states', type=int, default=3)
    ap.add_argument('--max-gaussians', type=int, default=20000)
    ap.add_argument('--flip-share', type=float, default=0.1)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    doc = {'schema': 'dexvantage.splat_measure/1', 'synthetic': True,
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'settings': vars(a), 'raster': raster_agreement(a.seed), 'samples': []}
    for name in a.samples:
        mont = os.path.join(os.path.dirname(os.path.abspath(a.out)), f'novel_q_{name}.png')
        doc['samples'].append(measure(name, a.iterations, a.width, a.height, a.views, a.q_states, a.max_gaussians,
                                      a.flip_share, a.device, a.seed, mont))
        print(json.dumps(doc['samples'][-1]['fit']), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, default=float)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
