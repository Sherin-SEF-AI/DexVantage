"""Measure the splat views' reconstruction quality (core/sim/splat_views.py) for docs/models/splat.md.

    systemd-run --user --scope -q -p MemoryMax=6G env QT_QPA_PLATFORM=offscreen XDG_CACHE_HOME=<a cache with the
        AlexNet weights, or none: LPIPS is then not measured> .venv/bin/python scripts/splat_quality.py \\
        --out out/splat/quality.json [--samples so100 ur5e] [--device auto]

For each sample, with the study's default settings (labelled defaults, core/sim/splat_views.DEFAULTS) unless named:
- demo: the study on a MuJoCo render of the project (48 views at 160 × 120 px, 3 joint states, 1,500 steps, 20,000
  Gaussians, seed 0): PSNR, SSIM and LPIPS on the held-out views and on holdout_q = 6 held-out joint states.
- episode: a DexVantage synthetic LeRobot episode of the sample (core/sim/episodes.py: one episode, no faults, its
  default pixel noise of 1 DN, at most 320 px wide, H.264), the study fitted to it (24 frames per camera at 160 px):
  the same figures on its held-out views and on the 6 frames per camera farthest in joint space from every frame
  used.
- offsets: the demo rig and the episode's rig through the target camera (the sample's wrist camera) moved by 0, 2, 5
  and 10 cm back along the sensor's x axis and 2 and 5 cm sideways (y), at 4 joint states of the capture's spread,
  against MuJoCo's render through the same moved camera (160 px wide); and the episode's rig at 4 of the episode's own
  joint states (its first, last and two between).
- spread: the demo rig at 4 joint states drawn within ±0.2, ±0.4, ±0.8 and ±1.2 rad of home (the capture's is 0.4)
  from new sphere poses, against MuJoCo.
Every capture here is synthetic (MuJoCo renders); the document says so. A real recording is not measured: none is
available offline in the repository (docs/models/splat.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)


def _scores(rig, p, views, dev, lp):
    import numpy as np

    from dexvantage.core import splat_reproject as RP
    out = []
    for cam, T, q, rgb in views:
        pred = np.clip(rig.render(p, cam, T, q, backend='torch', device=dev)['rgb'], 0, 1)
        tgt = np.asarray(rgb, float) / 255.0
        s = RP.image_scores(pred, tgt)
        out.append((s['psnr_db'], s['ssim'], lp(pred, tgt) if lp is not None else None))

    def m(k):
        xs = [o[k] for o in out if o[k] is not None]
        return round(float(np.mean(xs)), 4) if xs else None
    return {'psnr_db': m(0), 'ssim': m(1), 'lpips': m(2), 'views': len(out)}


def measure(name: str, device: str, out_dir: str) -> dict:
    import numpy as np

    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import episodes as EPS
    from dexvantage.core.sim import splat_rig as SRG
    from dexvantage.core.sim import splat_views as SV
    p = sample_project(name)
    res = {'sample': name}
    t0 = time.perf_counter()
    d = SV.analyse(p, {'source': 'demo', 'holdout_q': 6, 'device': device})
    res['demo'] = {'seconds': round(time.perf_counter() - t0, 1), 'rig': d['rig'], 'quality': _q(d['quality'])}
    print(name, 'demo', json.dumps(res['demo']['quality']), flush=True)
    st = SV.settings_of(p, {'source': 'demo', 'device': device})
    rig, info = SV.get_rig(p, st)
    dev = SV._torch_device(info['device'])
    lp, note = SV._lpips_for(st, dev)
    res['lpips_note'] = note
    root = os.path.join(out_dir, f'episodes_{name}')
    if not os.path.isdir(root):
        EPS.generate(p, root, {'episodes': 1, 'faults': [], 'max_width': 320, 'codec': 'h264'})
    st_e = SV.settings_of(p, {'source': 'episode', 'dataset': root, 'holdout_q': 6, 'device': device})
    erig = SV.get_rig(p, st_e)[0]
    E = SV._Episode(p, root, st_e)
    try:
        t = np.asarray(E.ep.t, float)
        eq = [{E.ruid: q} for q in E.q_at(t[np.linspace(0, len(t) - 1, 4).round().astype(int)])]
    finally:
        E.close()
    res['offsets'] = {'demo': _offsets(p, rig, st, dev, lp), 'episode': _offsets(p, erig, st_e, dev, lp),
                      'episode_at_its_q': _offsets(p, erig, st_e, dev, lp, eq)}
    for k, v in res['offsets'].items():
        for r in v['rows']:
            print(name, 'offset', k, r, flush=True)
    spread = []
    for sp in (0.2, 0.4, 0.8, 1.2):
        qn = SRG.random_qs(p, 5, 303, sp)[1:]
        nov = SRG.synthetic_capture(p, n_views=4, qs=qn, width=st['demo_width'], height=st['demo_height'], seed=404)
        spread.append({'spread_rad': sp, **_scores(rig, p, [(v.camera, v.T_world_optical, v.q, v.rgb)
                                                             for v in nov.views], dev, lp)})
        print(name, 'spread', spread[-1], flush=True)
    res['spread'] = spread
    t0 = time.perf_counter()
    d = SV.analyse(p, {'source': 'episode', 'dataset': root, 'holdout_q': 6, 'device': device})
    res['episode'] = {'seconds': round(time.perf_counter() - t0, 1), 'rig': d['rig'], 'quality': _q(d['quality']),
                      'capture': d['capture'], 'source': d['source']}
    print(name, 'episode', json.dumps(res['episode']['quality']), flush=True)
    return res


def _offsets(p, rig, st, dev, lp, qs=None) -> dict:
    """The target camera moved back along its x axis and sideways along y, at 4 joint states of the capture's
    spread: the rig's render against MuJoCo's through the same moved camera."""
    import numpy as np

    from dexvantage.core.frames import Pose
    from dexvantage.core.sim import splat_rig as SRG
    from dexvantage.core.sim import splat_views as SV
    from dexvantage.core.sim.render import Renderer
    s, im = SV.target_sensor(p, st)
    cam = im.camera
    cam = cam.scaled(160, int(round(cam.height * 160 / cam.width)))
    qs = qs or SRG.random_qs(p, 5, 11, st['demo_spread_rad'])[1:]
    rows = []
    with Renderer(p) as R:
        for axis, dists in (('x', (0.0, -0.02, -0.05, -0.10)), ('y', (0.02, 0.05))):
            for dist in dists:
                D = np.eye(4)
                D[{'x': 0, 'y': 1}[axis], 3] = dist
                views = []
                for q in qs:
                    img = R.render([{'sensor_uid': s.uid, 'q': q, 'outputs': ('rgb',), 'camera_model': cam,
                                     'extrinsic_delta': Pose.from_matrix(D)}])[0]
                    T = SV.target_pose_fn(p, {**st, 'target_offset_m': list(D[:3, 3]), 'target_rpy_deg': [0, 0, 0],
                                              'target_pose': None}, s, im, True)(q)
                    views.append((cam, T, q, img['rgb']))
                rows.append({'axis': axis, 'offset_m': dist, **_scores(rig, p, views, dev, lp)})
    return {'sensor': s.name, 'size': [cam.width, cam.height], 'joint_states': len(qs), 'rows': rows}


def _q(q: dict) -> dict:
    out = {}
    for k in ('held_out_views', 'held_out_q', 'train_views', 'before_fit'):
        v = q.get(k)
        if v:
            out[k] = {x: v.get(x) for x in ('psnr_db', 'ssim', 'lpips', 'robot_psnr_db', 'views', 'how')}
    out['lpips_note'] = q.get('lpips_note')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', nargs='+', default=['so100', 'ur5e'])
    ap.add_argument('--device', default='auto')
    a = ap.parse_args(argv)
    import torch
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    doc = {'schema': 'dexvantage.splat_quality/1', 'synthetic': True, 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'measured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'samples': []}
    for name in a.samples:
        doc['samples'].append(measure(name, a.device, out_dir))
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, default=str)
    print('wrote', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
