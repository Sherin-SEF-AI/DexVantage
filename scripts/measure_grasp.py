"""Measure the grasp self-occlusion study's performance budget (spec §16): N sampled grasps on a sample project.

    python scripts/measure_grasp.py [--sample ur5e] [--n 2000] [--seed 0] [--egl 'EGL device 1'] [--backend auto]
                                    [--cube-n 512] [--no-sensitivity] [--json OUT.json]

The grasps are sampled with core/grasps.sample (top-down, side and angled, round-robin, the given seed) over the
sample's task volume; the study runs with its default settings unless told otherwise. --egl picks the EGL device or
renderer for the GPU cube maps (DEXVANTAGE_EGL, core/occlusion.py). Prints the renderer, the phase timings and the
throughput; run it memory-capped, e.g. systemd-run --user --scope -p MemoryMax=4G env QT_QPA_PLATFORM=offscreen ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sample', default='ur5e')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--egl', default='')
    ap.add_argument('--backend', default='auto')
    ap.add_argument('--cube-n', type=int, default=512)
    ap.add_argument('--no-sensitivity', action='store_true')
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    if a.egl:
        os.environ['DEXVANTAGE_EGL'] = a.egl
    from dexvantage.core import grasps as GR
    from dexvantage.core import jobs
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.studies import grasp_visibility as GV
    p = sample_project(a.sample)
    p.tasks.grasp_sets = []                     # the sample's own set left out: sample a.n grasps
    r = p.robots[0]
    t0 = time.perf_counter()
    gs = GR.sample(p, r.uid, kinds=('top_down', 'side', 'angled'), n=a.n, rng_seed=a.seed)
    GR.add_set(p, gs)
    t_sample = time.perf_counter() - t0
    st = {'backend': a.backend, 'cube_n': a.cube_n}
    if a.no_sensitivity:
        st['sensitivity'] = False
    p.analysis.studies[GV.SID] = st
    partials = []
    ctx = jobs.Ctx(partial=lambda d: partials.append((time.perf_counter(), d.get('evaluated'))))
    t0 = time.perf_counter()
    doc = jobs.run_study(GV.SID, p, ctx)
    wall = time.perf_counter() - t0
    s, oc, th, tm = doc['summary'], doc['occlusion'], doc['throughput'], doc['timings']
    out = {'sample': a.sample, 'grasps': a.n, 'seed': a.seed, 'renderer': oc['renderer'], 'backend': oc['backend'],
           'gl_how': oc.get('gl_how'), 'gl_mode': oc.get('gl_mode'), 'cube_n': oc['cube_n'],
           'feasible': s['feasible'], 'cvs': s['cvs'], 'ci': [s['ci_low'], s['ci_high']],
           'sampling_seconds': round(t_sample, 3), 'wall_seconds': round(wall, 3), **tm,
           'grasps_per_s': th['grasps_per_s'], 'occlusion_grasps_per_s': th['occlusion_grasps_per_s'],
           'maps_built': oc['maps_built'], 'map_seconds': oc['map_seconds'], 'maps_per_s': th['maps_per_s'],
           'cache_hit_rate': oc['cache_hit_rate'], 'partial_updates': len(partials)}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
