"""Measure the occlusion models on the UR5e sample: cube-map build times for both backends at n = 256 and 512, the
cost per joint vector (posing the arm, digesting it and building the map, as Phase 3 will), the agreement of the
two backends and of the cube map with exact ray casting, and what vertex-clustering decimation changes. Writes
out/occlusion-measure.json and prints a summary (the figures in docs/models/occlusion.md come from this script).

    systemd-run --user --scope -q -p MemoryMax=7G env QT_QPA_PLATFORM=offscreen \\
        .venv/bin/python scripts/measure_occlusion.py [--quick]

DEXVANTAGE_EGL='EGL device 2' (or a renderer name such as llvmpipe) picks another EGL device; DEXVANTAGE_NO_GL=1
switches the GPU backend off.
"""
import json
import os
import platform
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402

from dexvantage.core import occlusion as OC  # noqa: E402
from dexvantage.core import raycast as RC  # noqa: E402
from dexvantage.core import scenegeom as SG  # noqa: E402
from dexvantage.core.robot.model import get_model  # noqa: E402
from dexvantage.core.robot.samples import sample_project  # noqa: E402


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return float(np.median(out)), float(np.min(out))


def main():
    quick = '--quick' in sys.argv
    p = sample_project('ur5e')
    robot = p.robots[0]
    m = get_model(p, robot.uid)
    st = OC.gl_status()
    res = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                           'machine': platform.machine(), 'system': platform.system(),
                           'processor': platform.processor() or platform.machine()},
           'gl': st, 'sample': 'ur5e', 'cameras': {}}
    occ = SG.scene_occluders(p, {})
    res['triangles'] = len(occ)
    lo, hi = p.workspace.volumes[0].bounds()
    rng = np.random.default_rng(20261106)
    P = rng.uniform(lo, hi, (5000 if quick else 20000, 3))
    for s in p.variant.sensors:
        o = SG.sensor_world_pose(p, s, {}).position
        so = SG.sensor_occluders(p, s, {}, base=occ)
        cam = {'origin_m': [float(v) for v in o], 'triangles': len(so)}
        for n in (256, 512):
            for be in (('gl', 'numpy') if st['available'] else ('numpy',)):
                OC.CubeMap.build(so, o, n, be)               # warm: geometry on the GPU, face directions cached
                med, best = timed(lambda: OC.CubeMap.build(so, o, n, be), 2 if be == 'numpy' else 10)
                cam[f'{be}_{n}_s'] = {'median': med, 'min': best}
            if st['available']:
                a, b = OC.CubeMap.build(so, o, n, 'gl').faces, OC.CubeMap.build(so, o, n, 'numpy').faces
                fa, fb = np.isfinite(a), np.isfinite(b)
                both = fa & fb
                diff = np.abs(a[both].astype(float) - b[both])
                cam[f'backends_{n}'] = {'coverage_mismatch_fraction': float((fa != fb).mean()),
                                        'depth_diff_over_1mm_fraction': float((diff > 1e-3).mean()),
                                        'depth_diff_median_m': float(np.median(diff)) if len(diff) else 0.0,
                                        'depth_diff_p99_m': float(np.percentile(diff, 99)) if len(diff) else 0.0}
        t = time.perf_counter()
        grid = RC.Grid.build(so.tris)
        tg = time.perf_counter() - t
        t = time.perf_counter()
        ve = RC.visible_exact(P, o, grid)
        tr = time.perf_counter() - t
        for n in (256, 512):
            c = OC.CubeMap.build(so, o, n)
            vc = c.visible(P)
            bad = np.nonzero(vc != ve)[0]
            sil = RC.near_silhouette(P[bad], o, grid, 2.0 / n) if len(bad) else np.zeros(0, bool)
            cam[f'cube_vs_raycast_{n}'] = {'points': len(P), 'visible_exact_fraction': float(ve.mean()),
                                           'disagree_fraction': float(len(bad) / len(P)),
                                           'disagree_near_silhouette': int(sil.sum()),
                                           'disagree_elsewhere': int(len(bad) - sil.sum())}
        cam['raycast'] = {'grid_s': tg, 'rays': len(P), 'rays_s': tr}
        for cell in (0.002, 0.005, 0.010):
            cam[f'decimate_{cell * 1000:.0f}mm'] = OC.decimation_report(so, o, P, cell, 512)
        res['cameras'][s.name] = cam
    # per joint vector, as Phase 3 will: pose the arm, build the occluders (digest), one map per camera
    nq = 5 if quick else 20
    Q = [m.clip(m.home + rng.normal(0, 0.3, m.nq)) for _ in range(nq)]
    for be in (('gl', 'numpy') if st['available'] else ('numpy',)):
        if be == 'numpy' and quick:
            Q2 = Q[:2]
        else:
            Q2 = Q if be == 'gl' else Q[:5]
        tp, tc = [], []
        for q in Q2:
            t = time.perf_counter()
            qq = {robot.uid: q}
            occq = SG.scene_occluders(p, qq)
            occq.digest()
            tp.append(time.perf_counter() - t)
            t = time.perf_counter()
            for s in p.variant.sensors:
                o = SG.sensor_world_pose(p, s, qq).position
                OC.CubeMap.build(SG.sensor_occluders(p, s, qq, base=occq), o, 512, be)
            tc.append(time.perf_counter() - t)
        res[f'per_q_{be}_512'] = {'joint_vectors': len(Q2), 'cameras': len(p.variant.sensors),
                                  'pose_and_digest_s_median': float(np.median(tp)),
                                  'cubes_s_median': float(np.median(tc)),
                                  'total_s_median': float(np.median(np.add(tp, tc)))}
    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    path = os.path.join(ROOT, 'out', 'occlusion-measure.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
