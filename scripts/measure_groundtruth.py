"""Measure MuJoCo render throughput of the ground-truth renderer (core/sim/render.py) on the UR5e sample.

    python scripts/measure_groundtruth.py [--device auto|N] [--views 120] [--lens pinhole|brown|equidistant]

Renders every camera of the sample at its stream size at random joint states (seeded), segmentation + depth + RGB,
in calls of 8 views, and prints: the GPU (OpenGL renderer string), frames per second inside the renderer (MuJoCo's
render calls alone), and views per second end to end (the worker round trip, the transfer of the images and the
remap and owner look-up in this process). --lens swaps every camera's lens model for a distorted one of the same
size (to time the remap path).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main() -> int:
    import numpy as np

    from dexvantage.core.optics import CameraModel
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim.render import Renderer
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--views', type=int, default=120)
    ap.add_argument('--lens', default='pinhole', choices=('pinhole', 'brown', 'equidistant'))
    a = ap.parse_args()
    dev = a.device if a.device == 'auto' else int(a.device)
    p = sample_project('ur5e')
    r0 = p.robots[0]
    m = get_model(p, r0.uid)
    rng = np.random.default_rng(0)
    with Renderer(p, device=dev) as R:
        info = R.info
        views = []
        for k in range(a.views):
            s = p.variant.sensors[k % len(p.variant.sensors)]
            q = np.clip(m.home + rng.normal(0, 0.3, m.nq), m.lower, m.upper)
            v = {'sensor_uid': s.uid, 'q': {r0.uid: q}, 'outputs': ('seg', 'depth', 'rgb')}
            if a.lens != 'pinhole':
                c = R.build.camera(s.uid).model
                v['camera_model'] = (CameraModel(c.width, c.height, c.fx, c.fy, c.cx, c.cy, 'brown',
                                                 (-0.2, 0.05, 0, 0, 0)) if a.lens == 'brown' else
                                     CameraModel(c.width, c.height, c.fx * 0.5, c.fy * 0.5, c.cx, c.cy,
                                                 'equidistant'))
            views.append(v)
        R.render(views[:2])                              # warm-up: the worker, the model, the renderers
        f0, s0 = R.stats['frames'], R.stats['render_seconds']
        t0 = time.perf_counter()
        for i in range(0, len(views), 8):
            R.render(views[i:i + 8])
        dt = time.perf_counter() - t0
        frames, rs = R.stats['frames'] - f0, R.stats['render_seconds'] - s0
    sizes = sorted({f'{R.build.camera(s.uid).model.width} × {R.build.camera(s.uid).model.height}'
                    for s in p.variant.sensors})
    print(f'GPU: {info["renderer"]} ({info["device"]}), MuJoCo {info["mujoco"]}')
    print(f'{len(views)} views ({", ".join(sizes)}; lens {a.lens}), {frames} frames (segmentation, depth, RGB)')
    print(f'renderer: {frames / rs:.0f} frames/s; end to end: {len(views) / dt:.1f} views/s ({dt:.2f} s)')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RuntimeError as e:                    # RenderUnavailable: said in words
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from None
