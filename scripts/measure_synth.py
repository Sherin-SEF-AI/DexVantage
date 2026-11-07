"""Measure the synthetic-episode generator's throughput (core/sim/episodes.py) on a sample project.

    python scripts/measure_synth.py [--sample ur5e] [--episodes 4] [--frames 0] [--max-width 640] [--size WxH]
                                    [--version v3.0] [--device auto|0|1|2] [--no-faults] [--json OUT.json]

--frames 0 plays each motion at its own length; --device picks the EGL device MuJoCo renders on ('auto': the first
that initialises). Prints the renderer, the rows and camera frames, the time spent rendering (including the remap and
the transfer from the render worker) and the rest (encoding, Parquet, the sidecar), and the camera frames per second.
The dataset goes to a temporary folder that is deleted afterwards. Run it memory-capped, e.g. systemd-run --user
--scope -p MemoryMax=4G env QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/measure_synth.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sample', default='ur5e')
    ap.add_argument('--episodes', type=int, default=4)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--max-width', type=int, default=640)
    ap.add_argument('--size', default='')
    ap.add_argument('--version', default='v3.0')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-faults', action='store_true')
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import episodes as SE
    p = sample_project(a.sample)
    st = {'episodes': a.episodes, 'frames': a.frames or None, 'max_width': a.max_width, 'version': a.version,
          'device': a.device if a.device == 'auto' else int(a.device)}
    if a.size:
        st['size'] = [int(x) for x in a.size.lower().split('x')]
    if a.no_faults:
        st['faults'] = []
    with tempfile.TemporaryDirectory(prefix='dexvantage-synth-') as d:
        s = SE.generate(p, os.path.join(d, 'ds'), st)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _ds, fs in os.walk(os.path.join(d, 'ds')) for f in fs)
    cams = s['truth']['cameras']
    out = {'sample': a.sample, 'settings': st, 'renderer': s['renderer'], 'codec': s['codec'], 'fps': s['fps'],
           'episodes': s['episodes'], 'rows': s['rows'], 'camera_frames': s['camera_frames'], 'renders': s['renders'],
           'cameras': {k: f'{c["intrinsics"]["width"]} x {c["intrinsics"]["height"]}' for k, c in cams.items()},
           'seconds': s['seconds'], 'render_seconds': s['render_seconds'],
           'other_seconds': round(s['seconds'] - s['render_seconds'], 3), 'gpu_render_seconds': s['gpu_render_seconds'],
           'camera_frames_per_s': s['camera_frames_per_s'], 'renders_per_s': s['renders_per_s'],
           'dataset_mb': round(size / 1e6, 2)}
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
