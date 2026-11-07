"""A writer process for the run store's concurrency check (validation checks_tracking.py, tests/test_tracking.py):

    python -m dexvantage.core.tracking._stress STORE_DIR WRITER N [GO_FILE]

waits until GO_FILE exists (so every writer starts at once), then adds N synthetic runs to the store, updating the
tags of every third one right after, and prints {'writer', 'ids', 'start', 'end'} (wall-clock seconds) as JSON. A
synthetic run is shaped like a real record (core/tracking/record.py) without a project: its figures are known numbers
(writer × 1000 + i), so the check can tell a lost or corrupted run from a stored one.
"""
from __future__ import annotations

import json
import os
import sys
import time

from .store import RunStore, new_run_id, utc_now


def synthetic(writer: int, i: int) -> tuple[dict, dict]:
    v = float(writer * 1000 + i)
    rec = {'schema': 'dexvantage.run/1', 'run_id': new_run_id(), 'kind': 'study', 'study': 'synthetic',
           'title': 'Synthetic run', 'name': f'writer {writer} run {i}', 'interface': 'batch',
           'started_utc': utc_now(), 'ended_utc': utc_now(), 'seconds': 0.0, 'status': 'finished', 'reason': '',
           'project': {'name': f'writer {writer}', 'sha256': '0' * 64}, 'variant': None, 'settings': {'i': i},
           'seeds': {}, 'metrics': {'synthetic.value': {'metric': 'synthetic.value', 'stat': None,
                                                        'title': 'Synthetic value', 'unit': 'count', 'better': 'higher',
                                                        'value': v, 'status': 'ok'}},
           'headline': {'text': f'value {v:g}', 'status': 'good', 'word': 'ok'}, 'result': None,
           'provenance': {}, 'tags': [f'w{writer}'], 'note': f'writer {writer}, run {i}', 'rerun_of': None,
           'reproduce': None, 'samples': {'synthetic.value': [v, v + 0.5]}}
    result = {'schema': 'dexvantage.synthetic/1', 'writer': writer, 'i': i, 'value': v, 'payload': 'x' * 2048}
    return rec, result


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    folder, writer, n = argv[0], int(argv[1]), int(argv[2])
    go = argv[3] if len(argv) > 3 else None
    t0 = time.monotonic()
    while go and not os.path.exists(go) and time.monotonic() - t0 < 60:
        time.sleep(0.001)
    st = RunStore(folder)
    ids = []
    start = time.time()
    for i in range(n):
        rec, res = synthetic(writer, i)
        st.add(rec, res)
        ids.append(rec['run_id'])
        if i % 3 == 0:
            st.update(rec['run_id'], add_tags=[f'updated-{writer}'])
    sys.stdout.write(json.dumps({'writer': writer, 'ids': ids, 'start': start, 'end': time.time()}) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
