"""Runs for Weights & Biases: a JSON file of the runs and a script that replays them through W&B's official Python
API (wandb.init, run.log, run.summary, run.finish), offline by default.

W&B's offline run folders (wandb/offline-run-*/run-*.wandb) are an internal format of the wandb client: they are not
written here. Instead export() writes

    wandb/wandb_runs.json    {'schema': 'dexvantage.wandb_replay/1', 'project', 'runs': [{'id', 'name', 'group',
                             'job_type', 'tags', 'notes', 'config', 'history', 'summary', 'units'}]}
    wandb/replay_wandb.py    a standalone script (no DexVantage import): python replay_wandb.py [--project P]
                             [--entity E] [--online] [--dir D]

and the script makes one W&B run per DexVantage run: wandb.init(project=…, entity=…, id=<the DexVantage run id>,
name=…, notes=<the note>, tags=<the tags>, config=<study, kind, project, variant, settings, seeds, provenance>,
group=<study>, job_type=<kind>, dir=…, mode='offline' unless --online, reinit='finish_previous', resume='allow'),
then run.log(<the figures>) once (one step), run.summary.update(<the figures>) and run.finish(). With
WANDB_MODE=offline (the default here) the wandb client writes its own offline folders, which `wandb sync` uploads
later. The parameters are those of wandb.init in wandb 0.30.0's wandb/sdk/wandb_init.py (read at
https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_init.py on 2026-11-07: entity, project, dir, id, name,
notes, tags, config, group, job_type, mode "online" / "offline" / "disabled", reinit "finish_previous", resume
"allow") and Run.log / Run.summary / Run.finish in wandb/sdk/wandb_run.py at the same tag.

wandb is an optional extra (pip install "dexvantage[tracking]"); writing the export never needs it, running the
script does (it says so in a sentence and exits 2 when wandb is missing).

Known limits: a figure not stated is left out of the history and the summary (it is listed in config as
dexvantage.not_stated); units go in config, since W&B has no unit field; re-running the script on the same W&B project
resumes the runs with the same ids (resume='allow') and logs their figures again as a new step.
"""
from __future__ import annotations

import json
import os

from .store import dumps, write_atomic

SCHEMA = 'dexvantage.wandb_replay/1'
WANDB_VERSION_READ = '0.30.0'

SCRIPT = '''#!/usr/bin/env python3
"""Replay DexVantage runs into Weights & Biases through its official Python API (written by DexVantage).

    python replay_wandb.py [--project NAME] [--entity TEAM] [--online] [--dir FOLDER]

Each run of wandb_runs.json (beside this script) becomes one W&B run with the same id: wandb.init(...),
run.log(figures), run.summary.update(figures), run.finish(). Offline by default (mode="offline", the W&B client
writes its offline run folders under --dir, default ./wandb_offline; `wandb sync` uploads them later); --online logs
to the W&B server directly. Needs the wandb package: pip install wandb.
"""
import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay DexVantage runs into Weights & Biases")
    ap.add_argument("--project", default=None, help="the W&B project (default: the one in wandb_runs.json)")
    ap.add_argument("--entity", default=None, help="the W&B entity (team or user); default: yours")
    ap.add_argument("--online", action="store_true", help="log to the W&B server now instead of offline")
    ap.add_argument("--dir", default=None, help="where the W&B client writes its files (default ./wandb_offline)")
    ap.add_argument("--runs", default=None, help="the runs file (default: wandb_runs.json beside this script)")
    args = ap.parse_args(argv)
    path = args.runs or os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_runs.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"replay_wandb: could not read {os.path.basename(path)} ({e}).", file=sys.stderr)
        return 2
    try:
        import wandb
    except ImportError:
        print("replay_wandb: the wandb package is not installed: pip install wandb, then run this script again.",
              file=sys.stderr)
        return 2
    mode = "online" if args.online else "offline"
    if not args.online:
        os.environ["WANDB_MODE"] = "offline"
    folder = os.path.abspath(args.dir or "wandb_offline")
    os.makedirs(folder, exist_ok=True)
    project = args.project or data.get("project") or "dexvantage"
    done = 0
    for r in data.get("runs", []):
        run = wandb.init(project=project, entity=args.entity, id=r["id"], name=r.get("name"), notes=r.get("notes"),
                         tags=r.get("tags") or None, config=r.get("config") or {}, group=r.get("group"),
                         job_type=r.get("job_type"), dir=folder, mode=mode, reinit="finish_previous",
                         resume="allow")
        if r.get("history"):
            run.log(r["history"])
        run.summary.update(r.get("summary") or {})
        run.finish()
        done += 1
    print(f"replay_wandb: {done} run(s) replayed ({mode}) into the W&B project {project}; files under {folder}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def wandb_run(r: dict) -> dict:
    """One run as the script logs it."""
    ms = r.get('metrics') or {}
    figs = {k: m['value'] for k, m in sorted(ms.items()) if isinstance(m.get('value'), (int, float))}
    proj, var = r.get('project') or {}, r.get('variant') or {}
    prov = r.get('provenance') or {}
    dv = prov.get('dexvantage') or {}
    config = {'dexvantage.study': r.get('study'), 'dexvantage.kind': r.get('kind'), 'dexvantage.title': r.get('title'),
              'dexvantage.status': r.get('status'), 'dexvantage.interface': r.get('interface'),
              'dexvantage.started_utc': r.get('started_utc'), 'dexvantage.ended_utc': r.get('ended_utc'),
              'dexvantage.project': proj.get('name'), 'dexvantage.project_sha256': proj.get('sha256'),
              'dexvantage.variant': var.get('name'), 'dexvantage.variant_sha256': var.get('sha256'),
              'dexvantage.version': dv.get('version'), 'dexvantage.git': dv.get('git'),
              'dexvantage.runtime': prov.get('runtime'),
              'dexvantage.result_sha256': (r.get('result') or {}).get('sha256'),
              'dexvantage.headline': (r.get('headline') or {}).get('text'), 'dexvantage.rerun_of': r.get('rerun_of'),
              'dexvantage.not_stated': [k for k, m in sorted(ms.items()) if m.get('value') is None],
              'settings': r.get('settings') or {}, 'seeds': r.get('seeds') or {}}
    return {'id': r['run_id'], 'name': r.get('name') or r['run_id'], 'group': r.get('study'),
            'job_type': r.get('kind'), 'tags': list(r.get('tags') or []), 'notes': r.get('note') or '',
            'config': config, 'history': figs, 'summary': figs,
            'units': {k: m.get('unit') or '' for k, m in sorted(ms.items())}}


def export(runs, out_dir: str, project: str | None = None) -> dict:
    """Write wandb_runs.json and replay_wandb.py into out_dir; {'files', 'runs'}."""
    os.makedirs(out_dir, exist_ok=True)
    names = sorted({(r.get('project') or {}).get('name') or '' for r in runs} - {''})
    proj = project or ('dexvantage-' + '-'.join(n.lower().replace(' ', '-') for n in names[:1]) if names
                       else 'dexvantage')
    doc = {'schema': SCHEMA, 'project': ''.join(c if c.isalnum() or c in '-_.' else '-' for c in proj)[:128],
           'wandb_api_read': f'wandb {WANDB_VERSION_READ} wandb.init / Run.log / Run.summary / Run.finish',
           'runs': [wandb_run(r) for r in runs]}
    write_atomic(os.path.join(out_dir, 'wandb_runs.json'), dumps(doc, indent=1))
    write_atomic(os.path.join(out_dir, 'replay_wandb.py'), SCRIPT.encode('utf-8'))
    return {'files': ['wandb_runs.json', 'replay_wandb.py'], 'runs': len(runs), 'project': doc['project']}


def load(path: str) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
