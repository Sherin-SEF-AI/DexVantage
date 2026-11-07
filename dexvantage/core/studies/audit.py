"""Dataset and episode audit (spec §8.2): a recorded or synthetic dataset (LeRobot v2.1 or v3.0, RLDS / Open
X-Embodiment, a ROS 2 bag in MCAP or SQLite: core/datasets) checked from its own data against the requirements, per
episode and over the campaign.

    @jobs.study('audit', 'Dataset audit', quick=False, batch='audit')   → --audit OUT.json --dataset PATH
    run(project, ctx) -> dict           the audit (core/audit/campaign.audit_dataset)
    dataset_path(project) -> str        the dataset the study audits: settings 'dataset' (a path), else the dataset
                                        entry 'dataset_uid' of project.data, else its first dataset
    use_dataset(project, path)          point the study at a dataset (the batch option --dataset)
    status_for(doc) -> (sev, word)

The checks (core/audit/episode.py for each episode: dropped and duplicated frames and the frame rate, USB resets and
gaps, timestamp jitter, camera–joint offset and drift, exposure and gain steps, motion blur, the gripper hidden in a
wrist camera, action–observation alignment; core/audit/campaign.py over the episodes: mount slip, a step against a
gradual drift by the model selection ported from Vantage core/online_calib.py) and their references are in those
modules' docstrings and docs/models/audit.md. Each requirement is a labelled default the user can change
(project.data['requirements']); each episode gets a verdict: pass, fix (only an offset or drift fails, which
re-stamping corrects), recalibrate (only the mount moved) or drop (drop it or re-record it).

Output (JSON-able): 'summary' (episodes, pass, the verdict lists, status, word, advice), 'requirements' (per
requirement: checks, failures, the worst value, the limit and whether it is a default), 'episodes' (every check of
every episode, with per-frame scores when per_frame is on), 'mounts' (per camera the mount estimate per episode, the
step-against-gradual classification and its events), 'offsets', 'slips', 'dataset' (format, version, counts, the
dataset folder's name only), 'cameras' (how each camera was matched to the plan), 'plan_notes', 'requirements_used',
'settings'.

Known limits: the checks read what the dataset stores: LeRobot resamples camera timestamps onto the nominal grid, so
its drops show as duplicates and its jitter is invisible; RLDS without an fps has step counts for times; the desync
estimate needs a robot that moves at varying speed and is resolved to about ±12 ms on static cameras (synthetic
measurement), not at all well on cameras on the robot over textureless scenes; the in-episode drift is not resolved
below several thousand ppm (real clocks drift tens of ppm: the campaign's per-episode offsets show such a drift);
blur is relative to neighbouring frames and not tested on cameras on the robot by default; the gripper check needs
the plan's mount and lens; mount slip by registration measures rotation only, relative to the first episode; the
plan side (predicted image motion, the gripper mask) needs the project's cameras to be the recorded ones.
"""
from __future__ import annotations

import os

from .. import jobs
from ..jobs import StudyError

SID = 'audit'
WORDS = {'good': 'ready', 'warning': 'mostly usable', 'serious': 'needs work', 'critical': 'not usable'}


def dataset_entry(project, uid: str | None = None) -> dict | None:
    ds = (getattr(project, 'data', None) or {}).get('datasets') or []
    if uid:
        return next((d for d in ds if d.get('uid') == uid), None)
    return ds[0] if ds else None


def dataset_path(project) -> str:
    return dataset_path_for(project, SID)


def dataset_path_for(project, sid: str) -> str:
    """The dataset study sid works on: its settings' 'dataset', else its 'dataset_uid' entry of project.data, else the
    audit's choice, else project.data's first dataset."""
    st = project.analysis.studies.get(sid) or {}
    au = project.analysis.studies.get(SID) or {}
    p = None
    for d in (st, au):
        p = d.get('dataset') or (dataset_entry(project, d['dataset_uid']) or {}).get('path') \
            if d.get('dataset_uid') and not d.get('dataset') else d.get('dataset')
        if p:
            break
    if not p:
        p = (dataset_entry(project) or {}).get('path')
    if not p:
        raise StudyError('Choose a dataset to audit first (Data ▸ Audit a dataset…, or --dataset PATH in batch).')
    from ..datasets.hub import is_hub_ref
    if not os.path.exists(p) and not is_hub_ref(p):
        raise StudyError(f'The dataset {os.path.basename(os.path.normpath(p))} is not there (nothing at that path): '
                         'choose it again, or check the path.')
    return p


def use_dataset(project, path: str, sid: str = SID):
    """Point study sid at the dataset at path (for one run: the batch option --dataset)."""
    d = dict(project.analysis.studies.get(sid) or {})
    d['dataset'] = path
    project.analysis.studies[sid] = d


def fingerprint(path: str) -> str:
    """A cheap digest of a dataset folder's files (names, sizes, modification times): the study's cache key follows a
    dataset that changed on disk."""
    import hashlib
    from ..datasets.hub import is_hub_ref
    h = hashlib.blake2b(digest_size=12)
    if is_hub_ref(path) and not os.path.exists(path):  # a Hub dataset: its name and revision (pinned when read)
        h.update(path.strip().encode())
        return h.hexdigest()
    if os.path.isfile(path):
        st = os.stat(path)
        h.update(f'{os.path.basename(path)}|{st.st_size}|{st.st_mtime_ns}'.encode())
        return h.hexdigest()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            try:
                st = os.stat(os.path.join(root, f))
            except OSError:
                continue
            h.update(f'{os.path.relpath(os.path.join(root, f), path)}|{st.st_size}|{st.st_mtime_ns}'.encode())
    return h.hexdigest()


def status_for(doc: dict) -> tuple[str, str]:
    s = doc.get('summary') or {}
    sev = s.get('status') or 'warning'
    return sev, WORDS.get(sev, s.get('word', ''))


@jobs.study(SID, 'Dataset audit', quick=False, batch='audit')
def run(project, ctx: jobs.Ctx) -> dict:
    """The dataset's episodes and campaign against the requirements (module docstring)."""
    from ..audit import campaign as AC
    from ..audit import episode as AE
    path = dataset_path(project)
    st = dict(project.analysis.studies.get(SID) or {})
    settings = {k: st[k] for k in AC.SETTINGS if k in st}
    req = AE.requirements_of(project, st.get('requirements'))
    doc = AC.audit_dataset(path, project, req, settings, progress=ctx.report, cancel=ctx.cancelled)
    doc['fingerprint'] = st.get('fingerprint')
    return doc
