"""The sim-to-real gap (research tools part 2): one recorded episode rendered again through the project's rig in MuJoCo
and compared with the recording camera by camera: photometric gaps (mean, contrast, Earth mover's distance of each
colour channel's histogram, sharpness, noise), geometric gaps (the robot silhouette IoU, the mount offset from the
silhouette calibration), depth (the predicted valid-pixel share, and the measured one with the depth error on the
robot where a reader decodes depth), timing (latency and jitter against the time-sync study's prediction) and a kernel
MMD between the recorded and rendered frames (not FID); then a ranked list of what explains the gap (Shapley shares of
the image difference over the corrections) with the change that closes each. core/sim2real/gap.py has the method.

    @jobs.study('sim2real', 'Sim-to-real gap', quick=False, batch='sim2real', needs_robot=True)
                                                            → --sim2real OUT.json --dataset PATH
    run(project, ctx) -> dict

Settings: project.analysis.studies['sim2real'] (core/sim2real/gap.DEFAULTS: dataset, episode, cameras, frames, width,
lag_max_s, lag_step_s, mount, the status thresholds, appearance — the render's camera response per sensor that Apply
sets). The dataset is the setting 'dataset', else the audit's, else the project's first dataset entry (as the other
data studies: core/studies/audit.dataset_path_for); in batch, --dataset PATH.

Known limits: those of core/sim2real/gap.py (the render is the project's model of the cell, not a photograph of it:
the corrections explain the part of the gap they can; one episode; a constant latency; the mount from the silhouettes
of static cameras only; depth compared only when a reader decodes it; Shapley shares are attributions over the
corrections tried, not measurements of physical causes).
"""
from __future__ import annotations

from .. import jobs

SID = 'sim2real'


@jobs.study(SID, 'Sim-to-real gap', quick=False, batch='sim2real', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """The sim-to-real gap of the dataset's episode (module docstring)."""
    from ..sim2real import gap as G
    from . import audit as AU
    path = AU.dataset_path_for(project, SID)
    st = {k: v for k, v in (project.analysis.studies.get(SID) or {}).items() if k != 'fingerprint'}
    st['dataset'] = path
    doc = G.analyse(project, path, st, progress=ctx.report, cancel=ctx.cancelled)
    doc['fingerprint'] = (project.analysis.studies.get(SID) or {}).get('fingerprint')
    return doc
