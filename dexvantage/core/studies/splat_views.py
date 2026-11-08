"""Splat views (spec §9.3, Phase 12): a robot-aligned 3D Gaussian Splatting rig fitted to a recorded LeRobot episode
(or to a MuJoCo render of the project, a demonstration labelled synthetic), its reconstruction quality on frames the
fit did not use (PSNR, SSIM and LPIPS on held-out views and held-out joint states), and the before/after view of a
proposed camera pose. core/sim/splat_views.py has the method; replay (a video of the episode through the proposed
camera) and augment (perturbed copies as a LeRobot dataset) write files from the same rig.

    @jobs.study('splat_views', 'Splat views', quick=False, batch='splat-views', needs_robot=True)
                                                            → --splat-views OUT.json [--dataset PATH]
    run(project, ctx) -> dict

Settings: project.analysis.studies['splat_views'] (core/sim/splat_views.DEFAULTS, each a labelled default: source,
dataset, episode, the capture and fit settings, the target camera and its proposed pose, LPIPS, the status
thresholds). It needs the splat extra (PyTorch) and, unless the device is set to cpu, a CUDA device; it refuses in one
sentence otherwise.

Known limits: those of core/sim/splat_views.py and core/sim/splat_rig.py (no densification, no view-dependent colour
or lighting model; poses from the planned mounts at the logged q; quality falls for joint states and camera poses far
from the capture, on specular surfaces and in regions no view saw; LPIPS needs downloaded AlexNet weights).
"""
from __future__ import annotations

from .. import jobs
from ..sim.splat_views import DEFAULTS  # noqa: F401 — the settings the study reads (core/study_settings.py)

SID = 'splat_views'


@jobs.study(SID, 'Splat views', quick=False, batch='splat-views', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """The splat rig's quality and the proposed camera's before/after view (module docstring)."""
    from ..sim import splat_views as SV
    return SV.analyse(project, progress=ctx.report, cancel=ctx.cancelled)
