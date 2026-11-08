"""dv.splat: splat views from Python (core/sim/splat_views.py, spec §9.3; no Qt on this path; needs the splat extra).

    import dexvantage as dv
    rig = dv.sample('so100')
    res = dv.splat.fit(rig, dataset='episodes/', iterations=1500)    # a Result: the splat_views study's document
    res.headline['text']        # 'Held-out views: PSNR … dB, SSIM …, LPIPS … (… views the fit did not use; …)'
    res = dv.splat.fit(rig)                                            # no dataset: a MuJoCo render (a demonstration)
    out = dv.splat.replay(rig, 'out/', dataset='episodes/', target_offset_m=[0, 0, 0.02])   # the replay's manifest
    out = dv.splat.augment(rig, 'out/', dataset='episodes/', augment_n=8)                   # a LeRobot dataset
    srig = dv.splat.rig(rig, dataset='episodes/')                      # the fitted SplatRig (core/sim/splat_rig.py)
    dv.splat.load('cell.splat.npz'), dv.splat.available(), dv.splat.lpips_state(), dv.splat.fetch_lpips_weights()
    dv.splat.settings()                                                # the settings and their defaults

Settings are core/sim/splat_views.DEFAULTS (each a labelled default), laid over the rig's stored ones for this call
only; an unknown name is refused in one sentence. Every render is synthetic, and every file says so.
fetch_lpips_weights() downloads AlexNet's ImageNet weights (244 MB, download.pytorch.org) for LPIPS: call it only
when you mean to.

Known limits: those of core/sim/splat_views.py; the calls run in this thread (pass progress= to see them move).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:                                    # names for the type hints only (nothing imported at run time)
    from .api import Result, Rig
    from .core.project import Project
    from .core.sim.splat_rig import SplatRig


def _project(rig: Rig | Project, dataset, settings: dict) -> Project:
    from .api import DexVantageError, Rig
    from .core.project import Project
    from .core.sim import splat_views as SV
    if isinstance(rig, Rig):
        p = rig.project.snapshot()
    elif isinstance(rig, Project):
        p = rig.snapshot()
    else:
        raise DexVantageError('Pass a dv.Rig (dv.open, dv.sample, dv.new) or a DexVantage Project.')
    from .core import study_settings as SST
    SST.check(SV.SID, settings)
    st = dict(p.analysis.studies.get(SV.SID) or {})
    st.update(settings)
    if dataset is not None:
        st['dataset'] = os.fspath(dataset)
    p.analysis.studies[SV.SID] = st
    return p


class _Splat:
    """dv.splat: fit, replay, augment, rig, load, available, lpips_state, fetch_lpips_weights, settings."""

    def fit(self, rig: Rig | Project, dataset: str | os.PathLike | None = None,
            progress: Callable[[float, str], None] | None = None, **settings: object) -> Result:
        """The splat_views study (fit the rig or reuse the cached fit; its quality on held-out frames; the
        before/after preview of the target camera) on rig with settings laid over the stored ones."""
        from .api import _CACHE, Result, _autolog
        from .core import jobs
        from .core.sim import splat_views as SV
        p = _project(rig, dataset, settings)
        doc = jobs.run_study(SV.SID, p, ctx=jobs.Ctx(progress=progress), cache=_CACHE)
        _autolog(SV.SID, doc, p)
        return Result(doc, SV.SID, project=p)

    def replay(self, rig: Rig | Project, out_dir: str | os.PathLike, dataset: str | os.PathLike | None = None,
               progress: Callable[[float, str], None] | None = None, **settings: object) -> dict:
        """The episode re-rendered at its logged q(t) through the proposed camera: out_dir/splat_replay_synthetic/
        (a video read back after writing, per-frame images, the manifest). Returns the manifest."""
        from .core.sim import splat_views as SV
        return SV.replay(_project(rig, dataset, settings), os.fspath(out_dir), progress=progress)

    def augment(self, rig: Rig | Project, out_dir: str | os.PathLike, dataset: str | os.PathLike | None = None,
                progress: Callable[[float, str], None] | None = None, **settings: object) -> dict:
        """augment_n copies of the episode through §8.5's perturbed extrinsics, as a LeRobot dataset in
        out_dir/splat_augment_synthetic/ (every render marked synthetic). Returns the manifest."""
        from .core.sim import splat_views as SV
        return SV.augment(_project(rig, dataset, settings), os.fspath(out_dir), progress=progress)

    def rig(self, rig: Rig | Project, dataset: str | os.PathLike | None = None,
            progress: Callable[[float, str], None] | None = None, **settings: object) -> SplatRig:
        """The fitted SplatRig itself (fitted now, or the cached fit for the same inputs)."""
        from .core.sim import splat_views as SV
        return SV.get_rig(_project(rig, dataset, settings), progress=progress)[0]

    def load(self, path: str | os.PathLike) -> SplatRig:
        """A splat rig a fit saved (.npz and its .json sidecar)."""
        from .api import DexVantageError
        from .core.sim.splat_rig import SplatRig
        try:
            return SplatRig.load(os.fspath(path))
        except ValueError as e:
            raise DexVantageError(str(e)) from None

    def available(self) -> str | None:
        """None when the splat extra is installed; else the sentence saying it is not."""
        from .core.sim import splat_views as SV
        return SV.requirement()

    def lpips_state(self) -> str:
        """'ready', 'no package' or 'no weights' (AlexNet's weights are not in the cache)."""
        from .core.sim import splat_views as SV
        return SV.lpips_state()

    def fetch_lpips_weights(self) -> str:
        """Download AlexNet's ImageNet weights for LPIPS into the cache (network, 244 MB); returns the path."""
        from .core.sim import splat_views as SV
        return SV.fetch_lpips_weights(progress=True)

    def settings(self) -> dict:
        """The settings and their defaults (core/sim/splat_views.DEFAULTS)."""
        from .core.sim import splat_views as SV
        return dict(SV.DEFAULTS)

    def __repr__(self) -> str:
        return '<dv.splat: fit, replay, augment, rig, load, available, lpips_state, fetch_lpips_weights, settings>'


splat = _Splat()
