"""The canonical view study (spec §7.16): how well robot B's wrist camera can be warped to robot A's canonical
viewpoint through the fingertip contact plane, and where the warp stops being exact.

    @jobs.study('canonical_view', 'Canonical view', quick=False, batch='canonical-view', needs_robot=True)
        → --canonical-view OUT.json

Settings (project.analysis.studies['canonical_view'], every one optional): 'source' (a sensor uid: robot B's
camera), 'canonical' (a sensor uid, or a canonical view document {'camera', 'pose_in_contact', ...} as an export's
<stem>_canonical_view.json holds), 'q' ({robot uid: q}, default each robot's home), 'plane_offset_m' (default 0: the
plane through the fingertip centre), 'closing_sign_source' / 'closing_sign_canonical' (±1, default +1: which way the
contact frame's x axis runs), 'depth_range_m' ([beyond the plane (negative), towards the camera (positive)], default
from the task volume), 'thresholds_px' (the status words, DexVantage defaults good ≤ 1 px "aligned", warning ≤ 4 px
"mild parallax", serious ≤ 16 px "parallax", above that "large parallax"). Without 'source' and 'canonical' the
study picks a camera on a second robot's link and one on the primary robot's (core/exporters/canonical_warp.py
default_pair).

The result (JSON-able): both views (lens, pose in the contact frame, whether the camera is fixed in its gripper
frame), the homography H and its inverse with the convention, the relative pose and baseline, the contact plane in
each camera, the depth range used, the residual parallax at both depth extremes (max, P95, median over the canonical
pixels the warp fills) with a coarse map, the share of the canonical image the warp fills, the headline (P95 at the
worse extreme, px, with a status word) and the suggestion (mount the source camera at the canonical pose relative
to its fingertips: the parallax is then zero). The model, its equations and references are in
core/exporters/canonical_warp.py and docs/models/canonical_warp.md.

Known limits: exact only on the contact plane; the parallax is the error bound over the depth range, not a
correction; occlusion is not modelled (what B sees behind the fingers is warped as if it lay on the plane); the two
grippers' contact frames are identified by their measured TCP and axes (closing sign a user choice); cameras that
are not fixed in the gripper frame are related at the q given (home by default); the map is sampled on a grid of at
most 64 cells across (the export computes it at every pixel).
"""
from __future__ import annotations

from .. import jobs
from ..exporters import canonical_warp as CW

SID = CW.SID


@jobs.study(SID, 'Canonical view', quick=False, batch='canonical-view', needs_robot=True)
def canonical_view(project, ctx: jobs.Ctx) -> dict:
    """Robot B's wrist camera warped to the canonical viewpoint: the homography, its exactness on the contact plane
    and the residual parallax off it over the task volume's depth range."""
    ctx.report(0.05, 'the views')
    doc = CW.analyse(project, None, ctx)
    ctx.report(1.0, 'done')
    return doc
