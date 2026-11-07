"""The readiness scorecard and the variants compare as studies (spec §7.11, §11, §13), so the Requirements page runs
them in the background and batch mode writes them with provenance:

    @jobs.study('requirements', 'Readiness scorecard', quick=False, batch='scorecard')          → --scorecard OUT.json
    @jobs.study('variants_compare', 'Variants compare', quick=False, batch='variants-compare')  → --variants-compare

Both read the project's own requirements (project.requirements) and run each study they name once (core/requirements
.py, its shared result cache). batch --check requirements.yaml evaluates a requirements file instead and is the CI
gate (exit 0 pass, 1 fail, 2 error).

Known limits: those of core/requirements.py (the active variant for the scorecard; bounds decide in one direction
only; no priorities) and of every study a requirement names.
"""
from __future__ import annotations

from .. import jobs
from .. import requirements as RQ

STUDY_ID = 'requirements'
VARIANTS_ID = 'variants_compare'


@jobs.study(STUDY_ID, 'Readiness scorecard', quick=False, batch='scorecard', needs_robot=False)
def scorecard(project, ctx: jobs.Ctx) -> dict:
    """Every requirement of the project against its study's figure, and the verdict."""
    return RQ.check(project, None, ctx=ctx)


@jobs.study(VARIANTS_ID, 'Variants compare', quick=False, batch='variants-compare', needs_robot=False)
def variants_compare(project, ctx: jobs.Ctx) -> dict:
    """Every variant against every requirement, and the best variant per requirement."""
    return RQ.compare_variants(project, None, ctx=ctx)
