"""The optimiser and the design agent as studies (spec §7.11), so the page, the batch mode and the result cache run
them like every other study.

    @jobs.study('optimise', 'Optimiser', quick=False, batch='optimise')           -> --optimise OUT.json
    @jobs.study('design_agent', 'Design agent', quick=False, batch='design-agent') -> --design-agent OUT.json

Settings: project.analysis.studies['optimise'] and ['design_agent'] (core/optimise/rig.py and designer.py list them,
each a labelled default unless set); in batch mode --optimise-settings JSON (text or a .json file) is merged over the
project's for that run.

What they compute: core/optimise (the NSGA-II search, the objectives and constraints, the options and their
explanations). The result: the objectives used and those left out with the reason, the constraints (the project's
requirements) and the requirements that could not be used, the factors or mount sites searched, the current rig's
figures, up to `options` designs spread along the Pareto front, each with its figures, the requirements it meets, what
it changes and a plain-English explanation of its trade-offs, every design scored (for the front's scatter), the
search's figures (evaluations, generations, seed, results reused) and a headline with a status word.

Known limits: see core/optimise/__init__.py: an approximate front within a budget, the studies' own limits, screening
settings unless confirm, unstated figures ranked last, the cost from stated prices only, the design agent's mount
sites a sphere and a cone.
"""
from __future__ import annotations

from .. import jobs


@jobs.study('optimise', 'Optimiser', quick=False, batch='optimise', needs_robot=False)
def run_optimise(project, ctx: jobs.Ctx) -> dict:
    from ..optimise import rig
    return rig.optimise(project, rig.settings_of(project), ctx)


@jobs.study('design_agent', 'Design agent', quick=False, batch='design-agent', needs_robot=False)
def run_design(project, ctx: jobs.Ctx) -> dict:
    from ..optimise import designer
    return designer.design(project, designer.settings_of(project), ctx)
