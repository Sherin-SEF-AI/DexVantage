"""Optimisation (spec §7.11): an evolutionary multi-objective optimiser over the project's factors, and the design
agent that builds complete rigs from the library for the project's requirements.

    nsga2.py        NSGA-II in NumPy: non-dominated sorting, crowding distance, SBX and polynomial mutation for
                    continuous genes, uniform crossover and mutation for categorical ones, constrained domination, a
                    budget in evaluations, a seed, progress and cancel; the ZDT1, ZDT2 and CONSTR benchmarks and IGD
    objectives.py   the objectives as core.metrics figures of merit (fallback definitions for the ids
                    core/metric_defs.py does not define) and the requirements as constraints
    search.py       scoring a design (memo, shared result cache, screening settings), choosing the options from the
                    front, the explanation of each option's trade-offs
    rig.py          the optimiser: NSGA-II over core.factors of the current rig (mount shifts and turns, enabled,
                    device), locked sensors never changed; study 'optimise' (batch --optimise)
    designer.py     the design agent: requirements in, a Pareto set of complete rigs out (library devices on static
                    frames and robot links), each explained, nothing applied until accepted; study 'design_agent'
                    (batch --design-agent)

The studies are registered by core/studies/optimise.py; the page is ui/pages/optimise.py; the validation checks are
validation/checks_optimise.py; the model is written up in docs/models/optimise.md.

Known limits: the front is an approximation found by a seeded evolutionary search within a budget of evaluations,
not a proof of optimality, and its quality falls as the number of objectives grows beyond three (NSGA-II's crowding
distance); each design is scored by the studies themselves, so the search inherits their modelling limits, and at the
screening settings (fewer grasps, no bracket meshes) its figures carry the sampling error of those settings unless
'confirm' re-scores the options shown; a figure a study does not state ranks a design below every design that states
it, and a requirement it does not state counts as missed; the cost is the sum of stated device prices only; the design
agent's mount sites are a sphere around the task volume and a cone around the gripper's approach axis.
"""
from __future__ import annotations

from . import nsga2  # noqa: F401
