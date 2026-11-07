"""The designed-experiments study (spec §7.11): the experiment saved in the project (project.analysis.studies
['experiments'], the design.json layout of core/experiments.py, schema 'dexvantage.experiment/1') run on the project.

The model, the designs, the estimators and their references are in core/experiments.py (ported from Vantage
core/experiments.py); this module registers the study so the Analysis dock, batch mode (--experiment-saved OUT.json;
--experiment DESIGN.json runs a design file) and --reproduce reach it.

Known limits
------------
Those of core/experiments.py: every response carries its own study's limits; Sobol indices assume independent
factors and need their intervals read; main-effect intervals treat the other factors as noise; a resolution III
fraction aliases main effects with two-factor interactions; runs a study refuses have no value and are left out of
that response's analysis; locked sensors are not varied. The study keeps the response values of up to
core.experiments.ROW_CACHE_SIZE runs between experiments (keyed by the study cache's keys of each run's project), so
Run after a cancel evaluates only the runs not done; a process pool (workers > 1) does not use it.
"""
from __future__ import annotations

from .. import experiments as EX
from .. import jobs

_CACHE = jobs.ResultCache(EX.CACHE_SIZE)
_ROWS = EX.RowCache()


@jobs.study(EX.STUDY_ID, 'Designed experiments', quick=False, batch='experiment-saved')
def experiments(project, ctx: jobs.Ctx) -> dict:
    st = EX.settings_of(project)
    if not st.get('factors'):
        raise EX.ExperimentError('The experiment has no factor yet: add one on Analysis ▸ Designed experiments (pick '
                                 'a quantity of the rig), or run a design file with --experiment DESIGN.json.')
    return EX.run(project, st, ctx, _CACHE, _ROWS)
