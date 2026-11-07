"""The data-collection campaign study: the campaign spec saved in the project (project.analysis.studies['campaign'],
the layout of core/campaign.py) planned on the project: conditions, the counterbalanced schedule over operators and
days, total hours and storage, the expected episodes per condition and a checklist per session.

The model, the design and its references are in core/campaign.py; this module registers the study so the Data ▸
Campaign page, batch mode (--campaign-saved OUT.json; --campaign SPEC.json plans a spec file and writes the CSV, the
iCalendar file and the printable HTML beside its JSON), --reproduce, the metrics and the Python API reach it.

Known limits
------------
Those of core/campaign.py: episode and reset times are taken as fixed; operators as equally fast; the fatigue limits
are defaults until set; position and carry-over balance are exact only for the numbers of sequences the result
states; days are filled greedily; a compressed format's storage is the raw size without a measured ratio; a sensor
without a stated data rate adds nothing (the storage is then "at least"); times are local and without a time zone;
the power analysis is for two independent proportions.
"""
from __future__ import annotations

from .. import campaign as CP
from .. import jobs


@jobs.study(CP.STUDY_ID, 'Data-collection campaign', quick=False, batch='campaign-saved')
def campaign(project, ctx: jobs.Ctx) -> dict:
    return CP.plan(project, None, ctx)
