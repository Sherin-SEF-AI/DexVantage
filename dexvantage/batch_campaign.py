"""Batch mode for the data-collection campaign planner (core/campaign.py):

    dexvantage PROJECT.dexv --campaign SPEC.json [--campaign-out OUT.json]

plans the campaign spec (schema dexvantage.campaign_spec/1, docs/models/campaign.md) laid over the project's stored
one and writes OUT.json (the plan with provenance), OUT.csv, OUT.ics and OUT.html beside it (default SPEC.plan.*).
Exit 0 when the plan is within the budget or no budget is set, 1 when it is over budget, 2 when the spec cannot be
planned (one sentence on stderr). --campaign-saved OUT.json (core/jobs) plans the spec stored in the project.
"""
from __future__ import annotations

import sys


def add_arguments(g):
    g.add_argument('--campaign', metavar='SPEC.json', help='plan a data-collection campaign (schema '
                   'dexvantage.campaign_spec/1, docs/models/campaign.md) on the project; writes the plan as JSON with '
                   'provenance, CSV, an iCalendar .ics and a printable HTML (core/campaign.py)')
    g.add_argument('--campaign-out', metavar='OUT.json', help='with --campaign: the plan file (default '
                   'SPEC.plan.json beside the spec; .csv, .ics and .html go beside it)')


def wanted(args) -> bool:
    return bool(getattr(args, 'campaign', None) or getattr(args, 'campaign_out', None))


def run(args, project) -> int:
    from .batch import EXIT_ERROR, EXIT_FAIL, EXIT_OK, _err, _unexpected
    from .core import campaign as CP
    if not getattr(args, 'campaign', None):
        return _err('--campaign-out goes with --campaign SPEC.json')
    if project is None:
        return _err('--campaign needs a project file (or --sample)')
    try:
        doc, paths = CP.batch(project, args.campaign, getattr(args, 'campaign_out', None))
    except CP.CampaignError as e:
        from . import batch_track
        batch_track.record(CP.STUDY_ID, None, project, reason=str(e))      # --track: a failed run
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        return _unexpected('The campaign planner', e)
    from . import batch_track
    batch_track.record(CP.STUDY_ID, doc, project, paths[0] if paths else None)   # --track: a run
    h = doc['headline']
    print(f'Campaign: {h["text"]} ({h["status_word"]}) → {", ".join(paths)}')
    for a in doc.get('advice') or []:
        print(f'  · {a}')
    sys.stdout.flush()
    return EXIT_FAIL if h['status'] == 'critical' else (EXIT_OK if paths else EXIT_ERROR)
