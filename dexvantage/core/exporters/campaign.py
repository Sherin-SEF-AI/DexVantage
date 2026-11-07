"""The data-collection campaign plan as files (core/campaign.py): campaign/plan.json (the study's result), plan.csv
(one row per recording block), plan.ics (an iCalendar file, RFC 5545: one event per session with its checklist) and
plan.html (a printable page, one day per printed page).

Options: 'spec' (a campaign spec laid over the project's stored one for this export), 'start_date' (an ISO date for
the calendar when the plan has none; default the export's date, said in the notes), 'folder' (default 'campaign').
The round trip of the calendar is checked in validation/checks_campaign.py (campaign.ics_roundtrip: a strict
parser written there reads every event back).
"""
from __future__ import annotations

import json
import os

from ..jobs import StudyError
from . import exporter


@exporter('campaign', 'Campaign plan (CSV, iCalendar, printable HTML)', needs_robot=False,
          description='The data-collection campaign plan: the counterbalanced schedule over operators and days as '
                      'CSV, an iCalendar file with one event per session and its checklist, and a printable HTML '
                      'page, with the plan\'s JSON.')
def export_campaign(project, out_dir, options, ctx) -> dict:
    from .. import campaign as CP
    from .. import jobs
    snap = project.snapshot()
    if options.get('spec') is not None:
        if not isinstance(options['spec'], dict):
            raise StudyError('The campaign export\'s spec option must be a campaign spec object.')
        stored = {k: v for k, v in (snap.analysis.studies.get(CP.STUDY_ID) or {}).items() if k != 'plan'}
        snap.analysis.studies[CP.STUDY_ID] = {**stored, **CP.read_spec(options['spec'])}
    doc = jobs.run_study(CP.STUDY_ID, snap)
    doc = {k: v for k, v in doc.items() if k != 'cached'}
    folder = str(options.get('folder') or 'campaign')
    os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    base = os.path.join(out_dir, folder, 'plan')
    with open(base + '.json', 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, allow_nan=False)
        fh.write('\n')
    CP.write_files(doc, base, options.get('start_date'))
    notes = [f'{doc["headline"]["text"]} ({doc["headline"]["status_word"]}).']
    if not doc['totals'].get('first_date') and not options.get('start_date'):
        notes.append('The plan has no start date: the calendar counts its days from the export\'s date (set '
                     'start_date in the spec).')
    notes.append('Calendar times are floating local times (no time zone).')
    return {'files': [f'{folder}/plan.{x}' for x in ('json', 'csv', 'ics', 'html')], 'notes': notes}
