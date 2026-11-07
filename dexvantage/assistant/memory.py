"""The assistant's memory notes for a project: decisions, constraints, rejected ideas and notes, kept on this computer
(store.py: <data>/assistant/<project>/memory.json) so the assistant remembers between sessions and does not suggest
again what the user turned down. The user sees every note in the dock's Memory view and can delete any of them.

An entry: {id, kind: decision | rejected | constraint | note, text, date: YYYY-MM-DD, source: user | assistant
(proposed by the assistant and accepted by the user)}.

Who writes it:
- the user, in the Memory view (add, delete);
- the assistant only proposes (the remember tool): the note is written when the user accepts it in the dock;
- discarding a proposed edit writes a 'rejected' note by itself ("rejected: wrist camera pitch +10.0°"), so the
  assistant is told not to propose it again.

What reads it: the model gets the notes with each question (prompt_block, compact and capped). No Qt, no network.

Ported from Vantage assistant/memory.py (the design log): the same kinds, cleaning and prompt block; kept beside the
chats instead of in the project file, and without Vantage's camera-specific change matching."""
from __future__ import annotations

import re
import time
import uuid

from . import store as ST

KINDS = ('decision', 'rejected', 'constraint', 'note')
SOURCES = ('user', 'assistant')
KIND_LABEL = {'decision': 'Decision', 'rejected': 'Rejected', 'constraint': 'Constraint', 'note': 'Note'}
TEXT_MAX = 400
PROMPT_CAP = 2400               # characters of the notes sent with a question
MAX_ENTRIES = 500
ORDER = {'constraint': 0, 'rejected': 1, 'decision': 2, 'note': 3}
INSTRUCTIONS = ('[Memory notes for this project, kept by the user. Respect every constraint. Do not propose a change '
                'that was rejected unless the user asks for it again. Decisions stand unless the user reopens them.]')


def clean(d) -> dict | None:
    """One entry, made safe (known kind and source, text trimmed, a date); None when junk."""
    if not isinstance(d, dict):
        return None
    text = ' '.join(str(d.get('text') or '').split())[:TEXT_MAX]
    if not text:
        return None
    date = str(d.get('date') or '')[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        date = time.strftime('%Y-%m-%d')
    return {'id': str(d.get('id') or '')[:40] or uuid.uuid4().hex[:12],
            'kind': d.get('kind') if d.get('kind') in KINDS else 'note', 'text': text, 'date': date,
            'source': d.get('source') if d.get('source') in SOURCES else 'user'}


def new_entry(kind, text, source='user') -> dict | None:
    return clean({'kind': kind, 'text': text, 'source': source})


def path(project) -> str:
    return ST.project_dir(project, 'memory.json')


def entries(project) -> list:
    """The notes, cleaned (a damaged file never breaks the dock), oldest first."""
    doc = ST.read_json(path(project), {}) or {}
    raw = doc.get('entries') if isinstance(doc, dict) else None
    out, seen = [], set()
    for d in raw if isinstance(raw, list) else []:
        e = clean(d)
        if e is not None and e['id'] not in seen:
            seen.add(e['id'])
            out.append(e)
    return out


def _write(project, log):
    ST.write_json(path(project), {'schema': 'dexvantage.assistant_memory/1', 'project': getattr(project, 'name', ''),
                                  'entries': log[-MAX_ENTRIES:]})


def add(project, entry: dict) -> dict | None:
    e = clean(entry)
    if e is None:
        return None
    log = entries(project)
    if any(x['id'] == e['id'] for x in log):
        e['id'] = uuid.uuid4().hex[:12]
    log.append(e)
    _write(project, log)
    return e


def remove(project, eid) -> bool:
    log = entries(project)
    keep = [e for e in log if e['id'] != eid]
    if len(keep) == len(log):
        return False
    _write(project, keep)
    return True


def clear(project):
    _write(project, [])


def prompt_text(project, cap=PROMPT_CAP) -> str:
    """The notes for the model: constraints, rejected ideas, decisions, notes (newest first within a kind), one line
    each, capped; '' when there are none."""
    log = entries(project)
    if not log:
        return ''
    order = sorted(range(len(log)), key=lambda i: (ORDER[log[i]['kind']], -i))
    lines, used = [], 0
    for n, i in enumerate(order):
        e = log[i]
        line = f'- {e["kind"]} ({e["date"]}): {e["text"]}'
        if used + len(line) + 1 > cap:
            left = len(order) - n
            lines.append(f'- … {left} older note{"s" if left != 1 else ""} not shown')
            break
        lines.append(line)
        used += len(line) + 1
    return '\n'.join(lines)


def prompt_block(project, cap=PROMPT_CAP) -> str:
    body = prompt_text(project, cap)
    return f'MEMORY\n{INSTRUCTIONS}\n{body}' if body else ''


def rejected_entry(proposal) -> dict | None:
    """The 'rejected' note for a discarded edit proposal: "rejected: wrist cam z +20 mm — "Raise the wrist camera"."."""
    rows = list(getattr(proposal, 'rows', []) or [])
    if not rows:
        return None
    bits = []
    for r in rows[:6]:
        a, b, unit = r.get('before'), r.get('after'), r.get('unit', '')
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            d = float(b) - float(a)
            bits.append(f'{r.get("title", r.get("factor"))} {d * 1000:+.0f} mm' if unit == 'm' else
                        f'{r.get("title", r.get("factor"))} {d:+.1f}{unit}')
        else:
            bits.append(f'{r.get("title", r.get("factor"))} → {b}')
    text = 'rejected: ' + ', '.join(bits)
    title = str(getattr(proposal, 'title', '') or '').strip()
    if title:
        text += f' — "{title[:80]}"'
    return new_entry('rejected', text, 'user')
