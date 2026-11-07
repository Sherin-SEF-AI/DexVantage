"""The assistant's past conversations, kept on this computer per project (store.py:
<data>/assistant/<project>/chats/<id>.json, readable by the user alone), saved after every answer. A conversation keeps
what the dock shows (the questions, the answers, the tools called, the number-check flags), its title (the first
question), the provider and model, and when it started and last changed. Nothing here is ever sent as it is:
continuing a conversation starts a new one with a short hand-over summary of it (routing.handover_text).

Ported from Vantage assistant/chats.py: one folder per project instead of one folder with a project field."""
from __future__ import annotations

import os
import re
import time
import uuid

from . import store as ST

MAX_CHATS = 200                 # per project; the oldest beyond this are removed
TITLE_MAX = 90


def chat_dir(project) -> str:
    return ST.project_dir(project, 'chats')


def new_id() -> str:
    return time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def _path(project, cid: str) -> str:
    if not re.fullmatch(r'[0-9A-Za-z_-]{6,64}', cid or ''):
        raise ValueError(f'not a chat id: {cid!r}')
    return os.path.join(chat_dir(project), f'{cid}.json')


def title_of(question: str) -> str:
    t = ' '.join(str(question or '').split())
    return t if len(t) <= TITLE_MAX else t[:TITLE_MAX - 1] + '…'


def save(project, cid: str, *, title: str, items: list, provider: str = '', model: str = '',
         started: float | None = None) -> dict | None:
    """Write or update a conversation; None when there is nothing to keep. items: the dock's transcript, a list of
    {'role': 'user' | 'assistant' | 'tool' | 'note', 'text', ...} (masked by the caller)."""
    if not any(str(i.get('text') or '').strip() for i in items or [] if isinstance(i, dict)):
        return None
    old = load(project, cid) or {}
    rec = {'schema': 'dexvantage.assistant_chat/1', 'id': cid, 'title': title or old.get('title') or 'Conversation',
           'project': getattr(project, 'name', ''), 'provider': provider, 'model': model,
           'started': started or old.get('started') or time.time(), 'updated': time.time(), 'items': list(items)}
    ST.write_json(_path(project, cid), rec)
    _prune(project)
    return rec


def load(project, cid: str) -> dict | None:
    try:
        d = ST.read_json(_path(project, cid))
    except ValueError:
        return None
    return d if isinstance(d, dict) and d.get('id') == cid else None


def listing(project) -> list:
    """This project's conversations, newest first (without their transcripts)."""
    try:
        names = os.listdir(chat_dir(project))
    except OSError:
        return []
    out = []
    for n in names:
        if n.endswith('.json'):
            rec = load(project, n[:-5])
            if rec is not None:
                out.append({k: v for k, v in rec.items() if k != 'items'})
    return sorted(out, key=lambda r: -float(r.get('updated', 0)))


def delete(project, cid: str) -> bool:
    try:
        os.remove(_path(project, cid))
        return True
    except (OSError, ValueError):
        return False


def markdown(rec: dict) -> str:
    """A saved conversation as Markdown (for Save conversation… and the hand-over summary)."""
    lines = []
    for it in rec.get('items') or []:
        role, text = it.get('role'), str(it.get('text') or '')
        if role == 'user':
            lines += ['### You', '', text, '']
        elif role == 'assistant':
            lines += ['### Assistant', '', text, '']
            if it.get('flag'):
                lines += [f'> {it["flag"]}', '']
    return '\n'.join(lines)


def _prune(project):
    d = chat_dir(project)
    try:
        files = sorted((os.path.getmtime(os.path.join(d, n)), n) for n in os.listdir(d) if n.endswith('.json'))
    except OSError:
        return
    for _, n in files[:max(0, len(files) - MAX_CHATS)]:
        try:
            os.remove(os.path.join(d, n))
        except OSError:
            pass
