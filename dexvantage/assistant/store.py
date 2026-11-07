"""Where the assistant keeps what it remembers on this computer, per project: under XDG_DATA_HOME/DexVantage/assistant
(on every platform when XDG_DATA_HOME is set; else ~/.local/share/DexVantage/assistant), one folder per project:

    <data>/assistant/<project key>/chats/<chat id>.json      the conversations (chats.py)
    <data>/assistant/<project key>/memory.json               the memory notes (memory.py)
    <data>/assistant/<project key>/copilot.json              the Copilot's plans and approval records (copilot.py)

The project key is the project's file name and a short hash of its path (a saved project), or of its name and creation
time (a project not saved yet), so two projects with the same name do not share a chat history. Files are written
atomically and readable only by the user (0600, folders 0700). No keys or machine paths go into them: the transcripts
are masked (agent.mask) before they are saved."""
from __future__ import annotations

import hashlib
import json
import os
import re


def data_root() -> str:
    base = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(base, 'DexVantage', 'assistant')


def _slug(s: str) -> str:
    t = re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')
    return (t or 'untitled')[:40]


def project_key(project) -> str:
    """A folder name for the project's chats, memory and plans."""
    path = str(getattr(project, 'path', '') or '')
    name = str(getattr(project, 'name', '') or 'Untitled')
    ident = f'path:{os.path.abspath(path)}' if path else f'unsaved:{name}:{getattr(project, "created", "")}'
    stem = os.path.splitext(os.path.basename(path))[0] if path else name
    return f'{_slug(stem)}-{hashlib.sha256(ident.encode("utf-8")).hexdigest()[:10]}'


def project_dir(project, *parts) -> str:
    return os.path.join(data_root(), project_key(project), *parts)


def read_json(path: str, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path: str, doc) -> str:
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + '.part'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, default=str)
        fh.write('\n')
    os.replace(tmp, path)
    return path
