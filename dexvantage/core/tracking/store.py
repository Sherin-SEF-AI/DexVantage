"""The run store: a folder of logged runs, safe for several writers at once.

    <store>/
      store.json               {'schema': 'dexvantage.runstore/1', 'created_utc'} (written once)
      runs.jsonl               the index: one JSON object per line, appended under a file lock
                               {'op': 'add', 'run': {the run record without its samples}}
                               {'op': 'update', 'run_id', 'tags', 'note', 'at'}
                               {'op': 'delete', 'run_id', 'at'}
      runs/<run_id>/run.json   the full run record (core/tracking/record.py, schema dexvantage.run/1)
      runs/<run_id>/result.json  the result the run wrote, with its provenance block
      deleted/<run_id>/        a deleted run's folder (kept: deleting is reversible by hand)
      .lock                    the lock file appends and updates take

How a run is added: its folder is written under a temporary name inside runs/ and renamed to runs/<run_id> in one
step (rename within a folder is atomic on POSIX and on NTFS), then its index line is appended under an exclusive lock
(fcntl.flock on POSIX, msvcrt.locking on Windows) with a single write and an fsync. A writer that dies between the two
leaves a folder the index does not name: readers take such folders from their run.json ("recovered"), so no run is
lost. A writer that dies inside its append leaves a line without its newline: the next writer starts on a new line
and readers skip what does not parse ("damaged lines", counted). run.json is rewritten only by update() (tags and
note), under the same lock, by write-then-rename. Nothing is ever edited in place.

Run ids are uuid4 hex strings (32 characters): unique without coordination between writers, and valid MLflow run
ids (core/tracking/mlruns.py).

Known limits: the lock is advisory (another program that ignores it can still write); a store on a network file
system is only as safe as that file system's flock and rename (NFS v4 honours both; SMB shares may not); the index
is read whole on every listing (some 2 kB a run: 10 000 runs read in about 0.1 s).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import secrets
import shutil
import time

from ..jobs import StudyError

try:                                     # POSIX
    import fcntl
except ImportError:                      # pragma: no cover - Windows
    fcntl = None
try:                                     # Windows
    import msvcrt
except ImportError:
    msvcrt = None

STORE_SCHEMA = 'dexvantage.runstore/1'
INDEX = 'runs.jsonl'
RUNS = 'runs'
DELETED = 'deleted'
LOCK = '.lock'
META = 'store.json'
RUN_FILE = 'run.json'
RESULT_FILE = 'result.json'
LOCK_TIMEOUT_S = 60.0
INDEX_OMIT = ('samples',)                # kept in run.json only (they can be long)


class StoreError(StudyError):
    """The run store cannot do what was asked; the message says why in a plain sentence."""


def utc_now() -> str:
    """Now in UTC as 'YYYY-MM-DDTHH:MM:SS.mmmZ' (strings of this form sort as the times they name)."""
    return utc_text(_dt.datetime.now(_dt.timezone.utc))


def utc_text(t: _dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.astimezone()
    t = t.astimezone(_dt.timezone.utc)
    return t.strftime('%Y-%m-%dT%H:%M:%S.') + f'{t.microsecond // 1000:03d}Z'


def parse_utc(s) -> _dt.datetime | None:
    """A time stamp the store wrote (or any ISO 8601 text; a date alone is its midnight UTC) as an aware datetime."""
    if isinstance(s, _dt.datetime):
        return s if s.tzinfo else s.astimezone()
    if isinstance(s, _dt.date):
        return _dt.datetime(s.year, s.month, s.day, tzinfo=_dt.timezone.utc)
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().replace('Z', '+00:00')
    try:
        d = _dt.datetime.fromisoformat(t)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


def epoch_ms(s) -> int | None:
    d = parse_utc(s)
    return None if d is None else int(round(d.timestamp() * 1000))


def new_run_id() -> str:
    import uuid
    return uuid.uuid4().hex


def _fsync_dir(path: str):
    if os.name != 'posix':
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: str, data: bytes):
    """Write a file by writing a temporary file beside it, fsync, and renaming it over the target."""
    folder = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(folder, f'.{os.path.basename(path)}.{os.getpid()}.{secrets.token_hex(4)}.part')
    with open(tmp, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def dumps(o, indent=None) -> bytes:
    return (json.dumps(o, ensure_ascii=False, allow_nan=False, indent=indent, separators=None if indent else
                       (',', ':')) + '\n').encode('utf-8')


@contextlib.contextmanager
def file_lock(path: str, timeout: float = LOCK_TIMEOUT_S):
    """An exclusive advisory lock on path (created if missing), held for the with block."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    locked = False
    try:
        t0 = time.monotonic()
        if fcntl is not None:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError:
                    if time.monotonic() - t0 > timeout:
                        raise StoreError(f'The run store is locked by another writer for over {timeout:.0f} s '
                                         f'({os.path.basename(os.path.dirname(path))}): try again, or remove its '
                                         '.lock file if no DexVantage process is using it.') from None
                    time.sleep(0.002)
        elif msvcrt is not None:         # pragma: no cover - Windows
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() - t0 > timeout:
                        raise StoreError('The run store is locked by another writer: try again.') from None
                    time.sleep(0.005)
        yield
    finally:
        if locked:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        os.close(fd)


def _valid_id(run_id) -> bool:
    return isinstance(run_id, str) and 8 <= len(run_id) <= 64 and all(c in '0123456789abcdef' for c in run_id)


class RunStore:
    """A run store folder (created on first write). Every method is safe to call from several processes at once."""

    def __init__(self, folder: str):
        if not folder:
            raise StoreError('No run store folder was given.')
        self.folder = os.path.abspath(os.fspath(folder))

    # ---------------------------------------------------------------- paths
    @property
    def index_path(self) -> str:
        return os.path.join(self.folder, INDEX)

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.folder, RUNS)

    def run_dir(self, run_id: str) -> str:
        if not _valid_id(run_id):
            raise StoreError(f'{run_id!r} is not a run id (32 hexadecimal characters).')
        return os.path.join(self.runs_dir, run_id)

    def result_path(self, run_id: str) -> str:
        return os.path.join(self.run_dir(run_id), RESULT_FILE)

    @property
    def name(self) -> str:
        return os.path.basename(self.folder.rstrip(os.sep)) or self.folder

    def exists(self) -> bool:
        return os.path.isfile(self.index_path) or os.path.isdir(self.runs_dir)

    def _ensure(self):
        try:
            os.makedirs(self.runs_dir, exist_ok=True)
        except OSError as e:
            raise StoreError(f'Could not create the run store {self.name}: {e.strerror or e}. Choose a folder you '
                             'can write to.') from None
        meta = os.path.join(self.folder, META)
        if not os.path.exists(meta):
            try:
                write_atomic(meta, dumps({'schema': STORE_SCHEMA, 'created_utc': utc_now(),
                                          'layout': 'runs.jsonl index + runs/<run_id>/{run.json,result.json}'},
                                         indent=1))
            except OSError:
                pass                     # another writer made it at the same moment; harmless

    # ---------------------------------------------------------------- writing
    def _append(self, event: dict, locked: bool = False):
        """Append one index line under the lock (a single write, fsync'd), after a newline if the file's last line
        was left without one. locked: the caller holds the lock already."""
        if not locked:
            with file_lock(os.path.join(self.folder, LOCK)):
                return self._append(event, locked=True)
        line = dumps(event)
        fd = os.open(self.index_path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size:
                os.lseek(fd, size - 1, os.SEEK_SET)
                if os.read(fd, 1) != b'\n':
                    line = b'\n' + line
            n = os.write(fd, line)
            while n < len(line):                             # a short write (rare): finish it, still under lock
                n += os.write(fd, line[n:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def add(self, record: dict, result: dict | None = None) -> dict:
        """Store a run: its folder (run.json, result.json) then its index line. Returns the record as stored."""
        rid = record.get('run_id') or new_run_id()
        if not _valid_id(rid):
            raise StoreError(f'{rid!r} is not a run id (32 hexadecimal characters).')
        rec = dict(record, run_id=rid)
        rec.setdefault('stored_utc', utc_now())
        if result is not None:
            rec['result'] = dict(rec.get('result') or {}, file=RESULT_FILE)
        self._ensure()
        final = self.run_dir(rid)
        if os.path.exists(final):
            raise StoreError(f'The run store already has a run {rid}.')
        tmp = os.path.join(self.runs_dir, f'.tmp-{rid}-{os.getpid()}-{secrets.token_hex(3)}')
        try:
            os.makedirs(tmp)
            if result is not None:
                write_atomic(os.path.join(tmp, RESULT_FILE), dumps(result))
            write_atomic(os.path.join(tmp, RUN_FILE), dumps(rec, indent=1))
            os.rename(tmp, final)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise StoreError(f'Could not write the run to {self.name}: {e.strerror or e}.') from None
        except (TypeError, ValueError) as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise StoreError(f'The run could not be written as JSON ({e}).') from None
        _fsync_dir(self.runs_dir)
        self._append({'op': 'add', 'run': {k: v for k, v in rec.items() if k not in INDEX_OMIT}})
        return rec

    def update(self, run_id: str, tags=None, note=None, add_tags=None, remove_tags=None) -> dict:
        """Change a run's tags and note (the only fields that change after a run is logged). Returns the record."""
        folder = self.run_dir(run_id)
        path = os.path.join(folder, RUN_FILE)
        with file_lock(os.path.join(self.folder, LOCK)):
            try:
                with open(path, encoding='utf-8') as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                raise StoreError(f'The run store {self.name} has no readable run {run_id}.') from None
            t = list(rec.get('tags') or []) if tags is None else clean_tags(tags)
            t += [x for x in clean_tags(add_tags or []) if x not in t]
            t = [x for x in t if x not in set(clean_tags(remove_tags or []))]
            rec['tags'] = t
            if note is not None:
                rec['note'] = str(note)
            write_atomic(path, dumps(rec, indent=1))
            self._append({'op': 'update', 'run_id': run_id, 'tags': rec['tags'], 'note': rec.get('note', ''),
                          'at': utc_now()}, locked=True)
        return rec

    def delete(self, run_id: str):
        """Take a run out of the store (its folder moves to deleted/)."""
        src = self.run_dir(run_id)
        if not os.path.isdir(src):
            raise StoreError(f'The run store {self.name} has no run {run_id}.')
        dst_dir = os.path.join(self.folder, DELETED)
        os.makedirs(dst_dir, exist_ok=True)
        with file_lock(os.path.join(self.folder, LOCK)):
            # checked again under the lock: another writer may have deleted it since (two deletes at once)
            if not os.path.isdir(src):
                raise StoreError(f'The run store {self.name} has no run {run_id} (another writer deleted it).')
            dst = os.path.join(dst_dir, run_id)
            if os.path.exists(dst):                        # an earlier deletion of a run with this id: keep both
                dst = f'{dst}.{secrets.token_hex(4)}'
            try:
                os.replace(src, dst)
            except OSError as e:
                raise StoreError(f'Could not delete run {run_id} from {self.name}: {e.strerror or e}.') from None
            self._append({'op': 'delete', 'run_id': run_id, 'at': utc_now()}, locked=True)

    # ---------------------------------------------------------------- reading
    def _events(self) -> tuple[list, int]:
        try:
            with open(self.index_path, 'rb') as fh:
                data = fh.read()
        except FileNotFoundError:
            return [], 0
        except OSError as e:
            raise StoreError(f'Could not read the run store {self.name}: {e.strerror or e}.') from None
        out, bad = [], 0
        parts = data.split(b'\n')
        for k, ln in enumerate(parts):
            if not ln.strip():
                continue
            try:
                ev = json.loads(ln.decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                if k < len(parts) - 1:      # a whole line that does not parse; a last one may be mid-write
                    bad += 1
                continue
            if isinstance(ev, dict):
                out.append(ev)
            else:
                bad += 1
        return out, bad

    def scan(self) -> dict:
        """{'runs': {run_id: record}, 'damaged_lines', 'recovered' (folders the index does not name), 'missing'
        (indexed runs whose folder is gone)}."""
        events, bad = self._events()
        runs: dict = {}
        for ev in events:
            op = ev.get('op')
            if op == 'add' and isinstance(ev.get('run'), dict) and _valid_id(ev['run'].get('run_id')):
                runs[ev['run']['run_id']] = dict(ev['run'])
            elif op == 'update' and ev.get('run_id') in runs:
                r = runs[ev['run_id']]
                if isinstance(ev.get('tags'), list):
                    r['tags'] = list(ev['tags'])
                if 'note' in ev:
                    r['note'] = ev['note']
            elif op == 'delete':
                runs.pop(ev.get('run_id'), None)
        deleted = {ev.get('run_id') for ev in events if ev.get('op') == 'delete'}
        recovered = 0
        on_disk = set()
        try:
            names = os.listdir(self.runs_dir)
        except OSError:
            names = []
        for n in names:
            if not _valid_id(n):
                continue
            on_disk.add(n)
            if n in runs or n in deleted:
                continue
            try:
                with open(os.path.join(self.runs_dir, n, RUN_FILE), encoding='utf-8') as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and rec.get('run_id') == n:
                runs[n] = {k: v for k, v in rec.items() if k not in INDEX_OMIT}
                recovered += 1
        missing = [r for r in runs if r not in on_disk]
        for r in missing:
            runs[r]['folder_missing'] = True
        return {'runs': runs, 'damaged_lines': bad, 'recovered': recovered, 'missing': len(missing)}

    def runs(self, study=None, tag=None, kind=None, since=None, until=None, status=None, project=None) -> list:
        """The runs (index records), oldest first, filtered: study (an id or a list of ids), tag (one tag or a list:
        a run carrying any of them), kind, status, project (its name), since / until (a datetime, a date or ISO
        text; a date alone for until means the whole of that day)."""
        rows = list(self.scan()['runs'].values())
        return filter_runs(rows, study=study, tag=tag, kind=kind, since=since, until=until, status=status,
                           project=project)

    def get(self, run_id: str) -> dict:
        """The full record of a run (run.json, with its samples)."""
        folder = self.run_dir(run_id)                       # StoreError for what is not a run id
        try:
            with open(os.path.join(folder, RUN_FILE), encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            rec = self.scan()['runs'].get(run_id)
            if rec is None:
                raise StoreError(f'The run store {self.name} has no run {run_id}.') from None
            return rec

    def result(self, run_id: str) -> dict | None:
        """The result a run wrote (with its provenance), or None when the run kept none."""
        path = self.result_path(run_id)
        try:
            with open(path, encoding='utf-8') as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            raise StoreError(f'The result of run {run_id} could not be read ({getattr(e, "strerror", None) or e}).') \
                from None

    def fingerprint(self) -> str:
        """Changes whenever a run is added, updated or deleted (the index's size and modification time)."""
        try:
            st = os.stat(self.index_path)
            n = len(os.listdir(self.runs_dir))
        except OSError:
            return 'empty'
        return f'{st.st_size}-{st.st_mtime_ns}-{n}'


def clean_tags(tags) -> list:
    """Tags as a list of short trimmed strings, without repeats (a comma-separated text is split; a comma inside a
    tag becomes a space, so the tables can list tags comma-separated)."""
    if isinstance(tags, str):
        tags = tags.split(',')
    out = []
    for t in tags or []:
        t = ' '.join(str(t).replace(',', ' ').split())[:64]
        if t and t not in out:
            out.append(t)
    return out


def _until(v):
    d = parse_utc(v)
    if d is None:
        return None
    if isinstance(v, str) and len(v.strip()) == 10 or (isinstance(v, _dt.date) and not isinstance(v, _dt.datetime)):
        d = d + _dt.timedelta(days=1) - _dt.timedelta(microseconds=1)
    return d


def filter_runs(rows, study=None, tag=None, kind=None, since=None, until=None, status=None, project=None) -> list:
    def as_set(v):
        if v in (None, '', [], ()):
            return None
        return {v} if isinstance(v, str) else set(v)
    st, tg, kd, ss, pj = as_set(study), as_set(tag), as_set(kind), as_set(status), as_set(project)
    lo = parse_utc(since) if since not in (None, '') else None
    hi = _until(until) if until not in (None, '') else None
    if since not in (None, '') and lo is None:
        raise StoreError(f'since={since!r} is not a date or time (write it as 2026-11-07 or 2026-11-07T14:00).')
    if until not in (None, '') and hi is None:
        raise StoreError(f'until={until!r} is not a date or time (write it as 2026-11-07 or 2026-11-07T14:00).')
    out = []
    for r in rows:
        if st is not None and r.get('study') not in st:
            continue
        if kd is not None and r.get('kind') not in kd:
            continue
        if ss is not None and r.get('status') not in ss:
            continue
        if pj is not None and (r.get('project') or {}).get('name') not in pj:
            continue
        if tg is not None and not (set(r.get('tags') or []) & tg):
            continue
        if lo is not None or hi is not None:
            t = parse_utc(r.get('started_utc'))
            if t is None or (lo is not None and t < lo) or (hi is not None and t > hi):
                continue
        out.append(r)
    # by start time; runs that started in the same millisecond keep the store's order (the order they were added:
    # a stable sort), not their random run ids, which put a later run first when two quick runs shared a timestamp
    out.sort(key=lambda r: str(r.get('started_utc') or ''))
    return out
