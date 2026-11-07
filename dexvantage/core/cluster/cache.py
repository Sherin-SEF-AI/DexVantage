"""The shared on-disk result cache of the cluster sweeps: study results keyed by core.jobs.ResultCache's own key, kept
in a folder that every worker, shard and machine can reach (a job folder on a shared filesystem).

    DiskResultCache(folder, size)      a core.jobs.ResultCache: memory first, then the folder; put() writes both
    atomic_write(path, data)           write bytes to a unique temporary name beside path, fsync, rename over path
    prune(folder, older_than_s)        remove temporary files left by writers that were killed part way

The key. A result is stored under sha256 of the canonical JSON (core.provenance.canonical) of the ResultCache key:
(study id, the project's content hash, the study settings' hash, the library entries' hash, the active variant's
index, a recorded dataset's fingerprint). Any change to the project, the settings, a library entry the project names or
the dataset gives another key, so an entry is never stale; it is only ever written with the same content again.

The file. `<folder>/<first two hex digits>/<digest>.json` holds one header line (JSON: schema, the key, the payload's
length and sha256) and then the payload (the study's result as JSON). A reader checks the length and the sha256 before
it trusts the payload: a file that is short or corrupt (copied by hand, a disk that filled) reads as a miss, never as
a result.

Concurrent writers, without locks. A writer writes the whole file under a name no other writer uses (the digest, the
process id and a random UUID, in the same folder), flushes and fsyncs it, then renames it over the final name. rename(2)
replaces an existing name atomically ("there is no point at which another process attempting to access newpath will
find it missing"), so a reader sees either no file, an earlier complete file or this complete file, never a mixture.
Two writers of one key write the same bytes (the key is the content), so the last rename wins harmlessly. On NFS the
server performs the rename; rename(2)'s BUGS section warns that a retransmitted RPC can report a failure for a rename
that was done, so a failed rename whose target now exists is taken as done. NFS's close-to-open consistency (nfs(5): on
close the client writes back pending changes, on open it checks with the server) is what makes a file written and
closed on one node readable in full on another.

Why no flock: file locks are not dependable on the filesystems clusters share. On NFS, flock(2) is emulated with fcntl
byte-range locks since Linux 2.6.12 and needs the file open for writing; a mount with `nolock` gives locks that exclude
only processes on the same client (nfs(5)); over SMB, flock is not propagated before Linux 5.5 and is mandatory after
(flock(2)); parallel filesystems differ again. A lock that silently does not lock is worse than none, so correctness
here rests on unique temporary names and atomic renames alone.

Known limits: a result that does not survive a JSON round trip exactly (a tuple, an integer dictionary key, a NumPy
array) is kept in memory only, never written; the folder grows without bound (delete it to clear it: nothing else
depends on it); rename is atomic only within one filesystem, so the temporary file is always made beside its target;
filesystems that do not give close-to-open consistency (some object-store mounts) are not supported.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid

from .. import jobs

SCHEMA = 'dexvantage.cluster-cache/1'
TMP_SUFFIX = '.tmp'


def atomic_write(path: str, data: bytes) -> None:
    """Write data to path through a uniquely named temporary file in the same folder and an atomic rename."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f'.{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}{TMP_SUFFIX}')
    try:
        with open(tmp, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.replace(tmp, path)
        except FileNotFoundError:
            # NFS: a retransmitted rename RPC can fail after the server did the rename (rename(2), BUGS)
            if not os.path.exists(path):
                raise
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def prune(folder: str, older_than_s: float = 3600.0) -> int:
    """Remove temporary files older than older_than_s (writers killed part way); the count removed."""
    n = 0
    now = time.time()
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(TMP_SUFFIX):
                p = os.path.join(root, f)
                try:
                    if now - os.path.getmtime(p) > older_than_s:
                        os.remove(p)
                        n += 1
                except OSError:
                    pass
    return n


def same(a, b) -> bool:
    """True when b is a exactly (types included; NaN equals NaN; a tuple is not a list)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def encode(doc) -> bytes | None:
    """The payload bytes of doc, or None when doc does not survive a JSON round trip exactly."""
    try:
        text = json.dumps(doc, ensure_ascii=False, allow_nan=True, separators=(',', ':'))
    except (TypeError, ValueError):
        return None
    if not same(doc, json.loads(text)):
        return None
    return text.encode('utf-8')


def key_digest(key) -> str:
    from .. import provenance as PV
    return PV.sha256(PV.canonical(list(key)))


def write_entry(path: str, key, payload: bytes) -> None:
    from .. import provenance as PV
    head = {'schema': SCHEMA, 'key': list(key), 'bytes': len(payload), 'sha256': PV.sha256(payload)}
    atomic_write(path, json.dumps(head, separators=(',', ':')).encode('utf-8') + b'\n' + payload)


def read_entry(path: str, key=None):
    """The payload of a cache file as a Python object, or None when it is missing, short, corrupt or another key's."""
    from .. import provenance as PV
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError:
        return None
    nl = raw.find(b'\n')
    if nl < 0:
        return None
    try:
        head = json.loads(raw[:nl])
    except ValueError:
        return None
    payload = raw[nl + 1:]
    if not isinstance(head, dict) or head.get('schema') != SCHEMA or head.get('bytes') != len(payload) or \
            head.get('sha256') != PV.sha256(payload):
        return None
    if key is not None and head.get('key') != json.loads(PV.canonical(list(key))):
        return None
    try:
        return json.loads(payload)
    except ValueError:
        return None


class DiskResultCache(jobs.ResultCache):
    """core.jobs.ResultCache with a folder behind its memory: get() looks in memory, then on disk; put() keeps the
    result in memory and writes it to disk (when it survives JSON exactly). Safe for many writers at once on one
    machine or many (module docstring). hits / misses count both levels; disk_hits and disk_writes the folder's."""

    def __init__(self, folder: str, size: int = jobs.CACHE_SIZE):
        super().__init__(size)
        self.folder = os.fspath(folder)
        self.disk_hits = 0
        self.disk_writes = 0
        self.memory_only = 0

    def path(self, key) -> str:
        d = key_digest(key)
        return os.path.join(self.folder, d[:2], d + '.json')

    def get(self, key):
        hit = super().get(key)
        if hit is not None:
            return hit
        doc = read_entry(self.path(key), key)
        if doc is None:
            return None
        with self._lock:
            self.misses -= 1                       # the memory level counted a miss; the folder had it
            self.hits += 1
            self.disk_hits += 1
        super().put(key, doc)
        return doc

    def put(self, key, value):
        super().put(key, value)
        payload = encode(value)
        if payload is None:
            self.memory_only += 1
            return
        p = self.path(key)
        if os.path.exists(p) and read_entry(p, key) is not None:   # the same key is the same content: nothing to
            return                                                  # write (a short or corrupt file is replaced)
        try:
            write_entry(p, key, payload)
            self.disk_writes += 1
        except OSError:                            # a full or read-only folder: the result is still in memory
            self.memory_only += 1

    def stats(self) -> dict:
        return {'folder': self.folder, 'hits': self.hits, 'misses': self.misses, 'disk_hits': self.disk_hits,
                'disk_writes': self.disk_writes, 'memory_only': self.memory_only}
