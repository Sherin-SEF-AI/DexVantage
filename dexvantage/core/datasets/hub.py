"""LeRobot datasets on the Hugging Face Hub, streamed file by file (spec §8.2: "Hub streaming"): the reader
(core/datasets/lerobot.py) opens a Hub dataset by its name and fetches only the files it needs, when it needs them.

    is_hub_ref(s) -> bool                 'hf://datasets/<org>/<name>[@<revision>]' or
                                          'https://huggingface.co/datasets/<org>/<name>[/tree/<revision>]'
    parse_hub_ref(s) -> (repo_id, revision) | None
    hub_ref(repo_id, revision='main') -> 'hf://datasets/<repo_id>@<revision>'
    HubMirror(repo_id, revision, endpoint=None, cache=None, opener=None)
        .sha, .root, .license, .fetch(rel, optional=False) -> local path, .list(prefix) -> [rel paths]
        .remote(rel) -> RemoteFile | None     a seekable file over HTTP range requests (None: no range support)

What is fetched, and when. Opening reads the dataset's revision (the commit it pins, so every file comes from one
commit), meta/info.json and the episode metadata (v2.1: meta/episodes.jsonl and tasks.jsonl; v3.0: the files under
meta/episodes/, meta/tasks.parquet); loading an episode fetches its data file; its cameras' videos are read in place
with HTTP range requests (RemoteFile: blocks of BLOCK bytes, the last MAX_BLOCKS kept in memory), so only the bytes of
that episode's frames and the MP4 index travel, not the whole packed v3.0 file (LeRobot's default rolls those over at
200 MB; measured on lerobot/svla_so100_pickplace at commit 728583b, 2026-11-07, over a home connection: opening
2.3 s; episode 10's two cameras 11.9 s and 33 MB by range requests, where their two packed files are 449 MB; then its
364 frames of one camera decoded in 0.6 s); a
server that ignores range requests gets the whole file fetched instead. The stats and the synthetic ground-truth
sidecar come only when asked for. Nothing else is downloaded. Files are kept in
XDG_CACHE_HOME/DexVantage/hub/<org>--<name>/<commit>/ (each written to a ".part" file and renamed when complete), so a
second read of the same commit is local. Only public datasets are read: no token is read or sent (a gated or private
dataset is refused with a sentence that says so).

The Hub's HTTP interface, as huggingface_hub uses it (read from github.com/huggingface/huggingface_hub at commit
fd5dccbb731e212df4f1dfa53b8f406b866070d3, main, 2026-11-05, read 2026-11-07: constants.py ENDPOINT (the HF_ENDPOINT
environment variable, else https://huggingface.co) and REPO_TYPES_URL_PREFIXES; file_download.py hf_hub_url (the
revision quoted with safe="", the filename with quote()); hf_api.py dataset_info and _iter_repo_tree_pages;
utils/_pagination.py, the "Link: <...>; rel="next"" header):

    {endpoint}/api/datasets/{repo_id}/revision/{revision}              JSON: 'sha', 'private', 'gated', 'cardData'
    {endpoint}/api/datasets/{repo_id}/tree/{revision}/{path}?recursive=true&expand=false
                                                                       JSON list of {'type': 'file' | 'directory',
                                                                       'path', 'size', ...}, paged by Link next
    {endpoint}/datasets/{repo_id}/resolve/{revision}/{path}            the file

Network errors are explained in plain words (DatasetError); tests pass a fake `opener` and never touch the network.

Known limits: streamed video bytes are not kept on disk (a second pass over an episode's frames reads them again,
unless they are still among the blocks in memory); LFS files are fetched through the resolve URL, which redirects to
the Hub's storage (urllib follows the redirect); a revision that is a branch is pinned to its commit at open, so a push
during a read does not mix commits; gated and private datasets are not read (no token).
"""
from __future__ import annotations

import collections
import io
import json
import os
import re
import socket
import threading
import ssl
import urllib.error
import urllib.parse
import urllib.request

from .episode import DatasetError

DEFAULT_ENDPOINT = 'https://huggingface.co'
TIMEOUT = 60.0                        # s per request
USER_AGENT = 'DexVantage (LeRobot dataset reader)'
SOURCE = ('github.com/huggingface/huggingface_hub at commit fd5dccbb731e212df4f1dfa53b8f406b866070d3 (main, '
          '2026-11-05), read 2026-11-07')
_REPO = r'([A-Za-z0-9][\w.\-]*/[\w.\-]+)'
_HF = re.compile(r'^hf://datasets/' + _REPO + r'(?:@([^\s/]+))?/?$')
_WEB = re.compile(r'^https?://[^/]+/datasets/' + _REPO + r'(?:/tree/([^\s/]+))?/?$')


def parse_hub_ref(s) -> tuple[str, str] | None:
    """(repo_id, revision) of a Hub dataset reference, or None when s is not one."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = _HF.match(s) or _WEB.match(s)
    if not m:
        return None
    return m.group(1), urllib.parse.unquote(m.group(2) or 'main')


def is_hub_ref(s) -> bool:
    return parse_hub_ref(s) is not None


def hub_ref(repo_id: str, revision: str = 'main') -> str:
    """The reference DexVantage stores for a Hub dataset: hf://datasets/<repo_id>@<revision>."""
    return f'hf://datasets/{repo_id.strip().strip("/")}' + (f'@{revision}' if revision and revision != 'main' else '')


def cache_dir() -> str:
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'DexVantage', 'hub')


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                  # noqa: BLE001 — the system's certificates then
        return ssl.create_default_context()


def _default_opener(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=TIMEOUT, context=_context())


BLOCK = 4 << 20                       # bytes per range request (a default of this build)
MAX_BLOCKS = 24                       # blocks kept in memory per mirror (96 MB)


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests of one repository file (for av.open)."""

    def __init__(self, mirror: 'HubMirror', rel: str, size: int):
        super().__init__()
        self.m, self.rel, self.size, self.pos = mirror, rel, int(size), 0
        self.name = rel

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + int(off))
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.m.read_range(self.rel, self.pos, n, self.size)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


class HubMirror:
    """One revision of a Hub dataset repository, fetched on demand into a local folder that mirrors the repository's
    layout (module docstring)."""

    def __init__(self, repo_id: str, revision: str = 'main', endpoint: str | None = None, cache: str | None = None,
                 opener=None):
        self.repo_id = repo_id.strip().strip('/')
        self.revision = revision or 'main'
        self.endpoint = (endpoint or os.environ.get('HF_ENDPOINT') or DEFAULT_ENDPOINT).rstrip('/')
        self._open = opener or _default_opener
        info = self._json(f'{self.endpoint}/api/datasets/{self.repo_id}/revision/'
                          f'{urllib.parse.quote(self.revision, safe="")}', 'the dataset\'s revision')
        if not isinstance(info, dict):
            raise DatasetError(f'The Hub did not describe the dataset {self.repo_id}.')
        if info.get('private') or info.get('gated'):
            raise DatasetError(f'The Hub dataset {self.repo_id} is ' + ('private' if info.get('private') else 'gated')
                               + ': DexVantage reads public datasets only (it sends no token). Download it with '
                                 'your own account (huggingface-cli download) and open the folder.')
        self.sha = str(info.get('sha') or self.revision)
        card = info.get('cardData') if isinstance(info.get('cardData'), dict) else {}
        lic = card.get('license')
        self.license = ', '.join(lic) if isinstance(lic, list) else (str(lic) if lic else None)
        self.root = os.path.join(cache or cache_dir(), self.repo_id.replace('/', '--'), self.sha)
        os.makedirs(self.root, exist_ok=True)
        self._sizes: dict = {}
        self._blocks: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self.bytes_ranged = 0                     # bytes read by range requests (for the tests and the docs)

    @property
    def ref(self) -> str:
        return hub_ref(self.repo_id, self.revision)

    # ---------------------------------------------------------------- HTTP, explained
    def _get(self, url: str, what: str, headers: dict | None = None):
        try:
            return self._open(url, headers) if headers else self._open(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise DatasetError(f'The Hub has no {what} for {self.repo_id} ({self.revision}): check the dataset\'s '
                                   'name and revision.') from None
            if e.code in (401, 403):
                raise DatasetError(f'The Hub refused {what} for {self.repo_id} ({e.code}): the dataset is private or '
                                   'gated, and DexVantage reads public datasets only.') from None
            raise DatasetError(f'The Hub answered {e.code} ({e.reason}) to {what} for {self.repo_id}. Try again '
                               'later.') from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ssl.SSLError, OSError) as e:
            reason = getattr(e, 'reason', e)
            host = urllib.parse.urlsplit(url).hostname or url
            if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(e, (socket.timeout, TimeoutError)):
                raise DatasetError(f'{host} did not answer within {TIMEOUT:.0f} s while fetching {what}. The '
                                   'connection may be slow or blocked; try again.') from None
            raise DatasetError(f'{host} could not be reached while fetching {what} ({reason}). Check the network '
                               'connection, or download the dataset and open its folder.') from None

    def _json(self, url: str, what: str):
        r = self._get(url, what)
        try:
            with r:
                return json.loads(r.read().decode('utf-8'))
        except ValueError:
            raise DatasetError(f'The Hub\'s answer for {what} of {self.repo_id} is not JSON.') from None

    def url(self, rel: str) -> str:
        return (f'{self.endpoint}/datasets/{self.repo_id}/resolve/{urllib.parse.quote(self.sha, safe="")}/'
                f'{urllib.parse.quote(rel)}')

    # ---------------------------------------------------------------- files
    def local(self, rel: str) -> str:
        rel = rel.replace('\\', '/').lstrip('/')
        if '..' in rel.split('/'):
            raise DatasetError(f'The dataset names a file outside itself ({rel}).')
        return os.path.join(self.root, *rel.split('/'))

    def fetch(self, rel: str, optional: bool = False) -> str | None:
        """The local path of the repository file rel, downloaded first when it is not in the cache. A file that is not
        there is a DatasetError, or None when optional."""
        dest = self.local(rel)
        if os.path.isfile(dest):
            return dest
        try:
            r = self._get(self.url(rel), rel)
        except DatasetError:
            if optional:
                return None
            raise
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + '.part'
        try:
            with r, open(tmp, 'wb') as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            os.replace(tmp, dest)
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise DatasetError(f'{rel} of {self.repo_id} could not be saved in the cache ({e.strerror or e}).') \
                from None
        return dest

    def remote(self, rel: str) -> RemoteFile | None:
        """rel as a seekable file read by HTTP range requests, or None when the server ignores range requests (then
        fetch() it whole)."""
        if rel not in self._sizes:
            r = self._get(self.url(rel), rel, {'Range': 'bytes=0-0'})
            with r:
                cr = r.headers.get('Content-Range', '') if getattr(r, 'headers', None) is not None else ''
                code = getattr(r, 'status', None) or getattr(r, 'code', None)
                m = re.match(r'bytes\s+0-0/(\d+)', cr or '')
                self._sizes[rel] = int(m.group(1)) if (m and code in (206, None)) else None
        size = self._sizes[rel]
        return RemoteFile(self, rel, size) if size else None

    def read_range(self, rel: str, start: int, n: int, size: int) -> bytes:
        """Bytes [start, start + n) of rel, through blocks of BLOCK bytes (the last MAX_BLOCKS kept)."""
        out = []
        b0, b1 = start // BLOCK, (start + n - 1) // BLOCK
        for b in range(b0, b1 + 1):
            key = (rel, b)
            with self._lock:
                data = self._blocks.get(key)
                if data is not None:
                    self._blocks.move_to_end(key)
            if data is None:
                lo, hi = b * BLOCK, min((b + 1) * BLOCK, size) - 1
                r = self._get(self.url(rel), rel, {'Range': f'bytes={lo}-{hi}'})
                with r:
                    data = r.read()
                if len(data) != hi - lo + 1:
                    raise DatasetError(f'The Hub sent {len(data)} bytes of {rel} where {hi - lo + 1} were asked for; '
                                       'try again.')
                with self._lock:
                    self.bytes_ranged += len(data)
                    self._blocks[key] = data
                    while len(self._blocks) > MAX_BLOCKS:
                        self._blocks.popitem(last=False)
            a = start - b * BLOCK if b == b0 else 0
            z = (start + n) - b * BLOCK if b == b1 else BLOCK
            out.append(data[a:z])
        return b''.join(out)

    def list(self, prefix: str) -> list[str]:
        """The repository's files under prefix (recursive), as paths relative to the repository."""
        url = (f'{self.endpoint}/api/datasets/{self.repo_id}/tree/{urllib.parse.quote(self.sha, safe="")}/'
               f'{urllib.parse.quote(prefix.strip("/"), safe="")}?recursive=true&expand=false')
        out = []
        for _ in range(10000):                                  # pages
            r = self._get(url, f'the file list of {prefix}')
            with r:
                try:
                    page = json.loads(r.read().decode('utf-8'))
                except ValueError:
                    raise DatasetError(f'The Hub\'s file list of {prefix} for {self.repo_id} is not JSON.') from None
                link = r.headers.get('Link', '') if getattr(r, 'headers', None) is not None else ''
            out += [str(x['path']) for x in page if isinstance(x, dict) and x.get('type') == 'file' and x.get('path')]
            m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link or '')
            if not m:
                break
            url = m.group(1)
        return sorted(out)
