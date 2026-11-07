"""MuJoCo Menagerie: list its robot models, download one with its own licence, import it (spec §1.2, §9.4).

    idx = index()                        {'models': [...], 'source': 'github' | 'offline', 'note': str}
    dl = download('unitree_g1')          MenagerieDownload: the folder, the LICENSE text, the commit, the MJCF files
    ... show dl.license_text and ask the user to accept it ...
    robot, sources = import_download(dl, accepted_on='2026-11-06')

Menagerie's licences differ per model, so nothing here assumes one. At commit 98a70c5 (every model folder's LICENSE
read 2026-11-06) the 70 models carry Apache-2.0, BSD-3-Clause, BSD-2-Clause, BSD-3-Clause-Clear or MIT and none is
non-commercial, but folders are added and changed, and other licences (Creative Commons, non-commercial terms) are
recognised and flagged in case one arrives. download() reads the model folder's own LICENSE file and returns its
text, and the UI shows it and asks for consent before import_download() reads the model into the project. A folder
without a LICENSE is refused. license_name() recognises the common licence texts so the project can record a short
name; anything it does not recognise is recorded as "see the LICENSE text", never guessed. The LICENSE travels with
the project ("<uid>/LICENSE" in its sources) and RobotSource.license names it, the commit and the date the user
accepted it.

The index comes from the GitHub API (the repository's top-level folders). Without a network connection, or when
GitHub refuses (its limit for requests without signing in is 60 an hour), index() falls back to OFFLINE_MODELS, a
list of folder names only, and says so. A download pins the commit it starts from (GET .../commits/main), lists the
folder's files from that commit's tree and fetches each from raw.githubusercontent.com, so every file comes from one
commit. Files go to XDG_DATA_HOME/DexVantage/menagerie/<model>/<commit>/ (written to a ".part" folder and renamed
when complete), and a second download of the same commit uses those files.

Network errors are explained in plain words (MenagerieError); tests pass a fake `opener` and never touch the network.

Known limits: Menagerie's gallery pictures (images at the top of a model folder) are not downloaded; models that need
MuJoCo plugins or <model> attachments may not import (core/robot/importer.py).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import posixpath
import re
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

REPO = 'google-deepmind/mujoco_menagerie'
WEB = f'https://github.com/{REPO}'
API = f'https://api.github.com/repos/{REPO}'
RAW = f'https://raw.githubusercontent.com/{REPO}'
BRANCH = 'main'
TIMEOUT = 30.0                        # s per request
# a model folder larger than this is refused (a default of this build). At commit 98a70c5 (read 2026-11-06) the
# largest folder, pndbotics_adam_lite, is 178 MB, then flybody (144 MB) and rainbow_robotics_rby1 (127 MB).
MAX_DOWNLOAD_MB = 300.0
_NONCOMMERCIAL = re.compile(r'\bnon[- ]?commercial\b')
MARKER = '.dexvantage-download.json'
USER_AGENT = 'DexVantage (MuJoCo Menagerie import)'

# Folder names of Menagerie robot models, for when GitHub cannot be reached (read from the repository's top level on
# 2026-11-06 at commit 98a70c5). Names only: what each model is and its licence come from the download itself.
OFFLINE_MODELS = (
    'agilex_piper', 'agility_cassie', 'aloha', 'anybotics_anymal_b', 'anybotics_anymal_c', 'apptronik_apollo',
    'arx_l5', 'berkeley_humanoid', 'bitcraze_crazyflie_2', 'booster_t1', 'boston_dynamics_spot', 'dynamixel_2r',
    'fanuc_crx5ia', 'flexiv_rizon4', 'flexiv_rizon4s', 'flybody', 'fourier_n1', 'franka_emika_panda', 'franka_fr3',
    'franka_fr3_v2', 'google_barkour_v0', 'google_barkour_vb', 'google_robot', 'hello_robot_stretch',
    'hello_robot_stretch_3', 'i2rt_yam', 'iit_softfoot', 'kinova_gen3', 'kuka_iiwa_14', 'leap_hand',
    'low_cost_robot_arm', 'ms_human_700', 'pal_talos', 'pal_tiago', 'pal_tiago_dual', 'pndbotics_adam_lite',
    'rainbow_robotics_rby1', 'realsense_d435i', 'rethink_robotics_sawyer', 'robot_soccer_kit', 'robotiq_2f85',
    'robotiq_2f85_v4', 'robotis_omy_3m', 'robotis_op3', 'robotstudio_so101', 'seeed_rebot_devarm', 'shadow_dexee',
    'shadow_hand', 'sharpa_wave', 'skydio_x2', 'stanford_tidybot', 'tetheria_aero_hand_open', 'toddlerbot_2xc',
    'toddlerbot_2xm', 'trossen_vx300s', 'trossen_wx250s', 'trossen_wxai', 'trs_so_arm100', 'ufactory_lite6',
    'ufactory_xarm7', 'umi_gripper', 'unitree_a1', 'unitree_g1', 'unitree_go1', 'unitree_go2', 'unitree_h1',
    'unitree_z1', 'universal_robots_ur10e', 'universal_robots_ur5e', 'wonik_allegro',
)
NOT_MODELS = {'.github', 'assets', 'python', 'test', 'docs', 'scripts'}
GALLERY_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.webm')


class MenagerieError(Exception):
    """Menagerie could not be listed, downloaded or imported; the message says why and what to do, in plain words."""


@dataclass
class MenagerieDownload:
    model: str
    folder: str                       # the model folder on disk
    commit: str                       # the Menagerie commit every file came from
    license_text: str
    license_name: str                 # a recognised licence name, or 'see the LICENSE text'
    noncommercial: bool               # the licence text restricts commercial use
    mjcf: list = field(default_factory=list)       # the folder's MJCF files (relative), the likely robot file first
    files: list = field(default_factory=list)      # every downloaded file (relative)
    cached: bool = False

    @property
    def url(self) -> str:
        return f'{WEB}/tree/{self.commit}/{self.model}'


# ====================================================================== the network, explained
def data_dir() -> str:
    base = os.environ.get('XDG_DATA_HOME') or os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(base, 'DexVantage', 'menagerie')


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                  # noqa: BLE001 — the system's certificates then
        return ssl.create_default_context()


def _default_opener(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/vnd.github+json'
                                               if url.startswith(API) else '*/*'})
    return urllib.request.urlopen(req, timeout=TIMEOUT, context=_context())


def _explain(e: Exception, url: str, what: str) -> MenagerieError:
    host = url.split('/')[2] if '://' in url else url
    if isinstance(e, urllib.error.HTTPError):
        hdr = getattr(e, 'headers', None) or {}
        if e.code in (403, 429) and str(hdr.get('X-RateLimit-Remaining', '')) == '0':
            when = ''
            try:
                when = time.strftime(' after %H:%M', time.localtime(int(hdr.get('X-RateLimit-Reset'))))
            except (TypeError, ValueError):
                pass
            return MenagerieError(f'GitHub refused {what}: it allows 60 requests an hour without signing in, and '
                                  f'this computer has used them. Try again{when}.')
        if e.code == 404:
            return MenagerieError(f'GitHub has no {what} ({url} answered "not found"). The model may have been '
                                  'renamed or removed from MuJoCo Menagerie: refresh the list.')
        return MenagerieError(f'GitHub answered {e.code} ({e.reason}) to {what}. Try again later.')
    reason = getattr(e, 'reason', e)
    if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(e, (socket.timeout, TimeoutError)):
        return MenagerieError(f'{host} did not answer within {TIMEOUT:.0f} s while fetching {what}. The connection may '
                              'be slow or blocked; try again.')
    if isinstance(reason, ssl.SSLError) or isinstance(e, ssl.SSLError):
        return MenagerieError(f'The secure connection to {host} failed ({reason}). A proxy or firewall may be '
                              'intercepting it, or the system clock may be wrong.')
    if isinstance(reason, socket.gaierror) or isinstance(e, socket.gaierror):
        return MenagerieError(f'DexVantage could not find {host} to fetch {what}: this computer seems to be '
                              'offline, or its DNS is not answering. Check the internet connection.')
    if isinstance(e, (urllib.error.URLError, OSError)):
        return MenagerieError(f'DexVantage could not reach {host} to fetch {what} ({reason}). Check the internet '
                              'connection or the proxy settings.')
    return MenagerieError(f'Fetching {what} from {host} failed ({type(e).__name__}: {e}).')


def _get(url: str, what: str, opener=None, limit: int | None = None) -> bytes:
    op = opener or _default_opener
    try:
        with op(url) as r:
            data = r.read() if limit is None else r.read(limit + 1)
    except MenagerieError:
        raise
    except Exception as e:                             # noqa: BLE001 — every failure explained in plain words
        raise _explain(e, url, what) from None
    if limit is not None and len(data) > limit:
        raise MenagerieError(f'{what} is larger than DexVantage accepts ({limit / 1e6:.0f} MB).')
    return data


def _json(url: str, what: str, opener=None):
    raw = _get(url, what, opener)
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        raise MenagerieError(f'GitHub sent something other than the expected list for {what}; try again later.') \
            from None


# ====================================================================== the index
def index(opener=None) -> dict:
    """{'models': [folder names], 'source': 'github' | 'offline', 'note': plain words}: Menagerie's model folders,
    from the GitHub API, or OFFLINE_MODELS (names only) with the reason when GitHub cannot be reached."""
    try:
        items = _json(f'{API}/contents/?ref={BRANCH}', 'the list of Menagerie models', opener)
        names = sorted(str(i.get('name')) for i in items if isinstance(i, dict) and i.get('type') == 'dir'
                       and str(i.get('name')) not in NOT_MODELS and not str(i.get('name')).startswith('.'))
        if not names:
            raise MenagerieError('GitHub listed no model folders in MuJoCo Menagerie.')
        return {'models': names, 'source': 'github', 'note': f'{len(names)} models listed by GitHub just now.'}
    except MenagerieError as e:
        return {'models': sorted(OFFLINE_MODELS), 'source': 'offline',
                'note': f'{e} Showing the list built into DexVantage (checked 2026-11-06); downloading needs the '
                        'network.'}


# ====================================================================== licences
def license_name(text: str) -> tuple[str, bool]:
    """(a short licence name, restricts commercial use) recognised from a LICENSE text; ('see the LICENSE text',
    False) when it is not one of the texts below. Recognised by their own wording, not guessed."""
    t = ' '.join(str(text or '').split())
    low = t.lower()
    # "NonCommercial" / "non-commercial" as a word; GPL-3.0 section 6(c) says "noncommercially" about a way of
    # conveying copies, which is not a use restriction, so that word does not count
    nc = bool(_NONCOMMERCIAL.search(low)) or 'not for commercial' in low
    flat = low.replace('-', '')
    # the CC BY legal codes mention CC0 in their closing notice, so whichever is named first is the licence
    cc0, attr = low.find('cc0'), low.find('attribution')
    if 'creative commons' in low and cc0 >= 0 and (attr < 0 or cc0 < attr):
        return 'CC0-1.0', nc
    if 'creative commons' in low and attr >= 0:
        near = t[attr:attr + 200]
        ver = next((v for v in ('4.0', '3.0', '2.5', '2.0', '1.0') if v in near), '')
        kind = 'CC-BY' + ('-NC' if nc else '') + ('-SA' if 'sharealike' in flat else '') \
            + ('-ND' if 'noderivatives' in flat or 'noderivs' in flat else '')
        return (f'{kind}-{ver}' if ver else kind), nc
    if 'apache license' in low and 'version 2.0' in low:
        if 'redistribution and use in source and binary forms' in low:     # a BSD text after it (franka_fr3 at 98a70c5)
            bsd = 'BSD-3-Clause' if 'neither the name' in low or 'endorse or promote' in low else 'BSD-2-Clause'
            return f'Apache-2.0 AND {bsd}', nc
        return 'Apache-2.0', nc
    if 'redistribution and use in source and binary forms' in low:
        if 'clear bsd license' in low:
            return 'BSD-3-Clause-Clear', nc
        if 'advertising materials' in low:
            return 'BSD-4-Clause', nc
        return ('BSD-3-Clause' if 'neither the name' in low or 'endorse or promote' in low else 'BSD-2-Clause'), nc
    if 'boost software license' in low:
        return 'BSL-1.0', nc
    if 'sil open font license' in low:                 # its grant also opens "Permission is hereby granted, free…"
        return ('OFL-1.1' if 'version 1.1' in low else 'OFL'), nc
    if 'permission is hereby granted, free of charge' in low:
        return 'MIT', nc
    # the GNU texts name each other (GPL-3.0 section 13 names the Affero licence, the Lesser licence builds on the
    # GPL), so the one named first is the licence, and its version is read just after that name
    gnu = [(low.find(k), name) for k, name in (('gnu general public license', 'GPL'),
                                                ('gnu lesser general public license', 'LGPL'),
                                                ('gnu library general public license', 'LGPL'),
                                                ('gnu affero general public license', 'AGPL')) if k in low]
    if gnu:
        at, name = min(gnu)
        near = low[at:at + 160]
        ver = next((v for k, v in (('version 3', '3.0'), ('version 2.1', '2.1'), ('version 2', '2.0')) if k in near),
                   '')
        return (f'{name}-{ver}' if ver else name), nc
    if 'mozilla public license' in low and '2.0' in t:
        return 'MPL-2.0', nc
    return 'see the LICENSE text', nc


# ====================================================================== the download
def _safe(rel: str) -> bool:
    parts = rel.split('/')
    return bool(rel) and not rel.startswith('/') and '..' not in parts and '\\' not in rel and all(parts)


def _mjcf_order(model: str, xmls: list) -> list:
    """The folder's XML files, the likely robot file first: the top-level file named like the folder, then other
    top-level files that are not scenes, then scenes, then the rest."""
    tail = model.split('_', 1)[-1]

    def rank(p):
        base = posixpath.basename(p)
        top = '/' not in p
        stem = os.path.splitext(base)[0]
        return (not top, base.startswith('scene'), not (stem == model or stem == tail or tail.startswith(stem)
                                                        or stem in model), p)
    return sorted(xmls, key=rank)


def download(model: str, dest: str | None = None, opener=None, progress=None) -> MenagerieDownload:
    """Download the Menagerie folder `model` (its MJCF, meshes, textures and LICENSE) and read its LICENSE.

    progress(fraction, text) as files arrive. Raises MenagerieError, in plain words, for a model without a LICENSE,
    one larger than MAX_DOWNLOAD_MB, or any network failure. Nothing is imported: show license_text first."""
    model = str(model or '').strip().strip('/')
    if not model or not _safe(model) or '/' in model or model in NOT_MODELS:
        raise MenagerieError(f'"{model}" is not a MuJoCo Menagerie model folder name.')
    root = os.path.join(dest or data_dir(), model)
    try:
        commit = _json(f'{API}/commits/{BRANCH}', 'the current Menagerie version', opener).get('sha')
    except MenagerieError:
        done = _cached(model, root)                    # offline: the newest copy downloaded before, if any
        if done is not None:
            return done
        raise
    if not isinstance(commit, str) or len(commit) < 7:
        raise MenagerieError('GitHub did not say which Menagerie version is current; try again later.')
    folder = os.path.join(root, commit[:12])
    marker = os.path.join(folder, MARKER)
    if os.path.isfile(marker):
        try:
            with open(marker, encoding='utf-8') as fh:
                info = json.load(fh)
            return _result(model, folder, commit, info.get('files') or [], cached=True)
        except (OSError, ValueError, MenagerieError):
            shutil.rmtree(folder, ignore_errors=True)
    top = _json(f'{API}/git/trees/{commit}', 'the Menagerie folder list', opener)
    entry = next((e for e in top.get('tree') or [] if e.get('path') == model and e.get('type') == 'tree'), None)
    if entry is None:
        raise MenagerieError(f'MuJoCo Menagerie has no model named {model} at its current version: refresh the list '
                             'and pick it again.')
    tree = _json(f'{API}/git/trees/{entry["sha"]}?recursive=1', f'the files of {model}', opener)
    blobs = [(str(e['path']), int(e.get('size') or 0)) for e in tree.get('tree') or []
             if e.get('type') == 'blob' and _safe(str(e.get('path', '')))]
    if not any(p == 'LICENSE' for p, _s in blobs):
        raise MenagerieError(f'The Menagerie folder {model} has no LICENSE file of its own. Menagerie licences differ '
                             'per model, so DexVantage does not import a model whose licence it cannot show you.')
    want = [(p, s) for p, s in blobs
            if not ('/' not in p and (p.lower().endswith(GALLERY_EXT) or (p.lower().endswith('.md'))))]
    total = sum(s for _p, s in want)
    if total > MAX_DOWNLOAD_MB * 1e6:
        raise MenagerieError(f'{model} is {total / 1e6:.0f} MB, more than DexVantage downloads at once '
                             f'({MAX_DOWNLOAD_MB:.0f} MB).')
    part = folder + '.part'
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    got = 0
    try:
        for i, (rel, size) in enumerate(want):
            if progress:
                progress(got / total if total else i / max(1, len(want)), f'{model}/{rel}')
            data = _get(f'{RAW}/{commit}/{model}/{rel}', f'{model}/{rel}', opener,
                        limit=int(MAX_DOWNLOAD_MB * 1e6))
            out = os.path.join(part, *rel.split('/'))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'wb') as fh:
                fh.write(data)
            got += size or len(data)
        files = [p for p, _s in want]
        with open(os.path.join(part, MARKER), 'w', encoding='utf-8') as fh:
            json.dump({'format': 'dexvantage.menagerie-download', 'model': model, 'commit': commit,
                       'source': f'{WEB}/tree/{commit}/{model}', 'files': files,
                       'downloaded': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()},
                      fh, indent=1)
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(part, folder)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    if progress:
        progress(1.0, f'{model}: {len(want)} files')
    return _result(model, folder, commit, files)


def _cached(model: str, root: str) -> MenagerieDownload | None:
    best = None
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for n in names:
        mk = os.path.join(root, n, MARKER)
        try:
            with open(mk, encoding='utf-8') as fh:
                info = json.load(fh)
            t = os.path.getmtime(mk)
        except (OSError, ValueError):
            continue
        if best is None or t > best[0]:
            best = (t, n, info)
    if best is None:
        return None
    _t, n, info = best
    try:
        return _result(model, os.path.join(root, n), str(info.get('commit') or n), info.get('files') or [],
                       cached=True)
    except MenagerieError:
        return None


def _result(model, folder, commit, files, cached=False) -> MenagerieDownload:
    lic = os.path.join(folder, 'LICENSE')
    if not os.path.isfile(lic):
        raise MenagerieError(f'The downloaded {model} has no LICENSE file; download it again.')
    with open(lic, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    name, nc = license_name(text)
    xmls = _mjcf_order(model, [f for f in files if f.lower().endswith('.xml')])
    return MenagerieDownload(model, folder, commit, text, name, nc, xmls, list(files), cached)


# ====================================================================== the import
def import_download(dl: MenagerieDownload, main: str | None = None, accepted_on: str | None = None):
    """(Robot, sources) of a downloaded model whose licence the user accepted: core.robot.import_robot on its MJCF
    (main, relative to the folder; default the likely robot file), with the licence recorded in
    Robot.source.license and the LICENSE file stored in the project ("<uid>/LICENSE")."""
    from .importer import import_robot
    rel = main or (dl.mjcf[0] if dl.mjcf else '')
    if not rel or (rel not in dl.files and not os.path.isfile(os.path.join(dl.folder, *rel.split('/')))):
        raise MenagerieError(f'{dl.model} has no MuJoCo model file (.xml) to import.')
    robot, sources = import_robot(os.path.join(dl.folder, *rel.split('/')))
    day = accepted_on or _dt.date.today().isoformat()
    robot.source.license = (f'{dl.license_name} (MuJoCo Menagerie {dl.model}/LICENSE at {dl.commit[:12]}; '
                            f'accepted {day})')
    note = f'Imported from {WEB}/tree/{dl.commit}/{dl.model}/{rel}.'
    robot.source.note = ' '.join(x for x in (robot.source.note, note) if x)
    sources[f'{robot.uid}/LICENSE'] = dl.license_text.encode('utf-8')
    return robot, sources
