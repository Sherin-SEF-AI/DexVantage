"""The hardware library: cameras, depth cameras, tactile sensors and hosts, from makers' datasheets only (§6, §1.1).

One JSON file per device, schema dexvantage/resources/library.schema.json, written up in docs/models/library.md:

    dexvantage/resources/library/<kind>/<id>.json        built-in entries (shipped, read-only)
    $XDG_DATA_HOME/DexVantage/library/<kind>/<id>.json   the user's custom devices (Library ▸ New custom device…)

Every figure in an entry comes from a cited source (its `sources`: URL or document, title, the date read, which
fields it covers); a figure the source does not state is null, shown as "—" with the tooltip "not stated by the
maker" (NOT_STATED). Assumptions an entry needs (an RGB offset the maker does not give, say) are labelled in its
`note` and flagged on the figure (`pose_stated: false`). Nothing here fills a gap.

API
    entries(kind=None) -> [Entry]           sorted by kind, maker, name
    get(model_id) -> Entry | None
    add_custom(dict) -> Entry               validated, written to the user library; LibraryError says why not
    remove_custom(model_id) -> bool
    reload()                                re-read both folders (a bad file is skipped and logged: problems())
    generation() -> int                     changes on every (re)read: for caches built from entries
    entries_digest(model_ids) -> str        the entries' content digest: for result cache keys
    validate(dict) -> [str]                 plain messages, [] when the entry is valid
    sensor_imagers(sensor, mode=None) -> [Imager]   Imager = (name, role, pose_in_body, camera), a 4-tuple
    projector(sensor) -> dict | None        the IR projector: FOV and pose (frames.Pose) in the body frame
    depth_spec(sensor) -> dict | None       the entry's depth block (baseline, min-Z table, range, RMS error, ...)
    min_z(sensor, width, height) -> float | None

Frames: an imager's pose is its OPTICAL frame (z forward, x right, y down) in the device body frame, which is REP-103
(x forward along the optical axis, y left, z up) with its origin at the optical centre of the reference (depth)
imager. A sensor's mount places the body frame (core/project.py Mount).

Known limits: entries are what the makers publish, so lens distortion is rarely stated and the built-in imagers are
pinhole models fitted to the stated FOV (calibrate a unit and store its K and D in sensor.calibration); FOVs vary
unit to unit (Intel: about ±3°); a stream whose aspect ratio differs from every stated FOV gets one derived by a
centred crop of the native image, which is an assumption the Imager's label says; the JSON Schema is checked with
jsonschema when installed, else by a smaller structural check (validate_minimal) that does not test every bound.
"""
from __future__ import annotations

import copy
import glob
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading

import numpy as np

from ..frames import OPTICAL_TO_BODY, Pose, quat_from_matrix

log = logging.getLogger(__name__)

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resources')
BUILTIN_DIR = os.path.join(RESOURCES, 'library')
SCHEMA_PATH = os.path.join(RESOURCES, 'library.schema.json')

KINDS = ('depth', 'camera', 'tactile', 'host')
ROLES = ('ir_left', 'ir_right', 'rgb', 'mono', 'tof')
LENS_MODELS = ('pinhole', 'brown', 'kb', 'equidistant', 'equisolid', 'stereographic')
NOT_STATED = 'not stated by the maker'

# the imager a sensor gets when it has no library device: a DEFAULT, labelled as one wherever it is shown
DEFAULT_HFOV_DEG = 90.0
DEFAULT_VFOV_DEG = 60.0
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_IMAGER_NAME = 'Default camera'
DEFAULT_LABEL = (f'Default camera (no device chosen): a pinhole {DEFAULT_RESOLUTION[0]} × {DEFAULT_RESOLUTION[1]} px, '
                 f'{DEFAULT_HFOV_DEG:g}° × {DEFAULT_VFOV_DEG:g}° — a default, not a device figure. Choose a library '
                 'device for the sensor.')

_OPTICAL_Q = quat_from_matrix(OPTICAL_TO_BODY)        # the optical frame of a camera looking along body x
_DATE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
_ID = re.compile(r'^[a-z0-9][a-z0-9._-]*$')


class LibraryError(ValueError):
    """A library operation DexVantage cannot do; the message says why in plain words."""


# ====================================================================== folders
def user_dir() -> str:
    """The user's custom devices: $XDG_DATA_HOME/DexVantage/library (when XDG_DATA_HOME is set, on every platform),
    else ~/.local/share (Linux), ~/Library/Application Support (macOS) or %LOCALAPPDATA% (Windows)."""
    root = os.environ.get('XDG_DATA_HOME')
    if not root:
        if sys.platform.startswith('win'):
            root = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
        elif sys.platform == 'darwin':
            root = os.path.expanduser('~/Library/Application Support')
        else:
            root = os.path.join(os.path.expanduser('~'), '.local', 'share')
    return os.path.join(root, 'DexVantage', 'library')


# ====================================================================== validation
_schema = None
_validator = None


def schema() -> dict:
    global _schema
    if _schema is None:
        with open(SCHEMA_PATH, encoding='utf-8') as fh:
            _schema = json.load(fh)
    return _schema


def _where(path) -> str:
    out = ''
    for p in path:
        out += f'[{p}]' if isinstance(p, int) else (f'.{p}' if out else str(p))
    return out or 'the entry'


def validate(d) -> list:
    """Plain messages saying what is wrong with an entry ([] when it is valid). Uses the JSON Schema through
    jsonschema when it is installed, else validate_minimal."""
    global _validator
    try:
        import jsonschema
    except ImportError:                                      # an optional dependency: the structural check instead
        return validate_minimal(d)
    if _validator is None:
        _validator = jsonschema.Draft202012Validator(schema())
    msgs = []
    for e in sorted(_validator.iter_errors(d), key=lambda e: list(map(str, e.absolute_path))):
        msgs.append(f'{_where(e.absolute_path)}: {e.message}')
    return msgs


def _is_num(v, nullable=True) -> bool:
    if v is None:
        return nullable
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_minimal(d) -> list:
    """The structural check used when jsonschema is not installed: required keys and their types, the kinds, roles
    and lens models, source dates, numeric figures (number or null) and pose shapes. It does not test every bound
    the schema sets."""
    if not isinstance(d, dict):
        return ['the entry must be a JSON object']
    msgs = []
    for k in ('id', 'kind', 'maker', 'name', 'note'):
        if k not in d:
            msgs.append(f'{k}: is required')
        elif not isinstance(d[k], str):
            msgs.append(f'{k}: must be text')
    if isinstance(d.get('id'), str) and not _ID.match(d['id']):
        msgs.append('id: must be lower case letters, digits, ".", "_" or "-"')
    if isinstance(d.get('kind'), str) and d['kind'] not in KINDS:
        msgs.append(f'kind: must be one of {", ".join(KINDS)}')
    src = d.get('sources')
    if not isinstance(src, list) or not src:
        msgs.append('sources: at least one source (url and the date read) is required')
    else:
        for i, s in enumerate(src):
            if not isinstance(s, dict) or not isinstance(s.get('url'), str) or not s.get('url'):
                msgs.append(f'sources[{i}].url: is required')
            if not isinstance(s, dict) or not isinstance(s.get('read'), str) or not _DATE.match(s.get('read', '')):
                msgs.append(f'sources[{i}].read: the date read is required, as YYYY-MM-DD')
    for k in ('power_w', 'mass_g'):
        if not _is_num(d.get(k)):
            msgs.append(f'{k}: must be a number or null')
    if d.get('size_mm') is not None and not (isinstance(d['size_mm'], list) and len(d['size_mm']) == 3
                                             and all(_is_num(v, False) for v in d['size_mm'])):
        msgs.append('size_mm: must be [x, y, z] or null')
    ims = d.get('imagers', [])
    if not isinstance(ims, list):
        msgs.append('imagers: must be a list')
        ims = []
    for i, im in enumerate(ims):
        w = f'imagers[{i}]'
        if not isinstance(im, dict):
            msgs.append(f'{w}: must be an object')
            continue
        if not isinstance(im.get('name'), str) or not im.get('name'):
            msgs.append(f'{w}.name: is required')
        if im.get('role') not in ROLES:
            msgs.append(f'{w}.role: must be one of {", ".join(ROLES)}')
        if im.get('lens_model') not in LENS_MODELS + (None,):
            msgs.append(f'{w}.lens_model: must be one of {", ".join(LENS_MODELS)} or null')
        if im.get('shutter') not in ('global', 'rolling', None):
            msgs.append(f'{w}.shutter: must be global, rolling or null')
        for k in ('hfov_deg', 'vfov_deg', 'dfov_deg', 'pixel_um', 'f_number'):
            if not _is_num(im.get(k)):
                msgs.append(f'{w}.{k}: must be a number or null')
        if 'pose' in im:
            msgs += _pose_msgs(im['pose'], f'{w}.pose')
        r = im.get('resolution')
        if r is not None and not (isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) and v > 0 for v in r)):
            msgs.append(f'{w}.resolution: must be [width, height] or null')
    pj = d.get('projector')
    if pj is not None:
        if not isinstance(pj, dict):
            msgs.append('projector: must be an object or null')
        elif 'pose' in pj:
            msgs += _pose_msgs(pj['pose'], 'projector.pose')
    dp = d.get('depth')
    if dp is not None:
        if not isinstance(dp, dict):
            msgs.append('depth: must be an object or null')
        else:
            for k in ('baseline_m', 'max_range_m', 'subpixel_px'):
                if not _is_num(dp.get(k)):
                    msgs.append(f'depth.{k}: must be a number or null')
            for i, row in enumerate(dp.get('min_z') or []):
                if not (isinstance(row, dict) and isinstance(row.get('width'), int)
                        and isinstance(row.get('height'), int) and _is_num(row.get('m'))):
                    msgs.append(f'depth.min_z[{i}]: must be {{"width", "height", "m"}}')
    for i, s in enumerate(d.get('streams') or []):
        if not (isinstance(s, dict) and isinstance(s.get('imager'), str) and isinstance(s.get('width'), int)
                and isinstance(s.get('height'), int)):
            msgs.append(f'streams[{i}]: needs imager, width and height')
    for k in ('tactile', 'host'):
        if d.get(k) is not None and not isinstance(d[k], dict):
            msgs.append(f'{k}: must be an object or null')
    return msgs


def _pose_msgs(p, where) -> list:
    if not (isinstance(p, dict) and isinstance(p.get('position'), list) and len(p['position']) == 3
            and isinstance(p.get('quaternion'), list) and len(p['quaternion']) == 4
            and all(_is_num(v, False) for v in p['position'] + p['quaternion'])):
        return [f'{where}: must be {{"position": [x, y, z], "quaternion": [x, y, z, w]}}']
    if float(np.linalg.norm(p['quaternion'])) < 1e-9:
        return [f'{where}.quaternion: has zero length']
    return []


# ====================================================================== entries
class Entry:
    """One library device. .data is the entry's JSON (do not change it in place: copy it)."""

    def __init__(self, data: dict, path: str = '', is_custom: bool = False):
        self.data = data
        self.path = path
        self.is_custom = bool(is_custom)

    id = property(lambda self: self.data['id'])
    kind = property(lambda self: self.data['kind'])
    maker = property(lambda self: self.data['maker'])
    name = property(lambda self: self.data['name'])
    note = property(lambda self: self.data.get('note', ''))
    sources = property(lambda self: list(self.data.get('sources') or []))

    @property
    def title(self) -> str:
        """'Maker name' unless the name already starts with the maker's name; a word the maker's name ends with and
        the device's name starts with is said once ('Intel RealSense' + 'RealSense Depth Camera D405' gives
        'Intel RealSense Depth Camera D405')."""
        n, m = self.name, self.maker
        if n.lower().startswith(m.lower()):
            return n
        mw, nw = m.split(), n.split()
        if mw and nw and mw[-1].lower() == nw[0].lower():
            return ' '.join(mw + nw[1:])
        return f'{m} {n}'

    @property
    def imagers(self) -> list:
        return list(self.data.get('imagers') or [])

    def imager(self, name: str) -> dict | None:
        """An imager by name (case and spacing ignored); 'depth' names the reference imager, 'rgb' the first RGB
        imager."""
        key = _norm(name)
        ims = self.imagers
        for im in ims:
            if _norm(im.get('name')) == key:
                return im
        if key == 'depth':
            ref = (self.data.get('depth') or {}).get('reference_imager')
            return next((im for im in ims if _norm(im.get('name')) == _norm(ref)), ims[0] if ims else None)
        return next((im for im in ims if im.get('role') == key), None)

    @property
    def reference_imager(self) -> dict | None:
        return self.imager('depth') if self.data.get('depth') else (self.imagers[0] if self.imagers else None)

    def streams(self, imager: str | None = None) -> list:
        """The entry's stream modes, or those of one imager (by name; a 'depth' stream belongs to the reference
        imager, an 'rgb' stream to the RGB imager)."""
        ss = list(self.data.get('streams') or [])
        if imager is None:
            return ss
        im = self.imager(imager)
        return [s for s in ss if im is not None and self.imager(s.get('imager', '')) is im]

    def __repr__(self):
        return f'Entry({self.id!r}{", custom" if self.is_custom else ""})'


def _norm(s) -> str:
    return re.sub(r'[\s_-]+', ' ', str(s or '')).strip().lower()


_lock = threading.RLock()
_entries: dict | None = None
_problems: list = []


def _read_dir(root: str, is_custom: bool, seen: dict, problems: list):
    files = sorted(glob.glob(os.path.join(root, '**', '*.json'), recursive=True))
    for f in files:
        rel = os.path.relpath(f, root)
        where = ('your library' if is_custom else 'the built-in library') + f' ({rel})'
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            problems.append(f'Skipped {where}: it is not a readable JSON file ({e}).')
            continue
        msgs = validate(d)
        if msgs:
            more = f' (and {len(msgs) - 3} more)' if len(msgs) > 3 else ''
            problems.append(f'Skipped {where}: it does not match the library schema: {"; ".join(msgs[:3])}{more}.')
            continue
        if d['id'] in seen:
            other = seen[d['id']]
            problems.append(f'Skipped {where}: the id "{d["id"]}" is already used by '
                            f'{"your library" if other.is_custom else "the built-in library"} '
                            f'({os.path.basename(other.path)}).')
            continue
        seen[d['id']] = Entry(d, f, is_custom)


_generation = 0


def _load() -> dict:
    global _entries, _problems, _generation
    with _lock:
        if _entries is None:
            seen, problems = {}, []
            _read_dir(BUILTIN_DIR, False, seen, problems)
            if os.path.isdir(user_dir()):
                _read_dir(user_dir(), True, seen, problems)
            for p in problems:
                log.warning('%s', p)
            _entries, _problems = seen, problems
            _generation += 1
        return _entries


def generation() -> int:
    """A number that changes whenever the library is (re)read, so a cache of anything built from entries (a frustum,
    a min-Z zone) can include it in its key."""
    _load()
    return _generation


def entries_digest(model_ids) -> str:
    """A content digest of the entries with these ids (an id the library lacks counts as missing), for result cache
    keys: a study result depends on the entry's figures, not only on its id, and a custom device can be changed in
    place under the same id."""
    import hashlib
    h = hashlib.sha256()
    for mid in sorted({str(m) for m in model_ids if m}):
        e = get(mid)
        h.update(mid.encode('utf-8') + b'\0')
        h.update(json.dumps(e.data, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8') if e is not None
                 else b'missing')
        h.update(b'\0')
    return h.hexdigest()


def reload():
    """Read the built-in and the user's folders again."""
    global _entries
    with _lock:
        _entries = None
        _load()


def problems() -> list:
    """Plain messages for the files the last load skipped."""
    _load()
    return list(_problems)


def entries(kind: str | None = None) -> list:
    """Every entry (of one kind), sorted by kind, maker and name."""
    if kind is not None and kind not in KINDS:
        raise LibraryError(f'There is no library kind "{kind}". The kinds are: {", ".join(KINDS)}.')
    es = [e for e in _load().values() if kind is None or e.kind == kind]
    return sorted(es, key=lambda e: (KINDS.index(e.kind), e.maker.lower(), e.name.lower()))


def get(model_id: str) -> Entry | None:
    if not model_id:
        return None
    return _load().get(str(model_id))


def add_custom(d: dict) -> Entry:
    """Validate a custom device and write it to the user library (replacing an earlier custom device with the same
    id). Raises LibraryError when it is not valid or its id is a built-in one."""
    if not isinstance(d, dict):
        raise LibraryError('A custom device must be a set of named fields (a JSON object).')
    d = copy.deepcopy(d)
    msgs = validate(d)
    if msgs:
        raise LibraryError('The custom device was not saved: ' + '; '.join(msgs[:5])
                           + (f' (and {len(msgs) - 5} more)' if len(msgs) > 5 else '') + '.')
    old = get(d['id'])
    if old is not None and not old.is_custom:
        raise LibraryError(f'"{d["id"]}" is a built-in device; give the custom device an id of its own.')
    folder = os.path.join(user_dir(), d['kind'])
    os.makedirs(folder, exist_ok=True)
    if old is not None and os.path.dirname(old.path) != folder:
        _remove_file(old.path)                            # its kind changed: drop the file in the old folder
    path = os.path.join(folder, d['id'] + '.json')
    fd, tmp = tempfile.mkstemp(prefix='.' + d['id'] + '-', suffix='.part', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(d, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        os.replace(tmp, path)
    except OSError as e:
        _remove_file(tmp)
        raise LibraryError(f'The custom device could not be written to {path}: {e.strerror or e}.') from e
    reload()
    return get(d['id'])


def _remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def remove_custom(model_id: str) -> bool:
    """Delete a custom device; False when there is none with that id. A built-in device cannot be removed."""
    e = get(model_id)
    if e is None:
        return False
    if not e.is_custom:
        raise LibraryError(f'"{model_id}" is a built-in device; only your own custom devices can be removed.')
    try:
        os.remove(e.path)
    except OSError as err:
        raise LibraryError(f'The custom device could not be removed ({err.strerror or err}).') from err
    reload()
    return True


# ====================================================================== sensors -> imagers
class Imager(tuple):
    """(name, role, pose_in_body, camera): the imager's optical frame in the sensor body frame (frames.Pose) and its
    optics.CameraModel. It unpacks as that 4-tuple; the attributes add what the numbers are:

        label        what the camera model is and where its figures come from (a default or an assumption says so)
        is_default   True for the default imager of a sensor with no library device
        pose_stated  False when the maker does not state this imager's offset (a labelled assumption)
        calibrated   True when sensor.calibration gave the intrinsics
        mode         (width, height) the camera model is for
    """

    def __new__(cls, name, role, pose_in_body, camera, *, label='', is_default=False, pose_stated=True,
                calibrated=False, model_id=''):
        t = super().__new__(cls, (name, role, pose_in_body, camera))
        t.label, t.is_default, t.pose_stated, t.calibrated, t.model_id = (label, bool(is_default), bool(pose_stated),
                                                                          bool(calibrated), model_id)
        t.mode = (camera.width, camera.height)
        return t

    name = property(lambda self: self[0])
    role = property(lambda self: self[1])
    pose_in_body = property(lambda self: self[2])
    camera = property(lambda self: self[3])

    def __getnewargs_ex__(self):                       # copy and pickle keep the labels
        return tuple(self), {'label': self.label, 'is_default': self.is_default, 'pose_stated': self.pose_stated,
                             'calibrated': self.calibrated, 'model_id': self.model_id}

    def __repr__(self):
        tag = ', default' if self.is_default else ''
        return f'Imager({self.name!r}, {self.role!r}, {self.mode[0]}x{self.mode[1]}{tag})'


def _optics():
    from ..optics import CameraModel
    return CameraModel


def default_imager(reason: str = '') -> Imager:
    """The labelled default for a sensor without a (known) library device."""
    cam = _optics().from_fov(DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], DEFAULT_HFOV_DEG, DEFAULT_VFOV_DEG)
    label = DEFAULT_LABEL if not reason else f'{DEFAULT_LABEL} {reason}'
    return Imager(f'{DEFAULT_IMAGER_NAME} (no device chosen)', 'rgb', Pose(np.zeros(3), _OPTICAL_Q), cam,
                  label=label, is_default=True, pose_stated=False)


def _aspect_eq(a, b) -> bool:
    return abs(a[0] / a[1] - b[0] / b[1]) < 0.01 * (b[0] / b[1])


def _mode_for(entry: Entry, im: dict, mode):
    """(w, h) for this imager: the requested mode when the imager streams it (or a dict naming the imager), else
    the imager's largest listed stream, else its native resolution, else the default resolution."""
    if isinstance(mode, dict):
        m = next((v for k, v in mode.items() if _norm(k) == _norm(im.get('name'))), None)
        if m is not None:
            return int(m[0]), int(m[1]), 'chosen'
        mode = None
    ss = entry.streams(im.get('name'))
    if mode is not None:
        w, h = int(mode[0]), int(mode[1])
        if any(s['width'] == w and s['height'] == h for s in ss) or (not ss and im.get('resolution')
                                                                     and _aspect_eq((w, h), im['resolution'])):
            return w, h, 'chosen'
    if ss:
        s = max(ss, key=lambda s: (s['width'] * s['height'], s['width']))
        return s['width'], s['height'], 'largest stream'
    if im.get('resolution'):
        return int(im['resolution'][0]), int(im['resolution'][1]), 'native'
    return DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], 'default'


def _fov_for(im: dict, w: int, h: int):
    """(hfov, vfov or None, where the FOV comes from) for a w × h stream of this imager."""
    native = im.get('resolution')
    hf, vf = im.get('hfov_deg'), im.get('vfov_deg')
    # a FOV the maker states for streams of this shape (a depth FOV for 16:9 streams, say) comes first; the
    # sensor's own FOV is for its native image
    if native is None or (int(w), int(h)) != tuple(native):
        for fm in im.get('fov_modes') or []:
            if fm.get('hfov_deg') is not None and _aspect_eq((w, h), fm['aspect']):
                return fm['hfov_deg'], fm.get('vfov_deg'), 'as stated: ' + (
                    fm.get('what') or f'{fm["aspect"][0]}:{fm["aspect"][1]} streams')
    if hf is not None and (native is None or _aspect_eq((w, h), native)):
        return hf, vf, 'as stated' + (f' for {native[0]} × {native[1]}' if native else '')
    if hf is not None and native is not None:
        # ASSUMPTION: the stream is a centred crop of the native image, scaled uniformly so it fits inside it
        W, H = native
        fx = (W / 2) / math.tan(math.radians(hf) / 2)
        fy = (H / 2) / math.tan(math.radians(vf) / 2) if vf is not None else fx
        s = max(w / W, h / H)
        hh = 2 * math.degrees(math.atan((w / 2) / (fx * s)))
        vv = 2 * math.degrees(math.atan((h / 2) / (fy * s))) if vf is not None else None
        return hh, vv, (f'derived from the stated {W} × {H} FOV by a centred crop (an assumption: the maker states '
                        f'no FOV for {w} × {h})')
    df = im.get('dfov_deg')
    if hf is None and df is not None and (im.get('lens_model') or 'pinhole') == 'pinhole' and 0 < df < 180:
        # only the diagonal is stated: a pinhole with square pixels has one focal length that shows it across the
        # native image's diagonal (or the stream's, when no native size is stated); the stream is then a centred
        # crop of that image, as above. Both steps are assumptions the label says.
        W, H = native if native is not None else (w, h)
        f = (math.hypot(W, H) / 2) / math.tan(math.radians(df) / 2)
        s = max(w / W, h / H)
        hh = 2 * math.degrees(math.atan((w / 2) / (f * s)))
        vv = 2 * math.degrees(math.atan((h / 2) / (f * s)))
        return hh, vv, (f'derived from the stated diagonal FOV {df:g}° (an assumption: square pixels and no '
                        'distortion; the maker states no horizontal or vertical FOV)')
    return None, None, ''


def _calibration_for(sensor, im: dict, first: bool):
    cal = getattr(sensor, 'calibration', None) or {}
    per = cal.get('imagers') if isinstance(cal.get('imagers'), dict) else {}
    for k, v in per.items():
        if _norm(k) == _norm(im.get('name')) and isinstance(v, dict):
            return v
    if first and 'K' in cal:
        return cal
    return None


def _camera_from_calibration(c: dict, w: int, h: int, name: str):
    CameraModel = _optics()
    try:
        cw, ch = int(c.get('width') or w), int(c.get('height') or h)
        cam = CameraModel.from_opencv(c['K'], c.get('D'), cw, ch, str(c.get('model') or 'brown'))
    except (KeyError, TypeError, ValueError) as e:
        raise LibraryError(f'The calibration of "{name}" cannot be used: {e}.') from e
    if (cw, ch) != (w, h):
        if not _aspect_eq((w, h), (cw, ch)):
            return cam                                    # a crop we cannot infer: keep the calibrated image size
        cam = cam.scaled(w, h)
    return cam


def sensor_imagers(sensor, mode=None) -> list:
    """The imagers of a project Sensor: [Imager(name, role, pose_in_body, camera)].

    mode: None (each imager's largest listed stream), (width, height) (used by every imager that streams it; the
    others keep their own) or {imager name: (width, height)}. A sensor without a model_id (or with one the library
    does not have) gets one labelled default imager; a device with no imagers (a tactile pad without a listed
    camera, a host) gives []. sensor.calibration overrides the intrinsics (and optionally the
    pose) per imager: {'imagers': {'IR left': {'K': 3 × 3, 'D': [...], 'width': w, 'height': h, 'model': 'brown',
    'pose': {'position', 'quaternion'}}}}; a flat {'K', 'D', 'width', 'height', 'model'} applies to the first
    imager."""
    mid = getattr(sensor, 'model_id', '') or ''
    entry = get(mid) if mid else None
    if entry is None:
        reason = f'"{mid}" is not in the library.' if mid else ''
        return [default_imager(reason)]
    CameraModel = _optics()
    out = []
    for i, im in enumerate(entry.imagers):
        w, h, why = _mode_for(entry, im, mode)
        cal = _calibration_for(sensor, im, i == 0)
        pose = Pose.from_dict(im['pose']) if isinstance(im.get('pose'), dict) else Pose(np.zeros(3), _OPTICAL_Q)
        stated = bool(im.get('pose_stated', 'pose' in im))
        if cal is not None and isinstance(cal.get('pose'), dict):
            pose, stated = Pose.from_dict(cal['pose']), True
        if cal is not None:
            cam = _camera_from_calibration(cal, w, h, im.get('name', ''))
            label = f'{im["name"]}: from the unit\'s calibration, {cam.width} × {cam.height} px'
        else:
            hf, vf, src = _fov_for(im, w, h)
            model = im.get('lens_model') or 'pinhole'
            if hf is None:
                hf, vf = DEFAULT_HFOV_DEG, DEFAULT_VFOV_DEG
                src = (f'the maker states no FOV: the default {DEFAULT_HFOV_DEG:g}° × {DEFAULT_VFOV_DEG:g}° is used '
                       '(a default, not a device figure)')
            try:
                cam = CameraModel.from_fov(w, h, hf, vf, model=model)
            except ValueError as e:                       # a custom entry's FOV the lens model cannot show
                raise LibraryError(f'The {im["name"]} imager of {entry.title} cannot be modelled: {e}. Edit the '
                                   'device (a pinhole lens shows less than 180°; choose a fisheye lens model for '
                                   'wider).') from None
            label = (f'{im["name"]}: {MODEL_NAMES.get(model, model)} {w} × {h} px ({why}), FOV {hf:.4g}°'
                     + (f' × {vf:.4g}°' if vf is not None else ' (square pixels)') + f', {src}')
        if not stated:
            label += '; its position in the device is an assumption (not stated by the maker, see the entry note)'
        out.append(Imager(im['name'], im['role'], pose, cam, label=label, pose_stated=stated,
                          calibrated=cal is not None, model_id=entry.id))
    return out


MODEL_NAMES = {'pinhole': 'pinhole', 'brown': 'Brown-Conrady', 'kb': 'Kannala-Brandt', 'equidistant': 'equidistant',
               'equisolid': 'equisolid', 'stereographic': 'stereographic'}


def projector(sensor) -> dict | None:
    """The sensor's IR projector: {'hfov_deg', 'vfov_deg', 'dfov_deg', 'pose' (frames.Pose in the body frame),
    'pose_stated', 'model_id'}, or None when its device has none (or the sensor has no device)."""
    e = get(getattr(sensor, 'model_id', '') or '')
    pj = e.data.get('projector') if e is not None else None
    if not pj:
        return None
    out = copy.deepcopy(pj)
    out['pose'] = Pose.from_dict(pj['pose']) if isinstance(pj.get('pose'), dict) else Pose(np.zeros(3), _OPTICAL_Q)
    out['pose_stated'] = bool(pj.get('pose_stated', False))
    out['model_id'] = e.id
    return out


def depth_spec(sensor) -> dict | None:
    """A copy of the device's depth block (technology, reference_imager, stereo_pair, baseline_m, min_z table,
    max_range_m, rms_error, subpixel_px, depth_fov, ...) plus 'model_id'; None for a sensor with no depth device."""
    e = get(getattr(sensor, 'model_id', '') or '')
    dp = e.data.get('depth') if e is not None else None
    if not dp:
        return None
    out = copy.deepcopy(dp)
    out['model_id'] = e.id
    return out


def min_z(sensor, width: int, height: int) -> float | None:
    """The stated minimum Z (m) for a depth stream of width × height; None when the maker does not state one."""
    dp = depth_spec(sensor)
    for row in (dp or {}).get('min_z') or []:
        if row.get('width') == int(width) and row.get('height') == int(height):
            return row.get('m')
    return None
