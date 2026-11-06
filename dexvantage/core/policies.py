"""Policies: the robot-learning policies the data is for (spec §5, §1.1.6), and what each takes from a camera.

A project policy (core/project.py Policy) names a policy library entry (core/library, kind 'policy': one JSON file per
policy in dexvantage/resources/library/policy, written from the model card, the paper and the official code, each
read on a stated date), the cameras it takes ([] = every camera and depth sensor of the active rig) and the user's own
figures over the entry's ('overrides', each with where it comes from: "measured by the user ...").

Every field of an entry is {"value": ..., "source": <index into the entry's sources> | null, "quote": what the source
says}. A value no source states is null and is shown as "not stated by the source"; nothing here fills a gap (§1.1.1,
§1.1.6: model timings and input sizes are cited or measured, never a constant in the code).

API
    FIELDS                                   field -> Field(label, unit, kind, text), in display order
    CROP_RULES                               crop rule -> what it does to a camera image
    library_entries() -> [library.Entry]     the policy entries
    resolve(project, policy_uid) -> {field: (value, source text)}     the overrides applied
    resolve_detail(project, policy_uid) -> {field: {'value', 'source', 'origin', 'quote', 'hardware'}}
    policy_cameras(project, policy) -> [Sensor]
    check_value(field, value) -> value       a user's figure, normalised; PolicyError says what is wrong
    user_source(text) -> str                 'measured by the user: ...'
    crop_geometry(rule, src_size, input_size, crop_side_fraction=None, pre_resize=None) -> CropGeometry
    input_geometry(project, policy_uid, src_size) -> CropGeometry      from the policy's resolved fields
    summary(project) -> dict                 every policy resolved (JSON-able: the Policies page's export)
    entry_problems(entry_data) -> [str]      what is wrong with a policy entry's policy block

The crop geometry (docs/models/policies.md) maps camera pixels to the policy's input pixels for each crop rule, with
the integer rounding of the code each rule comes from (openpi's resize_with_pad, LeRobot's top-left variant,
torchvision's CenterCrop, the GR00T N1 VideoCrop). Coordinates are continuous pixel-edge coordinates: x runs from 0 at
the left edge of the image to W at its right edge (an OpenCV pixel centre u is at x = u + 0.5).

Known limits: entries hold what the sources publish, and most publish one configuration (a fine-tune can change the
input size, the crop or the chunk length: enter your own as overrides); inference times are for the hardware each
source names, and not every source names one; control_hz is null where a source gives several robots' rates and none
is the policy's own; the crop geometry is the inference-time crop (training augmentation such as random crops,
rotations and colour jitter is not modelled); interpolation filters (bilinear, bicubic, Lanczos, antialiasing) are not
modelled, only where each camera pixel lands; Diffusion Policy's stretch from the camera stream to 320 × 240 is an
assumption (the paper does not say whether it stretches or crops); a policy's per-camera-role inputs (Octo's 128 ×
128 wrist image) are listed, not yet applied per camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .jobs import StudyError

NOT_STATED = 'not stated by the source'
USER_PREFIX = 'measured by the user'


class PolicyError(StudyError):
    """A policy figure or request DexVantage cannot use; the message says why in plain words."""


@dataclass(frozen=True)
class Field:
    label: str
    unit: str
    kind: str          # 'size' [w, h] | 'int' | 'num' | 'fraction' | 'crop' | 'ensemble' (a number or false)
    text: str


FIELDS: dict = {
    'input_resolution': Field('Input resolution', 'px', 'size', 'The width × height of each image the model takes.'),
    'crop_rule': Field('Crop rule', '', 'crop', 'How a camera image is turned into the input (CROP_RULES).'),
    'crop_side_fraction': Field('Crop fraction', '', 'fraction',
                                'center_crop_resize: the fraction of each image side the centred crop keeps.'),
    'pre_resize': Field('Resize before the crop', 'px', 'size',
                        'center_crop: the width × height the image is resized to before the crop.'),
    'patch_px': Field('ViT patch', 'px', 'int', 'The edge of one ViT patch in input pixels (null for a CNN).'),
    'tokens_per_image': Field('Tokens per image', '', 'int', 'Visual tokens per camera image.'),
    'wrist_input_resolution': Field('Wrist input', 'px', 'size', 'Wrist images, where they are sized differently.'),
    'wrist_tokens_per_image': Field('Wrist tokens', '', 'int', 'Visual tokens per wrist image.'),
    'n_cameras': Field('Cameras', '', 'int', 'The number of camera images per step.'),
    'action_chunk': Field('Action chunk H', 'steps', 'int', 'Actions predicted per inference.'),
    'execute_steps': Field('Executed per chunk', 'steps', 'int', 'Actions executed before the next inference.'),
    'temporal_ensemble_m': Field('Temporal ensemble m', '', 'ensemble',
                                 'm in the weights exp(−m·i); "not used" when the source says it uses none.'),
    'inference_ms': Field('Inference time', 'ms', 'num', 'One inference, on the hardware named.'),
    'control_hz': Field('Control rate', 'Hz', 'num', 'The rate actions are sent to the robot.'),
}

# the fields the studies need (the Policies page counts these when they are not stated)
NEEDED = ('input_resolution', 'crop_rule', 'action_chunk', 'inference_ms', 'control_hz')

CROP_RULES: dict = {
    'none': 'The camera image as captured: no resize, no crop (the policy was trained at the camera\'s own size).',
    'resize': 'Resized to the input size, aspect ratio not kept (stretched).',
    'resize_pad': 'Resized to fit the input keeping its aspect ratio; the rest padded with black, evenly on both '
                  'sides (openpi resize_with_pad).',
    'resize_pad_top_left': 'Resized to fit the input keeping its aspect ratio; padded on the left and top (LeRobot '
                           'resize_with_pad, the SmolVLA convention).',
    'center_crop': 'Optionally resized to pre_resize, then a centred crop of the input size (torchvision '
                   'CenterCrop).',
    'center_crop_resize': 'A centred crop of crop_side_fraction of each side, then resized to the input size '
                          '(GR00T N1 VideoCrop + VideoResize; Octo at inference).',
}
CROP_LABEL = {'none': 'as captured', 'resize': 'resized (stretched)', 'resize_pad': 'resized with padding',
              'resize_pad_top_left': 'resized, padded left and top', 'center_crop': 'centre crop',
              'center_crop_resize': 'centre crop, then resize'}


# ====================================================================== the library
def library_entries() -> list:
    from . import library
    return library.entries('policy')


def _entry(library_id: str):
    from . import library
    e = library.get(library_id) if library_id else None
    return e if e is not None and e.kind == 'policy' else None


def entry_problems(d) -> list:
    """Plain messages for what is wrong with a policy entry's policy block (the library schema checks its shape;
    this checks each value against its field, and that each source index names a source)."""
    if not isinstance(d, dict):
        return ['the entry must be a JSON object']
    msgs = []
    pb = d.get('policy')
    if not isinstance(pb, dict):
        return ['policy: a policy entry needs its policy block']
    n_src = len(d.get('sources') or [])
    for k, f in pb.items():
        if k not in FIELDS:
            msgs.append(f'policy.{k}: is not a policy field')
            continue
        if not isinstance(f, dict) or 'value' not in f or 'source' not in f:
            msgs.append(f'policy.{k}: must be {{"value", "source"}}')
            continue
        v, s = f['value'], f['source']
        if v is None:
            if s is not None:
                msgs.append(f'policy.{k}: a value no source states has no source index')
            continue
        if not (isinstance(s, int) and not isinstance(s, bool) and 0 <= s < n_src):
            msgs.append(f'policy.{k}: a stated value needs the index of its source (0 to {n_src - 1})')
        try:
            if check_value(k, v) != v:
                msgs.append(f'policy.{k}: {v!r} is not written in its normal form')
        except PolicyError as e:
            msgs.append(f'policy.{k}: {e}')
    for k in NEEDED:
        if k not in pb:
            msgs.append(f'policy.{k}: missing (write null when no source states it)')
    return msgs


# ====================================================================== values
def in_sentence(label: str) -> str:
    """A field label inside a sentence: lower case, acronyms and one-letter symbols kept ("ViT patch", "action
    chunk H")."""
    return ' '.join(w if len(w) == 1 or any(c.isupper() for c in w[1:]) else w.lower() for w in label.split())


def check_value(fieldname: str, value):
    """A figure for a field, normalised (a size as [w, h] whole pixels, a count as an int). Raises PolicyError."""
    f = FIELDS.get(fieldname)
    if f is None:
        raise PolicyError(f'"{fieldname}" is not a policy field. The fields are: {", ".join(FIELDS)}.')
    if value is None:
        return None
    name = in_sentence(f.label)
    if f.kind == 'size':
        if isinstance(value, str):
            value = value.replace('×', 'x').replace(',', 'x').lower().split('x')
        try:
            w, h = (float(v) for v in value)
        except (TypeError, ValueError):
            raise PolicyError(f'The {name} must be a width and a height in pixels, such as 224 × 224.') from None
        if not (w >= 1 and h >= 1 and w == int(w) and h == int(h)):
            raise PolicyError(f'The {name} must be whole pixels, at least 1 × 1.')
        return [int(w), int(h)]
    if f.kind == 'crop':
        v = str(value).strip()
        if v not in CROP_RULES:
            raise PolicyError(f'"{v}" is not a crop rule. The rules are: {", ".join(CROP_RULES)}.')
        return v
    if f.kind == 'ensemble' and (value is False or str(value).strip().lower() in ('false', 'not used', 'none used')):
        return False
    if isinstance(value, bool):
        raise PolicyError(f'The {name} must be a number.')
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise PolicyError(f'The {name} must be a number' + (f' in {f.unit}' if f.unit else '') + '.') from None
    if not math.isfinite(x):
        raise PolicyError(f'The {name} must be a finite number.')
    if f.kind == 'int':
        if x < 1 or x != int(x):
            raise PolicyError(f'The {name} must be a whole number, at least 1.')
        return int(x)
    if f.kind == 'fraction':
        if not 0 < x <= 1:
            raise PolicyError(f'The {name} must be more than 0 and at most 1.')
        return x
    if f.kind == 'ensemble':
        if x < 0:
            raise PolicyError('The temporal ensemble m must be 0 or more (or "not used").')
        return x
    if x <= 0:
        raise PolicyError(f'The {name} must be more than 0' + (f' {f.unit}' if f.unit else '') + '.')
    return x


def user_source(text: str) -> str:
    """Where a user's figure comes from, always saying it is the user's: 'measured by the user: <text>'."""
    t = ' '.join(str(text or '').split())
    if not t:
        raise PolicyError('Say where the figure comes from (how it was measured, on what hardware): a figure without '
                          'a source is not entered.')
    if t.lower().startswith(USER_PREFIX):
        return t
    return f'{USER_PREFIX}: {t}'


def format_value(fieldname: str, value) -> str:
    """A value for display: '—' when not stated."""
    f = FIELDS.get(fieldname)
    if value is None:
        return '—'
    if f is not None and f.kind == 'size':
        return f'{value[0]} × {value[1]}' + (f' {f.unit}' if f.unit else '')
    if f is not None and f.kind == 'crop':
        return CROP_LABEL.get(value, str(value))
    if value is False:
        return 'not used'
    if isinstance(value, float):
        txt = f'{value:.4f}'.rstrip('0').rstrip('.')
    else:
        txt = str(value)
    return txt + (f' {f.unit}' if f is not None and f.unit else '')


# ====================================================================== resolving
def policy(project, uid: str):
    return next((p for p in getattr(project, 'policies', []) if p.uid == uid), None)


def _source_text(entry, idx: int, quote: str) -> str:
    srcs = entry.sources
    s = srcs[idx] if isinstance(idx, int) and 0 <= idx < len(srcs) else {}
    title = s.get('title') or s.get('url') or entry.title
    text = f'{title}, {s.get("url", "")}, read {s.get("read", "?")}'
    return text + (f': {quote}' if quote else '')


def resolve_detail(project, policy_uid: str) -> dict:
    """{field: {'value', 'source' (text), 'origin' ('library' | 'user' | 'not_stated'), 'quote', 'hardware'}} for
    every field in FIELDS, the user's overrides over the library entry's figures."""
    pol = policy(project, policy_uid)
    if pol is None:
        raise PolicyError('That policy is not in the project (it may have been removed).')
    e = _entry(pol.library_id)
    block = (e.data.get('policy') or {}) if e is not None else {}
    out = {}
    for k in FIELDS:
        f = block.get(k) if isinstance(block.get(k), dict) else {}
        quote = str(f.get('quote') or '')
        hw = f.get('hardware')
        if f.get('value') is not None:
            d = {'value': f['value'], 'source': _source_text(e, f.get('source'), quote), 'origin': 'library',
                 'quote': quote, 'hardware': hw}
        else:
            why = NOT_STATED if e is not None else (f'the library has no policy "{pol.library_id}"'
                                                    if pol.library_id else 'no library entry chosen')
            d = {'value': None, 'source': why + (f' ({quote})' if quote else ''), 'origin': 'not_stated',
                 'quote': quote, 'hardware': None}
        ov = pol.overrides.get(k)
        if isinstance(ov, dict) and 'value' in ov:
            d = {'value': ov['value'], 'source': str(ov.get('source') or USER_PREFIX), 'origin': 'user',
                 'quote': '', 'hardware': None, 'library_value': d['value'], 'library_source': d['source']}
        out[k] = d
    return out


def resolve(project, policy_uid: str) -> dict:
    """field -> (value, source text) for every field in FIELDS, the overrides applied; value None is "not stated by
    the source" (the text says so)."""
    return {k: (d['value'], d['source']) for k, d in resolve_detail(project, policy_uid).items()}


def policy_cameras(project, pol) -> list:
    """The sensors of the active rig the policy takes: its cameras, or every camera and depth sensor when it names
    none (sensors no longer in the rig are left out)."""
    sensors = list(project.variant.sensors)
    if pol.cameras:
        return [s for s in sensors if s.uid in set(pol.cameras)]
    return [s for s in sensors if s.kind in ('camera', 'depth')]


def title_of(pol) -> str:
    e = _entry(pol.library_id)
    return e.name if e is not None else (pol.library_id or 'no library entry')


# ====================================================================== crop geometry
@dataclass(frozen=True)
class CropGeometry:
    """Where camera pixels land in the policy's input (pixel-edge coordinates, see the module docstring).

    src_rect: (x0, y0, x1, y1) of the camera image the input shows; dst_rect: where that region lands in the input
    (the rest of the input is padding); sx, sy: input pixels per camera pixel."""
    rule: str
    src_size: tuple
    input_size: tuple
    src_rect: tuple
    dst_rect: tuple
    sx: float
    sy: float
    note: str = ''

    @property
    def aspect_kept(self) -> bool:
        return abs(self.sx - self.sy) <= 1e-9 * max(self.sx, self.sy)

    @property
    def used_fraction(self) -> float:
        """The fraction of the camera image the input shows."""
        x0, y0, x1, y1 = self.src_rect
        return (x1 - x0) * (y1 - y0) / float(self.src_size[0] * self.src_size[1])

    @property
    def padded_fraction(self) -> float:
        """The fraction of the input that is padding."""
        x0, y0, x1, y1 = self.dst_rect
        return 1.0 - (x1 - x0) * (y1 - y0) / float(self.input_size[0] * self.input_size[1])

    def to_input(self, x, y):
        """Camera edge coordinates -> input edge coordinates (numpy arrays welcome)."""
        return (self.dst_rect[0] + (x - self.src_rect[0]) * self.sx,
                self.dst_rect[1] + (y - self.src_rect[1]) * self.sy)

    def to_source(self, x, y):
        """Input edge coordinates -> camera edge coordinates."""
        return (self.src_rect[0] + (x - self.dst_rect[0]) / self.sx,
                self.src_rect[1] + (y - self.dst_rect[1]) / self.sy)

    def to_dict(self) -> dict:
        return {'rule': self.rule, 'src_size': list(self.src_size), 'input_size': list(self.input_size),
                'src_rect': [float(v) for v in self.src_rect], 'dst_rect': [float(v) for v in self.dst_rect],
                'sx': float(self.sx), 'sy': float(self.sy), 'aspect_kept': self.aspect_kept,
                'used_fraction': self.used_fraction, 'padded_fraction': self.padded_fraction, 'note': self.note}


def _fit(W, H, w, h):
    """openpi / LeRobot resize_with_pad: the resized size, int() of the size divided by the larger ratio."""
    ratio = max(W / w, H / h)
    return int(W / ratio), int(H / ratio)


def crop_geometry(rule: str, src_size, input_size, crop_side_fraction=None, pre_resize=None) -> CropGeometry:
    """Where a src_size (W, H) camera image lands in an input_size (w, h) policy input under a crop rule."""
    W, H = (int(v) for v in src_size)
    w, h = (int(v) for v in input_size)
    if W < 1 or H < 1 or w < 1 or h < 1:
        raise PolicyError('Image sizes must be at least 1 × 1 pixels.')
    if rule not in CROP_RULES:
        raise PolicyError(f'"{rule}" is not a crop rule. The rules are: {", ".join(CROP_RULES)}.')
    if rule in ('none', 'resize'):
        note = ''
        if rule == 'none' and (W, H) != (w, h):
            note = (f'The policy takes images as captured at {w} × {h}; this stream is {W} × {H}, so it is drawn '
                    'stretched to the trained size.')
        return CropGeometry(rule, (W, H), (w, h), (0.0, 0.0, float(W), float(H)), (0.0, 0.0, float(w), float(h)),
                            w / W, h / H, note)
    if rule in ('resize_pad', 'resize_pad_top_left'):
        if (W, H) == (w, h):
            rw, rh = w, h
        else:
            rw, rh = _fit(W, H, w, h)
        if rule == 'resize_pad':
            x0, y0 = (w - rw) // 2, (h - rh) // 2
        else:
            x0, y0 = w - rw, h - rh
        return CropGeometry(rule, (W, H), (w, h), (0.0, 0.0, float(W), float(H)),
                            (float(x0), float(y0), float(x0 + rw), float(y0 + rh)), rw / W, rh / H)
    if rule == 'center_crop':
        if pre_resize is not None:
            Wp, Hp = (int(v) for v in pre_resize)
        else:
            Wp, Hp = W, H
        if w > Wp or h > Hp:
            raise PolicyError(f'A {w} × {h} centre crop does not fit in a {Wp} × {Hp} image: choose a larger stream or '
                              'set the resize before the crop.')
        kx, ky = Wp / W, Hp / H
        left, top = int(round((Wp - w) / 2.0)), int(round((Hp - h) / 2.0))
        return CropGeometry(rule, (W, H), (w, h), (left / kx, top / ky, (left + w) / kx, (top + h) / ky),
                            (0.0, 0.0, float(w), float(h)), kx, ky,
                            '' if pre_resize is None else f'resized to {Wp} × {Hp} first (stretched)')
    # center_crop_resize
    f = 1.0 if crop_side_fraction is None else float(crop_side_fraction)
    if not 0 < f <= 1:
        raise PolicyError('The crop fraction must be more than 0 and at most 1.')
    cw, ch = max(1, int(W * f)), max(1, int(H * f))
    left, top = int(round((W - cw) / 2.0)), int(round((H - ch) / 2.0))
    return CropGeometry(rule, (W, H), (w, h), (float(left), float(top), float(left + cw), float(top + ch)),
                        (0.0, 0.0, float(w), float(h)), w / cw, h / ch)


def input_geometry(project, policy_uid: str, src_size) -> CropGeometry:
    """crop_geometry from the policy's resolved input resolution, crop rule, crop fraction and resize."""
    r = resolve(project, policy_uid)
    size, rule = r['input_resolution'][0], r['crop_rule'][0]
    if size is None or rule is None:
        what = 'input resolution' if size is None else 'crop rule'
        raise PolicyError(f'The policy\'s {what} is not stated by its source: enter it on the Policies page (a figure '
                          'you measured or read in the model\'s code).')
    return crop_geometry(rule, src_size, size, r['crop_side_fraction'][0], r['pre_resize'][0])


# ====================================================================== the export
def summary(project) -> dict:
    """Every policy of the project, resolved, with its cameras: JSON-able."""
    out = []
    for pol in getattr(project, 'policies', []):
        det = resolve_detail(project, pol.uid)
        cams = policy_cameras(project, pol)
        out.append({'uid': pol.uid, 'name': pol.name, 'library_id': pol.library_id, 'entry': title_of(pol),
                    'cameras': [{'uid': s.uid, 'name': s.name} for s in cams],
                    'cameras_rule': 'listed' if pol.cameras else 'every camera and depth sensor',
                    'fields': {k: {'value': d['value'], 'source': d['source'], 'origin': d['origin'],
                                   'hardware': d.get('hardware')} for k, d in det.items()},
                    'not_stated': [k for k in NEEDED if det[k]['value'] is None]})
    return {'study': 'policies', 'title': 'Policies', 'policies': out,
            'not_stated': sum(len(p['not_stated']) for p in out)}
