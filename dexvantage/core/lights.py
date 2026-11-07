"""Light sources of the workspace (spec §7.8): project.workspace.lights, their photometry and the irradiance they give.

    KINDS                                  'panel', 'ring', 'window', 'point', 'spot'
    clean_light(d) -> dict                 a light normalised to the contract (below); junk dropped, figures checked
    new_light(kind, position, name='')     a light with no stated figures (the user enters them, or picks an entry)
    from_library(entry_id, pose=None)      a light from a 'light' library entry, every figure sourced from it
    emitters(light, band, n_area=4)        Emitter list: point emitters standing for the light (area lights split)
    irradiance(emitters, P) -> (wi, E_perp)   per emitter: unit directions to it (K, N, 3) and the irradiance on a
                                           plane facing it (K, N), lux (band 'vis') or W/m² (band 'ir')
    sun_beam(light, band)                  a window's direct-sun beam: (direction, irradiance normal to it) or None
    describe(light) -> dict                what the photometry is and where each figure comes from (for the page)

The contract (project.workspace.lights, each a dict):
    {'uid', 'name', 'kind', 'pose' {'position', 'quaternion' (x, y, z, w)}, 'size_m': [w, h] | null,
     'luminance_cd_m2': {'value', 'source'} | null, 'luminous_flux_lm': {'value', 'source'} | null,
     'ir_irradiance_w_m2': {'value', 'source', 'at_m' (optional)} | null, 'cct_k': {'value', 'source'} | null}
plus the optional keys this module keeps: 'library_id' (the entry the figures came from), 'beam_deg' {'value',
'source'} (a spot's full beam angle), 'inner_m' (a ring's inner diameter), 'dim' (0..1, the share of the stated output
the light is run at; 1 by default), 'enabled' (true), 'sun' (a window only: {'direction': [x, y, z] towards the sun,
world frame, 'illuminance_lux': {'value', 'source'} | null, 'ir_irradiance_w_m2': {'value', 'source'} | null}).
A figure is a value with its source (a datasheet with its URL and date, or "your measurement"); a figure nobody states
is null, and the study says what it could not compute without it. Nothing here fills a gap.

Geometry. The light's pose is its centre; its emitting face looks along its local +z (a panel or window faces into the
room along +z; a spot points along +z; a ring light's ring lies in its local x-y plane). size_m is [w, h] of the face
along local x and y (a ring: its outer diameter twice).

Photometry (visible band, photometric units: candela, lux):
- point: isotropic, I = Φ / 4π; with a luminance and a size, I = L · A instead.
- spot: a uniform cone of full angle β (beam_deg): I = Φ / (2π (1 − cos(β/2))) inside it, 0 outside. A spot without
  a stated beam angle has no intensity (null, said).
- panel, window, ring: Lambertian area emitters of uniform luminance, L as stated or L = Φ / (π A) from the flux
  (A = w · h; a ring: the annulus π (R² − r²), r from inner_m, else the disc), each split into n_area × n_area patches
  (a ring: 4 n_area patches round the annulus), each patch a point emitter of intensity I(θ) = L · A_patch · cos θ on
  the face's side (Lambert's cosine law), 0 behind it.
The irradiance from a point emitter on a plane facing it is E⊥ = I(θ) / r² (the inverse-square law); on a surface of
normal n it is E⊥ · max(0, n · ωi). Splitting an area light is exact as the patch shrinks; the validation check
(lighting.area_vs_closed_form) compares the split with the closed-form irradiance of a rectangle and with Monte Carlo.

IR (W/m², the band the depth sensors' projectors use): ir_irradiance_w_m2 is the in-band irradiance the light gives on
a surface facing it on its axis at 'at_m' (a meter reading or a datasheet figure at a stated distance): a point-like
light then has the on-axis intensity J = E · at_m², spread like its visible light; an area light the radiance
J / A. An area light without 'at_m' gives its IR as the irradiance at its own face (a window: the IR that passes
through it, measured at the glass): radiance E / π (a Lambertian exitance M = E). The sun through a window (its 'sun'
block) is a parallel beam of the stated illuminance and IR irradiance, normal to the beam, that reaches a point when
the line from the point towards the sun passes through the window's rectangle (and nothing in the scene blocks it).

'dim' scales every figure of the light (the stated figure stays as stated).

Known limits: emitters are Lambertian or uniform cones (no measured beam profile, no IES file); an area light's shadow
is taken from its centre (a hard shadow; soft shadow edges are not modelled); inter-reflections (the walls, the table
lighting the object) are not modelled, so shadowed parts are darker than in a real room; a window is a diffuse
emitter plus the direct sun beam, with no sky model; the visible band is photometric (the camera's spectral response
is taken as the eye's, V(λ)), and the IR band is whatever band the user's figures are for.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .frames import Pose, quat_to_matrix

KINDS = ('panel', 'ring', 'window', 'point', 'spot')
AREA_KINDS = ('panel', 'ring', 'window')
FIGURES = ('luminance_cd_m2', 'luminous_flux_lm', 'ir_irradiance_w_m2', 'cct_k')
KIND_TITLE = {'panel': 'LED panel', 'ring': 'Ring light', 'window': 'Window', 'point': 'Point light', 'spot': 'Spot'}
DEFAULT_N_AREA = 4                    # patches per side of an area light: a DEFAULT (the check measures its error)
DEFAULT_SIZE_M = {'panel': [0.6, 0.6], 'window': [1.0, 1.2], 'ring': [0.45, 0.45]}   # a new light's size: a DEFAULT


class LightError(ValueError):
    """A light DexVantage cannot use; the message says why in plain words."""


# ====================================================================== normalising
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def clean_figure(f, extra=()) -> dict | None:
    """{'value' ≥ 0, 'source': text (+ extra keys)} or None (a figure without a value, a source or a sane number)."""
    if not isinstance(f, dict):
        return None
    v = _num(f.get('value'))
    if v is None or v < 0:
        return None
    out = {'value': v, 'source': str(f.get('source') or 'your entry')}
    for k in extra:
        x = _num(f.get(k))
        if x is not None and x > 0:
            out[k] = x
    return out


def _pose(d) -> dict:
    try:
        p = Pose.from_dict(d)
    except Exception:                                    # noqa: BLE001 — junk pose: the identity
        p = Pose()
    q = np.asarray(p.quaternion, float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    pos = np.asarray(p.position, float)
    if not np.isfinite(pos).all():
        pos = np.zeros(3)
    return {'position': [float(x) for x in pos], 'quaternion': [float(x) for x in q / np.linalg.norm(q)]}


def clean_light(d) -> dict:
    """The light d normalised to the contract (module docstring)."""
    from .project import new_uid
    d = d if isinstance(d, dict) else {}
    kind = str(d.get('kind') or 'point')
    kind = kind if kind in KINDS else 'point'
    out = {'uid': str(d.get('uid') or new_uid()), 'name': str(d.get('name') or KIND_TITLE[kind]), 'kind': kind,
           'pose': _pose(d.get('pose'))}
    size = d.get('size_m')
    s = None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        a, b = _num(size[0]), _num(size[1])
        if a is not None and b is not None and a > 0 and b > 0:
            s = [a, b]
    out['size_m'] = s
    out['luminance_cd_m2'] = clean_figure(d.get('luminance_cd_m2'))
    out['luminous_flux_lm'] = clean_figure(d.get('luminous_flux_lm'))
    out['ir_irradiance_w_m2'] = clean_figure(d.get('ir_irradiance_w_m2'), extra=('at_m',))
    out['cct_k'] = clean_figure(d.get('cct_k'))
    if d.get('library_id'):
        out['library_id'] = str(d['library_id'])
    bd = clean_figure(d.get('beam_deg'))
    if bd is not None and 0 < bd['value'] < 180:
        out['beam_deg'] = bd
    inner = _num(d.get('inner_m'))
    if inner is not None and inner > 0:
        out['inner_m'] = inner
    dim = _num(d.get('dim'))
    out['dim'] = 1.0 if dim is None else min(max(dim, 0.0), 1.0)
    out['enabled'] = bool(d.get('enabled', True))
    sun = d.get('sun')
    if kind == 'window' and isinstance(sun, dict):
        v = np.asarray([_num(x) for x in (sun.get('direction') or [None] * 3)][:3], dtype=object)
        if len(v) == 3 and all(x is not None for x in v) and np.linalg.norm(np.asarray(v, float)) > 1e-9:
            dv = np.asarray(v, float)
            out['sun'] = {'direction': [float(x) for x in dv / np.linalg.norm(dv)],
                          'illuminance_lux': clean_figure(sun.get('illuminance_lux')),
                          'ir_irradiance_w_m2': clean_figure(sun.get('ir_irradiance_w_m2'))}
    return out


def lights_of(project) -> list:
    """The project's lights, normalised (the stored dicts are not changed)."""
    return [clean_light(x) for x in (project.workspace.lights or []) if isinstance(x, dict)]


def new_light(kind: str, position, name: str = '', quaternion=None) -> dict:
    """A light of kind at position (world, m), facing down (local +z along world −z) unless quaternion is given, with
    no stated figures."""
    if kind not in KINDS:
        raise LightError(f'There is no light kind "{kind}". The kinds are: {", ".join(KINDS)}.')
    q = quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0]      # 180° about x: +z points down
    return clean_light({'kind': kind, 'name': name or KIND_TITLE[kind],
                        'pose': {'position': [float(x) for x in position], 'quaternion': list(q)},
                        'size_m': copy.deepcopy(DEFAULT_SIZE_M.get(kind))})


def from_library(entry_id: str, pose=None, name: str = '') -> dict:
    """A light made from a 'light' library entry: its kind, size and every stated figure, each sourced 'library:
    <id> (<url>, read <date>)'. LightError when there is no such light entry."""
    from . import library as LB
    e = LB.get(entry_id)
    if e is None or e.kind != 'light' or not isinstance(e.data.get('light'), dict):
        raise LightError(f'"{entry_id}" is not a light in the library.')
    L = e.data['light']
    src = e.sources[0] if e.sources else {}
    where = f'library: {e.title} ({src.get("url", "")}, read {src.get("read", "")})'

    def fig(v, **extra):
        return None if v is None else {'value': float(v), 'source': where, **extra}
    kind = L.get('type') if L.get('type') in KINDS else 'panel'
    d = {'kind': kind, 'name': name or e.title, 'library_id': e.id,
         'pose': pose.to_dict() if isinstance(pose, Pose) else (pose or {'position': [0.0, 0.0, 1.5],
                                                                         'quaternion': [1.0, 0.0, 0.0, 0.0]}),
         'size_m': L.get('emitting_size_m'), 'luminance_cd_m2': fig(L.get('luminance_cd_m2')),
         'luminous_flux_lm': fig(L.get('luminous_flux_lm')), 'cct_k': fig(L.get('cct_k')),
         'ir_irradiance_w_m2': (fig(L.get('ir_irradiance_w_m2'), at_m=float(L['ir_at_m'])) if L.get('ir_at_m')
                                else fig(L.get('ir_irradiance_w_m2'))),
         'beam_deg': fig(L.get('beam_deg')), 'inner_m': L.get('ring_inner_m')}
    return clean_light(d)


# ====================================================================== photometry
def area_m2(light: dict) -> float | None:
    s = light.get('size_m')
    if not s:
        return None
    if light['kind'] == 'ring':
        R = 0.5 * float(min(s))
        r = 0.5 * float(light.get('inner_m') or 0.0)
        r = r if 0 < r < R else 0.0
        return math.pi * (R * R - r * r)
    return float(s[0]) * float(s[1])


def visible_photometry(light: dict) -> dict:
    """{'model': 'iso' | 'cone' | 'lambert', 'I_cd' (on-axis, a point or spot), 'L_cd_m2' (an area light), 'from':
    what it was derived from, 'missing': why it could not be (or ''), 'half_angle_rad' (a spot)}."""
    k, dim = light['kind'], float(light.get('dim', 1.0))
    L, F = light.get('luminance_cd_m2'), light.get('luminous_flux_lm')
    A = area_m2(light)
    out = {'model': 'lambert' if k in AREA_KINDS else ('cone' if k == 'spot' else 'iso'), 'I_cd': None,
           'L_cd_m2': None, 'from': '', 'missing': '', 'half_angle_rad': None}
    if k in AREA_KINDS:
        if A is None:
            out['missing'] = 'its size is not given'
        elif L is not None:
            out['L_cd_m2'], out['from'] = L['value'] * dim, f'the stated luminance ({L["source"]})'
        elif F is not None:
            out['L_cd_m2'] = F['value'] * dim / (math.pi * A)
            out['from'] = f'L = Φ / (π A) from the stated flux ({F["source"]}), a Lambertian emitter of {A:.3g} m²'
        else:
            out['missing'] = 'neither its luminance nor its luminous flux is stated'
        return out
    if k == 'spot':
        bd = light.get('beam_deg')
        if bd is None:
            out['missing'] = 'its beam angle is not stated'
            return out
        half = math.radians(bd['value']) / 2
        out['half_angle_rad'] = half
        if F is not None:
            out['I_cd'] = F['value'] * dim / (2 * math.pi * (1 - math.cos(half)))
            out['from'] = (f'I = Φ / (2π (1 − cos(β/2))) from the stated flux ({F["source"]}) and beam angle '
                           f'({bd["source"]})')
        elif L is not None and A is not None:
            out['I_cd'], out['from'] = L['value'] * A * dim, f'I = L · A from the stated luminance ({L["source"]})'
        else:
            out['missing'] = 'its luminous flux is not stated'
        return out
    if F is not None:
        out['I_cd'], out['from'] = F['value'] * dim / (4 * math.pi), f'I = Φ / 4π from the stated flux ({F["source"]})'
    elif L is not None and A is not None:
        out['I_cd'], out['from'] = L['value'] * A * dim, f'I = L · A from the stated luminance ({L["source"]})'
    else:
        out['missing'] = 'its luminous flux is not stated'
    return out


def ir_photometry(light: dict) -> dict:
    """{'J_w_sr' (on-axis W/sr, point-like), 'L_w_m2_sr' (an area light), 'from', 'missing'} for the IR band."""
    k, dim = light['kind'], float(light.get('dim', 1.0))
    E = light.get('ir_irradiance_w_m2')
    A = area_m2(light)
    out = {'J_w_sr': None, 'L_w_m2_sr': None, 'from': '', 'missing': ''}
    if E is None:
        out['missing'] = 'its IR irradiance is not stated'
        return out
    at = E.get('at_m')
    if k in AREA_KINDS:
        if A is None:
            out['missing'] = 'its size is not given'
        elif at:
            out['L_w_m2_sr'] = E['value'] * at * at * dim / A
            out['from'] = f'J = E · d² at {at:g} m on its axis, spread over its face ({E["source"]})'
        else:
            out['L_w_m2_sr'] = E['value'] * dim / math.pi
            out['from'] = f'the stated IR irradiance at its face, as a Lambertian exitance: L = E / π ({E["source"]})'
        return out
    if not at:
        out['missing'] = 'the distance its IR irradiance was stated at is not given'
        return out
    out['J_w_sr'] = E['value'] * at * at * dim
    if k == 'spot' and light.get('beam_deg') is None:
        out['J_w_sr'], out['missing'] = None, 'its beam angle is not stated'
        return out
    out['from'] = f'J = E · d² at {at:g} m on its axis ({E["source"]})'
    return out


@dataclass
class Emitters:
    """Point emitters standing for one light: positions (K, 3), the face's axis (3,) and model, the on-axis intensity
    per emitter (K,) (cd or W/sr) and, for a spot, its half angle."""
    light_uid: str
    pos: np.ndarray
    axis: np.ndarray
    model: str
    I0: np.ndarray
    half_angle: float | None = None

    def __len__(self):
        return len(self.pos)


def _frame(light):
    R = quat_to_matrix(np.asarray(light['pose']['quaternion'], float))
    return R, np.asarray(light['pose']['position'], float)


def patch_centres(light: dict, n_area: int = DEFAULT_N_AREA) -> tuple[np.ndarray, np.ndarray]:
    """(centres (K, 3) world, areas (K,)) of the patches an area light is split into."""
    R, t = _frame(light)
    n = max(1, int(n_area))
    w, h = (float(v) for v in light['size_m'])
    if light['kind'] == 'ring':
        Ro = 0.5 * min(w, h)
        ri = 0.5 * float(light.get('inner_m') or 0.0)
        ri = ri if 0 < ri < Ro else 0.0
        m = 4 * n
        ang = 2 * np.pi * (np.arange(m) + 0.5) / m
        rc = math.sqrt(0.5 * (Ro * Ro + ri * ri)) if ri > 0 else Ro * math.sqrt(0.5)
        loc = np.stack([rc * np.cos(ang), rc * np.sin(ang), np.zeros(m)], 1)
        a = np.full(m, math.pi * (Ro * Ro - ri * ri) / m)
    else:
        u = (np.arange(n) + 0.5) / n - 0.5
        X, Y = np.meshgrid(u * w, u * h)
        loc = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1)
        a = np.full(n * n, w * h / (n * n))
    return loc @ R.T + t, a


def emitters(light: dict, band: str = 'vis', n_area: int = DEFAULT_N_AREA) -> Emitters | None:
    """The light's point emitters in band 'vis' (cd) or 'ir' (W/sr); None when the figures it needs are not stated
    (visible_photometry / ir_photometry say which) or the light is switched off."""
    if not light.get('enabled', True):
        return None
    R, t = _frame(light)
    axis = R[:, 2].copy()
    ph = visible_photometry(light) if band == 'vis' else ir_photometry(light)
    if light['kind'] in AREA_KINDS:
        L = ph.get('L_cd_m2') if band == 'vis' else ph.get('L_w_m2_sr')
        if L is None:
            return None
        P, a = patch_centres(light, n_area)
        return Emitters(light['uid'], P, axis, 'lambert', L * a)
    I0 = ph.get('I_cd') if band == 'vis' else ph.get('J_w_sr')
    if I0 is None:
        return None
    model = 'cone' if light['kind'] == 'spot' else 'iso'
    half = None
    if model == 'cone':
        half = math.radians(light['beam_deg']['value']) / 2
    return Emitters(light['uid'], t[None, :].copy(), axis, model, np.array([float(I0)]), half)


def irradiance(em: Emitters, P) -> tuple[np.ndarray, np.ndarray]:
    """(wi (K, N, 3) unit directions from the points towards each emitter, E⊥ (K, N) the irradiance on a plane facing
    the emitter: I(θ) / r², with I(θ) = I0 (iso), I0 inside the cone (cone), I0 max(0, cos θ) (lambert))."""
    P = np.asarray(P, float).reshape(-1, 3)
    d = em.pos[:, None, :] - P[None, :, :]                # (K, N, 3) point -> emitter
    r2 = np.maximum(np.einsum('knj,knj->kn', d, d), 1e-12)
    wi = d / np.sqrt(r2)[..., None]
    cos_e = -np.einsum('knj,j->kn', wi, em.axis)          # the angle at the emitter from its axis
    I = np.broadcast_to(em.I0[:, None], r2.shape)
    if em.model == 'lambert':
        I = I * np.maximum(cos_e, 0.0)
    elif em.model == 'cone':
        I = np.where(cos_e >= math.cos(em.half_angle), I, 0.0)
    return wi, I / r2


def sun_beam(light: dict, band: str = 'vis'):
    """(unit direction towards the sun (3,), irradiance normal to the beam) of a window's sun, or None."""
    sun = light.get('sun') if light.get('enabled', True) else None
    if not sun:
        return None
    f = sun.get('illuminance_lux') if band == 'vis' else sun.get('ir_irradiance_w_m2')
    if f is None:
        return None
    return np.asarray(sun['direction'], float), float(f['value']) * float(light.get('dim', 1.0))


def through_window(light: dict, P, direction) -> np.ndarray:
    """(N,) True where the line from P towards direction crosses the window's rectangle (from inside the room, the
    side its +z faces)."""
    R, t = _frame(light)
    P = np.asarray(P, float).reshape(-1, 3)
    z = R[:, 2]
    dz = float(np.dot(direction, z))
    if dz >= -1e-9:                                      # the sun must lie behind the window (its −z side)
        return np.zeros(len(P), bool)
    s = ((P - t) @ z) / -dz                               # distance along the beam to the window's plane
    X = P + s[:, None] * np.asarray(direction, float) - t
    w, h = (float(v) for v in (light.get('size_m') or [0.0, 0.0]))
    return (s > 0) & (np.abs(X @ R[:, 0]) <= w / 2) & (np.abs(X @ R[:, 1]) <= h / 2)


def describe(light: dict) -> dict:
    """What the page shows about a light's photometry: {'vis', 'ir', 'sun'} each with 'text' and 'ok'."""
    v, i = visible_photometry(light), ir_photometry(light)
    if v['missing']:
        vt = f'Visible light not computed: {v["missing"]}.'
    elif v['L_cd_m2'] is not None:
        vt = f'Luminance {v["L_cd_m2"]:.4g} cd/m²: {v["from"]}.'
    else:
        vt = f'Intensity {v["I_cd"]:.4g} cd on its axis: {v["from"]}.'
    if i['missing']:
        it = f'IR not computed: {i["missing"]}.'
    elif i['L_w_m2_sr'] is not None:
        it = f'IR radiance {i["L_w_m2_sr"]:.4g} W/(m² sr): {i["from"]}.'
    else:
        it = f'IR intensity {i["J_w_sr"]:.4g} W/sr on its axis: {i["from"]}.'
    out = {'vis': {'text': vt, 'ok': not v['missing']}, 'ir': {'text': it, 'ok': not i['missing']}}
    if light.get('sun'):
        s = light['sun']
        parts = [f'{k} {s[k]["value"]:g} ({s[k]["source"]})' for k in ('illuminance_lux', 'ir_irradiance_w_m2')
                 if s.get(k)]
        out['sun'] = {'text': 'Direct sun through it: ' + ('; '.join(parts) if parts else 'no figures stated') + '.',
                      'ok': bool(parts)}
    if light.get('dim', 1.0) < 1.0:
        out['dim'] = {'text': f'Run at {light["dim"] * 100:.0f} % of its stated output.', 'ok': True}
    return out
