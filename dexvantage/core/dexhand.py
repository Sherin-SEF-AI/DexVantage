"""Dexterous hands (spec §6, §7.14): which links are which finger, the fingertip contact pads, and the power and
precision grasp postures the multi-finger occlusion study sweeps.

    structure(project, robot_uid) -> HandStructure    fingers (tip link, links, short name, title), palm, the rest
    pad_faces(model, tip_link, facing_deg=PAD_FACING_DEG) -> (M, 3, 3) triangles in the tip link's frame
    contact_pads_for(model, tip_links) -> [{'link', 'name', 'faces'}]    Robot.contact_pads records (the rule below)
    synergies(project, robot_uid) -> [Synergy]          the postures of the study's settings, else the defaults
    posture(model, synergy, closure) -> q              the joint vector at a closure in [0, 1]
    read_keyframes(hand_xml, keyframes_xml) -> {name: {joint: value}}

Fingers. Each fingertip frame (Robot.fingertip_frames) is the end of one finger. A finger's links are the chain of
MuJoCo bodies from its tip up to, not including, the first body that also carries another fingertip (the palm, for a
hand whose fingers all branch from it). The palm is the lowest common ancestor of every fingertip; the palm group is
that body and its descendants that belong to no finger. Every other link of the robot (the wrist, the forearm, the
arm) is "wrist and arm". A finger's short name is its tip link's name with the tips' common prefix and suffix taken
off at a word boundary (hand_rh_ffdistal ... hand_rh_thdistal give ff, mf, rf, lf, th; gripper_right_pad and
gripper_left_pad give right and left). A tip that is itself an ancestor of another tip (a parallel gripper's fixed
jaw) owns its own link. FINGER_TITLES spells out Shadow Robot's two-letter finger names; any other name is shown
as its tip link's name without the common prefix, in sentence case (Right pad, Fixed jaw).

Contact pads (pad_faces). The pad of a fingertip is the palmar side of its link's collision mesh (its visual mesh when
it has no collision mesh): the triangles whose outward normal lies within facing_deg (default PAD_FACING_DEG = 45°,
a default) of the direction the link's surface moves when its finger flexes. That direction is v = s · a × (c − o),
with a and o the axis and anchor of the nearest joint at or above the tip link (in the tip link's frame), c the
mesh's centroid and s the sign of the joint's larger travel (flexion is the direction with the larger range: Shadow's
distal joints run from 0 to 1.57 rad). Outward normals come from the mesh's own winding, flipped as a whole when its
signed volume is negative. The pads are stored on the robot as Robot.contact_pads 'faces' (triangles in the link
frame), so core/grasps.py's contact_patches samples them (its rule 1, "contact pad faces as tagged", 128 points per
pad by area).

Postures (synergies). A Synergy is a straight line in joint space from an open posture to a closed one; closure s
gives q = q_open + s (q_closed − q_open) on the joints it names (every other joint keeps the robot's home value),
clipped to the joint limits. For the bundled Shadow Hand the open and closed postures are keyframes of the model
folder's own keyframes.xml (MuJoCo Menagerie shadow_hand, commit 98a70c5): power = "open hand" to "grasp sphere",
precision = "open hand" to "two finger pinch", tripod = "open hand" to "three finger pinch". The keyframes are the
model's; the straight-line path between them is DexVantage's sample sweep, not a measured human or robot synergy
(no published synergy basis for this hand is used). For a hand without such postures, default_synergies() makes two
labelled sample postures: power closes every finger joint DEFAULT_CLOSE (70 %, a default) of the way from home to
the end of its range with the larger travel, precision does so for the thumb and the finger whose tip is nearest the
thumb's, the others left open.

Known limits: the finger rule needs the fingertip frames tagged (Robot page); a finger joint's flexion direction is
taken as its larger travel, which is wrong for a joint whose extension range is larger; joints coupled by a tendon
(Shadow's distal and middle joints, MuJoCo <tendon> J0) are independent entries of q, so a posture can pose them in
a way the real hand's coupling would not (the keyframes respect it); the pad is the whole palmar side of the distal
link, not the fingertip's actual contact area, which depends on the object.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

PAD_FACING_DEG = 45.0            # default: a pad triangle faces within this of the flexion direction
DEFAULT_CLOSE = 0.7              # default: the fallback postures close a joint this share of its travel
FINGER_TITLES = {'ff': 'First finger', 'mf': 'Middle finger', 'rf': 'Ring finger', 'lf': 'Little finger',
                 'th': 'Thumb'}  # Shadow Robot's finger names (its joint and link names: FFJ1 ... THJ5)
SYNERGY_KINDS = ('power', 'precision')


class HandError(ValueError):
    """The robot has no hand this module can work with; the message says what to tag."""


@dataclass
class Finger:
    name: str                    # short name (ff, th, ...)
    title: str
    tip: str                     # the fingertip link
    links: list                  # the finger's links, tip first


@dataclass
class HandStructure:
    robot_uid: str
    fingers: list                # [Finger]
    palm: str                    # the palm link (lowest common ancestor of the fingertips)
    palm_links: list
    other_links: list            # wrist, forearm, arm
    link_group: dict = field(default_factory=dict)      # link -> finger name | 'palm' | 'arm'

    def finger(self, name: str) -> Finger | None:
        return next((f for f in self.fingers if f.name == name), None)

    def hand_joints(self, model) -> list:
        """Joint names that move a finger link (the joints of the finger chains)."""
        links = {ln for f in self.fingers for ln in f.links}
        return [model.joint_names[i] for i, b in enumerate(_joint_bodies(model)) if model.body_link[b] in links]

    def to_dict(self) -> dict:
        return {'palm': self.palm, 'palm_links': list(self.palm_links),
                'fingers': [{'name': f.name, 'title': f.title, 'tip': f.tip, 'links': list(f.links)}
                            for f in self.fingers]}


@dataclass
class Synergy:
    id: str
    title: str
    kind: str                    # 'power' | 'precision'
    open: dict                   # joint -> value (rad / m)
    closed: dict
    source: str                  # where the postures come from, in words
    sample: bool = True          # a sample posture (labelled), not a published synergy

    def to_dict(self) -> dict:
        return {'id': self.id, 'title': self.title, 'kind': self.kind, 'open': dict(self.open),
                'closed': dict(self.closed), 'source': self.source, 'sample': self.sample}

    @staticmethod
    def from_dict(d: dict) -> 'Synergy':
        if not isinstance(d, dict):
            raise HandError('A grasp posture must be a record with an id, its open and its closed joint values.')
        kind = str(d.get('kind') or 'power')
        if kind not in SYNERGY_KINDS:
            raise HandError(f'A grasp posture\'s kind is power or precision; got "{kind}".')

        def joints(x, what):
            if not isinstance(x, dict):
                raise HandError(f'The {what} posture of "{d.get("id", "?")}" must map joint names to values.')
            out = {}
            for k, v in x.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    raise HandError(f'The {what} posture of "{d.get("id", "?")}" gives {k} a value that is not a '
                                    'number.') from None
                if not math.isfinite(f):
                    raise HandError(f'The {what} posture of "{d.get("id", "?")}" gives {k} a value that is not a '
                                    'finite number.')
                out[str(k)] = f
            return out
        return Synergy(str(d.get('id') or kind), str(d.get('title') or d.get('id') or kind), kind,
                       joints(d.get('open') or {}, 'open'), joints(d.get('closed') or {}, 'closed'),
                       str(d.get('source') or 'given in the study settings'), bool(d.get('sample', True)))


# ====================================================================== the hand's structure
def _joint_bodies(model) -> np.ndarray:
    return np.array([int(model.m.jnt_bodyid[j]) for j in model._jid], int)


def _body_of(model, link: str) -> int:
    return int(model._lb[model.link_index(link)])


def _ancestors(model, b: int) -> list:
    """b, its parent, ..., up to (not including) the world body."""
    out = []
    while b > 0:
        out.append(b)
        b = int(model.m.body_parentid[b])
    return out


_SEP = '_-./: '
_LINK_WORDS = ('distal', 'tip', 'fingertip', 'pad', 'link', 'jaw')     # a common suffix that may follow a name directly


def _common_prefix(tips: list) -> str:
    pre = os.path.commonprefix(tips) if len(tips) > 1 else ''
    while pre and pre[-1] not in _SEP:
        pre = pre[:-1]
    return pre


def _short_names(tips: list) -> list:
    """The tips' names with their common prefix and suffix taken off, cut only at a word boundary (a separator, or a
    suffix that is a link word such as "distal"), so gripper_right_pad and gripper_left_pad give right and left, not
    "righ" and "lef"."""
    if len(tips) < 2:
        return list(tips)
    pre = _common_prefix(tips)
    suf = os.path.commonprefix([t[::-1] for t in tips])[::-1]
    while suf and suf[0] not in _SEP and suf.lower() not in _LINK_WORDS:
        suf = suf[1:]
    out = []
    for t in tips:
        s = t[len(pre):len(t) - len(suf)] if len(t) > len(pre) + len(suf) else t
        out.append(s or t)
    return out if len(set(out)) == len(out) else list(tips)


def structure(project, robot_uid: str, model=None) -> HandStructure:
    """The fingers, palm and other links of a robot whose fingertips are tagged (see the module docstring)."""
    from .robot.model import get_model
    r = project.robot(robot_uid)
    if r is None:
        raise HandError('There is no such robot in the project.')
    model = model or get_model(project, robot_uid)
    tips = [t for t in r.fingertip_frames if t in model.link_names]
    if len(tips) < 2:
        raise HandError(f'{r.name} has {len(tips)} fingertip frame{"s" if len(tips) != 1 else ""} tagged; the '
                        'multi-finger study needs two or more. Tag the fingertip links on the Robot page.')
    return structure_of(model, robot_uid, tips)


def structure_of(model, robot_uid: str, tips: list) -> HandStructure:
    tb = [_body_of(model, t) for t in tips]
    chains = [_ancestors(model, b) for b in tb]
    others = [set(c) for c in chains]
    fingers, used = [], set()
    names = _short_names(tips)
    pre = _common_prefix(tips)
    for i, (t, c) in enumerate(zip(tips, chains)):
        rest = set().union(*(others[k] for k in range(len(tips)) if k != i))
        links = []
        for b in c:
            if b in rest:
                break
            links.append(model.body_link[b])
        if not links:       # the tip is an ancestor of another tip (a gripper's fixed jaw): it owns its own link,
            links = [t]     # or its contact patch would belong to the palm and count for no finger
        used |= set(links)
        plain = t[len(pre):].replace('_', ' ').strip() or t      # the title keeps the suffix: Right pad, Fixed jaw
        fingers.append(Finger(names[i], FINGER_TITLES.get(names[i], plain[:1].upper() + plain[1:].lower()), t, links))
    common = set(chains[0]).intersection(*chains[1:])
    palm_b = next((b for b in chains[0] if b in common), None)
    palm = model.body_link[palm_b] if palm_b is not None else ''
    group, palm_links, other = {}, [], []
    for f in fingers:
        for ln in f.links:
            group[ln] = f.name
    for ln in model.link_names:
        if ln in group:
            continue
        b = int(model._lb[model.link_index(ln)])
        if palm_b is not None and palm_b in _ancestors(model, b):
            group[ln] = 'palm'
            palm_links.append(ln)
        else:
            group[ln] = 'arm'
            other.append(ln)
    return HandStructure(robot_uid, fingers, palm, palm_links, other, group)


# ====================================================================== contact pads
def _flexion(model, tip_link: str):
    """(axis a, anchor o, sign s) of the nearest joint at or above the tip link, in the tip link's frame."""
    m = model.m
    b = _body_of(model, tip_link)
    jb = _joint_bodies(model)
    T_tip = model.link_pose(model.home, tip_link)
    for bb in _ancestors(model, b):
        idx = np.flatnonzero(jb == bb)
        if not len(idx):
            continue
        i = int(idx[-1])                                  # the last joint of that body
        j = int(model._jid[i])
        T_b = model.link_pose(model.home, model.body_link[bb])
        M = np.linalg.inv(T_tip) @ T_b
        a = M[:3, :3] @ np.asarray(m.jnt_axis[j], float)
        o = M[:3, :3] @ np.asarray(m.jnt_pos[j], float) + M[:3, 3]
        lo, hi = float(model.lower[i]), float(model.upper[i])
        h0 = float(model.home[i])
        up = (hi - h0) if math.isfinite(hi) else 1.0
        down = (h0 - lo) if math.isfinite(lo) else 1.0
        return a, o, (1.0 if up >= down else -1.0)
    return None


def _outward(V, F) -> np.ndarray:
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vol = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum())
    if vol < 0:
        n = -n
    return n


def pad_faces(model, tip_link: str, facing_deg: float = PAD_FACING_DEG) -> tuple[np.ndarray, str]:
    """(triangles (M, 3, 3) in the tip link's frame, rule text): the palmar side of the tip's mesh (module
    docstring)."""
    geoms = [(V.astype(float), F) for ln, V, F, _c in model.collision_geoms() if ln == tip_link and len(F)]
    what = 'collision mesh'
    if not geoms:
        geoms = [(V.astype(float), F) for ln, V, F, _c in model.visual_meshes() if ln == tip_link and len(F)]
        what = 'visual mesh'
    if not geoms:
        raise HandError(f'The fingertip link {tip_link} has no mesh to put a contact pad on.')
    fl = _flexion(model, tip_link)
    tris, ns = [], []
    for V, F in geoms:
        tris.append(V[F])
        ns.append(_outward(V, F))
    tri = np.concatenate(tris)
    n = np.concatenate(ns)
    a = np.linalg.norm(n, axis=1)
    ok = a > 1e-14
    tri, n = tri[ok], n[ok] / a[ok, None]
    if fl is None:
        return tri, f'the whole {what} of {tip_link} (no joint moves it)'
    ax, o, s = fl
    c = tri.reshape(-1, 3).mean(axis=0)
    v = s * np.cross(ax, c - o)
    if np.linalg.norm(v) < 1e-12:
        return tri, f'the whole {what} of {tip_link} (its centroid lies on the joint axis)'
    v /= np.linalg.norm(v)
    keep = n @ v >= math.cos(math.radians(facing_deg))
    if not keep.any():
        return tri, f'the whole {what} of {tip_link} (no face looks along the flexion direction)'
    return tri[keep], (f'palmar side of {tip_link}\'s {what}: faces within {facing_deg:g}° of the direction it moves '
                       'when the finger flexes')


def contact_pads_for(model, tip_links, facing_deg: float = PAD_FACING_DEG, decimals: int = 4) -> list:
    """Robot.contact_pads records ('faces' in each tip link's frame, rounded to 0.1 mm) by pad_faces' rule."""
    out = []
    for t in tip_links:
        tri, rule = pad_faces(model, t, facing_deg)
        tri = np.round(tri, decimals)
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        tri = tri[area > 1e-12]                        # triangles the rounding collapsed are dropped
        out.append({'link': t, 'name': t, 'faces': tri.tolist(), 'rule': rule})
    return out


# ====================================================================== postures
def read_keyframes(hand_xml: str, keyframes_xml: str) -> dict:
    """{key name: {joint name: value}} from a MuJoCo keyframe file whose qpos follows hand_xml's joints in document
    order (MuJoCo's qpos order for a tree of hinge and slide joints). Raises HandError when the counts differ."""
    joints = [j.get('name') for j in ET.parse(hand_xml).getroot().iter('joint') if j.get('name')]
    out = {}
    for k in ET.parse(keyframes_xml).getroot().iter('key'):
        q = [float(x) for x in (k.get('qpos') or '').split()]
        if len(q) != len(joints):
            raise HandError(f'Keyframe "{k.get("name")}" has {len(q)} values; {os.path.basename(hand_xml)} has '
                            f'{len(joints)} joints.')
        out[k.get('name') or f'key {len(out)}'] = dict(zip(joints, q))
    return out


SHADOW_POSTURES = (            # (id, title, kind, open keyframe, closed keyframe) of Menagerie shadow_hand
    ('power', 'Power grasp (sphere)', 'power', 'open hand', 'grasp sphere'),
    ('precision', 'Precision pinch (two fingers)', 'precision', 'open hand', 'two finger pinch'),
    ('tripod', 'Precision tripod (three fingers)', 'precision', 'open hand', 'three finger pinch'),
)


def shadow_synergies(hand_dir: str, prefix: str = '') -> list:
    """The bundled Shadow Hand's sample sweeps between keyframes of its own keyframes.xml (module docstring)."""
    keys = read_keyframes(os.path.join(hand_dir, 'right_hand.xml'), os.path.join(hand_dir, 'keyframes.xml'))
    out = []
    for sid, title, kind, k_open, k_closed in SHADOW_POSTURES:
        out.append(Synergy(sid, title, kind, {prefix + j: v for j, v in keys[k_open].items()},
                           {prefix + j: v for j, v in keys[k_closed].items()},
                           f'MuJoCo Menagerie shadow_hand/keyframes.xml (commit 98a70c5), keyframes "{k_open}" and '
                           f'"{k_closed}"; the straight line between them is a sample sweep, not a measured synergy',
                           True))
    return out


def default_synergies(model, hand: HandStructure) -> list:
    """Labelled sample postures for a hand without its own (module docstring)."""
    jb = _joint_bodies(model)
    by_finger = {f.name: [] for f in hand.fingers}
    for i, b in enumerate(jb):
        g = hand.link_group.get(model.body_link[b])
        if g in by_finger:
            by_finger[g].append(i)

    def closed_of(idx):
        out = {}
        for i in idx:
            lo, hi, h0 = float(model.lower[i]), float(model.upper[i]), float(model.home[i])
            up = (hi - h0) if math.isfinite(hi) else 0.0
            down = (h0 - lo) if math.isfinite(lo) else 0.0
            end = hi if up >= down else lo
            if not math.isfinite(end):
                continue
            out[model.joint_names[i]] = h0 + DEFAULT_CLOSE * (end - h0)
        return out
    all_idx = [i for v in by_finger.values() for i in v]
    open_all = {model.joint_names[i]: float(model.home[i]) for i in all_idx}
    T = model.fk(model.home)
    thumb = next((f for f in hand.fingers if f.name in ('th', 'thumb')), None)
    if thumb is None:            # the thumb: the fingertip farthest from the others' centroid
        P = np.array([T[f.tip][:3, 3] for f in hand.fingers])
        d = [np.linalg.norm(P[k] - np.delete(P, k, axis=0).mean(axis=0)) for k in range(len(P))]
        thumb = hand.fingers[int(np.argmax(d))]
    pt = T[thumb.tip][:3, 3]
    other = min((f for f in hand.fingers if f is not thumb), key=lambda f: np.linalg.norm(T[f.tip][:3, 3] - pt))
    lab = (f'a sample posture: every finger joint closed {DEFAULT_CLOSE * 100:.0f} % of the way from home to the end '
           'of its larger travel (a default), not a published synergy')
    return [Synergy('power', 'Power grasp (sample)', 'power', open_all, closed_of(all_idx), lab, True),
            Synergy('precision', f'Precision pinch (sample: {thumb.title} and {other.title})', 'precision', open_all,
                    closed_of(by_finger[thumb.name] + by_finger[other.name]),
                    lab.replace('every finger joint', f'the joints of {thumb.title} and {other.title}'), True)]


def synergies(project, robot_uid: str, settings: dict | None = None, model=None) -> list:
    """The postures to sweep: settings['synergies'] (records, Synergy.to_dict) when given, else the defaults for
    this hand."""
    from .robot.model import get_model
    given = (settings or {}).get('synergies')
    if given:
        return [Synergy.from_dict(d) for d in given]
    model = model or get_model(project, robot_uid)
    return default_synergies(model, structure(project, robot_uid, model))


def posture(model, synergy: Synergy, closure: float, base=None) -> np.ndarray:
    """q at closure s in [0, 1] along the synergy, the other joints at base (default home), clipped to the limits."""
    s = min(max(float(closure), 0.0), 1.0)
    q = (model.home if base is None else np.asarray(base, float)).copy()
    for i, n in enumerate(model.joint_names):
        if n in synergy.open or n in synergy.closed:
            a = synergy.open.get(n, float(q[i]))
            b = synergy.closed.get(n, a)
            q[i] = a + s * (b - a)
    return model.clip(q)


def unknown_joints(model, synergy: Synergy) -> list:
    names = set(model.joint_names)
    return sorted((set(synergy.open) | set(synergy.closed)) - names)
