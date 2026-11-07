"""A recorded episode's joint state put onto a robot model of the project (not a reader: no format registers here).

Used by the data links: grasp sets found where the gripper closes in a dataset (core/grasps.from_dataset) and hand-eye
captures from a dataset (core/calib/capture.capture_dataset).

    on_model(episode, model, unit='rad') -> JointMap      Q (N, nq) in the model's joint order, how it was matched
    interpolate(t, Q, times) -> (M, nq)                   linear in time; NaN outside the rows' time span

Matching the dataset's joint names (Episode.joint_names) to the model's joints, in this order:
1. the same name;
2. the same name once both are normalised (lower case; a trailing '.pos', '_pos', '.position', '_position', '_joint'
   or '.joint' and a leading 'main_' or 'follower_' left out, as LeRobot's SO-100 and Koch recordings name them;
   '-', ' ' and '.' read as '_'), when that is unambiguous;
3. when no name matches at all and the state vector has exactly as many values as the model has joints, by order
   (RLDS states named 'state_0' ... ), said in the notes.
A model joint the dataset does not name is held at the model's home (said); dataset columns that name no joint are
ignored (said). Fewer than half of the model's joints found while the dataset names more joints than were found is
another robot's recording, and refused. unit 'deg' reads the revolute joints in degrees (prismatic joints stay in
metres). A revolute column reaching beyond a full turn while the joint's limits do not looks like degrees: the notes
say so (nothing is converted unless unit is 'deg').

Known limits: one robot per state vector (a bimanual vector matches the robot whose joints it names); joint values are
taken as the model's own convention (no zero offsets or gear ratios applied); a gripper recorded as a percentage of
its stroke (LeRobot's SO-100 follower) is not a joint angle and must be converted before its closing can be read.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .episode import DatasetError

DEGREES_HINT = 2 * math.pi + 0.01        # rad: a revolute column beyond one full turn looks like degrees
EDGE_FRAC = 1e-3                         # of the row spacing: a time this close outside the rows takes the end row
EDGE_MIN_S = 1e-6
_SUFFIXES = ('.position', '_position', '.pos', '_pos', '_joint', '.joint')
_PREFIXES = ('main_', 'follower_')


def normalise_name(name: str) -> str:
    """The name as rule 2 of the module docstring compares it."""
    s = str(name).strip().lower()
    for suf in _SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)]
            break
    s = re.sub(r'[\s.\-]+', '_', s)
    for pre in _PREFIXES:
        if s.startswith(pre) and len(s) > len(pre):
            s = s[len(pre):]
            break
    return s


@dataclass
class JointMap:
    Q: np.ndarray                          # (N, nq) in the model's joint order (rad / m); NaN where not recorded
    columns: dict                          # model joint index -> dataset column index
    how: str                               # 'names' | 'normalised names' | 'order'
    notes: list = field(default_factory=list)

    @property
    def found(self) -> int:
        return len(self.columns)


def on_model(episode, model, unit: str = 'rad') -> JointMap:
    """The episode's joint state on the model (module docstring). Raises DatasetError (a plain sentence) when no
    joint of the model can be found in the dataset."""
    names = [str(n) for n in (episode.joint_names or [])]
    q = np.asarray(episode.q, float)
    n = len(np.asarray(episode.t).reshape(-1))
    q = q.reshape(n, -1) if q.size else np.zeros((n, 0))
    if not names:
        names = [f'state_{i}' for i in range(q.shape[1])]
    mnames = list(model.joint_names)
    cols: dict = {}
    how = 'names'
    exact = {nm: i for i, nm in enumerate(names)}
    for j, nm in enumerate(mnames):
        if nm in exact:
            cols[j] = exact[nm]
    if len(cols) < len(mnames):
        norm_d: dict = {}
        for i, nm in enumerate(names):
            norm_d.setdefault(normalise_name(nm), []).append(i)
        norm_m: dict = {}
        for j, nm in enumerate(mnames):
            norm_m.setdefault(normalise_name(nm), []).append(j)
        taken = set(cols.values())
        added = 0
        for key, js in norm_m.items():
            ds = [i for i in norm_d.get(key, []) if i not in taken]
            if len(js) == 1 and len(ds) == 1 and js[0] not in cols:
                cols[js[0]] = ds[0]
                taken.add(ds[0])
                added += 1
        if added:
            how = 'normalised names' if added == len(cols) else 'names and normalised names'
    notes = []
    if not cols:
        if q.shape[1] == len(mnames) and q.shape[1] > 0:
            cols = {j: j for j in range(len(mnames))}
            how = 'order'
            notes.append(f'No joint name of the dataset ({", ".join(names[:4])}{"…" if len(names) > 4 else ""}) is a '
                         f'joint of {model.name}; its {q.shape[1]} values are taken in the robot\'s joint order '
                         '(the counts are equal). Check that the order is the same.')
        else:
            raise DatasetError(
                f'None of the dataset\'s joints ({", ".join(names[:6])}{"…" if len(names) > 6 else ""}) is a joint of '
                f'{model.name} ({", ".join(mnames[:6])}{"…" if len(mnames) > 6 else ""}), and the counts differ '
                f'({q.shape[1]} against {len(mnames)}): choose the robot that recorded the dataset.')
    unused_n = len(names) - len(set(cols.values()))
    if how != 'order' and len(cols) * 2 < len(mnames) and unused_n > len(cols):
        got = [mnames[j] for j in sorted(cols)]
        other = [n_ for i, n_ in enumerate(names) if i not in set(cols.values())]
        raise DatasetError(
            f'Only {len(cols)} of {model.name}\'s {len(mnames)} joints ({", ".join(got[:4])}) are in the dataset, '
            f'which names {unused_n} others ({", ".join(other[:4])}{"…" if len(other) > 4 else ""}): it looks like '
            'another robot\'s recording. Choose the robot that recorded it.')
    Q = np.tile(np.asarray(model.home, float), (n, 1))
    slide = [t == 'prismatic' for t in model.joint_types]
    deg = str(unit).lower().startswith('deg')
    looks_deg = []
    for j, i in cols.items():
        col = q[:, i].astype(float)
        if deg and not slide[j]:
            col = np.radians(col)
        Q[:, j] = col
        lo, hi = float(model.lower[j]), float(model.upper[j])
        fin = col[np.isfinite(col)]
        if not deg and not slide[j] and fin.size and np.abs(fin).max() > DEGREES_HINT and \
                not (math.isfinite(lo) and math.isfinite(hi) and max(abs(lo), abs(hi)) > DEGREES_HINT):
            looks_deg.append(mnames[j])
    missing = [mnames[j] for j in range(len(mnames)) if j not in cols]
    if missing:
        notes.append(f'{len(missing)} joint(s) of {model.name} are not in the dataset and are held at home: '
                     + ', '.join(missing[:6]) + ('…' if len(missing) > 6 else '') + '.')
    unused = [names[i] for i in range(len(names)) if i not in set(cols.values())]
    if unused and how != 'order':
        notes.append('Dataset joints that name no joint of the robot are ignored: ' + ', '.join(unused[:6])
                     + ('…' if len(unused) > 6 else '') + '.')
    if looks_deg:
        notes.append(f'{", ".join(looks_deg[:4])} reach more than a full turn: the values look like degrees; read '
                     'the joints in degrees.')
    if deg:
        notes.append('Revolute joints read in degrees.')
    return JointMap(Q, cols, how, notes)


def interpolate(t, Q, times) -> np.ndarray:
    """(M, nq): Q (N, nq) at rows t (N,) interpolated linearly at times (M,); NaN outside [t0, t_last] and where a
    neighbouring row is NaN. A time within EDGE_FRAC of the median row spacing (at least EDGE_MIN_S) outside the span
    takes the end row: the rounding of times stored as float32 (LeRobot's timestamp column), not a missing reading."""
    t = np.asarray(t, float).reshape(-1)
    Q = np.asarray(Q, float).reshape(len(t), -1)
    x = np.asarray(times, float).reshape(-1)
    out = np.full((len(x), Q.shape[1]), np.nan)
    if len(t) == 0:
        return out
    if len(t) == 1:
        on = np.abs(x - t[0]) <= EDGE_MIN_S
        out[on] = Q[0]
        return out
    tol = max(EDGE_MIN_S, EDGE_FRAC * float(np.median(np.diff(t))))
    x = np.where((x < t[0]) & (x >= t[0] - tol), t[0], np.where((x > t[-1]) & (x <= t[-1] + tol), t[-1], x))
    inside = (x >= t[0]) & (x <= t[-1])
    k = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
    dt = t[k + 1] - t[k]
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(dt > 0, (x - t[k]) / np.where(dt > 0, dt, 1.0), 0.0)
    v = Q[k] * (1.0 - w)[:, None] + Q[k + 1] * w[:, None]
    exact_hi = w >= 1.0                                     # avoid NaN * 0 from the row after a NaN row
    v[exact_hi] = Q[k + 1][exact_hi]
    exact_lo = w <= 0.0
    v[exact_lo] = Q[k][exact_lo]
    out[inside] = v[inside]
    return out
