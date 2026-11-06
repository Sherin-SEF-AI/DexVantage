"""Joint trajectories q(t): read from CSV, sampled at a time, mapped onto a robot's joints.

A trajectory is a plain record kept in project.tasks.trajectories:

    {'uid', 'name', 'robot_uid', 'joints': [joint names], 't': [s], 'q': [[value per joint] per sample]}

with joint values in the file's own units: radians for revolute and continuous joints, metres for prismatic ones (SI,
as URDF and MJCF store them).

The CSV format: a header row whose first column is t (or time), in seconds, then one column per joint, named as in the
robot's model file; one row per sample, times strictly increasing. A robot's joints the file does not name keep their
current value while the trajectory plays. Blank lines and lines starting with # are skipped. Columns may be separated
by commas, semicolons or tabs; a semicolon or tab file may write decimal commas (0,25).

sample(traj, t) interpolates linearly between samples and holds the first and last sample outside the time range.

Known limits: linear interpolation only (no splines, so velocities jump at the samples); recorded joint_states (ROS 2
bags) are read in Phase 8; values are not unwrapped across ±π for continuous joints.
"""
from __future__ import annotations

import csv
import io
import math
import os

import numpy as np

from ..project import ProjectFormatError, new_uid

TIME_NAMES = ('t', 'time', 't_s', 'time_s')
DEGREES_HINT = 2 * math.pi + 0.01        # rad: a revolute column beyond one full turn while its limits are not


class TrajectoryError(ProjectFormatError):
    """A trajectory file DexVantage cannot use; the message says why and what to change, in plain words."""


def _rows(text: str):
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
    if not lines:
        raise TrajectoryError('The file is empty: a trajectory CSV needs a header row (t, then joint names) and at '
                              'least one row of numbers.')
    sample = lines[0]
    delim = ';' if sample.count(';') > sample.count(',') else ('\t' if sample.count('\t') > sample.count(',') else ',')
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO('\n'.join(lines)), delimiter=delim)]
    if delim != ',':                                  # a semicolon or tab file may use decimal commas (0,25)
        rows = rows[:1] + [[c.replace(',', '.') if c.count(',') == 1 and '.' not in c else c for c in r]
                           for r in rows[1:]]
    return rows


def read_csv(source: str, robots, name: str | None = None, prefer_uid: str | None = None) -> tuple[dict, list]:
    """Parse a trajectory CSV (a path, or the text itself when it contains a newline) for one of the robots
    [(robot uid, joint names, lower, upper, joint types)]: (trajectory record, warnings). Raises TrajectoryError.

    The robot is the one whose joints include every column (prefer_uid first, e.g. the selected robot)."""
    if '\n' in source or not os.path.exists(source):
        if '\n' not in source:
            raise TrajectoryError(f'There is no file at {os.path.basename(source) or source}.')
        text, base = source, 'Trajectory'
    else:
        try:
            with open(source, encoding='utf-8-sig') as f:
                text = f.read()
        except UnicodeDecodeError:
            raise TrajectoryError(f'{os.path.basename(source)} is not a text file: save the trajectory as CSV '
                                  '(comma-separated text).') from None
        base = os.path.splitext(os.path.basename(source))[0]
    rows = _rows(text)
    head = rows[0]
    if not head or head[0].lower() not in TIME_NAMES:
        raise TrajectoryError(f'The first column is "{head[0] if head else ""}": it must be t, the time in seconds, '
                              'followed by one column per joint named as in the robot file (for example: t, '
                              'shoulder, elbow).')
    joints = head[1:]
    if not joints:
        raise TrajectoryError('The header names no joints: after t, add one column per joint, named as in the robot '
                              'file.')
    blank = [i + 2 for i, j in enumerate(joints) if not j]
    if blank:
        raise TrajectoryError(f'Column {blank[0]} of the header has no name: name every column after t with a joint '
                              'of the robot.')
    dup = sorted({j for j in joints if joints.count(j) > 1})
    if dup:
        raise TrajectoryError(f'The header names joint {dup[0]} twice: keep one column per joint.')
    robots = list(robots)
    if not robots:
        raise TrajectoryError('The project has no robot to play the trajectory on: import the robot first '
                              '(Robot ▸ Import robot…).')
    order = sorted(robots, key=lambda r: r[0] != prefer_uid)
    fit = next((r for r in order if set(joints) <= set(r[1])), None)
    if fit is None:
        best = max(order, key=lambda r: len(set(joints) & set(r[1])))
        unknown = [j for j in joints if j not in best[1]]
        raise TrajectoryError(
            f'No robot in the project has a joint named {", ".join(unknown[:4])}'
            f'{" …" if len(unknown) > 4 else ""}. The closest robot\'s moving joints are '
            f'{", ".join(best[1][:12])}{" …" if len(best[1]) > 12 else ""}: rename the columns to match.')
    uid, names, lower, upper, types = fit
    t, q = [], []
    for k, r in enumerate(rows[1:], start=2):
        if len(r) != len(head):
            raise TrajectoryError(f'Row {k} has {len(r)} values; the header has {len(head)} columns. Every row needs '
                                  'a time and one value per joint.')
        vals = []
        for c, cell in enumerate(r):
            try:
                v = float(cell.replace('−', '-'))
            except ValueError:
                col = 't' if c == 0 else joints[c - 1]
                raise TrajectoryError(f'Row {k}, column {col}: "{cell}" is not a number. Write times in seconds and '
                                      'joint values in radians (metres for sliding joints).') from None
            if not math.isfinite(v):
                raise TrajectoryError(f'Row {k}, column {"t" if c == 0 else joints[c - 1]} is not a finite number.')
            vals.append(v)
        if t and vals[0] <= t[-1]:
            raise TrajectoryError(f'Row {k} has time {vals[0]:g} s, not later than the row before ({t[-1]:g} s): '
                                  'times must increase from row to row.')
        t.append(vals[0])
        q.append(vals[1:])
    if not t:
        raise TrajectoryError('The file has a header but no rows of numbers.')
    Q = np.asarray(q, float)
    warnings = []
    idx = {n: i for i, n in enumerate(names)}
    for c, j in enumerate(joints):
        i = idx[j]
        lo, hi = float(lower[i]), float(upper[i])
        col = Q[:, c]
        if types[i] in ('revolute', 'continuous') and np.abs(col).max() > DEGREES_HINT and \
                not (math.isfinite(lo) and math.isfinite(hi) and max(abs(lo), abs(hi)) > DEGREES_HINT):
            warnings.append(f'{j} reaches {np.abs(col).max():.1f} rad: more than a full turn. The values look like '
                            'degrees; the file must be in radians.')
        out = (col < lo - 1e-9) | (col > hi + 1e-9)
        if out.any():
            warnings.append(f'{j} leaves its limits in {int(out.sum())} of {len(col)} samples; those poses are '
                            'clipped to the limits when played.')
    rec = {'uid': new_uid(), 'name': name or base, 'robot_uid': uid, 'joints': list(joints),
           't': [float(v) for v in t], 'q': [[float(v) for v in row] for row in q]}
    return rec, warnings


def duration(traj: dict) -> tuple[float, float]:
    t = traj.get('t') or [0.0]
    return float(t[0]), float(t[-1])


def sample(traj: dict, t: float) -> np.ndarray:
    """The trajectory's joint values at time t (its joints' order): linear between samples, held outside."""
    T = np.asarray(traj.get('t') or [], float)
    Q = np.asarray(traj.get('q') or [], float).reshape(len(T), -1)
    if not len(T):
        return np.zeros(len(traj.get('joints') or []))
    if t <= T[0]:
        return Q[0].copy()
    if t >= T[-1]:
        return Q[-1].copy()
    k = int(np.searchsorted(T, t, side='right')) - 1
    a = (t - T[k]) / (T[k + 1] - T[k])
    return Q[k] * (1 - a) + Q[k + 1] * a


def on_model(traj: dict, joint_names, q_base, t: float) -> np.ndarray:
    """A full joint vector in joint_names order: the trajectory's joints at time t, the others from q_base."""
    q = np.array(q_base, float, copy=True)
    v = sample(traj, t)
    idx = {n: i for i, n in enumerate(joint_names)}
    for j, x in zip(traj.get('joints') or [], v):
        i = idx.get(j)
        if i is not None:
            q[i] = x
    return q
