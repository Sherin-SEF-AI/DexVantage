"""Dexterity measures of a robot pose: Yoshikawa's manipulability, the condition number of the Jacobian and the
normalised distance to the joint limits, for one joint vector or a batch.

    yoshikawa(J)                       w = √det(J Jᵀ)          [Yoshikawa 1985]
    condition(J)                       κ = σ_max / σ_min       (numpy's 2-norm condition number)
    inverse_condition(J)               1/κ in [0, 1]           (0 at a singularity; easier to plot and threshold)
    limit_distance(q, lower, upper)    per joint, 2·min(q − lo, hi − q) / (hi − lo) in [0, 1]: 1 at mid-range,
                                       0 at a limit; 1 for an unlimited joint
    measures(model, q, link) / measures_batch(model, Q, link)
                                       all of the above for a robot model's link, as a dict of floats or arrays
    chain_mask(model, link)            the joints that move the link (not a gripper's fingers for its base)

The measures come from the singular values σ of J (numpy's SVD on stacks), so batches cost one call: w is the product
of the singular values when J has no more rows than columns (det(J Jᵀ) = Πσᵢ²) and 0 otherwise (a 6 × 5 Jacobian
cannot move the tool in all six directions, so det(J Jᵀ) = 0 exactly).

part='full' uses the 6 × n world Jacobian [linear; angular], whose rows mix metres and radians: w and κ then depend on
the length unit, which is the usual caveat of Yoshikawa's measure [Yoshikawa 1985]. part='position' uses the 3 × n
linear rows (metres; the measure to use for a 5-joint arm such as the SO-100, solved position-only), part='rotation'
the 3 × n angular rows.

Known limits: the measures are local (first order, at one pose) and say nothing about how far a singularity is in
joint space; κ of the mixed-unit Jacobian is not scale-free; the joint-limit distance treats every joint's range
alike, whatever the joint's travel or speed.
"""
from __future__ import annotations

import numpy as np

PARTS = ('full', 'position', 'rotation')


def _rows(J, part: str) -> np.ndarray:
    J = np.asarray(J, float)
    if part == 'full':
        return J
    if part == 'position':
        return J[..., :3, :]
    if part == 'rotation':
        return J[..., 3:6, :]
    raise ValueError(f'part must be one of {", ".join(PARTS)}; got {part!r}.')


def singular_values(J) -> np.ndarray:
    """(..., min(m, n)) singular values of J (..., m, n), largest first."""
    J = np.asarray(J, float)
    if J.shape[-1] == 0 or J.shape[-2] == 0:
        return np.zeros(J.shape[:-2] + (0,))
    return np.linalg.svd(J, compute_uv=False)


def yoshikawa(J) -> np.ndarray | float:
    """√det(J Jᵀ) for J (m, n) or a stack (..., m, n); 0 when m > n."""
    J = np.asarray(J, float)
    m, n = J.shape[-2], J.shape[-1]
    if m > n or n == 0:
        out = np.zeros(J.shape[:-2])
    else:
        out = np.prod(singular_values(J), axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def condition(J) -> np.ndarray | float:
    """σ_max / σ_min of J (inf at a singularity), as numpy.linalg.cond(J) with the 2-norm."""
    s = singular_values(J)
    if s.shape[-1] == 0:
        out = np.full(s.shape[:-1], np.inf)
    else:
        smin = s[..., -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(smin > 0, s[..., 0] / np.where(smin > 0, smin, 1.0), np.inf)
    return float(out) if np.ndim(out) == 0 else out


def inverse_condition(J) -> np.ndarray | float:
    """σ_min / σ_max in [0, 1]: 1 isotropic, 0 singular."""
    s = singular_values(J)
    if s.shape[-1] == 0:
        out = np.zeros(s.shape[:-1])
    else:
        smax = s[..., 0]
        out = np.where(smax > 0, s[..., -1] / np.where(smax > 0, smax, 1.0), 0.0)
    return float(out) if np.ndim(out) == 0 else out


def limit_distance(q, lower, upper) -> np.ndarray:
    """(..., nq) per-joint normalised distance to the nearer limit, 2·min(q − lo, hi − q)/(hi − lo): 1 at mid-range,
    0 at (or beyond) a limit, 1 for a joint with an infinite limit on either side."""
    q = np.asarray(q, float)
    lo = np.broadcast_to(np.asarray(lower, float), q.shape)
    hi = np.broadcast_to(np.asarray(upper, float), q.shape)
    finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    rng = np.where(finite, hi - lo, 1.0)
    d = 2.0 * np.minimum(q - np.where(finite, lo, 0.0), np.where(finite, hi, 0.0) - q) / rng
    return np.where(finite, np.clip(d, 0.0, 1.0), 1.0)


def min_limit_distance(q, lower, upper) -> np.ndarray | float:
    """The smallest of limit_distance over the joints (1 for a robot with no limited joint)."""
    d = limit_distance(q, lower, upper)
    out = d.min(axis=-1) if d.shape[-1] else np.ones(d.shape[:-1])
    return float(out) if np.ndim(out) == 0 else out


def chain_mask(model, link: str) -> np.ndarray:
    """(nq,) bool: the joints that move `link` (on its body or an ancestor's), e.g. not a gripper's finger joints
    for the gripper's base, nor the other arm of a bimanual robot."""
    m = model.m
    b = int(model._lb[model.link_index(link)])
    up = set()
    while b > 0:
        up.add(b)
        b = int(m.body_parentid[b])
    return np.array([int(m.jnt_bodyid[j]) in up for j in model._jid], bool).reshape(model.nq)


def _jacobians(model, Q, link, point) -> np.ndarray:
    Q = model.batch_q(Q)
    out = np.empty((len(Q), 6, model.nq))
    for k, q in enumerate(Q):
        out[k] = model.jacobian(q, link, point)
    return out


def measures_batch(model, Q, link: str, point=None, part: str = 'full', chain_only: bool = True) -> dict:
    """{'yoshikawa', 'condition', 'inverse_condition', 'limit_distance' (N, nq), 'min_limit_distance'} for joint
    vectors Q (N, nq) of a RobotModel, the Jacobian of `point` (link coordinates, default the link origin).
    chain_only: min_limit_distance over the joints that move the link only (chain_mask); limit_distance always has
    every joint. Joints off the chain add zero columns to J, which change neither w nor κ."""
    Q = model.batch_q(Q)
    J = _rows(_jacobians(model, Q, link, point), part)
    s = singular_values(J)
    m, n = J.shape[-2], J.shape[-1]
    w = np.prod(s, axis=-1) if (m <= n and n) else np.zeros(len(Q))
    smax = s[:, 0] if s.shape[-1] else np.zeros(len(Q))
    smin = s[:, -1] if s.shape[-1] else np.zeros(len(Q))
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(smin > 0, smax / np.where(smin > 0, smin, 1.0), np.inf)
        inv = np.where(smax > 0, smin / np.where(smax > 0, smax, 1.0), 0.0)
    ld = limit_distance(Q, model.lower, model.upper)
    use = chain_mask(model, link) if chain_only else np.ones(model.nq, bool)
    return {'yoshikawa': w, 'condition': kappa, 'inverse_condition': inv, 'limit_distance': ld,
            'min_limit_distance': ld[:, use].min(axis=-1) if use.any() else np.ones(len(Q))}


def measures(model, q, link: str, point=None, part: str = 'full', chain_only: bool = True) -> dict:
    """measures_batch for one joint vector: floats, and 'limit_distance' as an (nq,) array."""
    b = measures_batch(model, np.asarray(q, float).reshape(1, -1), link, point, part, chain_only)
    return {k: (v[0] if k == 'limit_distance' else float(v[0])) for k, v in b.items()}
