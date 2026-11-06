# Inverse kinematics, manipulability and collision distances {#ik}

Modules: `dexvantage/core/robot/ik.py`, `manipulability.py`, `collision.py`. Checks: `dexvantage/validation/checks_ik.py`
(model `kinematics`). Tests: `tests/test_ik.py`.

## Inverse kinematics

`ik.solve(model, link, target, q0=None, seeds=8, rng=None, pos_tol=1e-4, rot_tol=1e-3, max_iter=200, damping=0.01,
weight_rot=0.3, mode='auto', rule='nearest_home', collision=True, margin=0.0, tcp=None)` returns an `IKResult`
(`q, ok, pos_err, rot_err, iterations, reason, mode, orientation_ok, manipulability, min_distance, solutions, note`).
`ik.solve_batch(model, link, targets (N, 4, 4) or (N, 3), ...)` returns an `IKBatch` with the same fields as arrays,
`reason` as a list and `counts()` giving how many targets succeeded and how many failed for each reason.

**The step.** For the tool pose x(q) (the link's frame, optionally followed by a fixed offset `tcp`) and a target
(p\*, R\*):

    e  = [p* − p ;  w · log(R* Rᵀ)]          position error (m), rotation error (rad, world axis-angle)
    Jw = [J_lin ; w · J_ang]                 MuJoCo's world Jacobian of the tool point (mj_jac), rows weighted alike
    Δq = (Jwᵀ Jw + λ² I)⁻¹ Jwᵀ e             damped least squares [Wampler 1986; Nakamura and Hanafusa 1986]
    λ² = ½ |e|² + δ²                         error-dependent damping [Sugihara 2011]
    Δq += −k (I − Jw⁺ Jw) ∇H(q)              H = Σ ((q − q_mid)/(q_hi − q_lo))², away from the limits [Liégeois 1977]
    q  ← clip(q + s·Δq, lower, upper)        s caps the largest joint change at 0.5 rad per iteration

w = 0.3 m/rad, δ = 0.01, k = 0.05 and the step cap are defaults of the solver (tuning of the method, not figures of
any robot); the convergence tolerances 0.1 mm and 1 mrad are defaults the caller can change. Iteration stops when
both errors are within tolerance, when the error score has not dropped by 0.1 % for 15 iterations (stalled), or at
`max_iter`.

**Only the joints that move the link are solved.** A gripper's finger joints do not move the gripper's base, and one
arm of a bimanual robot does not move the other's flange; such joints (`manipulability.chain_mask`) stay at q0 (or
home) and never enter the joint count. The SO-100 sample has six joints, but its jaw does not move the fixed jaw, so
its gripper frame has five.

**Position-only mode.** A chain with fewer than six joints cannot set all six pose coordinates. In `mode='auto'` such
chains are solved position-first: the position by damped least squares, the orientation as a second task inside the
null space N_p = I − J_lin⁺ J_lin of the position task [Nakamura and Hanafusa 1986], so the position is reached exactly
and the orientation as closely as the joints allow. `ok` then means the position was reached; `orientation_ok` says
whether the orientation was too; `note` says how far off it is, in degrees. `mode='pose'` asks for the full pose and
fails with **orientation unreachable** when only the position could be reached. A target given as a 3-vector leaves
the orientation free (`rot_err` is nan).

**Seeds and the rule.** Each target is solved from `seeds` starting points (q0 when given, home, then uniform draws
within the limits from a seeded generator) in lock step: MuJoCo evaluates the Jacobians one at a time, and the linear
algebra runs batched over every seed and target. Targets no seed solved get `retry_seeds` (default 2 × seeds) fresh
seeds. Among the converged seeds (in position-first mode, those within rot_tol of the best orientation), `rule`
chooses **nearest_home** (smallest joint distance to home; unlimited revolute joints wrapped to within π of home),
**nearest_q0** (the same, to q0, and wrapped to within π of q0: smooth when dragging, never a jump of a full turn) or **most_manipulable** (largest Yoshikawa measure of the task
Jacobian). With `collision=True` the chosen solution is the first, in rule order, that is at least `margin` from
self-collision.

**Why a target fails**, decided in this order:

| reason | meaning |
|---|---|
| out of reach | farther from the first moving joint than the reach bound (the sum of the distances between consecutive joint anchors and to the tool point, plus each prismatic joint's travel from home, max(upper − home, home − lower): the triangle inequality), or every seed stalled short of it with and without the joint limits |
| joint limit | a solve with the limits removed converges, and shifting revolute joints by whole turns does not bring it back inside them |
| orientation unreachable | `mode='pose'` on a chain with fewer than six joints whose position alone can be reached |
| self-collision | every converged solution has the robot nearer to itself than `margin` |
| no convergence | none of the above: the seeds ran out of iterations (more seeds or iterations may help) |

**Speed** (measured on the development machine, one thread, 8 seeds): 4 ms per target for the scripted 6-joint arm,
4.4 ms for the UR5e sample, 3.1 ms for the SO-100 sample (position-first), so 10,000 grasps take about 40 s.

## Manipulability

From the singular values σ of the Jacobian J (m × n) [Yoshikawa 1985]:

    w = √det(J Jᵀ) = Π σᵢ   (0 when m > n)        κ = σ_max / σ_min   (numpy's 2-norm condition number)
    1/κ in [0, 1]                                  limit distance dᵢ = 2·min(qᵢ − loᵢ, hiᵢ − qᵢ)/(hiᵢ − loᵢ) in [0, 1]

`measures(model, q, link, part='full'|'position'|'rotation')` and `measures_batch(model, Q, link)` return all of them;
`min_limit_distance` is taken over the joints that move the link. The 6 × n Jacobian mixes metres and radians, so w
and κ of `part='full'` depend on the length unit; `part='position'` is in metres and is the measure for five-joint
arms.

## Collision distances

`collision.self_collision(model, q, margin=0.0, distmax=0.2)` returns `(in_collision, min_distance, pairs)`;
`environment_distance` does the same against the static geoms of the robot's own file; both have batch versions.
The distance of a geom pair is MuJoCo's signed `mj_geomDistance` (negative = penetration depth), evaluated for pairs
whose bounding spheres come within distmax (a default; a value of distmax means "at least distmax"). Pairs not
checked: geoms of one rigid group (bodies joined without a moving joint), adjacent links (one joint between them),
pairs the file excludes (MuJoCo's contype/conaffinity rule and `<exclude>`), planes and height fields. Pairs the file
names in `<contact><pair>` are always checked.

## Validation

| check | reference | tolerance | measured |
|---|---|---|---|
| kinematics.ik_round_trip | 150 random collision-free joint vectors of a scripted 6-joint arm → pose by a product of exponentials written out in the check → IK → the solution's pose by the same product | 0.1 mm (the solver's tolerance), all solved, rotation ≤ 1 mrad | 150/150 solved; largest 0.0997 mm, median 0.018 mm; rotation ≤ 0.84 mrad |
| kinematics.ik_position_only | the same on a 5-joint arm, position-first | 0.1 mm, all solved | 150/150; largest 0.093 mm; orientation also reached in 149 |
| kinematics.manipulability_2r | closed form \|l₁ l₂ sin q₂\| over 1,000 poses (and κ from the eigenvalues of the closed-form J Jᵀ) | 1e-12 m² | 1.1e-16 m²; κ to 4.5e-12 relative |
| kinematics.collision_spheres | \|c₁ − c₂\| − r₁ − r₂ over 1,000 placements (78 overlapping) | 1e-9 m | 1.1e-16 m; in-collision flag right every time |
| kinematics.collision_capsules | closest points of two segments [Ericson 2005, §5.1.9] minus the radii, 1,000 placements (90 overlapping) | 1e-9 m | 1.1e-16 m; flag right every time |

The round trip's error sits just under its tolerance by design: the solver stops as soon as it is within 0.1 mm,
and the check confirms that a solution reported as converged meets the tolerance by an independent route.

## Known limits

- IK is local: a reachable target can be missed when every seed falls into a local minimum (reported as no
  convergence, or as out of reach when the seeds also stalled without limits). The reach bound is an upper bound.
- Unlimited revolute joints are seeded within ±π of home only; URDF mimic joints move independently.
- MuJoCo collides meshes as their convex hulls, so distances to concave links (a U-shaped bracket, a palm) are short
  of the true ones (never long). Closed linkages (the Robotiq 2F-85's) have non-adjacent links a few millimetres
  apart by design, so a margin larger than that reports them.
- The environment is only what the robot's own file places in its world body; project obstacles are not avoided yet.
- Yoshikawa's measure and κ of the full Jacobian are not scale-free (metres against radians).
