# Robot kinematics: forward kinematics and Jacobians

DexVantage poses a robot's links with two separate engines:

- **MuJoCo** (`core/robot/model.py`, `RobotModel`) is the primary engine. MuJoCo's compiler reads the stored URDF or
  MJCF, and `mj_kinematics` and `mj_jac` give the link poses and Jacobians.
- **The product-of-exponentials chain** (`core/robot/kinematics.py`, `PoEChain`) is an independent second route. It
  is used only to check the first. It reads the same stored files with `xml.etree`, not MuJoCo's parser, and builds
  the kinematic tree itself. It writes its rotation and twist algebra out again and shares no rotation code with
  `core/frames.py` or MuJoCo.

The validation suite (`validation/checks_kinematics.py`, model id `kinematics`) compares the two engines with each
other and with closed forms written from the makers' published DH tables.

## Forward kinematics by the product of exponentials

For a link whose path from the root crosses the moving joints 1 … k, the space-form product of exponentials
(Lynch and Park 2017, chapter 4) is

    T_link(q) = e^[S_1]θ_1 · e^[S_2]θ_2 · … · e^[S_k]θ_k · M_link

where:

- `M_link` is the link's world pose at the zero configuration, with the robot's base pose applied.
- `S_i = (ω_i, v_i)` is joint i's screw axis in the world frame at that configuration:
  - revolute through point p with unit axis ω: `S = (ω, −ω × p)`
  - prismatic along unit axis u: `S = (0, u)`
- `θ_i` is the joint's displacement from its zero: q for URDF, and q − ref for MJCF, where ref is the joint's
  `ref` attribute (MuJoCo's qpos0).

The exponential of a twist, written out in `exp_twist`:

    e^[ω]θ = I + sin θ [ω] + (1 − cos θ) [ω]²                                     (Rodrigues)
    e^[S]θ = [[e^[ω]θ, (I θ + (1 − cos θ)[ω] + (θ − sin θ)[ω]²) v], [0, 1]]       (|ω| = 1)
    e^[S]θ = [[I, v θ], [0, 1]]                                                  (ω = 0)

The tree is handled by keeping one running product per body. A body's product is its parent's product times the
exponentials of the body's own joints, in file order.

## Jacobian

The space Jacobian (Lynch and Park 2017, chapter 5) has one twist column per joint on the link's path, and zero
columns for the other joints:

    J_s = [S_1, Ad(e^[S_1]θ_1) S_2, …, Ad(e^[S_1]θ_1 ⋯ e^[S_{k−1}]θ_{k−1}) S_k],   Ad(R, p) = [[R, 0], [[p]R, R]]

`PoEChain.jacobian(q, link, point)` returns the world-frame `[linear; angular]` Jacobian of a point x fixed in the
link. Its linear rows are `v_s + ω_s × x`. This is the same convention as `RobotModel.jacobian`.
`PoEChain.space_jacobian` returns the raw twists `[ω; v]`.

## File conventions the reader implements

| Format | Element | Convention |
|---|---|---|
| URDF | `<origin xyz rpy>` | R = Rz(yaw) Ry(pitch) Rx(roll): roll, pitch, yaw about the fixed x, y, z axes |
| URDF | `<axis xyz>` | in the joint (child) frame, default (1, 0, 0), normalised |
| URDF | joint types | revolute and continuous turn; prismatic slides; fixed does not move; floating is held at identity |
| MJCF | `quat` | (w, x, y, z), normalised |
| MJCF | angles | degrees unless `<compiler angle="radian">` (euler, axisangle, hinge ref) |
| MJCF | `euler` | `eulerseq` (default `xyz`): a lowercase letter turns about the moving axis (post-multiplied), an uppercase letter about the fixed axis (pre-multiplied) |
| MJCF | `xyaxes` | y is orthogonalised against x, and z = x × y |
| MJCF | `zaxis` | the shortest rotation that takes (0, 0, 1) to the vector |
| MJCF | `<joint pos axis ref type>` | the attribute, else the element's `class`, else the nearest `childclass`, else `main`, with class inheritance |
| MJCF | `<frame>` | composes with the bodies and joints inside it |
| MJCF | `<include>` | resolved from the stored files through the import manifest |
| MJCF | ball and free joints | held at the `home` keyframe's qpos, or at their zero |
| MJCF | unnamed bodies and joints | named `body_<n>` and `joint_<n>` with MuJoCo's index |

## DH tables used by the checks

The UR5e and Panda models in the checks are written into the check from these tables, as MJCF and URDF. They do not
use any downloaded robot model.

**UR5e, standard DH.** Source: Universal Robots, "DH Parameters for calculations of kinematics and dynamics",
UR5e/UR7e table,
<https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>,
read 2026-11-06. The link transform is A_i = Rz(θ_i) Tz(d_i) Tx(a_i) Rx(α_i) (Denavit and Hartenberg 1955, as
Spong et al. 2006 write it).

| Joint | a (m) | d (m) | α (rad) |
|---|---|---|---|
| 1 | 0 | 0.1625 | π/2 |
| 2 | −0.425 | 0 | 0 |
| 3 | −0.3922 | 0 | 0 |
| 4 | 0 | 0.1333 | π/2 |
| 5 | 0 | 0.0997 | −π/2 |
| 6 | 0 | 0.0996 | 0 |

**Franka Panda, modified DH (Craig's convention).** Source: Franka Emika, "Robot and interface specifications",
"Denavit–Hartenberg parameters" for "the Panda's kinematic chain".

- Read 2026-11-06 from the page's source file, `source/control_parameters.rst` in <https://github.com/frankarobotics/docs>
  (formerly `frankaemika/docs`), at commit `476a2ed2c1f457b87ba2fbb9213ebcd13825382d` (2021-11-11).
- The published page, <https://frankaemika.github.io/docs/control_parameters.html>, returned HTTP 404 that day.
- Its successor page gives the same figures for the Franka Research 3: "Control Interface Specification and Robot
  Limits", <https://frankarobotics.github.io/docs/robot_specifications.html>, read 2026-11-06.

The link transform is ^{i−1}T_i = Rx(α) Tx(a) Rz(θ_i) Tz(d) (Craig 2005).

| Row | a (m) | d (m) | α (rad) | θ |
|---|---|---|---|---|
| Joint 1 | 0 | 0.333 | 0 | θ1 |
| Joint 2 | 0 | 0 | −π/2 | θ2 |
| Joint 3 | 0 | 0.316 | π/2 | θ3 |
| Joint 4 | 0.0825 | 0 | π/2 | θ4 |
| Joint 5 | −0.0825 | 0.384 | −π/2 | θ5 |
| Joint 6 | 0 | 0 | π/2 | θ6 |
| Joint 7 | 0.088 | 0 | π/2 | θ7 |
| Flange | 0 | 0.107 | 0 | 0 |

The flanges at q = 0, from the sums of these lengths:

- UR5e: (−0.8172, −0.2329, 0.0628) m
- Panda: (0.088, 0, 0.926) m

The tests check both.

## Validation

All checks below use the suite's default seed. Measured on 2026-11-06 with MuJoCo 3.17.0, NumPy 2.5.4 and
Python 3.14.4.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `kinematics.fk_poe_mujoco` | `PoEChain` against MuJoCo. 200 random q per robot, every link, five built-in robots plus the project's robots (with no project given, the two samples) | 1e-9 m, 1e-9 rad | 8.7e-15 m, 3.1e-15 rad (worst on the UR5e + 2F-85 sample; the five built-in robots alone 2.4e-15 m) |
| `kinematics.dh_ur5e` | Standard DH product against both engines, on the UR5e as MJCF and as URDF. Frames 1–6, 200 random q | 1e-9 m, 1e-9 rad | 5.6e-16 m, 9.4e-16 rad |
| `kinematics.dh_panda` | Modified DH product against both engines, on the Panda as URDF. Frames 1–7 and the flange, 200 random q | 1e-9 m, 1e-9 rad | 6.3e-16 m, 1.1e-15 rad |
| `kinematics.jacobian_fd` | `PoEChain.jacobian` against central differences (h = 1e-6) of `PoEChain.fk`. Every link, 10 q per robot | 1e-7 | 4.4e-10 |
| `kinematics.jacobian_mujoco` | `PoEChain.jacobian` against MuJoCo `mj_jac`, at every link's origin and at a random point fixed in the link (up to 0.2 m out on each axis). 40 q per robot | 1e-9 | 8.8e-15 |
| `kinematics.fk_pinocchio` | Pinocchio FK against both engines, on the DH URDFs | 1e-9 m, 1e-9 rad | registered only when pinocchio is installed; it was not in this run |

**Why these tolerances.**

- Both FK engines are double-precision products of about twenty rigid transforms, so rounding stays near 1e-15. A
  convention error (an rpy order, a quaternion order, degrees read as radians, a joint anchor) shows as 1e-3 or more.
  `tests/test_kinematics.py` swaps in two such slips and checks that the comparison fails.
- Central differences carry a truncation error of about h²/6 times the third derivative (about 1e-13 here) and a
  rounding error of about 1e-16/h (about 1e-10). A wrong column is of the order of the link lengths.

**Robots in the checks.** The built-in robots are:

- the UR5e, written from the DH table as MJCF and as URDF
- the Panda, written from the DH table as URDF
- an MJCF model that uses every pose convention in the table above, including a ball joint and a free body held at
  the home keyframe
- a branching URDF with a `world` root, axes that are not unit length, prismatic fingers and fixed chains

The check also compares every robot in the project. A robot the independent reader cannot read is listed under
`not_compared`, not counted as a pass.

## Known limits

- Free and ball joints are not part of q. They are held at the `home` keyframe, or at their zero, as `RobotModel`
  holds them.
- URDF mimic joints move independently.
- The reader does not read URDF planar joints or MJCF `<replicate>`, `<attach>`, `<composite>` and `<flexcomp>`. It
  says so in a plain error, and the MuJoCo comparison lists such a robot as not compared.
- MJCF defaults are read only for a joint's type, pos, axis and ref, the attributes that change kinematics.
- The DH checks build the UR5e and the Panda from the tables, so they validate the engines' conventions, not the
  geometry of a vendor's downloaded model file. The MuJoCo comparison covers any robot in the project, including
  imported vendor files. The bundled Menagerie UR5e is not the DH table: its link offsets are rounded to the
  millimetre, and its flange is up to 1.3 mm (0.9 mm RMS) from the DH flange after the best rigid alignment
  ([robots.md](robots.md#known-limits), `tests/test_samples.py`).
- Joint ranges are not part of the DH checks: θ is drawn in [−π, π] for every joint.
