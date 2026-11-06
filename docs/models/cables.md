# Cable flex across joints {#cables}

Modules: `dexvantage/core/cables.py` (the cable record, the routed arc model, the slack loop, the bend estimate, the
snag test, suggested attachments) and `dexvantage/core/studies/cables.py` (study id `cables`, batch flag
`--cables OUT.json`; the sweeps, the flags and the harness sheet). Page: `dexvantage/ui/pages/cables.py` (Analysis ▸
Cable flex across joints, Placement workspace). Checks: `dexvantage/validation/checks_cables.py` (model `cables`).
Tests: `tests/test_cables.py`. Library entries: `dexvantage/resources/library/cable/*.json`. Spec §7.5. Ported from
Vantage `core/harness.py`: the harness sheet and its export (JSON / CSV, here also HTML), findings with a severity and
a plain message, and the Ramer–Douglas–Peucker `simplify` for the exported polylines. Vantage's vehicle-surface
routing graph, entry points, link-reach limits and RF and power checks are automotive and were left out. The
articulated-joint model below is new.

## The cable record

`project.cables` is a list of plain records, in the project section `cables` (one undo step per edit) and in both
copies of the project schema:

| Key | Meaning |
|---|---|
| `uid`, `name`, `sensor_uid` | identity; the sensor the cable serves |
| `type` | `usb3`, `usb2`, `gmsl2`, `gmsl`, `coax`, `ethernet`, `power` or `other` |
| `library_id` | a cable library entry, or `''` |
| `min_bend_radius_m` | `{value, source, kind}` or `null` (not stated: bends are not checked, and the study says so); `kind` is `dynamic` (moving), `static` (fixed) or `unspecified` |
| `diameter_m`, `length_m` | outer diameter; the cable's provided length; `null` when not stated or not set |
| `wrap_radius_m` | the radius it is guided round a joint at; `null` = the default (below) |
| `attachments` | `[{robot_uid, link, point [x, y, z] in the link frame, length_to_next_m}]`; `length_to_next_m` is the cable laid to the next attachment, `null` when not set |
| `connectors`, `strain_relief`, `note` | the user's entries for the harness sheet (not checked) |

A cable made from a library entry (`cables.from_library`) takes its type, diameter and connectors, and its minimum
bend radius: the moving (dynamic) figure when the maker states one, since a cable across a joint flexes; otherwise
the fixed (static) figure, labelled `static`, and the study notes that a flexing cable may need more (the two library
cables that state both, Basler's GMSL cable 13.8 / 34.5 mm and VA Imaging's 20 / 50 mm, give 2.5 times the fixed
figure for moving use). The
source string cites the entry's source title, URL and date read.

## Which joints a run crosses

A run joins consecutive attachments A (on link a) and B (on link b). The joints it crosses are the moving joints on
the bodies along the kinematic tree's path from a to b (`crossed_joints`), each with a side: +1 when b is on the
joint's child side (turning the joint by +δ turns B by +δ about the axis relative to A), −1 when a is. The joint
axes and anchors at q come from MuJoCo (`xaxis`, `xanchor` after `mj_kinematics`) with `Robot.base` applied.

- **no joint** (`rigid`): the run keeps its length.
- **one revolute joint** (`wrap`): the routed arc model, below.
- **a prismatic joint or several joints** (`chord`): the straight chord |B − A|. When the arm folds, this is shorter
  than a cable laid along it. The study says so and asks for an attachment on each link between them.

## The routed arc model

In cylindrical coordinates about the joint axis (O, u), A is at radius d_a and height z_a, and B at d_b and z_b. The
angle from A to B about u is θ. The cable is guided round the joint on a cylinder of radius ρ: the cable's
`wrap_radius_m`, or by default min(d_a, d_b), the nearer attachment's distance from the axis (a labelled default). A
set radius larger than that is limited to it. Seen along the axis, the cable runs on the tangent from A to the
circle, along the circle, and on the tangent to B, like a belt over a pulley:

    t_a = √(d_a² − ρ²),  t_b = √(d_b² − ρ²),  α_a = atan2(t_a, ρ) = arccos(ρ/d_a),  α_b likewise
    L_plane = t_a + t_b + ρ (|θ| − α_a − α_b)              when |θ| > α_a + α_b (it wraps)
    L_plane = √(d_a² + d_b² − 2 d_a d_b cos θ)               otherwise (the straight line clears the cylinder)
    L       = √(L_plane² + (z_b − z_a)²)

The tangent planes and the cylinder unroll flat, so the shortest path over them is a straight line in the unrolled
plane. The height therefore grows linearly with unrolled length, and the length is the hypotenuse. The two branches
meet continuously at |θ| = α_a + α_b. α is computed as atan2(t, ρ), not as arccos(ρ/d): arccos is ill-conditioned
near 1 and disagreed with t by about 10⁻⁸ rad when an attachment lay on the cylinder, which showed up as 4 × 10⁻¹⁰ m
in the validation check before the change.

**The angle is not folded to half a turn.** θ = θ_home + side · (q − q_home), where θ_home is the geometric angle at
home in (−π, π]. This assumes the cable is laid the short way round at home. A joint that turns further winds the
cable round it, so the length keeps growing past half a turn. With other joints moving (along a trajectory) the
geometry of a single-joint run depends only on that joint, so the formula holds there too. The validation checks
that θ equals the geometric angle modulo 2π at every sample.

**Where it wraps, the cable lies on a helix** of radius ρ whose unrolled line rises at angle φ, tan φ =
(z_b − z_a) / L_plane. Its bend radius is ρ / cos²φ (the helix's radius of curvature c²/a [do Carmo 1976]).

## Sweeps and trajectories

Each crossed joint is swept over its full range from the model file (`RobotModel.lower`, `upper`), with the other
joints at home. The sweep takes `sweep_samples` samples (default 181). An unlimited joint is swept one turn (±π)
around home, and is flagged `endless` ("a cable across it winds up without end"). Every trajectory of the robot in
`project.tasks.trajectories` is played at its samples, with poses added between two samples so that no revolute
joint moves more than `max_step_deg` (default 2°) and no prismatic one more than 10 mm. A trajectory is thinned
evenly to `max_poses` (default 3000), with a note when that happens.

For each run:

| Figure | Definition |
|---|---|
| slack per joint | the largest minus the smallest length over that joint's sweep; `slack_loop_m` adds the service loop |
| length min / max | over the sweeps, home and the trajectories |
| suggested laid length | the largest length plus `loop_m` (default 0.05 m), rounded up to 5 mm |
| laid length used | the attachment's `length_to_next_m`, or the suggested one when it is not set (flagged `unset`) |

## The slack loop and the bend estimate

At a pose where the route is shorter than the laid length s, the extra cable bows outwards, away from the joint
axis. The bow's direction is the route's normal as seen along the axis: the radial direction less its component
along the route's in-plane tangent. That direction is constant along a straight part, radial on the helix, and
square to both the route and the axis; a run with no revolute joint bows perpendicular to its chord, towards world
+z. The drawn loop (`bulge`) is a circular arc laid over the route, in the route's unrolled length ℓ and that offset
y:

    ℓ = L/2 + R sin ψ,   y = R (cos ψ − cos β),   ψ ∈ [−β, β],   sin β / β = L / σ,   R = σ / (2β)

Beyond a half circle the bow overhangs the ends and is continued along the end segments. σ is found by bisection so
that the polyline is s long. On a straight run this is exactly the circular arc of length s over the chord.

**The bend radius estimate** (`bend_estimate`) is the smaller of two radii. The first is the route's own bend on the
joint cylinder, ρ / cos²φ, where the run wraps. The second is the slack loop's radius, taken as a circular arc of
length s over the route length L, R = s / (2β) with sin β / β = L / s; a taut run (s ≤ L) has no loop. Its minimum
over every pose is compared with the allowed radius, the stated minimum × `bend_factor`. The bend factor defaults to
1.5, a DexVantage margin for flexing and not a maker's figure. The drawn polyline is not used for the figure: laying
the bow over a tightly wrapped route adds the two curvatures, which over-states the bend.

## Flags

| Kind | Severity | When |
|---|---|---|
| `pull` | critical | a run laid shorter than its largest length (with the pose that needs it), or a cable whose `length_m` is shorter than its runs need together, before the run to the host |
| `snag` | serious | the slack loop's polyline passes through one of the robot's links at home or along a trajectory |
| `bend` | serious | the bend estimate is below the allowed radius; the message says whether the wrap radius or the loop sets it |
| `endless` | serious | an unlimited joint is crossed |
| `unset` / `unknown` / `setup` | warning | no laid length yet; no stated minimum bend radius; fewer than two attachments, attachments on different robots, a link the model lacks |

A cable's status is its worst flag. The headline is the number of cables with no flag, out of the total, with the
word for the worst status: "clear", "check", "at risk" or "pulls".

**The snag test** (`snag_hits`) tests the loop polyline at home and at every trajectory pose. The polyline's
segments are cast as rays against each link's colliding triangles, in the link's own frame. This uses core/raycast:
Möller–Trumbore [Möller 1997] through a uniform grid [Amanatides and Woo 1987], built once per link. Segments within
`seat_m` (default 20 mm) of either attachment are left out: that is the clamp, where the cable touches its link by
design. The loop's middle point is also tested for lying inside a link, by the parity of crossings along a ray. The
joint sweeps are not tested for snags: with one joint moving and the rest at home, the arm passes through poses it
would not take, as the spec's "over a trajectory" implies.

**Suggested attachments** (`suggest_attachments`, the page's Suggest attachments, and the study when the project has
no cables) start with the sensor's mount position, then add one point on each link down the chain to the base.
Each point is at the middle of its link (between its two joint anchors at home; on the base, at its child joint's
anchor). It is pushed out past the farthest colliding vertex within 30 mm of the push line, plus half the diameter
(5 mm assumed when the cable states none: a default, not a cable's figure) and 5 mm (a default clearance). The push direction moves away from both joint axes: their common perpendicular, turned towards +z (or −x
when level). These are suggestions to check against the real robot, and the study labels them so.

## The harness sheet

`files(doc, project)` returns `cables.json` (the result with its provenance block), `cables.csv` and `cables.html`;
`export(doc, project, folder)` writes them. The CSV has one row per run with these columns: cable, sensor, type,
library entry, diameter, the stated minimum bend radius with its kind and source, the allowed radius, the cable's
length, the run, its links, route, joints, slack loop per joint (mm), length range, suggested and laid lengths,
smallest bend, snag poses, connectors, strain relief, status, and the flags in words. The HTML page holds the same
content as tables, with no scripts or external files. The page's Export button writes JSON (with provenance), CSV
or HTML by the file's extension.

## Cable library

Three entries, each figure from the maker's page or datasheet, read 2026-11-06 (`kind: "cable"`, the `cable` block of
`library.schema.json`):

| Entry | Diameter | Minimum bend radius, fixed | moving | Source |
|---|---|---|---|---|
| Basler Cable GMSL, FAKRA-Z, 1x F/F | 3.45 mm max. | 13.8 mm | 34.5 mm | docs.baslerweb.com/basler-cable-gmsl-fakra-z-1x-f-f |
| VA Imaging VA-CABLE-USB3-xxm-HF (active optical USB 3.0) | 4.8 mm | 20 mm | 50 mm | hs.va-imaging.com, VA-CABLE-USB3-xxm-HF.pdf |
| Newnex High Flex Active USB3 (ULHLU11AC12) | 6.3 mm | not stated | 75 mm ("12 × OD (≈ 75 mm)") | newnex.com/high-flex-active-usb-3-cable.php |

The Newnex page states no fixed figure; it stays `null`. Each entry's note quotes the figures and says what the
source leaves out.

## Validation

Measured with the default seed, `--validate` (full run, 1.2 s for the four checks):

| Check | Compared with | Measured | Tolerance |
|---|---|---|---|
| `cables.routed_arc` | the circle arc r·\|q − α₀\|, the helix √((r·\|q − α₀\|)² + dz²) and the belt-over-a-pulley length, written out in the check; 721 samples per case over ±2.8 rad (wrapping past half a turn), the robot base moved and rotated | 1.7 × 10⁻¹⁶ m; unwrapped angle against the geometric angle 8.9 × 10⁻¹⁶ rad | 10⁻⁹ m |
| `cables.helix_bend` | the helix's radius of curvature (r² + b²)/r, b = dz/θ [do Carmo 1976]; the study's estimate and the drawn 49-point polyline of a taut wrapped run, dz = 0, 40, 100 mm | 4.9 × 10⁻⁵ (relative) | 5 × 10⁻³ |
| `cables.loop_arc` | the circular arc of length s over c, β from `scipy.optimize.brentq`; the study's loop radius and the drawn loop, 200 random (c, s), s/c from 1.02 to 6 | 4.4 × 10⁻⁴ (polyline), 2.7 × 10⁻¹³ (estimate); loop length 5.9 × 10⁻¹³ | 5 × 10⁻³ |
| `cables.snag_bruteforce` | brute force: the same loops resampled every 0.5 mm, each point tested against the scripted robot's boxes in closed form (base pose and Rz(q) written out); 162 trajectory poses | 0 poses differ (22 snagged on both sides, 2 grazing poses within 1 mm left out); share of time snagged 13.6 % at the study's 2° sampling against 12.8 % by brute force at 0.2° (bound 1.2 %) | 0 poses |

The tolerances are explained in each check's note. Both sides of `routed_arc` compute in float64 from the same link
poses, so 10⁻⁹ m leaves room for rounding, while a wrong side sign or a length folded to half a turn would show as
centimetres. The polyline radii carry a discretisation error of order (Δφ)² of the angle step between points.
Brute-force points 0.5 mm apart can miss a graze less than 1 mm deep, which is why grazing poses are left out of the
snag comparison.

## Worked example

See [../guide/cables.md](../guide/cables.md): the SO-100 sample with the Basler GMSL cable, and the UR5e sample's
wrist D405 with the VA Imaging USB 3 cable.

## Known limits

- The cable has no stiffness, weight, torsion or friction on the links. The slack loop is a circular bow, not an
  elastica, so the bend radius is an estimate.
- A run across several joints is a straight chord, which under-states the length when the arm folds. Add an
  attachment on each link.
- Each joint's sweep moves that joint alone. Combinations of joints come only from the trajectories.
- The wrap radius is one cylinder about the axis, not the joint housing's real shape.
- θ assumes the cable is laid the short way round at home.
- An unlimited joint is swept one turn only and flagged, because no external cable can cross it.
- Snags are tested at sampled poses (home and the trajectories), not over the continuous swept volume, and only
  against the cable's own robot: other robots, the task objects and the tables are not tested.
- The parity test assumes closed link meshes.
- The run from the last attachment to the host is not modelled. Connectors and strain relief are the user's
  entries and are not checked.
- The suggested attachments come from the link geometry at home, not from the real robot's cable channels.
