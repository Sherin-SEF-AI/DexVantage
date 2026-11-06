# Brackets and mounts {#brackets}

Modules: `dexvantage/core/brackets.py` (the generator, the clearance check, STL export) and `dexvantage/core/meshing.py`
(implicit solids and the watertight mesher), both ported from Vantage `core/brackets.py` and `core/meshing.py`.
Study: `dexvantage/core/studies/brackets.py` (study `brackets`, batch flag `--brackets OUT.json`). Page:
`dexvantage/ui/pages/brackets.py` (Analysis ▸ Brackets). Checks: `dexvantage/validation/checks_brackets.py` (model
`brackets`). Tests: `tests/test_brackets.py`. Spec §7.7 and the Brackets checks of §10.

What was ported and what was left out: Vantage's signed-distance solids, the marching-tetrahedra mesher, the mesh
check, binary STL, the camera plate with clearance holes and screw-access bores, the convex-hull support and the print
orientation came over unchanged in method. Vantage's automotive parts did not: the vehicle body as the mount surface,
the base fitted to a curved panel, the roof-bar tube clamp, VHB tape sized for road shocks, the vehicle frame (ISO
8855). DexVantage mounts brackets on robot flanges, T-slot profile, pan-tilt units and hands instead.

## Bracket types

A sensor's bracket type is `Mount.bracket` (`core/project.py`); its options are the study's per-sensor settings
(`project.analysis.studies['brackets']['sensors'][uid]`, the fields of `BracketSpec`).

| Type id | What it is | Interface figures from |
|---|---|---|
| `flange` | Wrist-flange adapter: a disc between the robot flange and the gripper, with an arm to the camera | `ISO9409` (cited per pattern, below) |
| `tslot2020`, `tslot4040` | Camera clamp on 20 × 20 or 40 × 40 T-slot profile: a pad with a key in the slot and two T-nut bolt holes | `TSLOT` (the maker's page, below) |
| `pantilt` | Pan-tilt neck mount: a pad for the unit's own hole pattern | the user's figures (none by default: no holes, flagged) |
| `palm`, `dorsal` | A sleeve around a micro camera (at most 5 mm) on a pad behind it (palm) or under it (back of the hand) | design defaults |
| `none` | No printed bracket | — |

## Frames and the camera body

The bracket frame is in millimetres, its origin at the centre of the contact face, z out of the mount surface into the
bracket (the STL prints z up). For a flange adapter it is the flange frame: z along the flange normal, x towards the
dowel (ISO 9409-1 clause 4.1: the location pin lies on +Xm). `T_parent_bracket` maps it (in metres) into the sensor's
mount frame.

**Flange face.** The mount link's fixed child frame (the gripper's coupling in a robot file), moved along its axis to
the lowest mesh point of the links below the flange within the pattern's pitch circle (where the coupling meets the
flange). On the UR5e sample that frame is 6.5 mm above the coupling's contact face; the face found lies 100.5 mm out
on wrist 3, against UR's DH d6 of 99.6 mm (`tests/test_brackets.py::test_flange_frame_matches_ur_dh`, within 1.5 mm). A
link with no fixed child uses its own frame (an assumption, said in the result).

**Other kinds.** The interface is placed from the camera: on its back, bottom or top side (`side`), `standoff_mm` from
that face, tilted by `tilt_deg` about the base's long axis; the clamp's position on the profile and a hand's surface
are not in the project.

**Camera body.** A box of the library entry's `size_mm` (x along the optical axis, y across, z up) with the reference
imager's optical centre at the middle of its front face — an ASSUMPTION, labelled in the result: makers state the size,
rarely where the optics sit in it. Mounting holes come from the entry's `mounting` block (added to the schema for this
study): per group the thread as written, count, face, spacing, maximum insertion depth and the centre where the drawing
dimensions it. A coordinate the drawing does not dimension is null; such holes are drawn at the face's middle and
flagged. Entries with stated holes (Intel RealSense D400 datasheet, document 337029-017, read 2026-11-06):

| Device | Holes | Where stated |
|---|---|---|
| D435, D435i | 2 × M3 on the back, 45 mm apart, maximum insertion 3 mm (centre not dimensioned); 1/4-20 underneath, 17.5 mm from the left imager | Figure 10-9, Table 4-20 |
| D405 | 2 × M3 × 0.5 on the back, 20 mm apart, maximum insert 4.0 mm (centre not dimensioned); 1/4-20, maximum insert 5.0 mm, 9 mm from the left imager (face taken as the bottom: an assumption) | Figure 10-14, Table 4-22 |

Other entries state no holes: their plate has none and the result says so.

## The solid

Every part is a signed distance function f (negative inside) with a bounding box (`core/meshing.py`): boxes, rods,
rounded rectangles and half-spaces are exact distances; the convex hull (the largest of its face-plane distances),
unions (min), subtractions (max(a, −b)) and fillets (`round_union`: max(r, min(a, b)) − ‖(max(r − a, 0), max(r − b,
0))‖) are lower bounds of the distance. The bracket is

    body  = interface ∪ holder ∪ (hull(holder's outer corners, interface anchor) − screw-access bores − keep-outs)
    solid = body − holes − camera box (grown 0.2 mm)

- **Holder:** a plate of thickness `wall_mm` on the chosen camera face, `margin_mm` beyond it, with clearance holes
  of nominal + `clearance_mm` (0.4 mm, a default: printed holes come out small) and a screwdriver bore behind each of
  twice the nominal plus 1 mm (a default); for a micro camera a sleeve with a tunnel of the camera's section plus
  `pocket_mm` (0.1 mm, a default), open at the front and the back (the camera slides in from the back).
- **Flange disc:** radius from the pattern's raised spigot d2 (when stated), thickness `disc_mm` (8 mm, a default);
  bolt clearance holes on the pitch circle at the pattern's angles; the dowel hole d5 + `fit_mm` on +x on the pitch
  circle; a centring spigot of d3 − `fit_mm` and `spigot_mm` (3 mm, a default) into the flange's recess and the same
  recess on top for the gripper's spigot; a centre hole for cables (20 mm, a default; ISO 9409-1 §4.4: at most d3).
  The arm leaves from a tab on the rim towards the camera, outside a keep-out cylinder above the disc that the gripper
  occupies (its footprint in the robot model 20 mm up from the adapter, plus 1 mm; 77 mm on the UR5e + 2F-85 sample).
- **T-slot pad:** the profile's width, a key of the groove width − `clearance_mm` and `key_mm` (2 mm, a default; flagged
  when as deep as the groove), and two holes as wide as the slot opening (any bolt that fits the slot passes), placed
  beyond the camera plate's ends so a screwdriver reaches them.
- Parts that join are made to overlap by up to 1 mm: a join on a shared plane meshes as two touching shells, which the
  validation found in the first version (a 2020 clamp in four pieces).

**Meshing** (`meshing.mesh`): f is sampled on a grid of `res_mm` (0.5 mm, a default; a micro camera's sleeve a quarter
of its smallest side) near the surface only, and each cube is meshed from a table derived from the six tetrahedra of a
Kuhn split around its main diagonal [Treece 1999]; neighbouring cubes split their shared face alike, so the surface is
a closed 2-manifold with no ambiguous cases, wound outward. **The check** (`meshing.check`): every undirected edge in
exactly two triangles, every directed edge once (consistent orientation), no zero-area triangle, the signed volume
V = Σ a · (b × c) / 6 over the triangles (a, b, c) positive (outward normals), the shells and the Euler characteristic
V − E + F = 2 − 2g per shell.

**Out:** binary STL (80-byte header that never starts with "solid", triangle count, per triangle the unit normal from
the winding and three float32 vertices) and a JSON sidecar `<name>.stl.json`: the STL's SHA-256, dimensions, holes,
interface figures with their sources, flags, print and assembly notes, the defaults used, the clearance and the
provenance block (`core/provenance.stamp`). The print orientation is the one of six (base on the bed, upside down, on
each side) with the least overhang (faces more than 45° down, not on the bed). The mass is the solid volume × 1.27
g/cm³, the density Prusa Polymers states for Prusament PETG (technical data sheet, "Dichte [g/cm3] 1.27", ISO 1183,
updated 16.02.2022, <https://storage.googleapis.com/prusa3d-content-prod-14e8-wordpress-prusament-prod/2023/10/c34f5f47-tds_prusament-petg_n_de.pdf>,
read 2026-11-07); other PETG filaments differ, and the figure is for an estimate only (infill lowers it).

## Clearance

The bracket and the camera body (one solid) against every robot link but the mount link, at the home pose, the pose
on screen and `samples` joint states (48, a default) drawn uniformly within the joint limits (±π for unlimited
joints), seeded; states in which the robot runs into itself (`core/robot/collision.self_collision_batch`) are dropped
and more drawn (at most 8 × samples; on the UR5e + 2F-85, 83 of 144 drawn were dropped, mostly the gripper's linkage
sampled joint by joint). Each link's mesh vertices and triangle centres are put in the bracket frame and the solid's
signed distance read there: its sign is exact, its value a lower bound of the clearance. For a flange adapter every
link below the flange is moved out by the adapter's thickness (the model has no adapter) and the gripper's contact
face on the adapter (within 1 mm of it, inside the keep-out) is its seat. Status: **critical** when a link is inside at
the home pose, the pose on screen or at every pose (fixed to the mount); **warning** when only some sampled states
collide (with their share), when a link comes within `margin_mm` (2 mm, a default), or a cable passes through.

**Cables:** each cable of `project.cables` laid along its attachments at the pose on screen
(`core/studies/cables.cable_polylines`: every run's slack loop at its laid length, as the Cables page draws it; the
radius is half the stated diameter, 0 when not stated); centre lines sampled 20 points per segment, minus the radius;
the sensor's own cable is ignored within 15 mm of the camera, where it leaves it by design. Without cables the result
says the check did not run. (Until the Phases 3, 6 and 7 review this looked for a module that does not exist and read
no cable at all; `tests/test_spec_review_p367.py` covers it now.)

**Image:** the share of each imager's pixel rays (about 64 across) that meet the bracket, by sphere tracing its solid
from the imager's optical centre.

## Sources

| Figure | Source (read 2026-11-06) |
|---|---|
| ISO 9409-1-50-4-M6: Ø50 ±0.1 pitch circle, M6-6H ↓ 6, 45° and 4 × 90°, Ø6 H7 ↓ 6.2 dowel on the circle, Ø31.5 H7 recess (depth not legible: null), Ø63 h8 spigot 6.5 high | Universal Robots, UR5e datasheet, drawing "Tool flange for UR5e", <https://www.universal-robots.com/manuals/latest/en/datasheets/ur5e/> |
| The pin on +Xm; 4 holes at 45° and 4 × 90°, 6 at 30° and 6 × 60°; designation ISO 9409-1-d1-n-thread | ISO 9409-1:2004 public preview, clauses 4.1 and 5, Figure 1, <https://cdn.standards.iteh.ai/samples/36578/348a837664f444bc83e2902c7a5acf4c/ISO-9409-1-2004.pdf> (Table 1 is not in the preview: the standard is sold) |
| UR bolts: M6, 8.8, ISO 4762, 8.0 Nm; "Do not use bolts that extend beyond 8mm" | <https://www.universal-robots.com/articles/ur/application-installation/end-effector-mounting-on-tool-output-flange/> |
| 20 × 20 profile, groove width 6 mm, depth 5.5 mm | Motedis, Profile 20x20 B-Type Slot 6, <https://www.motedis.es/en/Profile-20x20-B-type-slot-6> |
| 40 × 40 profile, groove width 10 mm, depth 12 mm | Motedis, Profile 40x40 B-type slot 10, <https://www.motedis.es/en/Profile-40x40-B-type-slot-10> |
| Camera holes | the library entries' `mounting` blocks (above) |

Another ISO 9409-1 size typed as a designation (e.g. `80-6-M8`) gets its pitch circle, hole count and thread from the
designation and the angles from Figure 1 (for 4 or 6 holes; other counts are spaced evenly, an assumption said in the
note); its centring diameters, dowel and depths are null (no public source found), so the adapter has no spigot and
no dowel hole, and the result says so. Misumi's and 80/20's profile pages refused automated reading on the day; Misumi
HFS5-2020 and 80/20 figures are therefore not used.

## Validation (measured 2026-11-06)

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| `brackets.watertight` | edge and orientation counts, Euler characteristic and divergence-theorem volume written out in the check (not `meshing.check`), on one bracket of each kind | 0 defects | 0 defects on all six (flange adapter, 2020 and 4040 clamps, pan-tilt plate, palm and dorsal sleeves); each one shell |
| `brackets.volume` | the flange disc's volume in closed form, π t (R² − rc² − 4 rb² − rp²) + π h (rs² − rc²) − π h (rr² − rc²), and V − E + F = 2 − 2g with g = 6 through holes | 1 % | 0.18 % at 0.25 mm (0.08 % at 0.35 mm, the quick run); V − E + F = −10 as expected |
| `brackets.iso_holes` | hole centres and diameters fitted on the mesh (Kåsa circle fit to the wall vertices) against the pattern typed again from UR's drawing and ISO Figure 1 | 0.05 mm (a third of UR's ±0.1 mm on the circle's diameter, halved for a radius) | 4.2 × 10⁻⁵ mm; diameters within 0.0015 mm of nominal + clearance |

Why the tolerances: a printable STL must be closed and consistently wound, so any defect fails; the mesher cuts sharp
edges by a fraction of a grid step, which costs the disc 0.1–0.2 % of its volume, while a wrong hole size (0.4 mm on a
bolt hole is 1.5 % of its area) or a missing feature exceeds 1 %; the bolt circle is held to ±0.1 mm on the robot,
and a circle fit to some 2 300 wall vertices locates a centre far better than the grid. The hole check is a second
reading of the same drawing, not a second source: an error in UR's drawing would pass.

The tests add: the mesher closed and manifold on 250 random fields; the STL round trip (normals agree with the
winding, the header never starts with "solid"); the camera holes read from the D435 and D405 entries; nothing of the
bracket inside the camera body; a link moved through a bracket is found (critical at home); a cable through the base is
found; the gripper on the UR5e sample, moved out by the adapter, sits on it and never inside it.

## Known limits

The bracket follows the planned pose and does not move the camera. The camera body is a box placed by the assumption
above; holes a drawing does not dimension are drawn at the face's middle (flagged). The flange face and the ISO +Xm
direction are taken from the robot file (check the dowel's angle on the robot); the adapter moves the gripper out by
its thickness, which the robot model does not include (the clearance check moves the links below the flange by it, the
tool centre point is not changed). Joint states are sampled, not searched: 48 states find a collision that a sixth of
the joint space shows with probability above 99.98 %, a rarer one less surely; coupled joints are sampled
independently (conservative). The clearance is read at mesh vertices and triangle centres, so a long thin triangle edge
through the bracket between its points is missed. A T-slot clamp's place on the profile, a pan-tilt unit's pattern and
a hand's surface are not in the project. Printed parts are not load-tested: no strength, stiffness or temperature
figure is given. The display mesh is thinned by vertex clustering (1.5 mm cell); the exported STL is not.
