# Brackets: a worked example on the UR5e sample

This walks through printable brackets for the two cameras of the UR5e + Robotiq 2F-85 sample: an adapter on the
robot's ISO 9409-1 flange for the wrist D405, and a 2020 T-slot clamp for the overhead D435. The model behind it is in
`docs/models/brackets.md`. Figures below are from a run on 2026-11-06 at the default settings.

## 1. Open the sample and run

    dexvantage --sample ur5e

Open Analysis ▸ Brackets and press **Run**. Neither sensor has a bracket type yet, so the headline reads "—" with the
status word **check** and the line "No sensor has a bracket type yet: Apply sets the suggested ones. Suggested: Wrist
D405: wrist-flange adapter (ISO 9409-1); Overhead D435: camera clamp, 2020 T-slot."

The suggestions come from the mounts: the D405 sits on `wrist_3_link`, whose fixed child is the gripper's coupling,
and the UR5e's flange is ISO 9409-1-50-4-M6 by UR's datasheet; the D435 is on the cage, a static frame.

**Preview** shows the suggestions with their reasons; **Apply** sets both types in one undo step (Edit ▸ Undo takes
them back) and runs the study again (about 17 s: most of it is the clearance check over the joint range).

## 2. Read the result

The headline now reads **0/2 ready**, status **check**. Both brackets are closed, printable solids; both carry a
warning that is about the camera, not the bracket:

| | Wrist D405 (flange adapter) | Overhead D435 (2020 clamp) |
|---|---|---|
| Size | 63 × 95 × 24 mm | 121 × 29 × 12 mm |
| Mass (solid PETG, an estimate) | 38 g | 32 g |
| Mesh | closed, one shell, V − E + F = −12 (7 through holes) | closed, one shell, V − E + F = −2 |
| Camera face | bottom, 1 × 1/4-20 (Intel Table 4-22: 9 mm from the left imager) | back, 2 × M3, 45 mm apart (Intel Figure 10-9) |
| Clearance | forearm_link 2.0 mm inside in 1 of 48 sampled states (2 %) | no link comes near |
| Image covered | 0.0 % | 0.0 % |

Select a row to see its holes and figures. For the adapter: four Ø6.4 mm bolt holes on the Ø50 circle at 45°, 135°,
225° and 315° and a Ø6.1 mm dowel hole on +x, all marked **cited** (the tooltip gives UR's datasheet and the ISO
preview); a Ø31.4 mm centring spigot 3 mm high; the gripper's footprint (77 mm) kept free above the disc. The flags
say what to check before printing:

- "The 1 × 1/4-20 on the bottom face: the maker's drawing does not dimension where they sit on the face, so they are
  drawn at its middle" — Intel gives the hole's distance from the left imager, not its position front to back.
- "The depth of the flange's Ø31.5 centring recess is not stated: the adapter's spigot is 3 mm high (a default)".

The notes give the assembly: the bolts get 8 mm longer, UR's limit on bolt engagement is quoted, the gripper sits
8 mm further out (add it to the tool centre point), a Ø6 mm dowel sets the camera's angle about the flange axis, and
the print orientation (on its side, about 10 cm² of overhang to support).

The clearance line means: in one of the 48 collision-free joint states sampled within the UR5e's limits, the forearm
folds back far enough to touch the camera. That is a warning, not a fault: avoid that arm pose, or move the camera.
The notes say 83 of 144 drawn states were dropped because the robot ran into itself.

## 3. Change it and export

- In the 3D view the brackets are drawn grey on the robot, with the camera body's box outlined in the status colour;
  they follow the arm as you move the joints. The Camera view outlines them too (useful to see whether the overhead
  camera sees the wrist bracket).
- Choose the sensor, then its **Bracket** type, flange pattern, side, standoff, tilt and wall: each change is one undo
  step and is kept in the project (the type on the sensor's mount, the options in the study settings). Run again.
- **Export STL…** writes the selected bracket at full resolution (0.5 mm grid) as `Wrist_D405.stl` with
  `Wrist_D405.stl.json` beside it: the STL's SHA-256, the dimensions, holes and their sources, flags, notes, the
  defaults used and the provenance block. Meshing at full resolution takes about a second, so the button builds and
  writes on a worker thread (the status bar shows it) and says when the files are written; a folder that does not
  exist is refused at once. **Export…** writes the whole result as JSON.

## 4. Batch

    dexvantage --sample ur5e --brackets brackets.json

writes the same result (with provenance) for the sensors that have a bracket type. Set the types first in the project
(or in the page) and save it; `--brackets` on a project file uses its types and options.

## What is not checked

Strength, stiffness and heat are not modelled. The camera body is a box of the maker's stated size with the optics at
the front face (an assumption). Where the T-slot clamp sits on the profile and the hand's surface under a palm mount
are not in the project; the base is placed from the camera (side, standoff, tilt). See the model's Known limits.
