# Lighting and glare: a worked example on the UR5e sample

Will the overhead camera see a shiny can as a white blob when the arm grasps it under a desk light? Will a window wash
out the D435's IR pattern? The Lighting page answers both. It reports the share of the object and gripper area each
camera sees in specular saturation over the gripper's working poses, and the valid depth an active-stereo sensor is
predicted to lose. It then tries the usual fixes and gives each one's effect. The model, its equations and its
validation are in [../models/lighting.md](../models/lighting.md).

## 1. Add a light and a shiny object

1. **File ▸ Open sample ▸ UR5e**. The rig has a Wrist D405 on the tool flange and an Overhead D435 on the cage. The
   sample has a grasp set of 200 tabletop grasps. It has no lights and no task objects.
2. Add a task object at the task volume's centre: a cylinder, radius 33 mm, height 115 mm, at (−134, 492, 57.5) mm,
   tagged **metallic** (a soda can). The window adds only boxes (**Workspace ▸ Add box object**), so this example adds
   the can by script (a box added in the window gets its tags on this page, under **Task object materials**: pick the
   object, tick its tags; each tick is one undo step and an object always keeps at least one tag, matte):

   ```python
   from dexvantage.core.project import TaskObject
   from dexvantage.core.frames import Pose
   p.workspace.objects.append(TaskObject(name='Soda can', shape='cylinder', size=(0.033, 0.115, 0),
                                         pose=Pose([-0.134, 0.492, 0.0575], [0, 0, 0, 1]), materials=['metallic']))
   ```
3. **Analysis ▸ Lighting and glare** opens the page (Placement workspace). The headline reads "not run yet". The
   lights table is empty.
4. **Add light ▸ Elgato Key Light (library)**. The light goes 1.2 m above the task volume's centre, at
   (−134, 492, 1350) mm, facing down. Its luminous flux, 2800 lm, comes from Elgato's product page; the source line
   names the page and the date it was read. The note under the editor gives the luminance the study uses:
   1.02 × 10⁴ cd/m², from L = Φ / (π A) over the 350 × 250 mm panel. That this is the emitting face is an assumption,
   and the library entry says so. Elgato states no IR figure, so the light's IR column reads "not stated". The light
   is drawn in the 3D view as a rectangle with a cross and a short line showing the way it faces. Adding it was one
   undo step.

## 2. Run

5. Press **Run**. It takes 1.4 s on the RTX 5080 through EGL device 0, IK included, and 18 s on the numpy
   rasteriser without a GPU, with the same figures. The study takes 6 reachable grasps from the sample's set as
   working poses: grasps 1, 48, 106, 118, 141 and 177.

The headline reads **34.9 % rel. · heavy glare**: "34.9 % of the object and gripper area Overhead D435 sees is in
specular saturation (relative model), 98.5 % at its worst pose (grasp 106)."

- **"rel."** means the relative model. Neither Intel nor any other maker in the library states its sensor's
  sensitivity in DN per lux second. So the page predicts saturation for an auto-exposure that meters the scene as
  18 % grey, which ISO 12232 puts at 12.7 % of full scale. A point saturates at 7.86 times the metered mean
  luminance. If you know the camera's sensitivity, f-number and exposure, enter them in the study settings
  (`sensors[uid]`). The model then becomes absolute, and the Cameras table says so.
- In the **Cameras** table, the Wrist D405 reads 0.0 %: at these grasps it sees the gripper and the table, not the
  can's highlight. The Overhead D435 reads 34.9 % on average and 98.5 % at grasp 106. At that grasp it looks straight
  down on the can's aluminium top, which mirrors the panel above it.
- The **Glare map** of the Overhead D435 fills the image cells where the can's highlight saturates and outlines the
  cells near saturation. While the page is open, the Camera view of the Overhead D435 outlines the same cells over
  its image. The 3D view marks the saturated points of the worst pose with small red crosses.
- **Active depth sensors**: the Overhead D435's projector is stated at 360 mW (Intel's D400 datasheet, Table 3-17).
  Its pattern gives a median 0.150 W/m² on the can and the gripper. The Key Light states no IR, so nothing washes the
  pattern out: 0.0 %. The D405 is passive stereo, and the notes say it has no wash-out figure.

**What to change** lists each candidate with the figure before and after:

| Change | Saturation |
|---|---|
| Move Elgato Key Light 30 cm along world −y | 34.9 % → 13.7 % |
| A linear polariser on Overhead D435 | 34.9 % → 34.9 %: it would not help, because aluminium reflects both polarisations nearly alike (an ideal polariser still passes 44–50 % of its specular light up to 80° incidence) and the auto-exposure makes up the diffuse light the polariser removes |
| Dim Elgato Key Light to 50 % | 34.9 % → 34.9 %: with one light, the auto-exposure makes up for it |
| Tilt Overhead D435 ±5° or ±10° | 34.9 % → 34.9–35.1 % |

Press **Apply** to move the light 30 cm along −y (one undo step), or keep going.

## 3. A window

6. **Add light ▸ Window (enter its figures)**. Set it at (−134, 1400, 800) mm, **faces −y** (into the room), size
   1200 × 1000 mm. Give it a luminance of 2000 cd/m² and an IR irradiance of 10 W/m², with no distance, so the IR is
   measured at the glass. Type the source of each figure; here, write "example figure, not a measurement". Use your
   own meter readings for a real lab. Each field is one undo step.
7. **Run** again (0.8 s).

The headline now reads **0.0 % rel. · low glare; depth washed out** (critical: the depth loss below sets the
status). The window brightens the scene the auto-exposure meters, so the
can no longer clips. The text adds: "Overhead D435 is predicted to lose valid depth on 73.8 % of what it sees of
them." In the depth table, the ambient IR median is 0.95 W/m² against the pattern's 0.150 W/m². Where the ambient IR
is as bright as the pattern (contrast under 0.5, a default), the pattern is washed out. The suggestion is:

| Change | Figure |
|---|---|
| Blind or curtain Window | depth loss 73.8 % → 0.0 % |

8. Press **Apply**. The window is set to 0 % of its stated output (`dim` 0, a blind) as one undo step, and the study
   runs again. The depth loss is now 0.0 %, and the glare is back at 34.9 %, with moving the Key Light the
   suggestion again. **Edit ▸ Undo** opens the blind.

## Settings worth knowing

The study settings live in `project.analysis.studies['lighting']`:

- `poses`: 'auto' (grasps when there is a set), 'grasps' or 'home'. `max_poses` defaults to 6.
- `robot_material`: the gripper's material tag. It defaults to 'matte', because model files carry no tags.
  `link_materials` sets it per link.
- `materials`: measured values per object, {uid: {'alpha', 'albedo', 'ir_albedo', 'depth_reads_through'}}. The
  defaults per tag are DexVantage defaults, not measurements.
- `sensors[uid]`: `exposure_s`, `gain`, `sensitivity_dn_per_lux_s`, `bits`, `f_number` (for the absolute model), and
  `polariser`.
- `contrast_min` (0.5) and `ir_albedo_min` (0.1): the wash-out and dark-surface thresholds, both defaults.
- `good_pct`, `warn_pct` and `serious_pct` (1, 5 and 15 %): the status thresholds, defaults.

Batch: `dexvantage project.dexv --lighting lighting.json` writes the same result with provenance. On a project
without lights, the result's headline says "no lights" and the command still exits 0.

## Known limits

The study models direct light only, with hard shadows from each light's centre. It assumes a photometric camera
response, and it takes the IR figures to be in the band the depth imagers see. The polariser's gain is an upper
bound. The materials' roughness and albedos are defaults, and points are sampled, so a very small highlight can be
missed. The full list is in [../models/lighting.md](../models/lighting.md#known-limits).
