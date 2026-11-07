# Robots: the robot catalogue {#robots}

The Library dock's **Robots** group holds two commands (Import a URDF or MJCF file…, Import from MuJoCo Menagerie…)
and, under them, the robot catalogue: 53 arms, hands, grippers, humanoids and mobile manipulators, each with the
figures its maker publishes and the official MuJoCo Menagerie model it imports from. The model behind it is in
[docs/models/robots.md](../models/robots.md#catalogue).

## Read a robot's figures

1. Open any project (File ▸ Open sample ▸ LeRobot SO-100 arm with a wrist and a third-person camera, for example) and show the Library dock.
2. Expand **Robots**. Each catalogue row reads like `Arm · 7 DoF · 855 mm · 3 kg`: type, degrees of freedom, reach
   and payload as the maker states them. A "—" means no source read states the figure ("not stated by the maker");
   nothing is guessed.
3. Hover **Franka Robotics Franka Research 3**. The tooltip lists every figure with the source it comes from
   (here Franka's datasheet R02212, release 2.4): 7 DoF, 855 mm, 3 kg, ± 0.1 mm (ISO 9283), 18.3 kg, 1 kHz, the
   interfaces and the joint ranges "A1 -166 to 166; … A7 -175 to 175 deg"; then the model (`franka_fr3/fr3.xml` at
   Menagerie commit `98a70c569c4c`, Apache-2.0) and what DexVantage measured when it imported that model: 7 joints, 7
   actuators, 9 links, DoF 7.
4. Type `hand` in the Library search box: the Allegro Hand V4 (16 DoF, CAN at 333 Hz), the LEAP Hand (16, 595 g), the
   Shadow Dexterous Hand (24 joints, 20 actuated, 1 kHz EtherCAT), the other hands and the hand-held UMI gripper
   remain.

The catalogue's joint ranges are the datasheet's, for comparison. The robot you import uses the limits in its own
model file (Robot page, joint sliders).

## Import a catalogue robot

1. Double-click **The Robot Studio SO-ARM100** (or right-click ▸ Import from MuJoCo Menagerie…). The Menagerie dialog
   opens on `trs_so_arm100` only and starts the download (it needs the internet: GitHub allows 60 requests an hour
   without signing in, and a download uses three).
2. When the files are here, the dialog shows the folder's own LICENSE (Apache-2.0) and the Menagerie commit, and
   picks the catalogue's file (`so_arm100.xml`) in the Model file box.
3. Read the licence. **Import** stays disabled until you tick *I accept the licence of this model*.
4. Import. The robot is added as one undo step (Edit ▸ Undo removes it); its licence, the Menagerie commit and the
   date you accepted it are recorded on the robot, and the LICENSE text travels with the project.

Each Menagerie model has its own licence. Every model the catalogue names carried a permissive licence (Apache-2.0,
BSD or MIT) when it was checked, but the dialog always shows the licence of the version you actually download.

## What "verified" means

Each entry records what DexVantage counted when the model was downloaded once at Menagerie commit `98a70c569c4c` and
imported: joints in q (the joints the sliders move), actuators and links. The DoF it compares with the maker's leaves
out a gripper's finger joints when an arm file includes its gripper (the Panda's `finger_joint1` and `finger_joint2`,
say), or counts actuators for a linkage gripper (the Robotiq 2F-85: eight hinges, one actuator). Validation (Help ▸
Validate the models…, model "Robot catalogue (data consistency)") checks these records against the makers' DoF and counts the bundled
Menagerie models again; it is a data-consistency check, not an accuracy figure.

## Known limits

The figures describe the product, not the simplified Menagerie model. 22 entries have no maker figure yet (the
maker's page could not be read or was not read for this build); their notes say which. A catalogue entry describes
one version of a product (the Z1's figures are the Z1 AIR's; the G1's DoF is the G1 EDU's 23–43): check the variant
line in the tooltip against your unit.
