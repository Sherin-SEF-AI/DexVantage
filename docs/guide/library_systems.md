# Systems from the library: hosts, policies, lights, cables and teleop devices

A worked example on the SO-100 sample. The entries, their sources and their checks are in
[`docs/models/library.md`](../models/library.md#library-systems); the teleop devices in
[`docs/models/teleop.md`](../models/teleop.md#library).

## 1. A teleop device from the library

    dexvantage --sample so100

Open Analysis ▸ Teleop overlap. Under **Teleop devices** a second row holds the library picker: choose
**SO-100 leader arm · Leader arm** and press **Add from library** (one undo step). The new device reads:

- *Leader*: the follower's own model, joint to joint. The entry is stated to lead an SO-100 (its `leads` words match
  the sample robot's name SO-ARM100 and its model file `so_arm100.xml`), and LeRobot's SO-100 guide states the
  leader is built from the follower's parts through the wrist, so `same_joints` is true.
- *Source*: `dof: 6` with LeRobot's docs at commit d490e47, read 2026-11-07, and the quote ("Remove the gears of the
  6 leader motors"); `leader:` why the follower's model is used.
- Below the source: *From the library: TheRobotStudio and Hugging Face (LeRobot), SO-100 leader arm. Not stated by
  the maker: tracking or sample rate, latency, workspace, ...*

Add two more from the picker: **Quantum Metagloves · Data glove** and **Koch v1.1 leader arm · Leader arm**, then
**Run**. Measured on 2026-11-07 (study defaults, CPU, 10.4 s):

| Device | Comfortable | Teleoperable | What the study says |
|---|---|---|---|
| SO-100 leader arm (library) | 96.7 % | 96.7 % | the same as the sample's own leader device: the same joints, joint to joint |
| Quantum Metagloves | — | — | "its tracking volume is not stated, so its comfortable workspace is not known. Enter a measured or published figure with its source; DexVantage does not guess one." |
| Koch v1.1 leader arm | — | — | "the library entry "Koch v1.1 leader arm" states the leader's figures but no kinematics, and it is not stated to lead this follower; add the leader arm as a robot (Robot ▸ Import) and choose it as the leader." |

The glove's figures (120 Hz sample rate, ≤ 7.5 ms signal latency, 138 g) are in its source with MANUS's page and the
date read; the reach model does not use them. To give the glove a region, measure the volume its wrist tracker covers
and enter it under *Volume* with its source.

## 2. A policy variant

Tasks ▸ Policies: the list now includes π0.5, GR00T N1.5 3B, RDT-1B, OpenVLA-OFT (LIBERO) and OpenVLA-OFT+
(ALOHA), the SmolVLA LIBERO and Meta-World fine-tunes, Diffusion Policy (CNN, Push-T), ACT (LeRobot defaults),
HPT-Base and Octo-Small. Each figure shows its source; a figure no source states shows "—". For example
**OpenVLA-OFT+ (ALOHA)**: three 224 × 224 images, the centre 90 % crop of the area (0.9487 of each side), chunk 25
executed open-loop, 321 ms per chunk on an NVIDIA A100, a 25 Hz controller — all from Kim, Finn and Liang (2025)
and the official code. **π0.5** states no inference time: the action-chunk study says so rather than assuming one.

## 3. A host and a cable

Analysis ▸ Bandwidth, hosts and power: **Basler USB 3.0 Card PCIe, Fresco FL1100, 4 HC, x4, 4 Ports (5 Gbit/s)** is a USB 3.2 Gen 1
controller card; its entry carries Basler's "Up to 380 MBps per port reliably supported" (3 040 Mbit/s) and "Up to
1.52 GBps total" (12 160 Mbit/s). Enter the per-port figure as the controller's practical rate (its source is in the
entry's note) to budget against it. The **Jetson AGX Thor Developer Kit** lists two USB 3.2 Gen 2 Type-A ports (10
Gbit/s each), two USB-C ports whose generation NVIDIA does not state, and a QSFP28 (4 × 25 GbE) for Holoscan Sensor
Bridge cameras.

Analysis ▸ Cable flex across joints: **Basler Cable USB 3.0, Micro B sl/A, P** states "Minimum Bending Radius 62 mm, fixed
installation" and "Suitable for Robotics Applications No"; across a moving joint the study uses the fixed radius and
says the maker states no moving figure. **Stereolabs GMSL2 Fakra Cable F-F** states 33 mm static and 66 mm "loaded"
(taken as the moving radius) and 10 000 flex cycles.

## 4. Check the entries

    dexvantage --validate out/validation

The `library` model's systems checks (`library.teleop_figures`, `library.policy_token_grid`, `library.stated_totals`)
pass with 0 problems, 0 tokens and 9.0 × 10⁻⁴ (the HaptX glove's "~ 1 lb" entered as 454 g). `dexvantage
--self-test` has the line "library teleop".
