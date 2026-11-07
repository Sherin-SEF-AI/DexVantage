# The hardware library: devices from datasheets {#library}

Module: `dexvantage/core/library/__init__.py`. Schema: `dexvantage/resources/library.schema.json` (JSON Schema draft
2020-12). Data: `dexvantage/resources/library/<kind>/<id>.json`. Tests: `tests/test_library.py`. The lens models the
library builds are in `dexvantage/core/optics` (see [optics.md](optics.md)).

## What an entry is

One JSON file per device, in the folder of its kind (`depth`, `camera`, `tactile`, `host`, `policy`, `cable`,
`light`, `robot`, `teleop`), named after its `id`.
Required: `id`, `kind`, `maker`, `name`, `sources` (at least one, each with `url` and the date `read`) and `note`.
The rest, each a figure or `null`:

| Block | What it holds |
|---|---|
| top level | `family`, `interface`, `power_w`, `mass_g`, `size_mm` ([x, y, z] in the device body frame: x the maker's depth or thickness, y the width, z the height) |
| `imagers[]` | `name`, `role` (`ir_left`, `ir_right`, `rgb`, `mono`, `tof`), `pose` + `pose_stated`, `shutter`, `hfov_deg`, `vfov_deg`, `dfov_deg`, `pixel_um`, `resolution` (native, the one the FOV is for), `lens_model`, `f_number`, `focus`; optional `fov_modes` (FOVs stated for other stream shapes, such as a depth FOV for 16:9 and 4:3 streams); optional, only where the source states them: `focal_length_mm`, `sensor_size_mm` ([w, h] active area), `sensor_diagonal_mm`, `optical_format` (the nominal type as written, e.g. `1/2.6"`, never checked as a size) and `figures_conflict` (the maker's own figures for the imager disagree: which, and which one the entry uses) |
| `projector` | `hfov_deg`, `vfov_deg`, `dfov_deg`, `pose`, `pose_stated` |
| `depth` | `technology`, `reference_imager`, `stereo_pair`, `baseline_m`, `min_z` (the table as stated: width, height, metres), `max_range_m`, `rms_error` (`text` as stated, `metric`: `rms`, `median_abs`, `std_dev` or `unspecified`, `pct`, `at_m`), `subpixel_px`, `depth_fov`; the Intel entries add `range_text` (the range line as written) and `depth_start_mm` (the depth zero plane behind the front glass) |
| `streams[]` | `imager` (`depth`, `rgb` or an imager's name), `width`, `height`, `fps`, `format`, `bits_per_pixel` |
| `tactile` | `pad_mm`, `camera_resolution`, `fps`, `sensing`, `taxels`; optional `area_mm2` (a curved pad whose source states only an area) |
| `host` | `type`, `ports`, `lanes`, `stated_mbps`, `practical_mbps` (only when a source states it), `gmsl_links`; optional `ingest[]` (each way cameras reach the host as stated: `bus` `usb3`/`usb2`/`pcie`/`csi`/`gmsl`/`ethernet`/`other`, `count`, `per_port_mbps`, `aggregate_mbps`, `practical_mbps` per port, `practical_aggregate_mbps`, `what` the source's words; nothing derived) |
| `cable` | `type`, `diameter_mm`, `min_bend_static_mm`, `min_bend_dynamic_mm`, `flex_cycles`, `lengths_m`, `connectors`, `active`; optional `stated_mbps`, `drag_chain_rated`, `robotics_rated` (true / false as the maker says, null when not stated) |
| `light` | `type`, `luminous_flux_lm`, `luminance_cd_m2`, `cct_k`, `cct_k_range`, `emitting_size_m`, `ring_inner_m`, `beam_deg`, `ir_irradiance_w_m2`, `ir_at_m`, `what`; optional `cri`, `illuminance_lux` at `illuminance_at_m` (on-axis, as stated), `wavelength_nm` (IR illuminators) |
| `policy` | each field `{value, source, quote}` (docs/models/policies.md) |
| `teleop` | `device_kind` (`leader_arm`, `gello`, `vr`, `spacemouse`, `glove`, `hand_tracker`), `mapping` (`joint`, `cartesian`), for arms `leads` and `same_joints`; each figure `{value, source, quote}`: `dof`, `tracking_hz`, `latency_ms`, `workspace` (a study volume, or null), `range_text`, `reach_mm`, `accuracy_mm`, `position_resolution_mm`, `joint_resolution_deg`, `feedback` ([teleop.md](teleop.md#library)) |

**Frames.** A pose is the imager's (or projector's) **optical** frame (z forward, x right, y down) in the device
**body** frame, which is REP-103 (x forward along the optical axis, y left, z up) with its origin at the optical
centre of the reference (depth) imager. A sensor's mount (`core/project.py` `Mount`) places that body frame. For a
stereo camera the right imager is one baseline along −y.

## Where the figures come from

Every number is copied from a cited source: the maker's datasheet or product page, with its URL, title, the date it
was read and which fields it covers. The source is read when the entry is written; nothing is filled from memory. When
a page cannot be fetched the figures stay `null` and the note says so. The six Intel RealSense entries (D405, D415,
D435, D435i, D455, D457) come from:

- *RealSense Product Family D400 Series Datasheet*, Revision 023, March 2026, document 337029-017 (the former Intel
  RealSense D400 datasheet; the product line is now RealSense, Inc.'s, under licence from Intel), read 2026-11-06:
  baselines and FOVs (Tables 3-11, 3-49), imager, projector and colour-sensor properties (Tables 3-12 to 3-20),
  sizes and masses (3-51 to 3-54), stream modes (4-2 to 4-4), min-Z (4-12), depth quality (4-15), depth start point
  (4-17), depth origin (4.8.1), power (7-7 to 7-10).
- *RealSense Product Family D457 Datasheet*, Revision 006, August 2025, document 337029-011, read 2026-11-06.
- The RealSense store product pages (`store.realsenseai.com/buy-realsense-depth-camera-<model>.html`), read
  2026-11-06: the RGB shutter of the D435/D435i (the datasheet gives none for the RGB camera outside the module) and
  the connectors; elsewhere a cross-check.

Where two sources disagree the entry uses the datasheet and the note gives both (examples: the D405 depth FOV is 84° ×
58° in the datasheet and 87° × 58° on the store page; the D405 connector is "Micro-B" in the datasheet and "USB-C" on
the store page; the D415 projector FOV is 67/41/75° in the SKU table and 64/41/72° in the projector table).

## Entries {#library-entries}

The camera, depth and tactile kinds hold **76 built-in entries** (33 camera, 33 depth, 10 tactile; the other kinds -
host, policy, cable, light - are written up in their own model documents). 44 of them were added on 2026-11-07, every
figure read that day from the source the entry names:

| Kind | Entries (★ added 2026-11-07) |
|---|---|
| Cameras, USB | e-con See3CAM_20CUG, _24CUG, ★_10CUG (AR0134 mono), ★_27CUG (OV2312 RGB-IR), ★_37CUG (IMX900 colour), ★_37CUGM (IMX900 mono); Basler dart daA1440-220um, ★dart daA1920-160uc, ★ace 2 a2A1920-160ucBAS; FLIR Blackfly S BFS-U3-16S2M-CS, ★BFS-U3-04S2M-C, ★BFS-U3-16S2C-CS, ★Firefly DL FFY-U3-16S2M-S-DL; ★Allied Vision Alvium 1800 U-240c; ★Luxonis OAK-1 (IMX378), ★OAK-1 OV9782, ★OAK-1 W OV9782 |
| Cameras, GMSL2 | e-con STURDeCAM25_CUOAGX, ★STURDeCAM29 (OV2311 mono), ★STURDeCAM57 (5 MP RGB-IR), ★NileCAM56 (IMX568), ★STURDeCAM39 (fisheye, 196.7° H); Leopard LI-AR0234CS-GMSL2-OWL (fisheye), LI-OV2312-GMSL2-110H, ★LI-AR0234CSC-GMSL2-060H, ★-120H; ★Stereolabs ZED X One GS wide and narrow |
| Cameras, MIPI and micro | ★Allied Vision Alvium 1800 C-158c; ★Arducam B0452 (OV9281), ★B0429 (AR0234, Jetvariety); OmniVision OVM6948, ams OSRAM NanEyeC |
| Depth, Intel RealSense | D405, D415, D435, ★D435f, D435i, ★D435if, ★D436 (global-shutter RGB), D455, ★D455f, ★D456 (IP65), D457 |
| Depth, Orbbec | Gemini 2, ★Gemini 2 L, Gemini 335, ★335L, 336, ★336L, Femto Bolt, ★Femto Mega |
| Depth, Luxonis | OAK-D, OAK-D Lite, OAK-D Pro, OAK-D SR, ★OAK-D S2, ★OAK-D W, ★OAK-D Pro W |
| Depth, Stereolabs | ZED 2i (wide, narrow), ZED Mini, ZED X (wide, narrow), ZED X Mini (wide, narrow) |
| Tactile | GelSight Mini, Meta DIGIT, ★Meta Digit 360, ★XELA uSkin uSPa 44 and uSPa 46, ★PaXini PX-6AX GEN3 fingertip DP-S2015 (Elite) and finger pad IP-M2324 (Core), ★AnySkin, ★ReSkin, ★Contactile PapillArray |

Sources of the added entries (all read 2026-11-07): the makers' product pages (e-con Systems, Teledyne FLIR,
XELA Robotics, PaXini), product documentation (Basler docs.baslerweb.com, Luxonis docs.luxonis.com Markdown pages and
sensor pages, Arducam docs.arducam.com), data sheets (Allied Vision per-model data sheets, Leopard Imaging
LI-AR0234CSC-GMSL2-xxxH rev. 1.0, Stereolabs ZED X One GS rev. 2.3, Orbbec Gemini 330 series V1.8, Gemini 2 / 2 L
V1.7 and Femto Mega v1.1, the RealSense D400 series datasheet rev. 023 and the RealSense store pages, the OAK-1 and
OAK-D S2 datasheets), Luxonis's depthai-boards files at commit bfa8f36 for the OAK camera positions, and for the
research sensors the papers (Digit 360: arXiv:2411.02479; AnySkin: arXiv:2409.08276; ReSkin: arXiv:2111.00071, CoRL
2021; PapillArray: arXiv:2307.04011, because contactile.com did not answer). Variants of an existing family (D435f,
Gemini 335L, OAK-D W, ...) start from that family's entry and change what the datasheet states differently; the note
of each says what differs.

Not added, and why: GelSight Mini variants (the product sheet gives one set of figures for all gels) and a GelSight
R1.5 (no figures found); Orbbec Gemini 2 XL and Gemini 305 (their pages are rendered by script and no data sheet text
could be read); the OAK-1 Max (its stated 45° H × 55° V for a 5312 × 6000 image was not entered without a reading of
the orientation); arducam.com product pages (they refuse automated reads: the Arducam entries come from
docs.arducam.com).

Many makers state a stream as "3MP @ 72 fps" without its width and height: such entries list no stream (the note says
so) rather than a guessed size; where the same page defines the resolution ("2MP (1600 × 1300 pixels)") it is entered.

## Systems entries: hosts, policies, lights, cables, teleop devices {#library-systems}

Added 2026-11-07, every figure read that day from the source the entry names (the entry's `note` quotes it and says
what is not stated). Tests: `tests/test_library_systems.py`; checks: `validation/checks_library_systems.py`.

| Kind | Entries (★ added 2026-11-07) | Sources |
|---|---|---|
| Hosts, Jetson | AGX Orin 64GB, AGX Orin devkit, Orin Nano 8GB, Orin Nano Super devkit, ★AGX Orin 32GB, ★Orin NX 16GB, ★Orin NX 8GB, ★Orin Nano 4GB, ★AGX Thor developer kit, ★T5000, ★T4000 | NVIDIA's Jetson Orin and Jetson Thor pages (the specifications tables, read column by column through their merged cells) |
| Hosts, capture | USB 3.2 Gen 1/2/2x2 references, Leopard LI-JAG-ADP-GMSL2-8CH, ★Basler USB 3.0 card PCIe FL1100 4 HC x4, ★Leopard LI-PCIEFG-GMSL2, ★Stereolabs ZED Link Quad | Basler technical specification DG001772 (via its distributor), Leopard datasheet, Stereolabs ZED Link Quad datasheet rev1.1 |
| Policies | act, diffusion-policy-cnn, gr00t-n1-2b, octo-base, openvla-7b, pi0, pi0-fast, smolvla-base, ★pi05, ★gr00t-n1.5-3b, ★rdt-1b, ★openvla-oft-libero, ★openvla-oft-plus-aloha, ★smolvla-libero, ★smolvla-metaworld, ★diffusion-policy-cnn-pusht, ★act-lerobot, ★hpt-base, ★octo-small | papers (arXiv HTML), model cards and the official code at a named commit |
| Lights | Elgato Key Light, Key Light Air MK.2, Ring Light, ★Neewer RL-18 55W ring, ★amaran P60c (45° panel), ★AXIS T90D20 (850 nm IR) | the makers' product pages and the Axis datasheet |
| Cables | Basler GMSL FAKRA-Z, VA Imaging USB3 HF, Newnex high-flex USB3, ★Stereolabs GMSL2 Fakra F-F, ★Basler USB 3.0 Micro-B/A P (fixed only), ★Basler GigE Cat 6 DrC (drag chain) | Stereolabs GMSL2 cables datasheet rev1.1, Basler product documentation |
| Teleop (new kind) | ★GELLO, ★SO-100 and ★SO-101 leader arms, ★Koch v1.1 leader, ★ALOHA leader (WidowX-250 6DOF), ★3Dconnexion SpaceMouse Compact, ★Meta Quest 3 and ★3S, ★Apple Vision Pro (M5), ★MANUS Quantum Metagloves, ★Rokoko Smartgloves II, ★HaptX Gloves G1, ★Ultraleap Leap Motion Controller 2 | the GELLO and ALOHA papers, LeRobot's docs and the Koch v1.1 README at named commits, Trossen's docs, the makers' specification pages |

**Hosts.** A host's ingest is modelled only from what its source states: `ports` and `stated_mbps` are the USB 3.2
ports whose generation is stated and the USB-IF per-port rate of that generation (the Thor developer kit's USB-C ports
say only "3.1", so they are listed in `ingest` without a rate; the Thor modules say "3x USB 3.2" with no generation, so
their `stated_mbps` is null), `lanes` the MIPI CSI-2 lanes, and `ingest` lists every way in with the source's words.
Two sources state a sustained figure (`practical_mbps`): Basler's card, "Up to 380 MBps per port reliably supported"
and "Up to 1.52 GBps total" (3 040 and 12 160 Mbit/s, decimal megabytes × 8), and Leopard's grabber, "PCIe X8 Gen3 (64
Gbit/s, ~51 Gbit/s netto)". No GMSL2 link rate is entered for a deserializer board: none of the boards' pages states
one. **Desktop GPUs** (RTX 4090/5080/5090) are not host entries: their stated figures (cores, memory, PCIe to the CPU)
are not camera-ingest figures, and the USB controllers of the machine they sit in are the motherboard's; a desktop
host is a USB 3.2 reference entry (or a capture card) with your machine's measured figure entered as a custom device.
The GPUs appear where they matter, as the `hardware` of a policy's stated inference time.

**Policies.** As [policies.md](policies.md): every field `{value, source, quote}`, the inference time only where the
paper or code states it, with the GPU named (π0.5, the SmolVLA fine-tunes, the Push-T diffusion policy and LeRobot's
ACT state none). Variants are separate entries because their figures differ: the OpenVLA-OFT LIBERO recipe (2 images,
chunk 8, 112 ms on an A100) and the ALOHA OFT+ recipe (3 images, chunk 25, 321 ms, 25 Hz); SmolVLA's LIBERO (2
cameras) and Meta-World (1 camera) checkpoints; Diffusion Policy's simulated Push-T (96 → 84 px, one camera) beside its
real-world setting; LeRobot's ACT defaults (whole chunk executed, no ensembling) beside the paper's ACT.

**Lights.** Makers of video lights state an on-axis illuminance at a distance ("5500lux at 0.5m", "6,300 lux @ 1m"),
a CRI and a beam angle, rarely a luminous flux; the illuminance is entered as stated (`illuminance_lux`,
`illuminance_at_m`) and not turned into a flux or an intensity, so the lighting study, which needs a flux or a
luminance, says those lights' output is not stated until one is entered. The AXIS T90D20 states its wavelength (850
nm), beam angles and an illumination range for cameras (144 m with the 10° lens), not an irradiance.

**Cables.** Basler states plainly which of its cables suit drag chains and robots (`drag_chain_rated`,
`robotics_rated`); the Micro-B/A P cable is "fixed installation only" (no moving radius, no flex cycles), which the
cables study then flags on a moving joint. The Stereolabs datasheet's "Bending radius (loaded) 66 mm" is entered as the
moving radius, an interpretation the note says.

**Teleop devices** ([teleop.md](teleop.md#library)): no maker states a tracking volume as a box or cylinder, so every
entry's `workspace` is null and says why; latencies are what each maker calls latency (MANUS "Signal latency ≤ 7.5ms",
Apple's "12‑millisecond photon‑to‑photon latency" of the passthrough); display refresh rates are not entered as
tracking rates.

## Gaps policy

- A figure the source does not state is `null`, shown as "—" with the tooltip "not stated by the maker"
  (`library.NOT_STATED`). No figure is derived from another to fill a gap in the file (pixel size from focal length and
  FOV, say).
- An assumption the model needs is labelled where it is made: an RGB camera whose offset the maker does not state is
  placed at the depth origin with `pose_stated: false`, and the note says "ASSUMPTION"; the Intel projectors are placed
  at the baseline midpoint the same way. The UI and the studies read `pose_stated` and say so.
- Intel publishes no lens distortion coefficients ("Distortion <= 1.5%"), so the built-in imagers are pinhole models
  fitted to the stated FOV (`lens_model: "pinhole"`, an assumption the note states). A unit's calibration replaces it.

## From a sensor to cameras

`library.sensor_imagers(sensor, mode=None)` returns `[Imager]`, each a 4-tuple `(name, role, pose_in_body, camera)`
(`frames.Pose`, `optics.CameraModel`) with the attributes `label` (what the model is and where its figures come from),
`is_default`, `pose_stated`, `calibrated` and `mode`.

1. **Which stream.** `mode=(w, h)` applies to every imager that streams that size; the others, and `mode=None`, use
   the imager's largest listed stream (then its native resolution). `mode={'RGB': (1280, 720)}` picks per imager.
2. **Which FOV.** A FOV the maker states for that stream shape (`fov_modes`, e.g. Intel's depth FOV for HD 16:9 and
   VGA 4:3) comes first; then the imager's own FOV when the stream has the native aspect ratio (scaled). Otherwise the
   FOV is derived from the native one by a **centred crop** at the same focal length scaled to fit (an assumption the
   label says). The camera is `CameraModel.from_fov(w, h, hfov, vfov, model)`: both stated angles are matched, so the
   pixels need not be square.
3. **Calibration.** `sensor.calibration` overrides per imager: `{'imagers': {'IR left': {'K': 3 × 3, 'D': [...],
   'width': w, 'height': h, 'model': 'brown' | 'kb' | 'pinhole' | ..., 'pose': {'position', 'quaternion'}}}}`; a flat
   `{'K', 'D', 'width', 'height', 'model'}` applies to the first imager. A calibration at another size of the same
   shape is scaled (`CameraModel.scaled`).
4. **No device.** A sensor without a `model_id` (or with one the library does not have) gets one imager, a pinhole
   1280 × 720 px, 90° × 60°, labelled "Default camera (no device chosen) … a default, not a device figure".

`library.projector(sensor)` gives the projector's FOV and pose (`frames.Pose`), `library.depth_spec(sensor)` a copy of
the depth block, `library.min_z(sensor, w, h)` the stated minimum Z for a depth stream (`None` when not stated).

## Custom devices

Sensors ▸ New custom device… (the spec's "Library ▸ New custom device…": the window has no Library menu; the Library panel's right-click menu has it too) writes `$XDG_DATA_HOME/DexVantage/library/<kind>/<id>.json` (on macOS and Windows the
platform's data folder when `XDG_DATA_HOME` is not set) through `library.add_custom(dict)`, after the same schema
check; a custom device cannot take a built-in id. `remove_custom(id)` deletes it. A file that does not parse or match
the schema, or reuses an id, is skipped with a plain message (`library.problems()`, also logged); the app never stops
on one.

## Validation

`tests/test_library.py` validates **every file in the folder** (any kind, whoever added it) against the schema with
`jsonschema` and with the fallback structural check, checks a set of figures against the datasheet tables (baselines,
min-Z, shutters, the D405's missing projector), and checks that the D435's imagers come out one stated baseline apart
along −y with optical frames per REP-103 and the stated 87° × 58° depth FOV.

### Data consistency (validation/checks_library.py, model `library`)

Three checks cross-check the figures of every built-in camera and depth imager against each other with closed forms
written out again in the check; they test the entries, not a model, and are labelled data-consistency:

| Check | Bound | Tolerance and why | Measured 2026-11-07 |
|---|---|---|---|
| `library.sensor_geometry` | pixel pitch × resolution against the stated active-area width, height and diagonal: w = pW, h = pH, d = p√(W² + H²) | 2 % (sizes are rounded to 0.1 mm: 0.05 / 3.8 mm = 1.3 %) | 1.2 % worst (Alvium 1800 C-158c height, 3.754 vs "3.8" mm), 8 figures on 4 entries |
| `library.fov_shape` | for a centred, radially symmetric lens whose field angle θ(r) is concave in the image radius (pinhole, barrel distortion, stereographic and, as the limit, equidistant fisheye), θ(r)/r cannot grow: θ_v ≤ θ_h ≤ θ_v·W/H and max(θ_h, θ_v) ≤ θ_d ≤ min(θ_h·r_d/r_h, θ_v·r_d/r_v) (full angles, square pixels) | 3° (makers state FOVs to about ±3°, Intel Table 3-49 note 3, Orbbec "± 3°", plus up to 1° of rounding) | 2.92° worst (Leopard LI-OV2312-GMSL2-110H, horizontal against vertical × W/H) over 182 FOV statements |
| `library.fov_focal` | a stated FOV is at least the pinhole FOV of the stated focal length, 2 atan(pN / 2f), for any lens without pincushion distortion; the implied barrel distortion r / (f tan θ) − 1 is reported | 3° (the FOV tolerance) | 0.48° worst (OAK-D Pro IMX378: EFL 4.81 mm gives 66.3° against the stated 66°), 16 statements |

Six imagers state angles that no such lens can show together - the diagonal is 4 to 14° wider than the horizontal and
vertical allow (Arducam B0429; e-con See3CAM_24CUG, See3CAM_37CUGM, STURDeCAM25, STURDeCAM29, STURDeCAM57), most
likely a diagonal quoted for the lens's image circle or angles measured on different crops. Their entries say so in
`figures_conflict`; the check lists them instead of failing, and the library uses the horizontal and vertical angles.
Fisheye imagers (a fisheye lens model or an angle of 180° or more) are listed, not counted: the equisolid and
orthographic projections grow faster than linearly and may exceed the bounds legitimately. The bounds are tested in
`tests/test_library_added_entries.py` on lenses built there (pinhole, equidistant, stereographic and barrel lenses
satisfy them to 10⁻⁹°; an equisolid fisheye and a strong pincushion break them), and a mistyped pixel pitch or swapped
H/V is shown to fail the checks.

### Systems entries (validation/checks_library_systems.py, model `library`)

| Check | Reference | Tolerance and why | Measured 2026-11-07 |
|---|---|---|---|
| `library.teleop_figures` | every teleop block consistent (`teleop_devices.entry_problems`) and every stated number found again in its own quote | 0 problems (each is a data error) | 0 problems; 13 entries, 17 stated numbers |
| `library.policy_token_grid` | the ViT patch grid (w / p)(h / p) / s² of the stated input size and patch (s = 4 for SmolVLM2's pixel shuffle) against the stated token count | 0 tokens (integers) | 0 over 13 counts (π0, π0-FAST, π0.5, Octo-Base and -Small with their wrist inputs, SmolVLA base and fine-tunes, both OpenVLA-OFT entries) |
| `library.stated_totals` | count × per-port figure against the stated total over the same ports, and each unit conversion redone from the quoted words (1 byte = 8 bit; 1 lb = 0.45359237 kg, NIST SP 811) | 0.005 relative (converted figures are rounded to 3–4 significant digits) | 9.0 × 10⁻⁴ worst (HaptX "~ 1 lb" entered as 454 g against 453.59 g); the Basler card 4 × 3 040 = 12 160 Mbit/s and its MB/s conversions 0; the Leopard grabber 4 × 10 Gbit/s = 40 Gbit/s per adapter 0 |

A typed figure that drifts from its quote (a glove's 120 Hz entered as 100) or a MB/s figure typed as Mbit/s is shown
to fail the checks in `tests/test_library_systems.py`.

## Known limits

- The figures are the makers' typical values; FOVs vary unit to unit (Intel: about ±3°, mechanical tolerance ±5%).
- Distortion and pixel size are not published for the Intel cameras: pinhole models fitted to the stated FOV, so pixel
  densities near the image corners carry the unmodelled ≤ 1.5 % distortion. Calibrate a unit for its own K and D.
- `max_range_m` is the upper figure of a line such as "0.2 m to over 3 m (varies with lighting conditions)": a lower
  bound on the usable range under good conditions, not a hard limit; `range_text` keeps the words.
- Stream tables are the USB 3.1 Gen 1 (or GMSL) ones; USB 2.0 modes are not entered. Simultaneous-stream limits
  (Tables 4-6, 4-7) are not modelled by the library.
- The RGB offsets of the D415, D435, D435i, D455 and D457 are not stated by the maker; their RGB imagers sit at the
  depth origin as a labelled assumption until a unit is calibrated.
- The IMUs of the D435i, D455 and D457 are mentioned in the notes but not modelled.
- Without `jsonschema` installed the fallback check does not test every bound of the schema.
- The data-consistency checks catch figures that contradict each other, not figures that are consistently wrong; a
  decentred or anamorphic lens, or a FOV measured on a crop other than the native image, can trip `library.fov_shape`
  without a typing error.
- Most added cameras state no distortion, so their imagers are pinhole fits to the stated FOV (wide lenses such as the
  OAK-1 W's 127° or the Leopard 120H with "< -66.5%" distortion are poorly described by a pinhole: calibrate). The
  STURDeCAM39 fisheye is a Kannala-Brandt fit to its stated angles.
- Tactile entries with magnetic or force-array sensing (XELA, PaXini, AnySkin, ReSkin, PapillArray) list no stream:
  their readings are not images, and the bandwidth study needs their data rate entered as a figure.
- Research-sensor figures (Digit 360, AnySkin, ReSkin) are what the papers report for their prototypes; units built from
  the open designs can differ.
- Host entries are specification tables, not measurements: NVIDIA's CSI figures are aggregates, its USB, PCIe and
  Ethernet share UPHY lanes ("not every listed port is available together"), and the module data sheets (downloads
  from NVIDIA's developer site) were not read. Only two capture cards state a sustained rate; measure yours.
- The Meta Quest pages are rendered by script and were read through a text fetch of the page; the Basler card's
  specification is Basler's document hosted by a distributor (Basler's own pages refused automated reads).
- Policy timings are each source's single measurement on the named GPU, batch 1; they do not carry over to other
  hardware, and the README's "slightly slower" or "3-4 times faster" are not entered.
- Lights stated only by an illuminance give the lighting study no flux; the illuminance is not converted because the
  beam profile behind it is not stated.
- Custom teleop devices cannot be made in the custom-device dialog (it covers the sensor and host kinds); write the
  JSON by hand into the user library, and the Library panel has no Teleop group (devices are added on the Teleop page).
