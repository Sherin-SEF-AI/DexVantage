# The hardware library: devices from datasheets {#library}

Module: `dexvantage/core/library/__init__.py`. Schema: `dexvantage/resources/library.schema.json` (JSON Schema draft
2020-12). Data: `dexvantage/resources/library/<kind>/<id>.json`. Tests: `tests/test_library.py`. The lens models the
library builds are in `dexvantage/core/optics` (see [optics.md](optics.md)).

## What an entry is

One JSON file per device, in the folder of its kind (`depth`, `camera`, `tactile`, `host`), named after its `id`.
Required: `id`, `kind`, `maker`, `name`, `sources` (at least one, each with `url` and the date `read`) and `note`.
The rest, each a figure or `null`:

| Block | What it holds |
|---|---|
| top level | `family`, `interface`, `power_w`, `mass_g`, `size_mm` ([x, y, z] in the device body frame: x the maker's depth or thickness, y the width, z the height) |
| `imagers[]` | `name`, `role` (`ir_left`, `ir_right`, `rgb`, `mono`, `tof`), `pose` + `pose_stated`, `shutter`, `hfov_deg`, `vfov_deg`, `dfov_deg`, `pixel_um`, `resolution` (native, the one the FOV is for), `lens_model`, `f_number`, `focus`; optional `fov_modes` (FOVs stated for other stream shapes, such as a depth FOV for 16:9 and 4:3 streams) |
| `projector` | `hfov_deg`, `vfov_deg`, `dfov_deg`, `pose`, `pose_stated` |
| `depth` | `technology`, `reference_imager`, `stereo_pair`, `baseline_m`, `min_z` (the table as stated: width, height, metres), `max_range_m`, `rms_error` (`text` as stated, `metric`: `rms`, `median_abs`, `std_dev` or `unspecified`, `pct`, `at_m`), `subpixel_px`, `depth_fov`; the Intel entries add `range_text` (the range line as written) and `depth_start_mm` (the depth zero plane behind the front glass) |
| `streams[]` | `imager` (`depth`, `rgb` or an imager's name), `width`, `height`, `fps`, `format`, `bits_per_pixel` |
| `tactile` | `pad_mm`, `camera_resolution`, `fps`, `sensing`, `taxels` |
| `host` | `type`, `ports`, `lanes`, `stated_mbps`, `practical_mbps` (only when a source states it), `gmsl_links` |

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
