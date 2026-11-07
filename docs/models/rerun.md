# Rerun bridge

`dexvantage/core/bridges/rerun.py`, the window command Export ▸ Stream to Rerun… (`dexvantage/ui/rerun_stream.py`),
the batch flags `--rerun OUT.rrd [--rerun-audit AUDIT.json]`, the check `formats.rerun_rrd_readback`
(`dexvantage/validation/checks_rerun.py`). Spec §9.4.

The bridge sends the project's scene to [Rerun](https://rerun.io), either saved as an `.rrd` file or streamed to a
live viewer. A viewer is started only when the user chooses "Open the Rerun viewer". It needs the optional extra
`rerun`, which pins `rerun-sdk==0.42.0`:

    pip install "dexvantage[rerun]"

The package is imported lazily. Without it the command and the batch flag say how to install it, in one sentence.
rerun-sdk is not bundled; its licence (MIT or Apache-2.0, per its package metadata) applies to the user's own
install.

## What is logged

Everything is in the world frame: the primary robot's base, REP-103, metres. The entity `world` carries
`ViewCoordinates.RIGHT_HAND_Z_UP`.

| Entity | Archetype | Content |
|---|---|---|
| `world/robots/<robot>/<link>` | Transform3D | the link frame in the world: at every trajectory sample on the timeline `time`, else once (static) at q |
| `…/<link>/mesh` | Mesh3D | the link's visual meshes in the link frame, with the model's colours per vertex |
| `world/robots/<robot>/<link>/sensors/<sensor>` or `world/sensors/<sensor>` | Transform3D | the sensor body's mount pose (under its link, so it rides with the arm) |
| `…/<sensor>/<imager>` | Transform3D | the imager's pose in the body |
| `…/<imager>/image` | Pinhole | image_from_camera = K, resolution, camera_xyz = RDF (OpenCV optical axes), image_plane_distance 0.15 m (a default) |
| `…/<imager>/lens` | TextDocument | the lens model, K and the distortion terms |
| `…/<imager>/field` | LineStrips3D | for a lens that is not a pinhole: the true image border unprojected through the lens to 0.15 m (Rerun's Pinhole has no distortion) |
| `…/<sensor>/min_z` and `…/min_z/edges` | Mesh3D, LineStrips3D | the part of the depth imager's field closer than its stated min-Z (`depth_validity.blind_zone_local`); only when the maker states a min-Z |
| `world/task_volumes/<volume>` | Boxes3D | each task volume |
| `audit/verdict` | TextLog on the timeline `episode` | "episode <id>: <verdict> (<failed flags>)", at level INFO for pass, WARN for fix or recalibrate, ERROR for drop |
| `audit/pass`, `audit/flags/<flag>` | Scalars on `episode` | 1 or 0; for each flag, the number of cameras failing that requirement in the episode |

Notes on the table:

- **Trajectories.** A robot's trajectory is its first one in `project.tasks.trajectories`, or the one named. It is
  sampled at 30 Hz (a default), at most 2 000 samples, and is linear in the joints between its own samples, as
  `core/robot/trajectory.py`.
- **Which links.** The links logged are those with meshes, those carrying a sensor, the gripper frame and the
  fingertip frames.
- **Pixel convention.** K is written in OpenCV's convention, with pixel centres at integers. Rerun's documentation
  read on 2026-11-07 does not state its pixel-centre convention, so the frustum may sit half a pixel from Rerun's
  own.
- **Audit flags.** These come from the audit result of the session (Export ▸ Stream to Rerun…), of the same batch run
  (`--audit OUT.json --rerun OUT.rrd`), or of a saved file (`--rerun-audit AUDIT.json`). The flag names are
  `core/audit/episode.py FLAG`, the same as the FiftyOne tags: drop, duplicate, fps, jitter, usb_reset, desync,
  drift, exposure_change, blur, gripper_hidden, action_lag, mount_slip, unreadable.

**API, rerun-sdk 0.42.0.** These calls were read from the installed wheel's source and
https://rerun.io/docs/reference/types/archetypes/pinhole (version 0.42.0) on 2026-11-07:

- `RecordingStream('dexvantage')` with `.save(path)`, `.spawn()` and `.connect_grpc(url)`;
- `.log(path, archetype, static=…)`, `.set_time('time', duration=s)`, `.set_time('episode', sequence=i)` and
  `.reset_time()`.

The bridge uses a recording stream of its own rather than Rerun's global one, so several exports can run in one
process. Files are read back with `rerun.chunk.RrdReader` (`read_back`).

Ported from Vantage `core/viz_bridge.py`, the Rerun part: the frames, the Pinhole with RDF axes, the lens as text,
the true field outline beside the pinhole frustum, and the three modes save, spawn and connect. The articulated robot
over a trajectory, the link-relative sensors, the min-Z zones and the audit flags are new.

## Validation

`formats.rerun_rrd_readback` (cost "study", registered only when rerun-sdk is installed) writes the UR5e sample, its
scripted pick and a two-episode audit result to an `.rrd` file, then reads it back with rerun's own reader. It
requires:

- every entity path the bridge logged is in the file;
- every camera's Pinhole holds the imager's K and resolution (Rerun stores a 3 × 3 column-major), and every imager
  transform holds the pose in the body;
- the gripper link's transform at the sample nearest 1.5 s equals MuJoCo forward kinematics at the trajectory's
  joint values for that time;
- the audit flags are on the `episode` timeline with the values logged.

The tolerance is 1e-4 px or m, because Rerun stores these components as float32: 2⁻²⁴ relative rounding at
intrinsics up to about 1 400 px is up to 8.3 × 10⁻⁵ px. Measured 2026-11-07, rerun-sdk 0.42.0: 83 entities, none
missing; 6 cameras; intrinsics 4.8 × 10⁻⁵ px; imager poses 7.7 × 10⁻¹⁰ m; link pose 1.3 × 10⁻⁸ m; audit flags on the
timeline. Run time 0.6 s.

Measured sizes and times (`--rerun`, start-up excluded):

| Sample | Time | File | Content |
|---|---|---|---|
| UR5e | 0.3 s | 2.8 MB | 21 links over 136 poses, 6 frusta, 2 min-Z zones |
| SO-100 | 0.01 s | 0.76 MB | static, 7 links, 2 frusta, no min-Z: its cameras have no device |
| dexhand | 0.09 s | 2.6 MB | 30 links, 5 frusta, 1 min-Z zone |

`tests/test_rerun_bridge.py` also logs through a fake rerun module, so the entity layout is tested without the
package and without any viewer or network.

## Known limits

- Rerun's Pinhole is a pinhole: a fisheye's frustum there is approximate. The true field is the `field` lines, and
  the distortion terms are in the `lens` text.
- Meshes are logged once per link, with no level of detail, so a large robot makes a large file.
- Contact pads, task objects, lights, grasp sets and study results such as coverage are not logged.
- Only one trajectory per robot is logged. Robots without a trajectory are static at q (home by default).
- The audit's per-frame scores are not logged, only per-episode flags.
- Streaming to a live viewer was exercised only through the fake module, because the tests start no viewer.
