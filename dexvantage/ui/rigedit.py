"""Rig edits and library figures shared by the Sensors and mounts page, the Library panel, the Inspector and the 3D
view's drop target: add a sensor of a library device, duplicate, remove, re-mount keeping its place in the world, and
the plain-text description of a library entry's figures ("—" for a figure the maker does not state).

Every edit is one undo step (store.edit on the active variant).
"""
from __future__ import annotations

import math

import numpy as np

from ..core import project as P
from ..core.frames import Pose, quat_from_rpy
from ..core.library import NOT_STATED

DASH = '—'
MIME = 'application/x-dexvantage-device'      # a library device id, dragged from the Library panel
SENSOR_KINDS = {'camera': 'camera', 'depth': 'depth', 'tactile': 'tactile'}   # library kind -> project sensor kind
STATIC_FRAMES = P.MOUNT_STATIC                 # world, cage, tripod, table


# ====================================================================== figures
def num(v, unit: str = '', dp: int = 1) -> str:
    """A figure with its unit, or "—" when the maker does not state it (v is None)."""
    if v is None:
        return DASH
    if isinstance(v, float) and not math.isfinite(v):
        return DASH
    s = f'{v:.{dp}f}' if isinstance(v, float) else str(v)
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if unit in ('°', '%'):
        return s + unit
    return f'{s} {unit}'.strip() if unit else s


def fov_text(im: dict) -> str:
    return ' × '.join(num(im.get(k), '°') for k in ('hfov_deg', 'vfov_deg', 'dfov_deg'))


def res_text(r) -> str:
    return f'{r[0]} × {r[1]}' if r else DASH


def device_title(e) -> str:
    """A device's display name: Entry.title, with a word the maker's name ends with and the device's name starts with
    said once ("Intel RealSense" + "RealSense Depth Camera D455" -> "Intel RealSense Depth Camera D455")."""
    m, n = e.maker.split(), e.name.split()
    if m and n and m[-1].lower() == n[0].lower():
        return ' '.join(m + n[1:])
    return e.title


def stream_modes(entry) -> list:
    """[(w, h, 'w × h @ fps, ... · imagers')] of a library entry, unique sizes, largest first."""
    seen: dict = {}
    for s in entry.streams() if entry is not None else []:
        k = (int(s['width']), int(s['height']))
        d = seen.setdefault(k, {'fps': set(), 'imagers': []})
        d['fps'].update(s.get('fps') or [])
        if s.get('imager') not in d['imagers']:
            d['imagers'].append(s.get('imager'))
    out = []
    for (w, h), d in sorted(seen.items(), key=lambda kv: (-kv[0][0] * kv[0][1], -kv[0][0])):
        fps = ', '.join(f'{f:g}' for f in sorted(d['fps']))
        out.append((w, h, f'{w} × {h}' + (f' @ {fps} fps' if fps else '') + f' · {", ".join(map(str, d["imagers"]))}'))
    return out


def entry_summary(e) -> str:
    """One short line of the device's headline figures (the first imager's FOV and resolution)."""
    ims = e.imagers
    if ims:
        im = e.reference_imager or ims[0]
        return f'{num(im.get("hfov_deg"), "°", 0)} · {res_text(im.get("resolution"))}'
    if e.kind == 'tactile':
        t = e.data.get('tactile') or {}
        pad = t.get('pad_mm')
        return f'pad {pad[0]:g} × {pad[1]:g} mm' if pad else 'pad ' + DASH
    if e.kind == 'host':
        h = e.data.get('host') or {}
        return num(h.get('stated_mbps'), 'Mb/s', 0)
    return ''


def entry_tooltip(e) -> str:
    """Every figure of a library entry, "—" marking those the maker does not state, and the sources with the date
    each was read."""
    d = e.data
    lines = [device_title(e) + (' (your custom device)' if e.is_custom else ''),
             f'{e.kind} · {d.get("family") or ""}'.rstrip(' ·')]
    for im in e.imagers:
        lines.append(f'{im["name"]} ({im["role"]}): FOV {fov_text(im)}, {res_text(im.get("resolution"))} px, '
                     f'{im.get("shutter") or DASH} shutter, pixel {num(im.get("pixel_um"), "µm", 2)}, '
                     f'f/{num(im.get("f_number"), "", 1)}, lens {im.get("lens_model") or DASH}'
                     + ('' if im.get('pose_stated', True) else ', offset assumed'))
    dp = d.get('depth')
    if dp:
        mz = ', '.join(f'{r["width"]}×{r["height"]}: {num(r.get("m"), "m", 3)}' for r in dp.get('min_z') or [])
        lines.append(f'Depth: {dp.get("technology") or DASH}, baseline {num(dp.get("baseline_m"), "m", 4)}, range up '
                     f'to {num(dp.get("max_range_m"), "m", 2)}, min-Z {mz or DASH}')
        rms = dp.get('rms_error') or {}
        if rms:
            lines.append(f'Depth error as stated: {rms.get("text") or DASH}')
    pj = d.get('projector')
    if pj:
        lines.append(f'IR projector: FOV {fov_text(pj)}')
    t = d.get('tactile')
    if t:
        pad = t.get('pad_mm')
        lines.append(f'Pad {pad[0]:g} × {pad[1]:g} mm' if pad else f'Pad {DASH}')
        lines.append(f'Camera {res_text(t.get("camera_resolution"))} at {num(t.get("fps"), "fps", 0)}, '
                     f'{t.get("sensing") or DASH}')
    h = d.get('host')
    if h:
        lines.append(f'Host: {h.get("type") or DASH}, ports {num(h.get("ports"))}, stated '
                     f'{num(h.get("stated_mbps"), "Mb/s", 0)}, practical {num(h.get("practical_mbps"), "Mb/s", 0)}')
    lines.append(f'Interface {d.get("interface") or DASH} · power {num(d.get("power_w"), "W", 2)} · mass '
                 f'{num(d.get("mass_g"), "g", 0)}')
    lines.append(f'"{DASH}": {NOT_STATED}.')
    for s in e.sources:
        lines.append(f'Source: {s.get("title") or s.get("url")} — {s.get("url")} (read {s.get("read")})')
    return '\n'.join(lines)


# ====================================================================== edits
def look_at_pose(pos, target) -> Pose:
    """A sensor body pose at pos with its x axis (the optical axis) towards target, no roll."""
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.atan2(d[1], d[0])
    pitch = -math.atan2(d[2], math.hypot(d[0], d[1]))
    return Pose(np.asarray(pos, float), quat_from_rpy(0.0, pitch, yaw))


def task_centre(project) -> np.ndarray:
    vols = project.workspace.volumes
    return np.asarray(vols[0].center if vols else (0.45, 0.0, 0.15), float)


def default_static_pose(project, n: int, at=None) -> Pose:
    """Where a new static sensor goes (as MainWindow.add_static_camera places one): around the task volume at 0.75 m,
    or above a dropped floor point; looking at the task volume's centre."""
    if at is not None:
        pos = np.array([float(at[0]), float(at[1]), 0.75])
    else:
        pos = np.array([0.45 - 0.5 * math.cos(0.6 * n), 0.55 * math.sin(0.6 * n + 1), 0.75])
    return look_at_pose(pos, task_centre(project))


def unique_name(project, base: str) -> str:
    names = {s.name for s in project.variant.sensors}
    if base not in names:
        return base
    k = 2
    while f'{base} ({k})' in names:
        k += 1
    return f'{base} ({k})'


def new_sensor(project, model_id: str = '', at=None) -> P.Sensor:
    """A sensor of a library device (or the default camera when model_id is ''): cameras and depth cameras on a
    tripod looking at the task volume; a tactile pad on the primary robot's gripper frame when it has one."""
    from ..core import library
    e = library.get(model_id) if model_id else None
    n = len(project.variant.sensors)
    kind = SENSOR_KINDS.get(e.kind, 'camera') if e is not None else 'camera'
    name = unique_name(project, e.name if e is not None else f'Camera {n + 1}')
    r = project.primary_robot
    if kind == 'tactile' and r is not None and (r.gripper_frame or r.fingertip_frames):
        mount = P.Mount((r.fingertip_frames or [r.gripper_frame])[0], r.uid, Pose())
    else:
        mount = P.Mount('tripod', '', default_static_pose(project, n, at))
    return P.Sensor(name=name, kind=kind, model_id=e.id if e is not None else '', mount=mount,
                    slot=_free_slot(project))


def _free_slot(project) -> int:
    used = [s.slot % 8 for s in project.variant.sensors]
    return next((k for k in range(8) if k not in used), len(used) % 8)


def add_sensor(store, model_id: str = '', at=None) -> P.Sensor:
    """Add a sensor of a library device to the active rig (one undo step) and select it."""
    s = new_sensor(store.project, model_id, at)
    store.edit('variant', lambda p: p.variant.sensors.append(P.Sensor.from_dict(s.to_dict())), f'Added {s.name}')
    store.select(('sensor', s.uid))
    return s


def duplicate_sensor(store, uid: str) -> P.Sensor | None:
    src = store.project.variant.sensor(uid)
    if src is None:
        return None

    def fn(p):
        s = P.Sensor.from_dict(src.to_dict())
        s.uid = P.new_uid()
        s.name = unique_name(p, src.name)
        s.slot = _free_slot(p)
        s.locked = False
        s.mount.pose = Pose(s.mount.pose.position + np.array([0.0, 0.05, 0.0]), s.mount.pose.quaternion)
        p.variant.sensors.insert(p.variant.sensors.index(p.variant.sensor(uid)) + 1, s)
        return s
    s = store.edit('variant', fn, f'Duplicated {src.name}')
    store.select(('sensor', s.uid))
    return s


def remove_sensor(store, uid: str) -> bool:
    s = store.project.variant.sensor(uid)
    if s is None:
        return False
    store.edit('variant', lambda p: p.variant.sensors.remove(p.variant.sensor(uid)), f'Removed {s.name}')
    if store.sel[:2] == ('sensor', uid):
        store.select(())
    return True


def parent_pose(project, robot_uid: str, frame: str, q: dict) -> Pose:
    """The world pose of a mount's parent frame: the identity for static frames, else the link at q."""
    if not robot_uid or frame in STATIC_FRAMES:
        return Pose()
    from ..core.robot.model import get_model
    m = get_model(project, robot_uid)
    v = q.get(robot_uid)
    return Pose.from_matrix(m.link_pose(m.home if v is None or len(v) != m.nq else v, frame))


def remount(store, uid: str, frame: str, robot_uid: str = '', keep_place: bool = True) -> bool:
    """Mount a sensor on another parent frame (a static frame, or a robot's link), one undo step. keep_place: the
    sensor stays where it is in the world (its mount pose is recomputed in the new parent), else the mount pose
    numbers are kept."""
    from ..core.scenegeom import sensor_world_pose
    st = store
    s = st.project.variant.sensor(uid)
    if s is None:
        return False
    robot_uid = '' if frame in STATIC_FRAMES else robot_uid
    world = None
    if keep_place:
        try:
            world = sensor_world_pose(st.project, s, st.q)
            new_pose = parent_pose(st.project, robot_uid, frame, st.q).inverse() @ world
        except Exception:                                # noqa: BLE001 — a robot that does not compile: keep numbers
            world = None
    where = frame if not robot_uid else f'{frame} of {st.project.robot(robot_uid).name}'

    def fn(p):
        o = p.variant.sensor(uid)
        o.mount.parent_frame = frame
        o.mount.robot_uid = robot_uid
        if world is not None:
            o.mount.pose = new_pose
    st.edit('variant', fn, f'Mounted {s.name} on {where}')
    return True
