"""MuJoCo MJCF export (spec §9.2): `<camera>` and `<site>` snippets to paste into the robot's own MJCF, the full
scene (core/sim/scene.save_scene), or both.

    @exporter('mjcf', ...)   options: mode 'both' (default) | 'snippets' | 'scene'

Snippets (mjcf/cameras_snippet.xml, one file): for every body that carries a sensor, a commented block of elements
to paste inside that `<body name="…">` of the robot's file (its own body names, no prefix), and a block for the
`<worldbody>` with the sensors on static frames, in the project's world frame (the primary robot's base). Each
sensor gets a `<site name="<sensor>_mount">` at its body frame (the mount pose; MuJoCo's default site size and type,
a drawing aid) and each imager a `<camera>` at its optical frame turned by diag(1, −1, −1) (MuJoCo's camera looks
along −z with +y up), with resolution, sensorsize, focalpixel and principalpixel set exactly as core/sim/scene.py sets
them (that module's docstring gives the attribute definitions read in MuJoCo 3.17.0's XML reference and the
principal-point sign measured on renders). Numbers are written with 17 significant digits (Python's repr, which
reads back to the same float64); MuJoCo's own writer, which save_scene uses, keeps 6.

A sensor on a robot that is not the primary sits, in that robot's own file, at the same body frame: a link mount's
snippet does not depend on where the robot stands. A static sensor's pose is in the project's world frame, which is
the primary robot's file frame (its base is the world origin); for the file of another robot, move it by the inverse
of that robot's base pose (the snippet says so).

MuJoCo cameras are pinhole: a distorted lens gets its pinhole part (fx, fy, cx, cy) and the notes say so (DexVantage
renders through the lens by remapping a wider pinhole, core/sim/render.py; the snippet alone does not).

Known limits: the snippets name the bodies of the robot's file as MuJoCo compiled them (fusestatic off); a URDF robot's
link that MuJoCo merges away is reached through its parent body with the fixed offset folded into the pose; the scene
inherits save_scene's limits (6 significant digits, no sensor housings).
"""
from __future__ import annotations

import os
from xml.sax.saxutils import quoteattr

import numpy as np

from ..jobs import StudyError
from . import _common as C
from . import exporter

MODES = ('both', 'snippets', 'scene')
SNIPPET = 'mjcf/cameras_snippet.xml'
SCENE_DIR = 'mjcf/scene'


def _attrs(d: dict) -> str:
    return ' '.join(f'{k}={quoteattr(v)}' for k, v in d.items())


def _vec(a) -> str:
    return ' '.join(C.fmt(v) for v in np.asarray(a, float).reshape(-1))


def camera_element(name: str, T_body_optical, cam) -> str:
    """The `<camera>` element of an imager at T_body_optical (optical frame in the body) with cam's pinhole part."""
    from ..sim.scene import OPTICAL_TO_MUJOCO, _intrinsic_attrs, _mjquat
    T = np.asarray(T_body_optical, float)
    R = T[:3, :3] @ OPTICAL_TO_MUJOCO
    ia = _intrinsic_attrs(cam)
    return '<camera ' + _attrs({'name': name, 'pos': _vec(T[:3, 3]), 'quat': _vec(_mjquat(R)),
                                'resolution': ' '.join(str(int(v)) for v in ia['resolution']),
                                'sensorsize': _vec(ia['sensor_size']), 'focalpixel': _vec(ia['focal_pixel']),
                                'principalpixel': _vec(ia['principal_pixel'])}) + '/>'


def site_element(name: str, T_body_sensor) -> str:
    from ..sim.scene import _mjquat
    T = np.asarray(T_body_sensor, float)
    return '<site ' + _attrs({'name': name, 'pos': _vec(T[:3, 3]), 'quat': _vec(_mjquat(T[:3, :3]))}) + '/>'


def snippet_blocks(project, cams) -> tuple[dict, list]:
    """({(robot uid, body name) or ('', 'worldbody'): [element text]}, notes) for the rig's cameras."""
    from ..robot.model import get_model
    blocks, notes, sites = {}, [], set()
    for c in cams:
        if c.robot_uid:
            rm = get_model(project, c.robot_uid)
            li = rm.link_index(c.parent)
            body = rm.body_link[int(rm._lb[li])]
            T_bs = rm._loff[li] @ c.T_parent_sensor
            key = (c.robot_uid, body)
            if not np.allclose(rm._loff[li], np.eye(4)):
                notes.append(f'{c.sensor.name} is on {c.parent}, which MuJoCo merged into {body}: its snippet is in '
                             f'{body}\'s frame.')
        else:
            T_bs, key = c.T_parent_sensor, ('', 'worldbody')
        els = blocks.setdefault(key, [])
        sname = f'{C.frame_name(c.sensor.name)}_mount'
        if sname not in sites:
            sites.add(sname)
            els.append(site_element(sname, T_bs))
        els.append(camera_element(c.key, T_bs @ c.T_sensor_optical, c.camera))
        if c.camera.model != 'pinhole':
            notes.append(f'{c.title}: MuJoCo cameras are pinhole, so the snippet\'s camera has the lens\'s pinhole '
                         f'part (fx, fy, cx, cy) only; its {c.camera.model} distortion is not in the MJCF.')
    return blocks, notes


def snippet_text(project, cams, q_text: str) -> tuple[str, list]:
    blocks, notes = snippet_blocks(project, cams)
    lines = ['<!-- DexVantage camera and site snippets for MuJoCo (MJCF).',
             f'     Project "{project.name}", rig variant "{project.variant.name}".',
             '     Paste each block inside the named <body> of the robot\'s own MJCF (or in <worldbody>).',
             '     Cameras: optical frame turned by diag(1, -1, -1) (MuJoCo looks along -z, +y up); intrinsics from',
             '     resolution, sensorsize (nominal 3 um pixels: only ratios matter), focalpixel and principalpixel.',
             f'     Link-mounted poses do not depend on the joint state; the export was made with {q_text}. -->',
             '<mujocoinclude>']
    names = {r.uid: r.name for r in project.robots}
    for (ruid, body), els in blocks.items():
        if ruid:
            lines.append(f'  <!-- inside <body name="{body}"> of {names.get(ruid, ruid)} -->')
        else:
            lines.append('  <!-- inside <worldbody> (the project\'s world frame: the primary robot\'s base) -->')
        lines += [f'  {e}' for e in els]
    lines.append('</mujocoinclude>')
    if any(not r.primary for r in project.robots) and ('', 'worldbody') in blocks:
        notes.append('The <worldbody> block is in the project\'s world frame (the primary robot\'s base); for the file '
                     'of another robot, move it by the inverse of that robot\'s base pose.')
    return '\n'.join(lines) + '\n', notes


@exporter('mjcf', 'MuJoCo MJCF', needs_robot=False,
          description='Camera and site snippets for the robot\'s own MJCF, and the full scene (robots, objects, '
                      'tables, lights and every camera) as a self-contained scene.xml with its assets.')
def export_mjcf(project, out_dir, options, ctx):
    mode = str(options.get('mode') or 'both')
    if mode not in MODES:
        raise StudyError(f'The MJCF mode must be one of {", ".join(MODES)}, not "{mode}".')
    q, q_text = C.joint_state(project, options)
    files, notes = [], []
    cams = C.rig_cameras(project, q)
    if mode in ('both', 'snippets'):
        C.need_cameras(cams, 'The MJCF camera snippets')
        text, n = snippet_text(project, cams, q_text)
        files.append(C.write_text(out_dir, SNIPPET, text))
        notes += n
    if mode in ('both', 'scene'):
        from ..sim.scene import save_scene
        ctx.report(0.2, 'Writing the MuJoCo scene')
        folder = os.path.join(out_dir, SCENE_DIR)
        try:
            path = save_scene(project, folder)
        except OSError as e:
            raise StudyError(f'Could not write the MuJoCo scene: {e.strerror or e}.') from None
        except RuntimeError as e:
            raise StudyError(str(e)) from None
        rel = os.path.relpath(path, out_dir).replace(os.sep, '/')
        files.append(rel)
        files.append(f'{SCENE_DIR}/scene_notes.txt')
        for name in sorted(os.listdir(os.path.join(folder, 'assets'))):
            files.append(f'{SCENE_DIR}/assets/{name}')
        notes.append('The scene file is written by MuJoCo\'s own XML writer, which keeps 6 significant digits '
                     '(camera poses within about 10⁻⁶ m and rad, check formats.mjcf_scene); the robots stand at '
                     'their home pose in the file (set qpos by joint name to pose them).')
    return {'files': files, 'notes': list(dict.fromkeys(notes))}
