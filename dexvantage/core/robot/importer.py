"""Import a robot model file (URDF or MJCF) and every file it references.

import_robot(path) reads the model file and the meshes, included XML files and textures it names into memory, so the
project carries them (the .dexv archive's sources/ folder) and the robot opens on any machine. It then compiles them
with MuJoCo (core/robot/model.py) and fills the Robot's plain records from the file:

- Robot.links:  [{'name', 'parent', 'parent_joint', 'mesh_keys', 'collision_keys'}]
- Robot.joints: [{'name', 'type', 'parent', 'child', 'axis', 'origin': {'position', 'quaternion'}, 'lower', 'upper',
                  'velocity', 'effort'}] with type one of revolute, prismatic, continuous, fixed, ball, free.
  Limits, velocities and efforts are the file's own figures (radians, metres, rad/s or m/s, N·m or N); a figure the
  file does not state is None. Nothing is typed in again or filled with a guess.
- Robot.home:   the file's keyframe named "home" when it has one (MJCF), else zero clipped to the limits.

Where the files go. Each file is stored under "<robot uid>/<path relative to the robot's folder>", so several robots
coexist in one project. The robot's folder is the model file's folder, or the folder that holds the ROS package a
package:// path names when that is higher up. A file outside that folder is stored under "<uid>/_ext/<name>". A small
manifest, "<uid>/.dexvantage/refs.json", records which stored file each reference in the model files means, so the
model compiles again from memory exactly as it was imported. Robot.source = RobotSource(kind, file=<the main file's
key>, sha256=<of the main file's bytes>).

How references resolve.
- URDF <mesh filename>: package://pkg/rest is looked for by walking up from the URDF's folder to a folder named pkg
  (or a folder that contains one); file:///abs and absolute paths are used as written; anything else is relative to
  the URDF's folder. <mesh scale> is kept (MuJoCo applies it).
- MJCF: <include file> relative to the main file's folder; mesh, skin and height-field files under the compiler's
  meshdir, textures under texturedir (assetdir sets both), as MuJoCo resolves them. The list comes from MuJoCo's own
  parser (mujoco.MjSpec), so nothing is missed by a hand-written scan.

A missing mesh or included file stops the import with a message that names the file and what to pick. A missing MJCF
texture does not: the robot is drawn without it and Robot.source.note says so (textures change no geometry).

Units check: a mesh more than 10 m across (MESH_MAX_EXTENT) is almost certainly in millimetres; the import stops and
says to set scale="0.001 0.001 0.001".

USD (.usd, .usda, .usdc, .usdz; needs the optional usd-core package): a UsdPhysics articulation is converted to MJCF
by core/robot/usd_import.py (bodies, revolute and prismatic joints with their axes and limits, meshes written as STL,
collision shapes, mass properties where the file states them) and that MJCF is imported as above; the stored files
are the converted ones, with <uid>/.dexvantage/usd_source.json recording the USD layers (SHA-256) and the conversion
notes, and Robot.source.note naming the USD file.

Known limits: xacro files must be expanded to plain URDF first (xacro is a ROS program, not a file format); URDF
<mimic> joints are imported as independent joints; MJCF <model> (attach) assets and plugins are not
collected.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import xml.etree.ElementTree as ET

from ..frames import Pose, quat_from_rpy
from ..project import ProjectFormatError, Robot, RobotSource, new_uid

MANIFEST = '.dexvantage/refs.json'      # under "<uid>/": which stored file each reference in the model files means
MANIFEST_FORMAT = 'dexvantage.robot-refs'
MESH_MAX_EXTENT = 10.0                  # m: a larger mesh is taken to be in millimetres (a plausibility default)
EXT_DIR = '_ext'
URDF_JOINT_TYPES = {'revolute': 'revolute', 'continuous': 'continuous', 'prismatic': 'prismatic', 'fixed': 'fixed',
                    'floating': 'free', 'planar': 'planar'}


class RobotImportError(ProjectFormatError):
    """A robot file DexVantage cannot import; the message says why and what to do, in plain words."""


# ====================================================================== small helpers
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _floats(text, n, default=None):
    if text is None:
        return default
    try:
        v = [float(x) for x in str(text).split()]
    except ValueError:
        return default
    return v if len(v) == n else default


def _num(el, attr):
    """A float attribute of el, or None when el or the attribute is absent (the file does not state it)."""
    if el is None or el.get(attr) is None:
        return None
    try:
        return float(el.get(attr))
    except ValueError:
        return None


USD_EXTS = ('.usd', '.usda', '.usdc', '.usdz')


def check_path(path: str):
    """Raise RobotImportError, in plain words, when path is not a robot file this build can read: missing, a folder,
    or USD without the optional usd-core package."""
    base = os.path.basename(os.path.normpath(path)) or path
    if os.path.isdir(path):
        raise RobotImportError(f'{base} is a folder: pick the robot\'s .urdf or MuJoCo .xml file inside it.')
    if not os.path.isfile(path):
        raise RobotImportError(f'There is no file at {base}.')
    if os.path.splitext(path)[1].lower() in USD_EXTS:
        from .usd_import import have_pxr, missing_pxr_message
        if not have_pxr():
            raise RobotImportError(missing_pxr_message(base))


def _import_usd(path: str, name: str | None, uid: str | None, check_units: bool):
    """A USD robot: converted to MJCF in a temporary folder (core/robot/usd_import.py), imported from there, the
    conversion record stored beside it."""
    import shutil
    import tempfile

    from . import usd_import as U
    tmp = tempfile.mkdtemp(prefix='dexvantage-usd-')
    try:
        try:
            conv = U.convert_to_folder(path, tmp)
        except U.UsdImportError as e:
            raise RobotImportError(str(e)) from None
        robot, sources = import_robot(os.path.join(tmp, U.MAIN), name or conv.name, uid=uid, check_units=check_units)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sources[f'{robot.uid}/{U.RECORD}'] = U.record_bytes(conv)
    base = os.path.basename(path)
    with open(path, 'rb') as fh:
        digest = _sha256(fh.read())
    robot.source.note = ' '.join(x for x in (
        f'Converted from the USD file {base} (SHA-256 {digest[:12]}…) by DexVantage\'s USD importer; the stored model '
        f'is the MJCF it wrote ({U.RECORD} lists the layers).', *conv.notes, robot.source.note) if x)
    return robot, sources


def detect_kind(path: str, data: bytes) -> str:
    """'urdf' or 'mjcf' from the root element (the extension is only a hint: MJCF files are .xml, URDFs sometimes)."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise RobotImportError(f'{os.path.basename(path)} is not valid XML ({e}), so it is neither a URDF nor an '
                               'MJCF file.') from None
    tag = root.tag.split('}')[-1]
    xacro = path.lower().endswith('.xacro') or 'xacro' in root.tag or (tag == 'robot' and any(
        'xacro' in str(el.tag) or any('$(' in v or '${' in v for v in el.attrib.values()) for el in root.iter()))
    if xacro:
        raise RobotImportError(f'{os.path.basename(path)} is a xacro file (it has xacro macros or $(…) / ${{…}} '
                               'expressions). Expand it to plain URDF first (ros2 run xacro xacro in.urdf.xacro > '
                               'out.urdf), then import the .urdf.')
    if tag == 'robot':
        if root.find('link') is None:
            raise RobotImportError(f'{os.path.basename(path)} is a URDF with no <link>: there is no robot in it. '
                                   'Pick the complete URDF.')
        return 'urdf'
    if tag == 'mujoco':
        return 'mjcf'
    raise RobotImportError(f'{os.path.basename(path)} starts with <{tag}>: a URDF starts with <robot> and an MJCF '
                           'file with <mujoco>. Pick the robot\'s .urdf or MuJoCo .xml file.')


def _find_package(pkg: str, start_dir: str) -> str | None:
    """The folder of ROS package pkg: walk up from start_dir to a folder named pkg or one that contains pkg/."""
    d = os.path.abspath(start_dir)
    while True:
        if os.path.basename(d) == pkg:
            return d
        if os.path.isdir(os.path.join(d, pkg)):
            return os.path.join(d, pkg)
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def _missing(kind_word: str, what: str, ref: str, main: str, hint: str) -> RobotImportError:
    return RobotImportError(f'The {kind_word} names {what} {os.path.basename(ref.rstrip("/"))} ({ref}), which is not '
                            f'in the folder you chose ({os.path.basename(os.path.dirname(os.path.abspath(main)))}): '
                            f'{hint}')


def _read(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


class _Files:
    """The files of one robot: absolute path -> stored relative path, given the robot's folder (anchor)."""

    def __init__(self, anchor: str):
        self.anchor = os.path.abspath(anchor)
        self.rel: dict[str, str] = {}

    def add(self, path: str) -> str:
        path = os.path.abspath(path)
        if path in self.rel:
            return self.rel[path]
        r = os.path.relpath(path, self.anchor)
        if r.startswith('..') or os.path.isabs(r):
            base = os.path.basename(path)
            taken = set(self.rel.values())
            r, i = f'{EXT_DIR}/{base}', 1
            while r in taken:
                r, i = f'{EXT_DIR}/{i}_{base}', i + 1
        r = r.replace(os.sep, '/')
        self.rel[path] = r
        return r


# ====================================================================== URDF
def _urdf_resolve(ref: str, main: str) -> tuple[str, str | None]:
    """(absolute path, package folder or None) of a URDF filename; raises RobotImportError when it is not found."""
    base = os.path.dirname(os.path.abspath(main))
    if ref.startswith('package://'):
        rest = ref[len('package://'):]
        pkg, _, sub = rest.partition('/')
        root = _find_package(pkg, base)
        if root is None:
            raise RobotImportError(
                f'The URDF names mesh {posixpath.basename(ref)} inside ROS package "{pkg}" ({ref}), but no folder '
                f'named {pkg} is above or beside {os.path.basename(main)}: keep the URDF inside its package folder '
                f'(the folder named {pkg}, with its meshes) and import it from there.')
        p = os.path.join(root, *sub.split('/'))
        if not os.path.isfile(p):
            raise _missing('URDF', 'mesh', ref, main, f'it should be at {pkg}/{sub}. Pick the URDF inside the complete '
                                                       f'package folder {pkg}, with its meshes.')
        return p, root
    if ref.startswith('file://'):
        p = ref[len('file://'):]
    else:
        p = ref if os.path.isabs(ref) else os.path.join(base, *ref.split('/'))
    if not os.path.isfile(p):
        raise _missing('URDF', 'mesh', ref, main, 'pick the package folder that holds the URDF and its meshes, or '
                                                  'copy the mesh next to the URDF.')
    return p, None


def _origin(el) -> dict:
    o = el.find('origin') if el is not None else None
    xyz = _floats(o.get('xyz') if o is not None else None, 3, [0.0, 0.0, 0.0])
    rpy = _floats(o.get('rpy') if o is not None else None, 3, [0.0, 0.0, 0.0])
    return Pose(xyz, quat_from_rpy(*rpy)).to_dict()


def _urdf_records(root, mesh_rel: dict) -> tuple[list, list]:
    """Robot.links and Robot.joints from the URDF's own elements (the file's figures, untouched)."""
    joints = []
    parent_of = {}
    for j in root.findall('joint'):
        jt = str(j.get('type', 'fixed'))
        p, c = j.find('parent'), j.find('child')
        lim = j.find('limit')
        axis = _floats(j.find('axis').get('xyz') if j.find('axis') is not None else None, 3, [1.0, 0.0, 0.0])
        rec = {'name': str(j.get('name', '')), 'type': URDF_JOINT_TYPES.get(jt, jt),
               'parent': p.get('link', '') if p is not None else '',
               'child': c.get('link', '') if c is not None else '',
               'axis': axis, 'origin': _origin(j),
               'lower': _num(lim, 'lower') if jt in ('revolute', 'prismatic') else None,
               'upper': _num(lim, 'upper') if jt in ('revolute', 'prismatic') else None,
               'velocity': _num(lim, 'velocity'), 'effort': _num(lim, 'effort')}
        joints.append(rec)
        parent_of[rec['child']] = rec
    links = []
    for ln in root.findall('link'):
        name = str(ln.get('name', ''))

        def keys(tag):
            out = []
            for g in ln.findall(tag):
                m = g.find('geometry/mesh')
                if m is not None and m.get('filename') in mesh_rel and mesh_rel[m.get('filename')] not in out:
                    out.append(mesh_rel[m.get('filename')])
            return out
        pj = parent_of.get(name)
        links.append({'name': name, 'parent': pj['parent'] if pj else '', 'parent_joint': pj['name'] if pj else '',
                      'mesh_keys': keys('visual'), 'collision_keys': keys('collision')})
    return links, joints


def _import_urdf(path: str, data: bytes):
    root = ET.fromstring(data)
    main = os.path.abspath(path)
    refs = []                                          # (filename as written, absolute path, package folder)
    for m in root.iter('mesh'):
        fn = m.get('filename')
        if fn and fn not in [r[0] for r in refs]:
            refs.append((fn, *_urdf_resolve(fn, main)))
    tex = []
    for t in root.iter('texture'):
        fn = t.get('filename')
        if fn:
            try:
                tex.append((fn, _urdf_resolve(fn, main)[0]))
            except RobotImportError:
                pass                                   # textures are optional: MuJoCo's URDF reader ignores them
    anchor = os.path.dirname(main)
    for _fn, _p, pkg in refs:
        if pkg is not None:
            up = os.path.dirname(pkg)
            if os.path.commonpath([up, anchor]) == up:  # the package's parent is above the URDF: anchor there
                anchor = up
    files = _Files(anchor)
    main_rel = files.add(main)
    manifest = {'mesh:' + fn: files.add(p) for fn, p, _pkg in refs}
    manifest.update({'texture:' + fn: files.add(p) for fn, p in tex})
    blobs = {files.rel[os.path.abspath(p)]: _read(p) for _fn, p, _pkg in refs}
    blobs.update({files.rel[os.path.abspath(p)]: _read(p) for _fn, p in tex})
    links, joints = _urdf_records(root, {k[5:]: v for k, v in manifest.items() if k.startswith('mesh:')})
    return main_rel, manifest, blobs, links, joints, str(root.get('name') or ''), []


# ====================================================================== MJCF
def _mjcf_includes(path: str) -> dict:
    """{file as written: absolute path} of every <include>, recursively (MuJoCo: relative to the main file's
    folder; the including file's own folder is tried too)."""
    main_dir = os.path.dirname(os.path.abspath(path))
    out = {}
    todo = [os.path.abspath(path)]
    while todo:
        cur = todo.pop()
        try:
            root = ET.fromstring(_read(cur))
        except ET.ParseError as e:
            raise RobotImportError(f'{os.path.basename(cur)} is not valid XML ({e}).') from None
        for inc in root.iter('include'):
            fn = inc.get('file')
            if not fn or fn in out:
                continue
            cands = [fn] if os.path.isabs(fn) else [os.path.join(main_dir, fn), os.path.join(os.path.dirname(cur), fn)]
            p = next((c for c in cands if os.path.isfile(c)), None)
            if p is None:
                raise _missing('MJCF file', 'the included file', fn, path,
                               'pick the robot\'s complete folder, with the files it includes.')
            out[fn] = os.path.abspath(p)
            todo.append(out[fn])
    return out


def _mjcf_asset_path(file: str, folder: str, main_dir: str) -> str | None:
    if os.path.isabs(file):
        return file if os.path.isfile(file) else None
    for c in (os.path.join(main_dir, folder, file), os.path.join(main_dir, file)):
        if os.path.isfile(c):
            return c
    return None


def mjcf_spec_files(spec) -> list:
    """[(category, element, attribute or index, file as written)] for every file a parsed MJCF spec names."""
    out = []
    for m in spec.meshes:
        if m.file:
            out.append(('mesh', m, 'file', m.file))
    for s in spec.skins:
        if s.file:
            out.append(('skin', s, 'file', s.file))
    for h in spec.hfields:
        if h.file:
            out.append(('hfield', h, 'file', h.file))
    for t in spec.textures:
        if t.file:
            out.append(('texture', t, 'file', t.file))
        for i, f in enumerate(list(t.cubefiles)):
            if f:
                out.append(('texture', t, i, f))
    return out


def _import_mjcf(path: str, data: bytes):
    import mujoco
    main = os.path.abspath(path)
    main_dir = os.path.dirname(main)
    incs = _mjcf_includes(main)
    try:
        spec = mujoco.MjSpec.from_string(data.decode('utf-8'), include={k: _read(p) for k, p in incs.items()})
    except Exception as e:                             # noqa: BLE001 — explained
        raise RobotImportError(f'MuJoCo could not read {os.path.basename(path)}: {str(e).strip().splitlines()[0]}.') \
            from None
    files = _Files(main_dir)
    main_rel = files.add(main)
    manifest = {'include:' + k: files.add(p) for k, p in incs.items()}
    blobs = {files.rel[p]: _read(p) for p in incs.values()}
    notes = []
    for cat, _el, _attr, fn in mjcf_spec_files(spec):
        folder = spec.texturedir if cat == 'texture' else spec.meshdir
        p = _mjcf_asset_path(fn, folder, main_dir)
        if p is None:
            if cat == 'texture':
                notes.append(fn)
                continue
            raise _missing('MJCF file', f'the {cat}', fn, path,
                           'pick the robot\'s complete folder (its meshes are usually in assets/ or meshes/ beside the '
                           '.xml).')
        manifest[f'{cat}:{fn}'] = files.add(p)
        blobs[files.rel[os.path.abspath(p)]] = _read(p)
    return main_rel, manifest, blobs, None, None, str(spec.modelname or ''), notes


def _mjcf_records(model) -> tuple[list, list]:
    """Robot.links and Robot.joints read back from MuJoCo's compiled MJCF (the file's own figures)."""
    import mujoco
    m = model.m
    links, joints = [], []
    for b in range(1, m.nbody):
        name = model.body_link[b]
        par = int(m.body_parentid[b])
        pname = model.body_link[par] if par > 0 else ''
        jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        T = Pose(m.body_pos[b], [*m.body_quat[b][1:], m.body_quat[b][0]]).to_dict()
        for j in jids:
            jt = int(m.jnt_type[j])
            limited = bool(m.jnt_limited[j])
            J = mujoco.mjtJoint
            kind = {int(J.mjJNT_FREE): 'free', int(J.mjJNT_BALL): 'ball', int(J.mjJNT_SLIDE): 'prismatic',
                    int(J.mjJNT_HINGE): 'revolute' if limited else 'continuous'}[jt]
            one = jt in (int(J.mjJNT_SLIDE), int(J.mjJNT_HINGE))
            frc = float(m.jnt_actfrcrange[j][1]) if bool(m.jnt_actfrclimited[j]) else None
            joints.append({'name': model.joint_name(j), 'type': kind, 'parent': pname, 'child': name,
                           'axis': [float(v) for v in m.jnt_axis[j]], 'origin': T,
                           'anchor': [float(v) for v in m.jnt_pos[j]],
                           'lower': float(m.jnt_range[j][0]) if limited and one else None,
                           'upper': float(m.jnt_range[j][1]) if limited and one else None,
                           'velocity': None, 'effort': frc})
        if not jids:
            joints.append({'name': f'{name}_fixed', 'type': 'fixed', 'parent': pname, 'child': name,
                           'axis': [1.0, 0.0, 0.0], 'origin': T, 'lower': None, 'upper': None, 'velocity': None,
                           'effort': None})
        links.append({'name': name, 'parent': pname, 'parent_joint': model.joint_name(jids[0]) if jids else '',
                      'mesh_keys': model.body_mesh_keys(b, visual=True),
                      'collision_keys': model.body_mesh_keys(b, visual=False)})
    return links, joints


# ====================================================================== the entry point
def import_robot(path: str, name: str | None = None, *, uid: str | None = None,
                 check_units: bool = True) -> tuple[Robot, dict[str, bytes]]:
    """Read a URDF, MJCF or USD file and the files it references: (Robot, {source key: bytes}).

    The Robot is not added to any project: the caller adds it and the sources (project.sources.update) as one undo
    step. Raises RobotImportError with a plain-English message for anything it cannot use."""
    from .model import RobotModel, RobotModelError
    check_path(path)
    if os.path.splitext(path)[1].lower() in USD_EXTS:
        return _import_usd(path, name, uid, check_units)
    data = _read(path)
    kind = detect_kind(path, data)
    main_rel, manifest, blobs, links, joints, file_name, notes = (_import_urdf if kind == 'urdf' else _import_mjcf)(
        path, data)
    uid = uid or new_uid()
    for ln in links or []:
        ln['mesh_keys'] = [f'{uid}/{k}' for k in ln['mesh_keys']]
        ln['collision_keys'] = [f'{uid}/{k}' for k in ln['collision_keys']]
    sources = {f'{uid}/{rel}': b for rel, b in blobs.items()}
    sources[f'{uid}/{main_rel}'] = data
    sources[f'{uid}/{MANIFEST}'] = json.dumps({'format': MANIFEST_FORMAT, 'version': 1, 'kind': kind,
                                               'main': main_rel, 'refs': manifest, 'missing': notes},
                                              indent=1, sort_keys=True).encode()
    note = ''
    if notes:
        note = 'Drawn without these textures, which were not found: ' + ', '.join(sorted(set(notes))) + '.'
    robot = Robot(uid=uid, name=name or file_name or os.path.splitext(os.path.basename(path))[0],
                  source=RobotSource(kind=kind, file=f'{uid}/{main_rel}', sha256=_sha256(data), note=note))
    robot.links, robot.joints = links or [], joints or []
    try:
        model = RobotModel(robot, sources)
    except RobotModelError as e:
        raise RobotImportError(str(e)) from None
    if kind == 'mjcf':
        robot.links, robot.joints = _mjcf_records(model)
        model = RobotModel(robot, sources)
    if check_units:
        for mesh_name, key, ext in model.mesh_extents():
            if ext > MESH_MAX_EXTENT:
                where = 'on its <mesh> in the URDF' if kind == 'urdf' else 'on its <mesh> asset in the MJCF file'
                raise RobotImportError(
                    f'The mesh {posixpath.basename(key) if key else mesh_name} is {ext:,.0f} m across, which looks '
                    f'like millimetres rather than metres: set scale="0.001 0.001 0.001" {where}, then import again.')
    robot.home = {n: float(v) for n, v in zip(model.joint_names, model.file_home())}
    return robot, sources
