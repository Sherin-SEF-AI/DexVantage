"""The import wizard's mesh check: every file a URDF or MJCF names, found or missing, before the import.

check_references(path) resolves each reference the way the importer does (core/robot/importer.py: the same package://
walk, MJCF meshdir and texturedir, MuJoCo's own parser for the MJCF asset list), but lists every reference instead of
stopping at the first missing one, so the wizard can show the whole picture:

    {'kind': 'urdf' | 'mjcf', 'items': [{'what', 'ref', 'found', 'file', 'required', 'message'}]}

'what' is mesh, include, texture, skin or hfield, or model for the model file itself when MuJoCo cannot parse it
(then its other files cannot be listed); 'file' the resolved file's name ('' when missing); 'required' False
for textures (a missing texture changes no geometry; the robot is drawn without it); 'message' the importer's plain
sentence for a missing file, '' when found.

A USD file (core/robot/usd_import.check_references) lists the layers its stage uses and the asset paths that do not
resolve ('what' layer; a missing one is not required: the stage composes without it).

Known limits: as the importer's (xacro is not read; MJCF <model> attach assets are not listed). When an MJCF
include is missing, MuJoCo cannot parse the rest, so its assets are not listed until the include is found.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from . import importer as I


def check_references(path: str) -> dict:
    """See the module docstring. Raises RobotImportError when the file itself cannot be read."""
    I.check_path(path)
    if os.path.splitext(path)[1].lower() in I.USD_EXTS:
        from . import usd_import as U
        try:
            return U.check_references(path)
        except U.UsdImportError as e:
            raise I.RobotImportError(str(e)) from None
    data = I._read(path)
    kind = I.detect_kind(path, data)
    return {'kind': kind, 'items': _urdf(path, data) if kind == 'urdf' else _mjcf(path, data)}


def _item(what, ref, p, required=True, message=''):
    return {'what': what, 'ref': ref, 'found': p is not None, 'file': os.path.basename(p) if p else '',
            'required': required, 'message': message}


def _urdf(path, data):
    root = ET.fromstring(data)
    out, seen = [], set()
    for tag, what in (('mesh', 'mesh'), ('texture', 'texture')):
        for el in root.iter(tag):
            fn = el.get('filename')
            if not fn or (what, fn) in seen:
                continue
            seen.add((what, fn))
            try:
                p, _pkg = I._urdf_resolve(fn, path)
                out.append(_item(what, fn, p, what == 'mesh'))
            except I.RobotImportError as e:
                msg = str(e) if what == 'mesh' else 'Not found: the robot is drawn without this texture.'
                out.append(_item(what, fn, None, what == 'mesh', msg))
    return out


def _mjcf(path, data):
    import mujoco
    main = os.path.abspath(path)
    main_dir = os.path.dirname(main)
    try:
        incs = I._mjcf_includes(main)
    except I.RobotImportError as e:
        return [_item('include', '', None, True, str(e))]
    out = [_item('include', k, p) for k, p in incs.items()]
    try:
        spec = mujoco.MjSpec.from_string(data.decode('utf-8'), include={k: I._read(p) for k, p in incs.items()})
    except Exception as e:                             # noqa: BLE001 — explained
        out.append(_item('model', '', None, True,
                         f'MuJoCo could not read {os.path.basename(path)}: {str(e).strip().splitlines()[0]}.'))
        return out
    seen = set()
    for cat, _el, _attr, fn in I.mjcf_spec_files(spec):
        if (cat, fn) in seen:
            continue
        seen.add((cat, fn))
        folder = spec.texturedir if cat == 'texture' else spec.meshdir
        p = I._mjcf_asset_path(fn, folder, main_dir)
        if p is not None:
            out.append(_item(cat, fn, p, cat != 'texture'))
        elif cat == 'texture':
            out.append(_item(cat, fn, None, False, 'Not found: the robot is drawn without this texture.'))
        else:
            out.append(_item(cat, fn, None, True, str(I._missing(
                'MJCF file', f'the {cat}', fn, path, 'pick the robot\'s complete folder (its meshes are usually in '
                'assets/ or meshes/ beside the .xml).'))))
    return out
