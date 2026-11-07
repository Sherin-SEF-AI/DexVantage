"""The Git-friendly project form: an unpacked folder next to the .dexv zip, and the stable text git diff shows.

    save_unpacked(project, 'cell.dexv.d')     one pretty-printed JSON file per section (sorted keys, two-space indent,
                                              UTF-8, a final newline), mesh arrays as .npy files, the stored model
                                              files as files
    load_unpacked('cell.dexv.d') -> Project
    unpack('cell.dexv', 'cell.dexv.d')        the zip as a folder;  pack('cell.dexv.d', 'cell.dexv') the reverse
    textconv(path) -> str                     a stable text of a .dexv file or folder for `git diff` (docs/guide/
                                              versioning.md: a textconv driver)
    is_unpacked(path) -> bool

The folder:

    cell.dexv.d/
        project.json        {"format": "dexvantage.project", "version": N, "layout": "dexvantage.unpacked/1",
                             "name", "notes", "created", "active_uid", "sections": {section: file},
                             "meshes": {key: file}, "sources": [names], "thumb": "thumb.png" | null}
        robots.json  workspace.json  tasks.json  variants.json  requirements.json  analysis.json  rig.json
        cables.json  policies.json  calibration.json  data.json  log.json      (one per section of Project.to_dict)
        meshes/<key>.npy    numpy's own format (allow_pickle off), one array per file
        sources/<name>      the original model files, under their stored names (which may hold folders)
        thumb.png

Project.load and Project.save (core/project.py) read and write this form for a folder path or a path ending in
.dexv.d, so the window, batch mode and the API open it as they open a .dexv file.

Byte stability: the same project always gives the same bytes in every file (sorted keys, Python's shortest float
repr, numpy's deterministic .npy header); saving writes a file only when its bytes change, so an unchanged section
keeps its file (and git sees nothing). The save time (saved_at) is not written: a time that changes on every save
would make every save a change; the folder's history (git log) says when it was saved. Files of a section, mesh or
model file that the project no longer has are removed; other files in the folder (a .gitattributes, a README) are left
alone.

Known limits: mesh arrays are binary .npy files (git stores them whole; the textconv text shows their SHA-256, dtype
and shape, not their values); a stored model file name that would leave the folder ("..", an absolute path) is
refused on save and on load; the folder is written file by file (each atomically), not as one atomic step, so a crash
mid-save can leave a mix of old and new section files (the zip form is atomic).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re

import numpy as np

LAYOUT = 'dexvantage.unpacked/1'
SUFFIX = '.dexv.d'
HEADER_KEYS = ('format', 'version', 'name', 'notes', 'created', 'active_uid')
SKIP_KEYS = ('saved_at', 'meshes', 'sources')          # meshes and sources are files; saved_at is not written
_SAFE = re.compile(r'[^A-Za-z0-9._-]+')


def dumps(obj) -> str:
    """The one JSON text of the unpacked form: sorted keys, two-space indent, UTF-8 kept, a final newline."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def is_unpacked(path) -> bool:
    return isinstance(path, str) and (path.rstrip('/\\').endswith(SUFFIX) or os.path.isdir(path))


def _err(text):
    from .project import ProjectFormatError
    return ProjectFormatError(text)


def _safe_rel(name: str) -> bool:
    if not isinstance(name, str) or not name or '\\' in name or name.startswith('/') or '\x00' in name:
        return False
    parts = name.split('/')
    return all(p not in ('', '.', '..') for p in parts) and not re.match(r'^[A-Za-z]:', name)


def _mesh_files(keys) -> dict:
    """{mesh key: file name under meshes/}: the key with characters other than letters, digits, '.', '_' and '-'
    replaced, and a short hash of the key added when that would collide or change it."""
    out, used = {}, set()
    for k in sorted(keys):
        stem = _SAFE.sub('_', k).strip('._') or 'mesh'
        if stem != k or stem.lower() in used:
            stem = f'{stem}-{hashlib.sha256(k.encode("utf-8")).hexdigest()[:8]}'
        used.add(stem.lower())
        out[k] = f'meshes/{stem}.npy'
    return out


def _npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return buf.getvalue()


def files_of(project) -> dict:
    """{relative path: bytes} of the unpacked form of the project (project.json first)."""
    doc = project.to_dict()
    sections = {}
    files: dict = {}
    for k, v in doc.items():
        if k in HEADER_KEYS or k in SKIP_KEYS:
            continue
        sections[k] = f'{k}.json'
        files[f'{k}.json'] = dumps(v).encode('utf-8')
    meshes = _mesh_files(project.meshes)
    for k, rel in meshes.items():
        files[rel] = _npy(project.meshes[k])
    for name, data in sorted(project.sources.items()):
        if not _safe_rel(name):
            raise _err(f'The stored model file "{name}" has a name that would leave the project folder, so the project '
                       'cannot be written as a folder. Save it as a .dexv file.')
        files[f'sources/{name}'] = bytes(data)
    if project.thumb:
        files['thumb.png'] = bytes(project.thumb)
    head = {k: doc[k] for k in HEADER_KEYS if k in doc}
    head.update(layout=LAYOUT, sections=sections, meshes=meshes, sources=sorted(project.sources),
                thumb='thumb.png' if project.thumb else None)
    return {'project.json': dumps(head).encode('utf-8'), **files}


def _write_if_changed(path: str, data: bytes) -> bool:
    try:
        with open(path, 'rb') as fh:
            if fh.read() == data:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def _owned(folder: str) -> set:
    """The files in folder this form wrote before (those it may remove): the sections project.json lists, meshes/
    and sources/ files and thumb.png."""
    out = set()
    try:
        with open(os.path.join(folder, 'project.json'), encoding='utf-8') as fh:
            head = json.load(fh)
        out |= {str(v) for v in (head.get('sections') or {}).values() if _safe_rel(str(v))}
    except (OSError, ValueError, AttributeError):
        pass
    for sub in ('meshes', 'sources'):
        root = os.path.join(folder, sub)
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                out.add(os.path.relpath(os.path.join(dp, f), folder).replace(os.sep, '/'))
    if os.path.isfile(os.path.join(folder, 'thumb.png')):
        out.add('thumb.png')
    return out


def save_unpacked(project, folder: str) -> dict:
    """Write the project as an unpacked folder; {'written': [rel], 'removed': [rel], 'unchanged': n}."""
    files = files_of(project)
    if os.path.exists(folder) and not os.path.isdir(folder):
        raise _err(f'{os.path.basename(folder)} is a file, not a folder: choose another name for the unpacked '
                   'project.')
    os.makedirs(folder, exist_ok=True)
    stale = _owned(folder) - set(files)
    written = [rel for rel, data in files.items() if _write_if_changed(os.path.join(folder, rel), data)]
    removed = []
    for rel in sorted(stale):
        try:
            os.remove(os.path.join(folder, rel))
            removed.append(rel)
        except OSError:
            pass
    for sub in ('meshes', 'sources'):                  # empty folders left by removed files
        root = os.path.join(folder, sub)
        for dp, _dn, _fn in sorted(os.walk(root), key=lambda t: -len(t[0])):
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
            except OSError:
                pass
    return {'written': written, 'removed': removed, 'unchanged': len(files) - len(written)}


def load_unpacked(folder: str):
    """The project an unpacked folder holds (ProjectFormatError, in plain words, when it cannot be read)."""
    from .project import Project
    base = os.path.basename(folder.rstrip('/\\')) or folder
    head_path = os.path.join(folder, 'project.json')
    if not os.path.isdir(folder):
        raise _err(f'There is no project folder at {base}.')
    if not os.path.isfile(head_path):
        raise _err(f'{base} has no project.json inside, so it is not an unpacked DexVantage project.')

    def read_json(rel):
        try:
            with open(os.path.join(folder, rel), encoding='utf-8') as fh:
                return json.load(fh)
        except FileNotFoundError:
            raise _err(f'{base} lists {rel}, but the file is not there. Restore it (for example with git checkout) '
                       'or save the project again.') from None
        except (UnicodeDecodeError, ValueError) as e:
            raise _err(f'{rel} in {base} is damaged: {e}.') from None
    head = read_json('project.json')
    if not isinstance(head, dict):
        raise _err(f'project.json in {base} holds no project (its JSON is not an object).')
    if head.get('layout') not in (None, LAYOUT):
        raise _err(f'{base} is laid out as "{head.get("layout")}", which this build does not read (it reads '
                   f'{LAYOUT}).')
    doc = {k: head[k] for k in HEADER_KEYS if k in head}
    for key, rel in sorted((head.get('sections') or {}).items()):
        if not _safe_rel(str(rel)):
            raise _err(f'project.json in {base} names a section file outside the folder ({rel}).')
        doc[key] = read_json(rel)
    p = Project.from_dict(doc)
    meshes = {}
    for key, rel in (head.get('meshes') or {}).items():
        if not _safe_rel(str(rel)):
            raise _err(f'project.json in {base} names a mesh file outside the folder ({rel}).')
        try:
            meshes[str(key)] = np.load(os.path.join(folder, rel), allow_pickle=False)
        except FileNotFoundError:
            raise _err(f'{base} lists the mesh file {rel}, but it is not there.') from None
        except ValueError as e:
            raise _err(f'The mesh file {rel} in {base} cannot be read: {e}.') from None
    p.meshes = meshes
    sources = {}
    for name in head.get('sources') or []:
        if not _safe_rel(str(name)):
            raise _err(f'project.json in {base} names a model file outside the folder ({name}).')
        try:
            with open(os.path.join(folder, 'sources', *str(name).split('/')), 'rb') as fh:
                sources[str(name)] = fh.read()
        except FileNotFoundError:
            raise _err(f'{base} lists the model file sources/{name}, but it is not there.') from None
    p.sources = sources
    thumb = head.get('thumb')
    p.thumb = b''
    if thumb and _safe_rel(str(thumb)) and os.path.isfile(os.path.join(folder, thumb)):
        with open(os.path.join(folder, thumb), 'rb') as fh:
            p.thumb = fh.read()
    p.path = folder
    return p


def unpack(src: str, folder: str) -> dict:
    """A .dexv file (or folder) written as an unpacked folder."""
    from .project import Project
    return save_unpacked(Project.load(src), folder)


def pack(folder: str, dest: str):
    """An unpacked folder (or .dexv file) written as a .dexv zip; the project's saved_at is set as Project.save does."""
    from .project import Project
    p = Project.load(folder)
    if is_unpacked(dest):
        raise _err(f'{os.path.basename(dest)} names a folder: pack writes a .dexv file.')
    p.save(dest)
    return p


def textconv(path: str) -> str:
    """A stable text of a .dexv file or an unpacked folder, for git diff: each section's JSON as the unpacked form
    writes it, then one line per mesh array and stored file with its SHA-256 (their bytes are not text)."""
    from .project import Project
    p = Project.load(path)
    files = files_of(p)
    out = []
    for rel in ['project.json'] + sorted(r for r in files if r.endswith('.json') and r != 'project.json'
                                        and '/' not in r):
        out.append(f'### {rel}\n')
        out.append(files[rel].decode('utf-8'))
    for rel in sorted(r for r in files if r.startswith('meshes/')):
        a = np.load(io.BytesIO(files[rel]), allow_pickle=False)
        out.append(f'### {rel}  {a.dtype.str} {list(a.shape)}  sha256 {hashlib.sha256(files[rel]).hexdigest()}\n')
    for rel in sorted(r for r in files if r.startswith('sources/') or r == 'thumb.png'):
        out.append(f'### {rel}  {len(files[rel])} bytes  sha256 {hashlib.sha256(files[rel]).hexdigest()}\n')
    return ''.join(out)
