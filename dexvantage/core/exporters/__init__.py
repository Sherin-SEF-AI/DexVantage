"""Exporters (spec §9.2): one module per target, each registering with @exporter; `--export OUT --formats a,b` and
the Export dialog run them through export().

    @exporter('opencv_yaml', 'OpenCV YAML, Kalibr camchain and JSON intrinsics', needs_robot=False)
    def _(project, out_dir, options, ctx) -> dict:
        ...write files under out_dir...
        return {'files': ['cameras/wrist.yaml', ...],          # relative to out_dir
                'notes': ['plain sentences: what is approximated, what the target cannot represent']}

    export(project, out_dir, ['opencv_yaml', 'mjcf'], options={'mjcf': {...}}) -> manifest dict
      manifest = {'schema': 'dexvantage.export/1', 'formats': {fid: {'title', 'files': [...], 'notes': [...],
                  'status': 'ok' | 'refused', 'reason'?}}, 'files': {rel: sha256}}
      formats may use the aliases in ALIASES ('usd' for 'isaac'); the manifest uses the registered ids;
      written to out_dir/export_manifest.json with the provenance block; every JSON file written is stamped too,
      except those an exporter lists under 'unstamped' (a target parser that refuses unknown keys, such as LeRobot's
      draccus train config): those keep only their sha256 in the manifest, and the format row lists them.

A format that cannot run on the project (no robot, no policy, an optional package missing) raises StudyError with
a plain sentence; export() records it as 'refused' and carries on with the other formats. Each exporter has a
round-trip format check in validation/checks_formats.py (through the target tool's own parser where installable).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pkgutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..jobs import Ctx, StudyError

if TYPE_CHECKING:                                    # names for the type hints only
    from ..project import Project


@dataclass
class ExporterInfo:
    id: str
    title: str
    fn: Callable
    needs_robot: bool = False
    description: str = ''


EXPORTERS: dict = {}
_LOADED = False
# the specification's names (§11 `--formats mjcf,usd,...`) for formats registered under another id
ALIASES = {'usd': 'isaac'}


def resolve(formats: str | list | tuple) -> list:
    """Format ids with the aliases (ALIASES) replaced by the registered id, in order, without repeats."""
    out = []
    for f in formats:
        f = ALIASES.get(f, f)
        if f not in out:
            out.append(f)
    return out


def exporter(fid: str, title: str, needs_robot: bool = False, description: str = '') -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        EXPORTERS[fid] = ExporterInfo(fid, title, fn, needs_robot, description)
        return fn
    return deco


def load_exporters() -> None:
    global _LOADED
    if _LOADED:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for m in pkgutil.iter_modules([here]):
        if not m.name.startswith('_'):
            importlib.import_module(f'{__package__}.{m.name}')
    _LOADED = True


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export(project: Project, out_dir: str, formats: str | list | tuple, options: dict | None = None,
           ctx: Ctx | None = None) -> dict:
    from .. import provenance as PV
    load_exporters()
    ctx = ctx or Ctx()
    options = {ALIASES.get(k, k): v for k, v in (options or {}).items()}
    formats = resolve(formats)
    unknown = [f for f in formats if f not in EXPORTERS]
    if unknown:
        raise StudyError(f'DexVantage has no export format {", ".join(unknown)} (it has: '
                         f'{", ".join(sorted(EXPORTERS))}).')
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise StudyError(f'Could not create the folder {os.path.basename(out_dir) or out_dir}: '
                         f'{e.strerror or e}.') from None
    man = {'schema': 'dexvantage.export/1', 'formats': {}, 'files': {}}
    for k, fid in enumerate(formats):
        info = EXPORTERS[fid]
        row = {'title': info.title, 'files': [], 'notes': []}
        try:
            if info.needs_robot and not project.robots:
                raise StudyError(f'{info.title} needs a robot: import one first or open a sample.')
            res = info.fn(project, out_dir, dict(options.get(fid) or {}), ctx) or {}
            row.update(files=list(res.get('files', [])), notes=list(res.get('notes', [])), status='ok')
            if res.get('unstamped'):
                row['unstamped'] = [r for r in res['unstamped'] if r in row['files']]
        except StudyError as e:
            row.update(status='refused', reason=str(e))
        man['formats'][fid] = row
        ctx.report((k + 1) / max(1, len(formats)), info.title)
    for row in man['formats'].values():
        for rel in row['files']:
            p = os.path.join(out_dir, rel)
            if os.path.isfile(p):
                man['files'][rel] = _sha256(p)
    keep = {r for row in man['formats'].values() for r in row.get('unstamped', ())}
    PV.stamp_paths([os.path.join(out_dir, r) for r in man['files'] if r.lower().endswith('.json') and r not in keep],
                   project, folder=out_dir)
    for rel in list(man['files']):
        if rel.lower().endswith('.json'):
            man['files'][rel] = _sha256(os.path.join(out_dir, rel))
    path = os.path.join(out_dir, 'export_manifest.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(PV.stamp(man, project, output={'option': 'export', 'name': 'export_manifest.json',
                                                 'in_folder': True}), fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    return man
