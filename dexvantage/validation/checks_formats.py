"""Checks of the project file: a populated project written to a .dexv and read back with Python's zipfile and json
modules alone (not DexVantage's reader) must validate against the published JSON Schema with an independent validator
(the jsonschema package), and DexVantage's own reader must give back the same content hash."""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np

from .suite import Outcome, check

SCHEMA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources',
                           'project.schema.json')


def reference_project():
    """A project with every container filled: two robots, sensors on a link and on a static frame, volumes, objects
    of each material tag, a requirement, a mesh array and a source file."""
    from ..core import project as P
    from ..core.frames import Pose, quat_from_rpy
    p = P.blank_project('Format reference')
    a = P.Robot(name='Left arm', primary=True)
    a.joints = [{'name': 'j1', 'type': 'revolute', 'parent': 'base', 'child': 'l1', 'axis': [0, 0, 1],
                 'lower': -2.9, 'upper': 2.9}]
    a.source = P.RobotSource('urdf', 'left.urdf', 'ab' * 32, 'BSD-3-Clause', 'reference')
    b = P.Robot(name='Right arm', base=Pose([0, -0.6, 0], quat_from_rpy(0, 0, np.pi)))
    p.robots += [a, b]
    v = p.variant
    v.sensors.append(P.Sensor(name='Wrist', kind='depth', mount=P.Mount('l1', a.uid, Pose([0.05, 0, 0.02]))))
    v.sensors.append(P.Sensor(name='Overhead', kind='camera', slot=1,
                              mount=P.Mount('cage', '', Pose([0.4, 0, 1.2], quat_from_rpy(0, np.pi / 2, 0)))))
    p.workspace.objects += [P.TaskObject(name=f'{m} cube', materials=[m]) for m in P.MATERIAL_TAGS]
    p.workspace.volumes.append(P.Volume(name='Handover', kind='handover', refine=True))
    p.requirements.append(P.Requirement(name='Contact visibility', metric='grasp_visibility.score', op='>=',
                                        threshold=0.95, study='grasp_visibility'))
    p.meshes['l1/visual'] = np.arange(36, dtype=np.float32).reshape(-1, 3)
    p.sources['left.urdf'] = b'<robot name="left"/>'
    return p


def _written(ctx):
    if 'formats.file' in ctx.cache:
        return ctx.cache['formats.file']
    p = reference_project()
    path = os.path.join(ctx.tmp(), 'reference.dexv')
    p.save(path)
    ctx.cache['formats.file'] = (p, path)
    return p, path


@check('formats.schema', 'formats', 'Project file against its JSON Schema',
       'number of schema violations in project.json of a fully populated project, read with zipfile and json only',
       'the published JSON Schema (resources/project.schema.json, also docs/project.schema.json) through the '
       'jsonschema package\'s Draft 2020-12 validator', 0, 'violations', refs=('json_schema', 'zip_appnote'))
def _schema(ctx):
    import jsonschema
    _, path = _written(ctx)
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read('project.json').decode('utf-8'))
        names = z.namelist()
    with open(SCHEMA_FILE, encoding='utf-8') as fh:
        schema = json.load(fh)
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    return Outcome(len(errors), {'members': sorted(names), 'first_errors': [e.message for e in errors[:5]]})


@check('formats.round_trip', 'formats', 'Project file round trip',
       'number of content sections whose hash differs after save and load, plus array or source files not read back '
       'bit for bit', 'the project in memory before saving (core.provenance content hash per section)', 0,
       'differences', refs=('zip_appnote',))
def _round_trip(ctx):
    from ..core import provenance as PV
    from ..core.project import Project
    p, path = _written(ctx)
    q = Project.load(path)
    a, b = PV.project_digest(p), PV.project_digest(q)
    diff = [k for k in a['sections'] if a['sections'][k] != b['sections'].get(k)]
    arrays = [k for k in p.meshes if k not in q.meshes or q.meshes[k].dtype != p.meshes[k].dtype
              or not np.array_equal(q.meshes[k], p.meshes[k])]
    files = [k for k in p.sources if q.sources.get(k) != p.sources[k]]
    return Outcome(len(diff) + len(arrays) + len(files), {'sections_differ': diff, 'arrays_differ': arrays,
                                                          'sources_differ': files, 'sha256': b['sha256']})
