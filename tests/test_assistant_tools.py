"""The assistant's tool layer (assistant/tools.py, figures.py, session.py, page_guide.py) and the edits as data
(mcp/edits.py): every tool on a sample project, figures read off the studies (never computed), locked items left as
the user set them, one undo step per edit, the window's store and page context."""
import copy
import json
import os

import pytest

from dexvantage.assistant import figures as F
from dexvantage.assistant import tools as T
from dexvantage.assistant.session import CLAUDE, FileSession, SessionError, StoreSession
from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core.robot.samples import sample_project
from dexvantage.mcp import edits as E

SPEC_TOOLS = ('get_overview', 'get_robot', 'get_sensor', 'get_coverage', 'get_depth_validity', 'get_grasp_visibility',
              'get_teleop_overlap', 'get_cable_flex', 'get_bandwidth', 'get_sync', 'get_vla_tokens',
              'get_action_chunk', 'get_handeye_plan', 'get_audit', 'get_page_context', 'list_metrics',
              'evaluate_metrics', 'edit_project', 'simulate_changes', 'propose_plan', 'run_study', 'export_files',
              'show_in_app', 'undo_last_change')
WRIST = 'Wrist camera (sample placement)'
TP = 'Third-person camera (sample placement)'


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


@pytest.fixture
def s(so100):
    return FileSession(project=so100.snapshot())


@pytest.fixture(scope='module', autouse=True)
def _test_metric():
    """A metric of this test's own (the shared definitions in core/metric_defs.py may change)."""
    @M.metric('test_tools.coverage_ge2', 'Seen by two or more (test)', '%', 'higher', study='coverage')
    def _(doc):
        v = (doc.get('summary') or {}).get('seen_ge2')
        return None if v is None else 100.0 * v
    yield
    M.METRICS.pop('test_tools.coverage_ge2', None)


def call(s, name, **args):
    out = T.call(s, name, args)
    json.dumps(out)                                       # every result is JSON-able
    return out


# ======================================================================= the registry
def test_every_tool_of_the_spec_is_there_with_a_closed_schema():
    assert set(SPEC_TOOLS) <= set(T.TOOLS)
    for name, t in T.TOOLS.items():
        sch = t.input_schema
        assert sch['type'] == 'object' and sch.get('additionalProperties') is False, name
        assert t.description and t.title and callable(t.fn)
        assert set(sch.get('required', ())) <= set(sch.get('properties', {})), name
    assert T.TOOLS['edit_project'].changes and not T.TOOLS['get_coverage'].changes


def test_bad_input_is_a_plain_sentence(s):
    assert 'no parameter' in call(s, 'get_overview', bogus=1)['error']
    assert 'is missing' in call(s, 'get_sensor')['error']
    assert 'a boolean is expected' in call(s, 'get_coverage', full='yes')['error']
    assert 'There is no tool' in call(s, 'nope')['error']
    r = call(s, 'get_sensor', sensor='nobody')
    assert 'no sensor "nobody"' in r['error'] and 'Traceback' not in r['error']


# ======================================================================= reading
def test_overview_robot_and_sensor_read_the_project_and_the_library(s, so100):
    ov = call(s, 'get_overview')
    assert ov['project'] == so100.name and [r['name'] for r in ov['robots']] == [r.name for r in so100.robots]
    assert {x['name'] for x in ov['sensors']} == {x.name for x in so100.variant.sensors}
    assert any(x['id'] == 'coverage' for x in ov['studies'])
    rb = call(s, 'get_robot')
    j = next(j for j in so100.primary_robot.joints if j.get('type') == 'revolute')
    row = next(x for x in rb['joints'] if x['name'] == j['name'])
    assert row['lower_rad'] == pytest.approx(j['lower'], abs=1e-6) and row['upper_rad'] == pytest.approx(j['upper'],
                                                                                                         abs=1e-6)
    sen = call(s, 'get_sensor', sensor=WRIST)
    assert sen['mount']['parent'] == next(x for x in so100.variant.sensors if x.name == WRIST).mount.parent_frame


def test_sensor_with_a_library_model_cites_its_sources():
    from dexvantage.core import library as LB
    p = sample_project('ur5e')
    s = FileSession(project=p)
    x = next(x for x in p.variant.sensors if x.model_id)
    out = call(s, 'get_sensor', sensor=x.uid)
    e = LB.get(x.model_id)
    assert out['model']['id'] == e.id and out['model']['sources'][0]['url'] == e.sources[0]['url']
    assert out['model']['mass_g'] == e.data.get('mass_g')            # as the entry states it (None: not stated)


@pytest.mark.parametrize('tool,sid', [('get_coverage', 'coverage'), ('get_bandwidth', 'bandwidth'),
                                      ('get_sync', 'time_sync'), ('get_vla_tokens', 'vla_tokens'),
                                      ('get_cable_flex', 'cables'), ('get_depth_validity', 'depth_validity')])
def test_study_figures_are_the_studys_own(s, so100, tool, sid):
    out = call(s, tool)
    assert 'error' not in out, out
    doc = jobs.run_study(sid, so100.snapshot())
    assert out['study'] == sid and out['status'] == F.brief(sid, doc)['status']
    full = call(s, tool, full=True)
    assert 'result' in full and full['figure'] == out['figure']
    if sid == 'coverage':
        assert out['figures']['blind_pct'] == pytest.approx(100 * doc['summary']['blind'], abs=1e-3)
        for row in out['rows']:
            cam = next(c for c in doc['cameras'] if c['sensor_uid'] == row['sensor_uid'])
            assert row['seen_pct'] == pytest.approx(100 * cam['seen'], abs=1e-3)
    if sid == 'time_sync':
        assert out['figure']['value'] == pytest.approx(doc['summary']['worst_skew_s'], rel=1e-3)


def test_a_study_that_refuses_says_why(s):
    out = call(s, 'get_action_chunk')                      # SmolVLA's inference time is not stated by its source
    assert 'not stated' in out['error']
    assert 'dataset' in call(s, 'get_audit')['error'].lower()


def test_long_studies_need_a_plan_where_the_session_says_so(so100):
    s = FileSession(project=so100.snapshot(), allow_long=False)
    for tool in ('get_grasp_visibility', 'get_teleop_overlap', 'get_handeye_plan'):
        assert 'propose_plan' in call(s, tool)['error']
    assert 'propose_plan' in call(s, 'run_study', study='grasp_visibility')['error']
    assert 'error' not in call(s, 'run_study', study='coverage')


def test_run_study_lists_and_applies_settings_for_one_run_only(s):
    lst = call(s, 'run_study')
    assert any(x['id'] == 'coverage' and x['batch_flag'] == '--coverage' for x in lst['studies'])
    out = call(s, 'run_study', study='coverage', settings={'cube_n': 256})
    assert 'error' not in out and 'this run only' in out['note']
    assert 'coverage' not in (s.project().analysis.studies or {}) or \
        'cube_n' not in s.project().analysis.studies.get('coverage', {})
    assert 'no study' in call(s, 'run_study', study='nope')['error']


def test_metrics_list_and_evaluate(s, so100):
    ids = [m['id'] for m in call(s, 'list_metrics')['metrics']]
    assert 'test_tools.coverage_ge2' in ids
    out = call(s, 'evaluate_metrics', metrics=['test_tools.coverage_ge2', 'nope.metric'])['metrics']
    doc = jobs.run_study('coverage', so100.snapshot())
    assert out['test_tools.coverage_ge2']['value'] == pytest.approx(100 * doc['summary']['seen_ge2'], abs=1e-4)
    assert out['nope.metric']['status'] == 'unknown'


# ======================================================================= changing
def test_simulate_changes_nothing_and_edit_is_one_undoable_step(s):
    z0 = next(x for x in s.project().variant.sensors if x.name == WRIST).mount.pose.position[2]
    sim = call(s, 'simulate_changes', changes={f'sensor:{WRIST}.dz': 0.02}, metrics=['test_tools.coverage_ge2'])
    assert sim['would_change'] and 'before' in sim['metrics']['test_tools.coverage_ge2']
    assert next(x for x in s.project().variant.sensors if x.name == WRIST).mount.pose.position[2] == z0
    out = call(s, 'edit_project', changes={f'sensor:{WRIST}.dz': 0.02}, metrics=['test_tools.coverage_ge2'])
    assert out['undo_step'].startswith(CLAUDE) and len(s.undo_stack) == 1
    m = out['metrics']['test_tools.coverage_ge2']
    assert m['before'] == sim['metrics']['test_tools.coverage_ge2']['before']
    assert m['after'] == pytest.approx(sim['metrics']['test_tools.coverage_ge2']['after'])
    z1 = next(x for x in s.project().variant.sensors if x.name == WRIST).mount.pose.position[2]
    assert z1 == pytest.approx(z0 + 0.02)
    assert call(s, 'undo_last_change')['result'].startswith('Undid')
    assert next(x for x in s.project().variant.sensors if x.name == WRIST).mount.pose.position[2] == pytest.approx(z0)
    assert call(s, 'undo_last_change')['result'] == 'Nothing to undo.'


def test_requirements_are_compared_before_and_after(so100):
    from dexvantage.core.project import Requirement
    p = so100.snapshot()
    p.requirements.append(Requirement(name='Two views', metric='test_tools.coverage_ge2', op='>=', threshold=40.0,
                                      unit='%', study='coverage'))
    s = FileSession(project=p)
    out = call(s, 'edit_project', changes={f'sensor:{TP}.enabled': False})
    req = out['requirements'][0]
    assert req['pass_before'] is True and req['after'] < req['before'] and req['default_threshold'] is True


def test_locked_items_stay_the_users(so100):
    p = so100.snapshot()
    wr = next(x for x in p.variant.sensors if x.name == WRIST)
    wr.locked = True
    s = FileSession(project=p)
    i = p.variant.sensors.index(wr)
    vi = p.variants.index(p.variant)
    for kw in ({'changes': {f'sensor:{WRIST}.x': 0.1}},
               {'ops': [{'op': 'remove_sensor', 'sensor': WRIST}]},
               {'changes': {f'json:/variants/{vi}/sensors/{i}/locked': False}},
               {'changes': {f'json:/variants/{vi}/sensors/{i}/mount/pose/position/0': 0.5}}):
        out = call(s, 'edit_project', **kw)
        assert 'locked' in out['error'], (kw, out)
    assert not s.undo_stack
    out = call(s, 'edit_project', changes={f'json:/variants/{vi}/sensors/{1 - i}/locked': True})   # locking is fine
    assert 'error' not in out and s.project().variant.sensors[1 - i].locked


@pytest.mark.parametrize('kw,why', [
    ({'changes': {f'sensor:{WRIST}.x': 'far'}}, 'a number'),
    ({'changes': {f'sensor:{WRIST}.model': 'no-such-device'}}, 'not in the library'),
    ({'changes': {f'sensor:{WRIST}.parent': 'nowhere'}}, 'neither a static frame'),
    ({'changes': {f'sensor:{WRIST}.stream': [640]}}, '[width, height]'),
    ({'changes': {'setting:nope.x': 1}}, 'no study'),
    ({'changes': {'json:/log': []}}, 'not edited from here'),
    ({'ops': [{'op': 'fly'}]}, 'one of'),
    ({'ops': [{'op': 'activate_variant', 'variant': 'Nope'}]}, 'no variant'),
    ({}, 'Give changes'),
])
def test_bad_changes_change_nothing(so100, kw, why):
    with pytest.raises(E.EditError) as ei:
        E.apply(so100, **kw)
    assert why in str(ei.value)


def test_structural_operations(so100):
    from dexvantage.core import library as LB
    mid = next(e.id for e in LB.entries('depth'))
    link = so100.primary_robot.links[-1]['name']
    ed = E.apply(so100, ops=[{'op': 'add_variant', 'name': 'Claude try'},
                             {'op': 'add_sensor', 'model_id': mid, 'name': 'Extra', 'parent': link, 'z': 0.05},
                             {'op': 'remove_sensor', 'sensor': TP}],
                 changes={'sensor:Extra.pitch': 30.0})
    p = ed.project
    assert len(p.variants) == len(so100.variants) + 1 and p.variant.name == 'Claude try'
    x = next(x for x in p.variant.sensors if x.name == 'Extra')
    assert x.kind == 'depth' and x.mount.robot_uid == so100.primary_robot.uid and x.mount.parent_frame == link
    assert TP not in [y.name for y in p.variant.sensors] and TP in [y.name for y in so100.variant.sensors]
    assert ed.keys == ['variants'] and len(ed.lines) == 4
    ed2 = E.apply(so100, changes={f'sensor:{WRIST}.model': mid})
    assert next(y for y in ed2.project.variant.sensors if y.name == WRIST).kind == 'depth'
    assert ed2.keys == [f'variant:{so100.active_uid}']


def test_file_session_saves_each_change_with_a_backup(tmp_path, so100):
    path = str(tmp_path / 'so100.dexv')
    so100.snapshot().save(path)
    s = FileSession(path=path)
    assert call(s, 'edit_project', changes={f'sensor:{WRIST}.enabled': False})['undo_step']
    from dexvantage.core.project import Project
    assert not next(x for x in Project.load(path).variant.sensors if x.name == WRIST).enabled
    assert next(x for x in Project.load(path + '.bak').variant.sensors if x.name == WRIST).enabled
    call(s, 'undo_last_change')
    assert next(x for x in Project.load(path).variant.sensors if x.name == WRIST).enabled
    assert s.redo().startswith('Redid')
    os.utime(path, (1, 1))                                 # someone else wrote the file
    assert 'changed on disk' in call(s, 'edit_project', changes={f'sensor:{WRIST}.dz': 0.01})['error']
    with pytest.raises(SessionError):
        FileSession(path=str(tmp_path / 'missing.dexv'))


def test_propose_plan_checks_each_step(s):
    out = call(s, 'propose_plan', title='Check the grasps', rationale='r',
               steps=[{'kind': 'study', 'study': 'grasp_visibility'},
                      {'kind': 'edit', 'changes': {f'sensor:{WRIST}.dz': 0.01}}])
    plan = out['plan']
    assert plan['status'] == 'proposed' and plan['steps'][0]['long'] and plan['steps'][1]['would_change']
    assert s.plans[-1] is plan or s.plans[-1] == plan
    from dexvantage.core import provenance as PV
    assert PV.plan_field(plan)['plan'] == plan['plan']
    assert 'no study' in call(s, 'propose_plan', title='x', steps=[{'kind': 'study', 'study': 'nope'}])['error']
    assert 'a number' in call(s, 'propose_plan', title='x', steps=[{'kind': 'edit', 'changes': {
        f'sensor:{WRIST}.x': 'a'}}])['error']
    assert not s.undo_stack                                 # a plan changes nothing


def test_export_files_lists_and_writes(s, tmp_path):
    from dexvantage.core import exporters as X

    @X.exporter('test_tools_json', 'A test format')
    def _(project, out_dir, options, ctx):
        with open(os.path.join(out_dir, 'x.json'), 'w', encoding='utf-8') as fh:
            json.dump({'name': project.name}, fh)
        return {'files': ['x.json'], 'notes': []}
    try:
        assert any(f['id'] == 'test_tools_json' for f in call(s, 'export_files')['formats'])
        out = call(s, 'export_files', formats=['test_tools_json'], folder=str(tmp_path / 'out'))
        assert out['formats']['test_tools_json']['status'] == 'ok' and 'x.json' in out['files']
        assert os.path.isfile(tmp_path / 'out' / 'export_manifest.json')
        assert 'no export format' in call(s, 'export_files', formats=['nope'], folder=str(tmp_path))['error']
    finally:
        X.EXPORTERS.pop('test_tools_json', None)


def test_show_in_app_and_page_context_need_the_window(s):
    assert 'none is open' in call(s, 'show_in_app', action='frame_all')['error']
    assert call(s, 'get_page_context')['page'] is None
    assert 'one of' in call(s, 'show_in_app', action='dance')['error']


def test_compact_leaves_out_paths_secrets_and_arrays():
    import numpy as np
    out = F.compact({'path': '/x/y', 'api_key': 'k', 'note': 'runs/one/file.txt', 'a': np.zeros(3),
                     'v': 1.23456789, 'small': 0.000123456, 'l': list(range(20))})
    assert set(out) == {'a', 'v', 'small', 'l'} and out['a'] == 'array (3,)'
    assert out['v'] == 1.2346 and out['small'] == 0.0001235 and out['l'][-1] == '… 14 more'


# ======================================================================= the window
def test_store_session_edits_as_one_undo_step_and_undoes_only_its_own(win, so100):
    win.store.set_project(so100.snapshot())
    s = StoreSession(win.ctx)
    n = win.store.undo.count()
    out = call(s, 'edit_project', changes={f'sensor:{WRIST}.dz': 0.01})
    assert win.store.undo.count() == n + 1 and win.store.undo.text(n).startswith(CLAUDE), out
    assert call(s, 'undo_last_change')['result'].startswith('Undid')
    assert win.store.undo.index() == n
    call(s, 'edit_project', changes={f'sensor:{WRIST}.dz': 0.01})
    win.store.edit('meta', lambda p: setattr(p, 'notes', 'mine'), 'User note')
    assert "user's" in call(s, 'undo_last_change')['result'] and win.store.project.notes == 'mine'


def test_page_context_reads_the_page_as_shown(win, so100, app):
    from tests.conftest import pump
    win.store.set_project(so100.snapshot())
    win.ctx.analysis.wait(30)
    win.open_page('bandwidth')
    pg = win.pages['bandwidth']
    pg.run()
    win.ctx.analysis.wait(30)
    pump(app, 0.1)
    s = StoreSession(win.ctx)
    pc = call(s, 'get_page_context')
    assert pc['page'] == 'bandwidth' and pc['study'] == 'bandwidth' and pc['status'] == 'done', pc
    assert pc['shown']['headline'] == pg.head.state and pc['figures']['status'] == pg.doc['headline']['status']
    assert pc['ms'] < 100
    s2 = StoreSession(win.ctx)
    assert call(s2, 'show_in_app', action='open_page', page='coverage')['done'] == 'open_page'
    assert win.current_page() is win.pages['coverage']
    assert 'no page' in call(s2, 'show_in_app', action='open_page', page='nope')['error']
    assert call(s2, 'show_in_app', action='select', sensor=WRIST)['done'] == 'select'
    assert win.store.sel[0] == 'sensor'


def test_store_session_uses_the_windows_own_result(win, so100):
    win.store.set_project(so100.snapshot())
    win.ctx.analysis.run('bandwidth')
    win.ctx.analysis.wait(30)
    s = StoreSession(win.ctx)
    doc = s.last_result('bandwidth')
    assert doc is not None and doc['revision'] == win.store.revision
    out = call(s, 'get_bandwidth')
    assert out['status'] == doc['headline']['status']
    _ = copy.deepcopy(out)
