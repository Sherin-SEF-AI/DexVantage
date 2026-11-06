"""Cable flex across joints: the cable model (core/cables.py), the study and harness sheet (core/studies/cables.py),
the cable library entries, the project section, the page (ui/pages/cables.py), the batch flag and the checks."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import cables as CB
from dexvantage.core import jobs
from dexvantage.core.studies import cables as CS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def ur5e():
    from dexvantage.core.robot.samples import sample_project
    return sample_project('ur5e')


@pytest.fixture(scope='module')
def ur5e_doc(ur5e):
    return jobs.run_study('cables', ur5e)


@pytest.fixture(scope='module')
def ur5e_model(ur5e):
    from dexvantage.core.robot.model import get_model
    return get_model(ur5e, ur5e.primary_robot.uid)


def _cable(project, atts, **kw):
    c = CB.new_cable('Test cable', project.variant.sensors[0].uid)
    c['attachments'] = atts
    c.update(kw)
    return c


def _wrist_cable(project, model, **kw):
    """Wrist D405 → wrist_2_link → wrist_1_link: two runs, each across one joint."""
    s = project.variant.sensors[0]
    atts, _n = CB.suggest_attachments(model, s)
    return _cable(project, atts[:3], **kw)


# ---------------------------------------------------------------------- records
def test_clean_cable_is_tolerant():
    c = CB.clean_cable({'type': 'nonsense', 'diameter_m': 'x', 'length_m': -1, 'min_bend_radius_m': 0.03,
                        'attachments': [{'link': 'a', 'point': [1, 'b']}, 'junk'], 'connectors': 'FAKRA'})
    assert c['type'] == 'other' and c['diameter_m'] is None and c['length_m'] is None
    assert c['min_bend_radius_m'] == {'value': 0.03, 'source': 'entered by you', 'kind': 'unspecified'}
    assert c['attachments'] == [{'robot_uid': '', 'link': 'a', 'point': [0.0, 0.0, 0.0], 'length_to_next_m': None}]
    assert c['connectors'] == ['FAKRA'] and len(c['uid']) == 10
    assert CB.clean_cable(c) == c                                   # idempotent
    assert CB.clean_bend({'value': float('nan')}) is None


def test_library_cables_from_datasheets():
    from dexvantage.core import library as L
    L.reload()
    assert L.problems() == []
    es = {e.id: e for e in L.entries('cable')}
    assert set(es) == {'basler-cable-gmsl-fakra-z-1x-f-f', 'va-imaging-va-cable-usb3-hf',
                       'newnex-high-flex-active-usb3'}
    for e in es.values():
        assert L.validate(e.data) == []
        assert all(s['url'].startswith('https://') and s['read'] for s in e.data['sources'])
    b = es['basler-cable-gmsl-fakra-z-1x-f-f'].data['cable']
    assert (b['diameter_mm'], b['min_bend_static_mm'], b['min_bend_dynamic_mm']) == (3.45, 13.8, 34.5)
    v = es['va-imaging-va-cable-usb3-hf'].data['cable']
    assert (v['diameter_mm'], v['min_bend_static_mm'], v['min_bend_dynamic_mm']) == (4.8, 20.0, 50.0)
    n = es['newnex-high-flex-active-usb3'].data['cable']
    assert n['min_bend_static_mm'] is None and n['min_bend_dynamic_mm'] == 75.0      # not stated stays null
    # a record from the library takes the moving figure, with its source
    f = CB.from_library(es['basler-cable-gmsl-fakra-z-1x-f-f'])
    assert f['min_bend_radius_m']['value'] == pytest.approx(0.0345) and f['min_bend_radius_m']['kind'] == 'dynamic'
    assert 'docs.baslerweb.com' in f['min_bend_radius_m']['source'] or 'Basler' in f['min_bend_radius_m']['source']
    assert '2026-11-06' in f['min_bend_radius_m']['source']
    assert f['diameter_m'] == pytest.approx(0.00345) and f['type'] == 'gmsl'
    # a static-only figure is labelled static
    static = CB.from_library({'id': 'x', 'maker': 'M', 'name': 'N', 'sources': [{'url': 'u', 'read': '2026-01-01'}],
                              'cable': {'min_bend_static_mm': 10}})
    assert static['min_bend_radius_m']['kind'] == 'static' and 'fixed' in static['min_bend_radius_m']['source']


# ---------------------------------------------------------------------- the routed arc model
def test_wrap_length_closed_forms():
    r = 0.08
    th = np.linspace(-6, 6, 101)
    w = CB.wrap_length(r, r, 0.0, th, r)
    assert np.allclose(w['L'], r * np.abs(th), atol=1e-14)                     # on the cylinder: the circle arc
    w = CB.wrap_length(r, r, 0.03, th, r)
    assert np.allclose(w['L'], np.hypot(r * np.abs(th), 0.03), atol=1e-14)      # the helix
    # unequal radii: straight chord until the line touches the cylinder, then tangents + arc; continuous
    da, db, rho = 0.1, 0.07, 0.05
    th = np.linspace(0, 2 * np.pi, 20001)
    w = CB.wrap_length(da, db, 0.0, th, rho)
    t0 = math.acos(rho / da) + math.acos(rho / db)
    assert not w['wraps'][th < t0 - 1e-6].any() and w['wraps'][th > t0 + 1e-6].all()
    assert np.max(np.abs(np.diff(w['L']))) < 1e-4                               # no jump at the tangency
    chord = np.sqrt(da ** 2 + db ** 2 - 2 * da * db * np.cos(th))
    assert np.allclose(w['L'][th < t0], chord[th < t0])
    assert np.all(w['bend_m'][w['wraps']] == pytest.approx(rho))
    # the polyline follows the same route and has the same length
    A = np.array([[da, 0.0, 0.0]])
    for t in (0.5, 2.5, 5.0, -4.0):
        B = np.array([[db * math.cos(t), db * math.sin(t), 0.02]])
        cyl = CB.cylindrical(A, B, np.zeros((1, 3)), np.array([[0, 0, 1.0]]))
        P = CB.wrap_polyline(cyl, np.array([t]), rho, 401)
        assert np.allclose(P[0, 0], A[0]) and np.allclose(P[0, -1], B[0], atol=1e-12)
        L = CB.wrap_length(da, db, 0.02, t, rho)['L']
        assert CB.poly_length(P)[0] == pytest.approx(float(L), rel=1e-4)


def test_crossed_joints_and_sides(ur5e_model):
    m = ur5e_model
    assert [(m.joint_names[j], s) for j, s in CB.crossed_joints(m, 'wrist_3_link', 'wrist_2_link')] == \
        [('wrist_3_joint', -1)]
    assert [(m.joint_names[j], s) for j, s in CB.crossed_joints(m, 'wrist_2_link', 'wrist_3_link')] == \
        [('wrist_3_joint', 1)]
    names = [m.joint_names[j] for j, _s in CB.crossed_joints(m, 'base', 'wrist_3_link')]
    assert names == ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                     'wrist_3_joint']
    assert CB.crossed_joints(m, 'gripper_base', 'gripper_base_mount') == []
    assert CB.route_kind(m, CB.crossed_joints(m, 'base', 'wrist_3_link')) == 'chord'


def test_unwrapped_angle_follows_the_geometry(ur5e, ur5e_model):
    """θ = θ_home ± (q − q_home) must agree with the geometric angle modulo 2π along a sweep (the side sign)."""
    m = ur5e_model
    c = _wrist_cable(ur5e, m)
    for i in range(2):
        run = CS._Run(m, c['attachments'][i], c['attachments'][i + 1], None)
        j, _s = run.crossed[0]
        Q = np.tile(m.home, (61, 1))
        Q[:, j] = np.linspace(m.lower[j], m.upper[j], 61)
        ev = run.evaluate(*CB.posed(m, Q), Q)
        d = np.angle(np.exp(1j * (ev['theta'] - ev['cyl']['angle'])))
        assert np.max(np.abs(d)) < 1e-9


def test_bulge_and_bend_radius():
    P = CB.chord_polyline(np.array([[0, 0, 0.0], [0, 0, 0.0]]), np.array([[0.1, 0, 0.0], [0, 0.2, 0.0]]), 49)
    s = np.array([0.15, 0.25])
    Q = CB.bulge(P, s, CB.outward(P))
    assert np.allclose(CB.poly_length(Q), s, rtol=1e-9)
    assert np.allclose(CB.bend_radius(Q), CB.arc_radius(s, np.array([0.1, 0.2])), rtol=5e-3)
    assert np.allclose(Q[:, 0], P[:, 0]) and np.allclose(Q[:, -1], P[:, -1])
    taut = CB.bulge(P, np.array([0.05, 0.2]), CB.outward(P))        # not longer than the route: unchanged
    assert np.allclose(taut, P)
    assert np.all(np.isinf(CB.bend_radius(P)))


def test_snag_hits_on_a_link(ur5e, ur5e_model):
    m = ur5e_model
    T, _a, _x = CB.posed(m, m.home[None])
    li = m.link_index('forearm_link')
    c = T[0, li, :3, 3] + T[0, li, :3, :3] @ np.array([0.0, 0.0, 0.2])     # inside the forearm
    P = CB.chord_polyline((c - [0.3, 0, 0])[None], (c + [0.3, 0, 0])[None], 49)
    hits = CB.snag_hits(m, T, P, P[:, 0], P[:, -1], 0.02)
    assert hits[0] and 'forearm_link' in {h[0] for h in hits[0]}
    far = P + np.array([0.0, 0.0, 3.0])
    assert CB.snag_hits(m, T, far, far[:, 0], far[:, -1], 0.02) == [[]]


def test_suggested_route_crosses_one_joint_per_run(ur5e, ur5e_model):
    atts, notes = CB.suggest_attachments(ur5e_model, ur5e.variant.sensors[0])
    assert [a['link'] for a in atts] == ['wrist_3_link', 'wrist_2_link', 'wrist_1_link', 'forearm_link',
                                         'upper_arm_link', 'shoulder_link', 'base']
    for a, b in zip(atts, atts[1:]):
        assert len(CB.crossed_joints(ur5e_model, a['link'], b['link'])) == 1
    assert atts[0]['point'] == [float(v) for v in ur5e.variant.sensors[0].mount.pose.position]


# ---------------------------------------------------------------------- the project
def test_project_section_file_and_digest(tmp_path, ur5e, ur5e_model):
    from dexvantage.core import provenance as PV
    from dexvantage.core.project import Project, blank_project
    p = blank_project('c')
    d0 = PV.project_digest(p)['sha256']
    assert p.cables == [] and 'cables' in p.to_dict()
    p.cables = [CB.new_cable('A')]
    assert PV.project_digest(p)['sha256'] != d0                    # the result cache misses when cables change
    p.cables = []
    assert PV.project_digest(p)['sha256'] == d0                    # no cables: the old hash
    q = ur5e.snapshot()
    q.cables = [_wrist_cable(q, ur5e_model)]
    path = str(tmp_path / 'c.dexv')
    q.save(path)
    r = Project.load(path)
    assert r.cables == q.cables
    st = r.section('cables')
    r.cables = []
    r.restore('cables', st)
    assert r.cables == q.cables
    try:
        import jsonschema
    except ImportError:
        pytest.skip('jsonschema not installed')
    for rel in ('dexvantage/resources/project.schema.json', 'docs/project.schema.json'):
        with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
            jsonschema.validate(json.loads(json.dumps(r.to_dict())), json.load(fh))


# ---------------------------------------------------------------------- the study
def test_study_on_the_sample_suggests_and_is_json(ur5e_doc):
    doc = ur5e_doc
    text = json.dumps(doc, allow_nan=False)
    assert doc['study'] == 'cables' and doc['suggested'] and len(doc['suggested_cables']) == 1
    assert doc['notes'][0].startswith('The project has no cables yet')
    c = doc['cables'][0]
    assert len(c['segments']) == 6
    for sg in c['segments']:
        assert sg['route'] == 'wrap' and len(sg['joints']) == 1
        j = sg['joints'][0]
        assert j['range_from'] == "the model file's limits"
        assert j['slack_m'] >= 0 and j['slack_loop_m'] == pytest.approx(j['slack_m'] + 0.05, abs=1e-4)
        assert sg['suggested_m'] >= sg['length_max_m'] + 0.05 - 1e-9
        assert sg['length_min_m'] <= sg['length_home_m'] <= sg['length_max_m']
        assert any(f['kind'] == 'unset' for f in sg['flags'])
        assert sg['bend_radius_min_m'] > 0 and sg['snag_poses_tested'] == 1
    assert any(f['kind'] == 'unknown' for f in c['flags'])          # no stated minimum bend radius
    assert doc['summary']['status'] in CS.SEV and doc['summary']['word'] == CS.WORDS[doc['summary']['status']]
    assert len(doc['sheet']) == 6 and '/home/' not in text


def test_pull_bend_and_apply(ur5e, ur5e_model):
    p = ur5e.snapshot()
    c = _wrist_cable(p, ur5e_model, min_bend_radius_m={'value': 0.5, 'source': 'test', 'kind': 'dynamic'},
                     length_m=0.2)
    c['attachments'][0]['length_to_next_m'] = 0.05                 # far too short
    p.cables = [c]
    doc = jobs.run_study('cables', p)
    r = doc['cables'][0]
    kinds = [f['kind'] for sg in r['segments'] for f in sg['flags']] + [f['kind'] for f in r['flags']]
    assert 'pull' in kinds and 'bend' in kinds and r['status'] == 'critical'
    assert doc['summary']['word'] == 'pulls' and doc['summary']['clear'] == 0
    pull = next(f for f in r['segments'][0]['flags'] if f['kind'] == 'pull')
    assert pull['at']['pose'] == 'sweep' and pull['at']['joint'] == 'wrist_3_joint'
    # Apply: every run laid at its suggested length, rounded up to 5 mm
    new = CS.apply_suggested(p.cables, doc)
    assert [a['length_to_next_m'] for a in new[0]['attachments'][:2]] == [sg['suggested_m'] for sg in r['segments']]
    assert all(abs(sg['suggested_m'] / CS.ROUND_UP_M - round(sg['suggested_m'] / CS.ROUND_UP_M)) < 1e-6
               for sg in r['segments'])
    p.cables = new
    p.cables[0]['min_bend_radius_m'] = {'value': 0.001, 'source': 'test', 'kind': 'dynamic'}
    p.cables[0]['length_m'] = 5.0
    doc2 = jobs.run_study('cables', p)
    k2 = [f['kind'] for sg in doc2['cables'][0]['segments'] for f in sg['flags']] + \
        [f['kind'] for f in doc2['cables'][0]['flags']]
    assert 'pull' not in k2 and 'bend' not in k2 and 'unset' not in k2


def test_trajectory_poses_are_tested(tmp_path):
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    r = p.primary_robot
    m = get_model(p, r.uid)
    p.tasks.trajectories = [{'uid': 'aaaaaaaaaa', 'name': 'wave', 'robot_uid': r.uid, 'joints': ['Wrist_Roll'],
                             't': [0.0, 1.0], 'q': [[-1.0], [1.0]]}]
    T, ts, note = CS.trajectory_poses(m, p.tasks.trajectories[0], m.home, 2.0, 3000)
    assert len(ts) == math.ceil(math.degrees(2.0) / 2.0) + 1 and note == ''
    assert np.allclose(T[:, m.joint_names.index('Wrist_Roll')], np.linspace(-1, 1, len(ts)))
    _T, ts2, note2 = CS.trajectory_poses(m, p.tasks.trajectories[0], m.home, 2.0, 10)
    assert len(ts2) == 10 and 'thinned' in note2
    doc = jobs.run_study('cables', p)
    seg = doc['cables'][0]['segments'][0]
    assert seg['snag_poses_tested'] == 1 + len(ts)
    assert not any('No trajectory' in n for n in doc['cables'][0]['notes'])


def test_joint_ranges_from_the_trajectories(ur5e_doc):
    """joint_ranges 'trajectories' sweeps each crossed joint over the span the trajectories use (home included), so
    the suggested length shrinks to the task; the default sweeps the model file's limits and says why the loop is long
    on a joint that turns a full circle."""
    from dexvantage.core.jobs import StudyError
    from dexvantage.core.robot.model import get_model
    from dexvantage.core.robot.samples import sample_project
    full = {sg['joints'][0]['name']: sg for sg in ur5e_doc['cables'][0]['segments'] if sg['joints']}
    assert any('turns 720°' in n for sg in full.values() for n in sg['notes'])      # the UR5e wrists: ±360°
    p = sample_project('ur5e')
    r = p.primary_robot
    m = get_model(p, r.uid)
    names = list(m.joint_names)
    q0 = np.asarray(m.home, float)
    q1 = q0.copy()
    q1[names.index('elbow_joint')] += 0.5
    p.tasks.trajectories = [{'uid': 'bbbbbbbbbb', 'name': 'reach', 'robot_uid': r.uid, 'joints': names,
                             't': [0.0, 1.0], 'q': [q0.tolist(), q1.tolist()]}]
    p.analysis.studies['cables'] = {'joint_ranges': 'trajectories'}
    doc = jobs.run_study('cables', p)
    task = {sg['joints'][0]['name']: sg for sg in doc['cables'][0]['segments'] if sg['joints']}
    el = task['elbow_joint']['joints'][0]
    assert el['range_from'].startswith("the project's trajectories")
    assert el['range'] == pytest.approx([q0[names.index('elbow_joint')], q1[names.index('elbow_joint')]], abs=1e-4)
    assert task['elbow_joint']['suggested_m'] < full['elbow_joint']['suggested_m']
    w = task['wrist_1_joint']['joints'][0]                  # not moved by the trajectory: no slack beyond the loop
    assert w['range'][0] == pytest.approx(w['range'][1]) and w['slack_m'] == pytest.approx(0.0, abs=1e-4)
    p.tasks.trajectories = []                               # none: the model's limits, said so
    doc = jobs.run_study('cables', p)
    assert any('has none in the project' in n for n in doc['cables'][0]['notes'])
    p.analysis.studies['cables'] = {'joint_ranges': 'some'}
    with pytest.raises(StudyError, match='joint_ranges'):
        jobs.run_study('cables', p)


def test_plain_errors_and_settings(ur5e):
    from dexvantage.core.jobs import StudyError
    p = ur5e.snapshot()
    rid = p.primary_robot.uid
    p.cables = [_cable(p, [{'robot_uid': rid, 'link': 'base', 'point': [0, 0, 0]}], name='One point'),
                _cable(p, [{'robot_uid': rid, 'link': 'base', 'point': [0, 0, 0]},
                           {'robot_uid': rid, 'link': 'no_such_link', 'point': [0, 0, 0]}], name='Bad link')]
    doc = jobs.run_study('cables', p)
    msgs = [f['message'] for c in doc['cables'] for f in c['flags']]
    assert any('1 attachment point' in m for m in msgs) and any('no link named no_such_link' in m for m in msgs)
    p.analysis.studies['cables'] = {'bend_factor': -1}
    with pytest.raises(StudyError, match='bend_factor'):
        jobs.run_study('cables', p)
    p.analysis.studies['cables'] = {'loop_m': 'lots'}
    with pytest.raises(StudyError, match='loop_m'):
        jobs.run_study('cables', p)
    p.analysis.studies['cables'] = {'bend_factor': 2.0}
    st = CS.settings_of(p)
    assert st['bend_factor'] == 2.0 and 'bend_factor' not in st['defaults'] and 'loop_m' in st['defaults']
    from dexvantage.core.project import blank_project
    with pytest.raises(StudyError, match='needs a robot'):
        jobs.run_study('cables', blank_project('x'))


def test_cancel_returns_what_it_has(ur5e):
    ctx = jobs.Ctx(cancel=lambda: True)
    doc = jobs.run_study('cables', ur5e.snapshot(), ctx)
    assert doc['cancelled'] and doc['cables'] == []


def test_harness_sheet_exports(tmp_path, ur5e, ur5e_doc):
    paths = CS.export(ur5e_doc, ur5e, str(tmp_path / 'out'))
    assert sorted(os.path.basename(x) for x in paths) == ['cables.csv', 'cables.html', 'cables.json']
    j = json.loads(open(paths[[os.path.basename(x) for x in paths].index('cables.json')]).read())
    assert 'provenance' in j and j['study'] == 'cables'
    csv_text = CS.to_csv(ur5e_doc)
    lines = csv_text.strip().split('\n')
    assert lines[0].split(',') == list(CS.CSV_COLUMNS) and len(lines) == 7
    assert 'slack_loop_per_joint_mm' in lines[0] and 'wrist_3_joint' in lines[1]
    html = CS.to_html(ur5e_doc, 'UR5e')
    assert html.startswith('<!doctype html>') and '<script' not in html and 'Harness sheet' in html
    assert 'Known limits' in html and 'not stated' in html
    for x in paths:
        assert str(tmp_path) not in open(x, encoding='utf-8').read()


def test_batch_flag_writes_the_result(tmp_path):
    out = tmp_path / 'cables.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--cables', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['study'] == 'cables' and 'provenance' in doc and doc['cables'][0]['segments']
    assert str(tmp_path) not in out.read_text()


# ---------------------------------------------------------------------- validation
def test_validation_checks_pass():
    from dexvantage.validation import ValidationSettings, run
    res = run(settings=ValidationSettings(models=('cables',), quick=True))
    ids = {c.id: c for c in res.checks}
    assert set(ids) == {'cables.routed_arc', 'cables.helix_bend', 'cables.loop_arc', 'cables.snag_bruteforce'}
    for c in res.checks:
        assert c.passed, (c.id, c.error, c.exception, c.detail)
    assert ids['cables.snag_bruteforce'].detail['snagged_study'] > 0          # the scene does snag
    assert ids['cables.snag_bruteforce'].detail['snagged_study'] < ids['cables.snag_bruteforce'].detail['poses']


# ---------------------------------------------------------------------- the page
def _wait(win, app):
    from conftest import pump
    win.ctx.analysis.wait(60)
    pump(app, 0.05)


def test_page_runs_applies_and_undoes(app, win, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('ur5e'))
    _wait(win, app)
    assert 'cables' in win.pages
    page = win.pages['cables']
    win.open_page('cables')
    assert page.head.status.text.text() == 'not run yet'
    assert page.cable_box.count() == 0 and not page.b_del.isEnabled()
    page.run()
    _wait(win, app)
    assert page.doc is not None and page.doc['suggested']
    assert page.head.status.text.text() in CS.WORDS.values()
    assert page.runs.topLevelItemCount() == 7 and page.b_apply.isEnabled()
    assert 'no cables yet' in page.none_hint.text() and 'suggested route' in page.none_hint.text()
    assert page.ranges_box.currentData() == 'model' and page.ranges_default.text() == 'default'
    assert page.overlays(win.viewport)                              # the suggested route is drawn
    # Apply adds the suggested cable, one undo step
    n0 = win.store.undo.count()
    page.apply()
    assert len(win.store.project.cables) == 1 and win.store.undo.count() == n0 + 1
    assert page.cable_box.count() == 1 and page.none_hint.text() == ''
    assert page.cable_box.count() == 1 and page.att.rowCount() == 7
    win.store.undo.undo()
    assert win.store.project.cables == []
    win.store.undo.redo()
    assert len(win.store.project.cables) == 1
    # choose a library cable: the bend radius comes with its source
    k = page.lib_box.findData('va-imaging-va-cable-usb3-hf')
    assert k > 0
    page.set_library('va-imaging-va-cable-usb3-hf')
    c = win.store.project.cables[0]
    assert c['library_id'] == 'va-imaging-va-cable-usb3-hf' and c['min_bend_radius_m']['value'] == pytest.approx(0.05)
    assert 'VA Imaging' in page.bend_src.text()
    page.run()
    _wait(win, app)
    assert not page.doc['suggested'] and page.doc['cables'][0]['library_id'] == 'va-imaging-va-cable-usb3-hf'
    # Apply now sets the laid lengths
    page.apply()
    laid = [a['length_to_next_m'] for a in win.store.project.cables[0]['attachments'][:-1]]
    assert all(v is not None and v > 0 for v in laid)
    # Preview toggles and draws dashed
    page.preview()
    assert all(d.dash for d in page.overlays(win.viewport))
    page.preview()
    # exports
    for ext in ('json', 'csv', 'html'):
        path = page.export(str(tmp_path / f'sheet.{ext}'))
        assert path and os.path.getsize(path) > 100
    assert 'provenance' in json.loads(open(tmp_path / 'sheet.json').read())
    # the Camera view overlay projects the cables
    from dexvantage.core.frames import Pose
    from dexvantage.core.optics import CameraModel
    from dexvantage.core.frames import OPTICAL_TO_BODY, quat_from_matrix
    from dexvantage.core.robot.samples import look_at
    cam = CameraModel.from_fov(640, 480, 90.0, 70.0)
    body = look_at([1.2, 1.2, 1.0], [0.0, 0.3, 0.4]).matrix()
    R = body[:3, :3] @ OPTICAL_TO_BODY                              # the optical frame looking along body x
    out = page.camview_overlays(cam, Pose(body[:3, 3], quat_from_matrix(R)))
    assert out and all(len(polys) and isinstance(label, str) for polys, _col, label in out)
    # deleting the cable is one undo step too
    page.delete_cable()
    assert win.store.project.cables == []
    win.store.undo.undo()
    assert len(win.store.project.cables) == 1


def test_page_new_cable_and_points(app, win):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('so100'))
    _wait(win, app)
    page = win.pages['cables']
    page.new_cable()
    assert len(win.store.project.cables) == 1
    c = win.store.project.cables[0]
    assert c['sensor_uid'] == win.store.project.variant.sensors[0].uid and len(c['attachments']) >= 2
    n = len(c['attachments'])
    page.att.setCurrentCell(0, 0)
    page.add_point()
    assert len(win.store.project.cables[0]['attachments']) == n + 1
    page.remove_point()
    assert len(win.store.project.cables[0]['attachments']) == n
    page.att.setCurrentCell(0, 0)
    page._set_point(0, 2, 0.123)
    assert win.store.project.cables[0]['attachments'][0]['point'][2] == pytest.approx(0.123)
    page.sensor_point()
    s = win.store.project.variant.sensors[0]
    assert win.store.project.cables[0]['attachments'][0]['point'] == [float(v) for v in s.mount.pose.position]
