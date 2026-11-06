"""Dexterous hands and the multi-finger occlusion study (spec §6, §7.14, §1.2): the bundled Shadow Hand, the hand
structure and pads (core/dexhand.py), the 'dexhand' sample, the study (core/studies/dex_hand_tactile.py), its page,
its batch flag and its validation checks.

Independent references: the bundled Menagerie files compiled directly by MuJoCo (joint order, limits, meshes), the
closed-form scripted scene of validation/checks_dexhand.py, and an all-pairs ray cast written out in that module."""
import json
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from dexvantage.core import dexhand as DH
from dexvantage.core import jobs
from dexvantage.core.jobs import StudyError
from dexvantage.core.robot import get_model
from dexvantage.core.robot import samples as S
from dexvantage.core.studies import dex_hand_tactile as D

from conftest import pump

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
HAND = os.path.join(S.ROBOTS_DIR, 'shadow_hand')


@pytest.fixture(scope='module')
def dex():
    return S.sample_project('dexhand')


@pytest.fixture(scope='module')
def doc(dex):
    return jobs.run_study(D.STUDY_ID, dex)


# ====================================================================== the bundled model and its licence
def test_shadow_hand_is_bundled_with_its_licence_and_only_used_meshes():
    from dexvantage.core.robot.menagerie import license_name
    text = open(os.path.join(HAND, 'LICENSE'), encoding='utf-8').read()
    assert license_name(text)[0] == 'Apache-2.0' == S.BUNDLED['shadow_hand']['license']
    assert 'Shadow Robot Company' in text
    used = {m.file for m in mujoco.MjSpec.from_file(os.path.join(HAND, 'right_hand.xml')).meshes}
    assert set(os.listdir(os.path.join(HAND, 'assets'))) == used
    assert sorted(os.listdir(HAND)) == ['LICENSE', 'assets', 'keyframes.xml', 'right_hand.xml']
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(HAND) for f in fs)
    assert total < 10e6


def test_third_party_records_every_hand_model_checked():
    text = open(os.path.join(ROOT, 'THIRD_PARTY.md'), encoding='utf-8').read()
    assert S.COMBINED_DEXHAND in text
    for folder in ('shadow_hand', 'wonik_allegro', 'unitree_g1', 'leap_hand', 'shadow_dexee'):
        assert f'`{folder}`' in text, folder
    for name in ('Inspire', 'Ability'):
        assert name in text


def test_combined_model_keeps_the_hand_files_joints_and_limits():
    hand = mujoco.MjModel.from_xml_path(os.path.join(HAND, 'right_hand.xml'))
    both = mujoco.MjModel.from_xml_path(S.robot_file(S.COMBINED_DEXHAND))
    names = [hand.joint(i).name for i in range(hand.njnt)]
    got = {both.joint(i).name: both.jnt_range[i] for i in range(both.njnt)}
    for i, n in enumerate(names):
        assert np.allclose(got['hand_' + n], hand.jnt_range[i]), n
    assert both.njnt == hand.njnt + 6
    assert both.nkey == 1 and np.allclose(both.key_qpos[0][6:], 0.0)        # the UR5e's home, the hand open


def test_keyframes_follow_mujocos_qpos_order():
    keys = DH.read_keyframes(os.path.join(HAND, 'right_hand.xml'), os.path.join(HAND, 'keyframes.xml'))
    m = mujoco.MjModel.from_xml_path(os.path.join(HAND, 'right_hand.xml'))
    order = [m.joint(i).name for i in range(m.njnt)]
    assert list(m.jnt_qposadr) == list(range(m.njnt))
    for k in ('open hand', 'grasp sphere', 'two finger pinch', 'three finger pinch'):
        assert list(keys[k]) == order
    raw = next(k for k in __import__('xml.etree.ElementTree', fromlist=['parse']).parse(
        os.path.join(HAND, 'keyframes.xml')).getroot().iter('key') if k.get('name') == 'grasp sphere')
    assert np.allclose([keys['grasp sphere'][n] for n in order], [float(x) for x in raw.get('qpos').split()])


# ====================================================================== the hand
def test_structure_fingers_palm_and_pads(dex):
    r = dex.robots[0]
    m = get_model(dex, r.uid)
    h = DH.structure(dex, r.uid, m)
    assert [f.name for f in h.fingers] == ['ff', 'mf', 'rf', 'lf', 'th']
    assert h.fingers[4].title == 'Thumb' and h.fingers[4].links[-1] == 'hand_rh_thbase'
    assert h.fingers[3].links[-1] == 'hand_rh_lfmetacarpal'
    assert h.palm == 'hand_rh_palm' and h.link_group['hand_rh_forearm'] == 'arm'
    assert h.link_group['shoulder_link'] == 'arm'
    assert len(h.hand_joints(m)) == 22                    # 24 hand joints less the two wrist joints
    T = m.fk(m.home)
    for pad in r.contact_pads:
        tri = np.asarray(pad['faces'], float)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        mean_w = T[pad['link']][:3, :3] @ n.mean(axis=0)
        if pad['link'] != 'hand_rh_thdistal':             # the four fingers' pads face the palm side (+x) at home
            assert mean_w[0] > 0.7, pad['link']
        # every pad triangle is a triangle of the link's collision mesh (rounded to 0.1 mm)
        V = np.concatenate([Vv[F] for ln, Vv, F, _c in m.collision_geoms() if ln == pad['link']]).astype(float)
        V = V.reshape(-1, 9)
        k = np.round(tri.reshape(-1, 9), 4)
        assert len({tuple(x) for x in k} - {tuple(x) for x in np.round(V, 4)}) == 0


def test_postures_interpolate_and_clip(dex):
    r = dex.robots[0]
    m = get_model(dex, r.uid)
    syns = DH.synergies(dex, r.uid, dex.analysis.studies[D.STUDY_ID], m)
    assert [s.id for s in syns] == ['power', 'precision', 'tripod']
    assert all(s.sample and 'keyframes.xml' in s.source for s in syns)
    p0, p1, ph = (DH.posture(m, syns[0], c) for c in (0.0, 1.0, 0.5))
    assert np.allclose(p0, m.home)
    i = m.joint_names.index('hand_rh_FFJ3')
    assert np.isclose(p1[i], syns[0].closed['hand_rh_FFJ3']) and np.isclose(ph[i], 0.5 * p1[i])
    assert np.allclose(p1[:6], m.home[:6])                # the arm is not part of a posture
    big = DH.Synergy('x', 'x', 'power', {}, {'hand_rh_FFJ3': 99.0}, 'test')
    assert DH.posture(m, big, 1.0)[i] == m.upper[i]
    assert DH.unknown_joints(m, DH.Synergy('y', 'y', 'power', {'nope': 0.0}, {}, 'test')) == ['nope']
    with pytest.raises(DH.HandError):
        DH.Synergy.from_dict({'id': 'z', 'kind': 'squeeze'})
    with pytest.raises(DH.HandError):
        DH.Synergy.from_dict({'id': 'z', 'closed': {'a': 'b'}})


def test_default_postures_without_keyframes_are_labelled_samples(dex):
    r = dex.robots[0]
    m = get_model(dex, r.uid)
    h = DH.structure(dex, r.uid, m)
    syns = DH.default_synergies(m, h)
    assert [s.kind for s in syns] == ['power', 'precision']
    assert all('sample posture' in s.source and s.sample for s in syns)
    assert 'Thumb' in syns[1].title
    for s in syns:
        q = DH.posture(m, s, 1.0)
        assert np.all(q >= m.lower - 1e-12) and np.all(q <= m.upper + 1e-12)
    assert set(syns[1].closed) < set(syns[0].closed)


def test_sample_cameras_name_library_devices(dex):
    from dexvantage.core import library as LB
    ids = [s.model_id for s in dex.variant.sensors]
    assert ids == ['omnivision-ovm6948', 'e-con-see3cam-24cug', 'intel-realsense-d435']
    assert all(LB.get(i) is not None for i in ids)
    assert all('sample placement' in s.name for s in dex.variant.sensors)
    assert dex.variant.sensors[0].mount.parent_frame == 'hand_rh_palm'
    assert dex.variant.sensors[1].mount.parent_frame == 'hand_rh_forearm'
    assert not dex.variant.sensors[2].mount.on_link
    assert 'dexhand' in [s['name'] for s in S.list_samples() if s['available']]


def test_sample_saves_and_reloads_with_its_pads_and_postures(dex, doc, tmp_path):
    from dexvantage.core import project as P
    from dexvantage.core.provenance import project_digest
    path = str(tmp_path / f'dexhand{P.EXTENSION}')
    dex.save(path)
    q = P.Project.load(path)
    assert q.robots[0].contact_pads == dex.robots[0].contact_pads
    assert q.analysis.studies[D.STUDY_ID] == dex.analysis.studies[D.STUDY_ID]
    assert project_digest(q)['sha256'] == project_digest(dex)['sha256']
    d2 = jobs.run_study(D.STUDY_ID, q)
    assert d2['rig'] == doc['rig'] and d2['matrix'] == doc['matrix']


# ====================================================================== the kernel on the scripted scene
def test_kernel_matches_the_closed_form():
    from dexvantage.validation import checks_dexhand as CK
    sc = CK.scripted_scene()
    cam = CK._camera()
    for path in ('raycast', 'numpy'):
        for (pad, j), _exp in sc['expected'].items():
            R, t = sc['cams'][j]
            P = sc['pads'][pad]
            if path == 'raycast':
                vis, own, _i = D.visibility_owner_raycast(P, t, sc['tris'], sc['group'], 3)
            else:
                vis, own, _i = D.visibility_owner(P, t, sc['tris'], sc['group'], 3, n=1024, backend='numpy')
            _uv, iv = cam.project((P - t) @ R)
            assert iv.all()
            assert np.array_equal(own, sc['point_owner'][(pad, j)]), (path, pad, j)
            c = D.patch_counts(iv, vis, own, 3)
            assert c.sum() == len(P)
    assert sc['expected'][('A', 0)][1] == pytest.approx(0.44)
    assert sc['expected'][('A', 1)][2] == pytest.approx(0.2)


def test_owner_from_group_maps_equals_one_map_of_the_scene():
    from dexvantage.core import occlusion as OC
    from dexvantage.validation import checks_dexhand as CK
    sc = CK.scripted_scene()
    o = sc['cams'][0][1]
    dirs = np.random.default_rng(3).normal(0, 1, (5000, 3))
    full = OC.CubeMap.build(sc['tris'], o, 128, backend='numpy').depth(dirs)
    best = np.full(len(dirs), np.inf)
    for g in range(3):
        best = np.minimum(best, OC.CubeMap.build(sc['tris'][sc['group'] == g], o, 128, backend='numpy').depth(dirs))
    assert np.array_equal(np.isfinite(full), np.isfinite(best))
    fin = np.isfinite(full)
    assert np.abs(full[fin] - best[fin]).max() < 1e-6


def test_brute_force_reference_finds_the_nearest_owner():
    from dexvantage.validation import checks_dexhand as CK
    tris = np.concatenate([CK._box([-0.01] * 3, [0.01] * 3), CK._box([-0.01, -0.01, 0.05], [0.01, 0.01, 0.06])])
    grp = np.r_[np.zeros(12, int), np.ones(12, int)]
    P = np.array([[0.0, 0.0, -0.01], [0.0, 0.0, 0.0101]])
    own = CK.brute_owner(P, [0.0, 0.0, 0.3], tris, grp)
    assert own.tolist() == [1, 1]
    assert CK.brute_owner(P[1:], [0.0, 0.0, 0.03], tris, grp).tolist() == [-1]


def test_patch_counts_and_bits():
    c = D.patch_counts([True, True, False, True], [True, False, False, False], [-1, 1, -1, 0], 2)
    assert c.tolist() == [1, 1, 1, 1]
    m = np.random.default_rng(0).random(37) > 0.5
    assert np.array_equal(D.unpack_bits(D.pack_bits(m), 37), m)


# ====================================================================== the study
def test_study_on_the_sample(dex, doc):
    json.dumps(doc, allow_nan=False)
    assert doc['study'] == D.STUDY_ID and doc['source'] == 'synergy'
    assert len(doc['cases']) == 1 + 3 * 4
    F, C = len(doc['fingers']), len(doc['cameras'])
    assert (F, C) == (5, 3)
    mx = doc['matrix']
    for i in range(F):
        for j in range(C):
            v = mx['not_seen_pct'][i][j]
            assert 0 <= v <= 100
            assert sum(mx['by_owner_pct'][i][j]) == pytest.approx(v, abs=0.05)
            assert mx['dominant'][i][j] in mx['categories'] + ['']
    rig = doc['rig']
    assert rig['status'] == D.status_for(rig['fingertips_seen_pct'], D.settings_of(dex))[0]
    # the rig sees at least what its best camera sees
    assert rig['fingertips_seen_pct'] >= max(c['fingertips_seen_pct'] for c in doc['cameras']) - 1e-9
    by = {c['name']: c for c in doc['cameras']}
    dorsal = by['Dorsal wrist camera (sample placement)']
    assert dorsal['hidden_pct'] > 90 and dorsal['main_occluder'] == 'palm'      # it looks at the back of the hand
    palm = by['Palm micro camera (sample placement)']
    assert palm['outside_image_pct'] > 10                  # its 120° view misses part of the spread fingers
    assert sorted(c['rank'] for c in doc['cameras']) == [1, 2, 3]
    n = sum(f['points'] for f in doc['fingers'])
    for pc in doc['per_case']:
        for j in range(C):
            bits = D.unpack_bits(pc['seen_bits'][j], n)
            at = 0
            for i, f in enumerate(doc['fingers']):
                assert bits[at:at + f['points']].mean() * 100 == pytest.approx(pc['seen_pct'][j][i], abs=0.01)
                at += f['points']
    assert doc['handoff'] is None and doc['occlusion']['backend'] == 'raycast'
    assert any('sample postures' in n_ for n_ in doc['notes'])
    assert 'q' not in dex.analysis.studies[D.STUDY_ID]      # the project is not changed


def test_suggestion_turns_about_the_optical_centre(dex, doc):
    from dexvantage.core import library as LB
    from dexvantage.core.frames import Pose
    sug = doc['suggestion']
    assert sug is not None and sug['to_pct'] >= sug['from_pct']
    s = dex.variant.sensor(sug['sensor_uid'])
    im = LB.sensor_imagers(s)[0]
    a = s.mount.pose.matrix() @ im.pose_in_body.matrix()
    b = Pose.from_dict(sug['pose']).matrix() @ im.pose_in_body.matrix()
    assert np.allclose(a[:3, 3], b[:3, 3], atol=1e-9)
    ang = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
    assert ang == pytest.approx(abs(sug['deg']), abs=1e-6)
    # applying it gives the figure the search promised
    p2 = S.sample_project('dexhand')
    p2.variant.sensors[[x.name for x in dex.variant.sensors].index(s.name)].mount.pose = Pose.from_dict(sug['pose'])
    d2 = jobs.run_study(D.STUDY_ID, p2)
    assert d2['rig']['fingertips_seen_pct'] == pytest.approx(sug['to_pct'], abs=0.01)


def test_cube_backend_runs_and_stays_close(dex, doc):
    p = S.sample_project('dexhand')
    p.analysis.studies[D.STUDY_ID].update({'backend': 'numpy', 'cube_n': 256, 'synergy_ids': ['power'], 'steps': 1,
                                           'suggest_deg': 0})
    d = jobs.run_study(D.STUDY_ID, p)
    assert d['occlusion']['backend'] == 'numpy' and d['suggestion'] is None and len(d['cases']) == 2
    assert np.isfinite(np.asarray(d['matrix']['not_seen_pct'], float)).all()


def test_grasp_set_cases(dex):
    from dexvantage.core import grasps as GK
    p = S.sample_project('dexhand')
    r = p.robots[0]
    gs = GK.sample(p, r.uid, kinds=('top_down',), n=4, rng_seed=2)
    uid = GK.add_set(p, gs)
    p.analysis.studies[D.STUDY_ID].update({'source': 'grasp_set', 'grasp_set_uid': uid, 'suggest_deg': 0})
    d = jobs.run_study(D.STUDY_ID, p)
    g = d['grasps']
    assert g['taken'] == 4 and g['feasible'] == len(d['cases']) >= 1 and g['synergy'] == 'power'
    m = get_model(p, r.uid)
    q_closed = DH.posture(m, DH.synergies(p, r.uid, p.analysis.studies[D.STUDY_ID], m)[0], 1.0)
    fingers = [m.joint_names.index(n) for n in DH.structure(p, r.uid, m).hand_joints(m)]
    for pc in d['per_case']:
        assert np.allclose(np.asarray(pc['q'])[fingers], q_closed[fingers], atol=1e-6)   # the hand stays closed
    p.analysis.studies[D.STUDY_ID]['grasp_set_uid'] = 'nope'
    with pytest.raises(StudyError, match='grasp set'):
        jobs.run_study(D.STUDY_ID, p)


def test_study_refuses_what_it_cannot_use():
    p = S.sample_project('dexhand')
    p.analysis.studies[D.STUDY_ID]['backend'] = 'magic'
    with pytest.raises(StudyError, match='backend'):
        jobs.run_study(D.STUDY_ID, p)
    p = S.sample_project('dexhand')
    p.analysis.studies[D.STUDY_ID]['steps'] = 0
    with pytest.raises(StudyError, match='steps'):
        jobs.run_study(D.STUDY_ID, p)
    p = S.sample_project('dexhand')
    p.variant.sensors = []
    with pytest.raises(StudyError, match='camera'):
        jobs.run_study(D.STUDY_ID, p)
    p = S.sample_project('ur5e')                         # two fingertips, two pads: it runs, with sample postures
    d = jobs.run_study(D.STUDY_ID, p)
    assert len(d['fingers']) == 2 and any('sample posture' in s['source'] for s in d['synergies'])
    p = S.sample_project('ur5e')
    p.robots[0].fingertip_frames = []
    with pytest.raises(StudyError, match='fingertip'):
        jobs.run_study(D.STUDY_ID, p)
    with pytest.raises(StudyError, match='Phase 5'):
        D.tactile_handoff(p)


def test_batch_flag_writes_the_result(tmp_path):
    out = tmp_path / 'dex.json'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'dexhand', '--dex-occlusion', str(out)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d['study'] == D.STUDY_ID and 'provenance' in d and d['rig']['fingertips_seen_pct'] > 0
    assert str(tmp_path) not in out.read_text()


def test_validation_checks_pass_quick():
    from dexvantage.validation import suite
    suite.load_checks()
    specs = [s for s in suite.CHECKS if s.model == 'dexhand']
    assert {s.id for s in specs} == {'dexhand.matrix_vs_analytic', 'dexhand.owner_vs_bruteforce',
                                     'dexhand.cube_owner_union'}
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        for s in specs:
            res = suite.run_one(s, ctx)
            assert res.passed, (s.id, res.error, res.exception, res.detail)
    finally:
        ctx.close()


# ====================================================================== the page
def test_page_runs_shows_and_applies(app, win, tmp_path):
    from dexvantage.core import camrender as CR
    from dexvantage.core import library as LB
    from dexvantage.ui.pages import dexhand as PG
    assert 'dexhand' in win.pages
    assert win.open_sample('dexhand')
    page = win.pages['dexhand']
    win.open_page('dexhand')
    pump(app, 0.05)
    assert page.head.status.text.text() == 'not run yet'
    assert page.export() is None
    page.run()
    assert win.ctx.analysis.wait(300)
    pump(app, 0.1)
    d = page.doc
    assert d is not None and page.head.figure.text() not in ('', '—')
    assert page.head.status.text.text() in D.WORDS.values()
    assert len(page.matrix.rows) == 5 and len(page.matrix.cols) == 3
    assert page.table.topLevelItemCount() == 3
    assert page.sweep.rows and len(page.sweep.rows) == len(d['per_case'])
    page.matrix.grab()
    # hover a matrix cell: its tooltip names the split
    from PyQt6.QtCore import QPointF
    r = page.matrix.cell_rect(0, 0)

    class _E:
        def position(self):
            return QPointF(r.center())
    page.matrix.mouseMoveEvent(_E())
    assert 'Not seen' in page.matrix.tip_text()
    # a case on screen draws its patch points
    k = len(d['per_case']) - 1
    page.c_case.setCurrentIndex(k)
    assert page.overlays(win.viewport) == []              # the arm on screen is not in that posture
    assert page.show_case(k)
    items = page.overlays(win.viewport)
    assert items and all(isinstance(x, PG.LineDraw) for x in items)
    # the Camera view of a studied camera draws them too
    s = win.store.project.variant.sensors[2]
    win.camview.view.sensor_uid = s.uid
    im = LB.sensor_imagers(s)[0]
    pose = CR.optical_pose(win.store.project, s, im, win.store.q)
    ov = page.camview_overlays(im.camera, pose)
    assert ov and all(len(poly) == 2 for polys, _c, _l in ov for poly in polys)
    # Apply turns the suggested camera (one undo step) and runs again
    sug = d['suggestion']
    assert sug and page.b_apply.isEnabled()
    page.preview()
    assert len(page.overlays(win.viewport)) >= 2
    before = win.store.project.variant.sensor(sug['sensor_uid']).mount.pose.to_dict()
    n_undo = win.store.undo.count()
    page.apply()
    assert win.store.undo.count() == n_undo + 1
    after = win.store.project.variant.sensor(sug['sensor_uid']).mount.pose.to_dict()
    assert np.allclose(after['quaternion'], sug['pose']['quaternion'])
    assert win.ctx.analysis.wait(300)
    pump(app, 0.1)
    win.store.undo.undo()
    assert np.allclose(win.store.project.variant.sensor(sug['sensor_uid']).mount.pose.to_dict()['quaternion'],
                       before['quaternion'])
    out = page.export(str(tmp_path / 'dex.json'))
    exp = json.loads(open(out, encoding='utf-8').read())
    assert 'provenance' in exp and exp['rig'] == page.doc['rig']
    # grasp-set mode without a set says what to do
    page.source.set_value('grasp_set')
    page.source.changed.emit('grasp_set')
    assert 'Tasks page' in page.case_note.text()


def test_page_explains_a_failed_run(app, win):
    assert win.open_sample('dexhand')
    page = win.pages['dexhand']
    win.store.project.variant.sensors = []
    page.run()
    assert win.ctx.analysis.wait(60)
    pump(app, 0.05)
    assert page.head.status.text.text() == 'could not run'
    assert 'StudyError' not in page.head.explain.text() and 'camera' in page.head.explain.text()
