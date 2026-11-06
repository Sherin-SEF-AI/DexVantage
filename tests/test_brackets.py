"""Brackets (core/brackets.py, core/meshing.py, core/studies/brackets.py, ui/pages/brackets.py): the mesher (closed and
manifold on random fields, binary STL round trip), the ISO 9409-1 and T-slot tables and their sources, the camera's
stated holes, every bracket kind a single closed shell, the flange disc's volume and holes against figures typed again
here, the flange frame on the UR5e sample against UR's DH figure, clearance (a link through the bracket, the gripper
moved out by the adapter, cables), the study (JSON, suggestions, batch), the STL export with its sidecar and the page
(Apply as one undo step, overlays)."""
import json
import math
import os

import numpy as np
import pytest
from conftest import pump

from dexvantage.core import brackets as B
from dexvantage.core import meshing as M
from dexvantage.core.project import Sensor
from dexvantage.validation.checks_brackets import edge_report, fit_circle, sample_designs


@pytest.fixture(scope='module')
def ur5e():
    from dexvantage.core.robot.samples import sample_project
    return sample_project('ur5e')


def d435():
    return B.camera_geom(Sensor(model_id='intel-realsense-d435'))


# ====================================================================== settings and tables
def test_spec_is_validated():
    s = B.BracketSpec.from_dict({'kind': 'nope', 'wall_mm': 0.1, 'res_mm': 'x', 'tilt_deg': 400, 'side': 'left',
                                 'pt_count': 3.6, 'size_mm': [1, -2, 3], 'standoff_mm': float('nan'), 'junk': 1})
    assert s.kind == 'none' and s.wall_mm == 0.8 and s.res_mm == 0.5 and s.tilt_deg == 90.0 and s.side == 'auto'
    assert s.pt_count == 4 and s.size_mm is None and s.standoff_mm == 10.0
    assert B.BracketSpec.from_dict(s.to_dict()) == s
    assert B.BracketSpec.from_dict(None) == B.BracketSpec()
    assert B.BracketSpec.from_dict({'size_mm': [1, 2, 3]}).size_mm == [1.0, 2.0, 3.0]


def test_thread_diameters():
    assert B.thread_diameter_mm('M6') == 6.0 and B.thread_diameter_mm('M2.5') == 2.5
    assert B.thread_diameter_mm('1/4-20') == pytest.approx(6.35) and B.thread_diameter_mm('1/4-20 UNC') == \
        pytest.approx(6.35)
    assert B.thread_diameter_mm('M3x0.5') == 3.0 and B.thread_diameter_mm('a screw') is None


def test_iso_table_is_cited_and_designations_parse():
    p = B.iso_pattern('ISO 9409-1-50-4-M6')
    assert p['stated'] and p['pcd_mm'] == 50.0 and p['count'] == 4 and p['pin_d_mm'] == 6.0
    assert p['pilot_d_mm'] == 31.5 and p['pilot_depth_mm'] is None
    for k in p['sources']:
        assert B.SOURCES[k]['url'].startswith('https://') and B.SOURCES[k]['read'] == '2026-11-06'
    q = B.iso_pattern('80-6-M8')                       # read from the designation only: the rest not stated
    assert not q['stated'] and q['pcd_mm'] == 80.0 and q['count'] == 6 and q['thread_d_mm'] == 8.0
    assert q['pilot_d_mm'] is None and q['pin_d_mm'] is None and (q['first_deg'], q['pitch_deg']) == (30.0, 60.0)
    assert 'not stated' in q['note']
    r = B.iso_pattern('40-5-M6')
    assert r['pitch_deg'] == 72.0 and 'ASSUMPTION' in r['note']
    with pytest.raises(B.BracketError, match='not an ISO 9409-1 designation'):
        B.iso_pattern('fifty')
    bolts, pin = B.iso_hole_positions(p)
    assert np.allclose(np.hypot(*bolts.T), 25.0) and np.allclose(pin, [25.0, 0.0])
    assert np.allclose(np.degrees(np.arctan2(bolts[:, 1], bolts[:, 0])) % 360, [45, 135, 225, 315])


def test_tslot_table_is_cited():
    for k, v in B.TSLOT.items():
        src = B.SOURCES[v['source']]
        assert src['url'].startswith('https://www.motedis') and str(int(v['groove_width_mm'])) in src['title']
        assert v['groove_depth_mm'] > 0 and k in B.TYPES


def test_robot_flange_lookup(ur5e):
    assert B.robot_flange(ur5e.robots[0]) == ('50-4-M6', 'ur5e_datasheet')

    class R:
        name = 'SO-ARM100'
    assert B.robot_flange(R()) is None


# ====================================================================== the mesher
def test_mesher_is_closed_on_random_fields():
    rng = np.random.default_rng(3)
    seen = 0
    for _ in range(250):
        n = int(rng.integers(2, 7))
        vals = (rng.random((n + 2,) * 3) - rng.random()).astype(np.float32)
        for ax in range(3):
            idx = [slice(None)] * 3
            for end in (0, -1):
                idx[ax] = end
                vals[tuple(idx)] = 1.0
        V, F = M.march(vals, np.zeros(3), 1.0)
        if not len(F):
            continue
        seen += 1
        r = edge_report(V, F)
        assert r['edges_not_two'] == 0 and r['directed_twice'] == 0 and r['zero_area'] == 0, r
        assert r['euler_ok'] and r['volume_mm3'] > 0
        c = M.check(V, F)
        assert c['ok'] and c['euler'] == r['euler']
    assert seen > 150


def test_stl_binary_round_trip():
    s = M.subtract(M.box([0, 0, 0], np.eye(3), [6, 4, 3], r=1), M.rod([0, 0, -9], [0, 0, 9], 1.5))
    V, F = M.mesh(s, 0.4)
    data = B.stl_bytes(V, F, 'solid test')
    assert len(data) == 84 + 50 * len(F) and not data[:80].lower().startswith(b'solid')
    V2, F2, N, head = M.read_stl(data)
    assert 'DexVantage bracket' in head and len(F2) == len(F)
    tri = V2[F2]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum('ij,ij->i', cr, N) > 0)
    assert M.check(V2, F2)['ok']


# ====================================================================== the camera
def test_camera_holes_from_the_library():
    c = d435()
    assert c.size_mm == (25.0, 90.0, 25.0) and not c.micro
    holes, flags = B.stated_holes(c, 'back')
    assert len(holes) == 2 and all(h['thread'] == 'M3' and h['max_insert_mm'] == 3.0 for h in holes)
    assert np.linalg.norm(holes[0]['centre'] - holes[1]['centre']) == pytest.approx(45.0)
    assert np.allclose([h['centre'][0] for h in holes], -25.0)             # on the back face
    assert flags and 'does not dimension' in flags[0]
    tri, _f = B.stated_holes(c, 'bottom')
    assert len(tri) == 1 and tri[0]['centre'][1] == pytest.approx(-17.5) and tri[0]['d_mm'] == pytest.approx(6.35)
    d405 = B.camera_geom(Sensor(model_id='intel-realsense-d405'))
    t405, _ = B.stated_holes(d405, 'bottom')
    assert t405[0]['centre'][1] == pytest.approx(-9.0) and t405[0]['max_insert_mm'] == 5.0
    assert B.choose_face(c, np.array([-1.0, 0, 0]))[0] == 'back'
    assert B.choose_face(c, np.array([0, 0, -1.0]))[0] == 'bottom'


def test_camera_without_size_or_holes():
    none = B.camera_geom(Sensor())
    assert none.size_mm is None and 'no library device' in none.size_note
    with pytest.raises(B.BracketError, match='size'):
        B.design(B.BracketSpec(kind='tslot2020'), none)
    nane = B.camera_geom(Sensor(model_id='ams-osram-naneyec'))
    assert nane.size_mm is None and 'not stated' in nane.size_note
    given = B.camera_geom(Sensor(model_id='ams-osram-naneyec'), B.BracketSpec(size_mm=[2.0, 1.0, 1.0]))
    assert given.size_mm == (2.0, 1.0, 1.0) and 'your figure' in given.size_note
    D = B.design(B.BracketSpec(kind='pantilt'), B.camera_geom(Sensor(model_id='e-con-see3cam-24cug')))
    assert any('No mounting holes are stated' in f for f in D.flags)
    assert any('pattern is not entered' in f for f in D.flags)


# ====================================================================== the brackets
def test_every_kind_is_one_closed_shell():
    for label, D in sample_designs(False):
        V, F, chk = B.mesh_design(D)
        r = edge_report(V, F)
        assert chk['ok'] and r['shells'] == 1 and r['edges_not_two'] == 0 and r['directed_twice'] == 0, label
        assert r['volume_mm3'] > 0 and chk['euler'] == r['euler'], label
        # nothing of the bracket inside the camera body
        c, A, half = D.camera_box
        inside = np.all(np.abs((V - c) @ A) < half - 0.3, axis=1)
        assert not inside.any(), label


def test_flange_disc_volume_and_topology():
    spec = B.BracketSpec(kind='flange', fillet_mm=0.0)
    solid, fi = B.flange_plate(spec)
    V, F = M.mesh(solid, 0.35)
    r = edge_report(V, F)
    R, t, rb, rp = 63 / 2, 8.0, (6 + 0.4) / 2, (6 + 0.1) / 2           # typed again: UR's drawing + the defaults
    rs, rr, rc, h = 31.5 / 2 - 0.05, 31.5 / 2 + 0.05, 10.0, 3.0
    exact = math.pi * t * (R * R - rc * rc - 4 * rb * rb - rp * rp) + math.pi * h * (rs * rs - rc * rc) - \
        math.pi * h * (rr * rr - rc * rc)
    assert r['volume_mm3'] == pytest.approx(exact, rel=0.01)
    assert r['shells'] == 1 and r['euler'] == 2 - 2 * 6
    assert fi['rc'] == rc and len(fi['holes']) == 5


def test_flange_holes_measured_on_the_mesh():
    _label, D = sample_designs(True)[0]
    V, F, _c = B.mesh_design(D, 0.35)
    for a in (45, 135, 225, 315, 0):
        x0, y0 = 25 * math.cos(math.radians(a)), 25 * math.sin(math.radians(a))
        d = 6.1 if a == 0 else 6.4
        wall = (np.abs(np.hypot(V[:, 0] - x0, V[:, 1] - y0) - d / 2) < 0.5) & (V[:, 2] > 1) & (V[:, 2] < 7)
        cx, cy, rad = fit_circle(V[wall, 0], V[wall, 1])
        assert math.hypot(cx - x0, cy - y0) < 0.05 and 2 * rad == pytest.approx(d, abs=0.1)


def test_flange_without_stated_centring_or_dowel():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    T[:3, 3] = [90.0, 0.0, 30.0]
    D = B.design(B.BracketSpec(kind='flange', pattern='80-6-M8'), d435(), T)
    assert D.interface['spigot_d_mm'] is None and D.interface['pin_hole_d_mm'] is None
    assert any('dowel' in f for f in D.flags) and any('centring' in f for f in D.flags)
    assert sum(h['role'] == 'flange_bolt' for h in D.holes) == 6


def test_tslot_key_and_bolts():
    D = B.design(B.BracketSpec(kind='tslot2020'), d435())
    assert D.interface['key_mm'][1] == pytest.approx(6.0 - 0.4) and D.interface['key_mm'][2] < 5.5
    xs = sorted(h['centre'][0] for h in D.holes if h['role'] == 'tslot_bolt')
    assert len(xs) == 2 and all(h.get('access') for h in D.holes if h['role'] == 'tslot_bolt')
    D2 = B.design(B.BracketSpec(kind='tslot2020', key_mm=6.0), d435())
    assert any('as deep as the groove' in f for f in D2.flags)
    D3 = B.design(B.BracketSpec(kind='tslot2020', standoff_mm=0.0), d435())
    assert any('into the base' in f for f in D3.flags)


def test_micro_sleeve_pocket():
    micro = B.camera_geom(Sensor(model_id='omnivision-ovm6948'))
    assert micro.micro
    D = B.design(B.BracketSpec(kind='palm', wall_mm=1.2, pocket_mm=0.1), micro)
    assert D.plate['kind'] == 'sleeve' and np.allclose(D.plate['pocket_mm'], np.array(micro.size_mm) + 0.2)
    assert D.res_mm <= min(micro.size_mm) / 2
    big = B.design(B.BracketSpec(kind='dorsal'), d435())
    assert any('micro camera' in f for f in big.flags)


# ====================================================================== on the UR5e sample
def test_flange_frame_matches_ur_dh(ur5e):
    s = ur5e.variant.sensors[0]
    T, note = B.flange_frame(ur5e, s)
    # UR's DH table (UR5e): d6 = 0.0996 m from wrist 3 to the flange (typed again here; read 2026-11-06)
    assert T[1, 3] == pytest.approx(0.0996, abs=0.0015) and np.allclose(T[:3, 2], [0, 1, 0], atol=1e-9)
    assert 'gripper' in note


def test_flange_bracket_on_the_ur5e(ur5e):
    s = ur5e.variant.sensors[0]
    spec = B.BracketSpec.from_dict({'kind': 'flange'})
    row = B.build(ur5e, s, spec, samples=12, seed=1)
    assert row['mesh']['ok'] and row['mesh']['shells'] == 1 and row['status'] in ('good', 'warning')
    assert row['pattern']['designation'] == 'ISO 9409-1-50-4-M6'
    assert any('8 mm further out' in n for n in row['notes'])
    links = {b['link']: b for b in row['clearance']['links']}
    # the gripper is moved out by the adapter: it sits on it, never inside it
    for name in ('gripper_base', 'gripper_base_mount'):
        if name in links:
            assert links[name]['min_mm'] > -0.5, links[name]
    assert 'wrist_3_link' not in links
    json.dumps(B.public(row) | {'_design': None}, default=str)


def test_clearance_finds_a_link_through_the_bracket(ur5e):
    import copy
    p = copy.deepcopy(ur5e)
    from dexvantage.core.robot.model import get_model
    s = p.variant.sensors[1]                                    # the overhead D435, on the cage
    m = get_model(p, p.robots[0].uid)
    T = m.link_pose(m.home, 'forearm_link')
    s.mount.pose.position = T[:3, 3] + np.array([0.0, 0.0, 0.0])
    row = B.build(p, s, B.BracketSpec(kind='tslot2020'), mesh_it=False, samples=4)
    cl = row['clearance']
    assert cl['status'] == 'critical' and cl['links'][0]['min_mm'] < 0 and row['status'] == 'critical'


def test_cable_clearance(ur5e, monkeypatch):
    import copy
    p = copy.deepcopy(ur5e)
    s = p.variant.sensors[1]
    D = B.design(B.BracketSpec(kind='tslot2020'), B.camera_geom(s))
    pl = B.placement(p, s, B.BracketSpec(kind='tslot2020'), B.camera_geom(s))
    solid = B.body_solid(D)
    assert B.cable_clearance(p, s, solid, pl['T_parent_bracket'])['checked'] == 0
    c = pl['T_parent_bracket'] @ np.array([0, 0, 10 / 1000.0, 1.0])        # through the base
    p.cables = [{'name': 'USB', 'points': [list(c[:3] - [0.2, 0, 0]), list(c[:3] + [0.2, 0, 0])], 'diameter_mm': 5}]
    r = B.cable_clearance(p, s, solid, pl['T_parent_bracket'])
    assert r['checked'] == 1 and r['min_mm'] < 0 and r['cable'] == 'USB'


def test_suggestions(ur5e):
    k0, _ = B.suggest_kind(ur5e, ur5e.variant.sensors[0])
    k1, _ = B.suggest_kind(ur5e, ur5e.variant.sensors[1])
    assert (k0, k1) == ('flange', 'tslot2020')
    from dexvantage.core.robot.samples import sample_project
    so = sample_project('so100')
    kinds = [B.suggest_kind(so, s)[0] for s in so.variant.sensors]
    assert kinds == ['', 'none']


# ====================================================================== the study, export and batch
def test_study_and_export(ur5e, tmp_path):
    import copy
    from dexvantage.core import jobs
    p = copy.deepcopy(ur5e)
    p.analysis.studies['brackets'] = {'samples': 6}
    doc = jobs.run_study('brackets', p)
    assert doc['rig']['brackets'] == 0 and doc['rig']['unset'] == 2 and len(doc['suggestion']) == 2
    p.variant.sensors[0].mount.bracket = 'flange'
    p.variant.sensors[1].mount.bracket = 'tslot2020'
    doc = jobs.run_study('brackets', p)
    json.dumps(doc)
    assert doc['rig']['brackets'] == 2 and doc['suggestion'] is None
    for r in doc['sensors']:
        assert r['mesh']['ok'] and r['display']['triangles'] > 100 and len(r['placement']['T_parent_bracket']) == 4
    s = p.variant.sensors[1]
    row = B.build(p, s, B.spec_for(p, s), clear=False)
    paths = B.export_stl(row, p, str(tmp_path / 'overhead.stl'))
    V, F, _N, _h = M.read_stl(open(paths[0], 'rb').read())
    assert M.check(V, F)['ok']
    side = json.load(open(paths[1]))
    assert side['stl'] == 'overhead.stl' and side['provenance']['project']['name'] == p.name
    assert side['profile']['source']['url'].startswith('https://www.motedis')
    assert len(side['stl_sha256']) == 64


def test_batch_flag(tmp_path):
    import subprocess
    import sys
    out = tmp_path / 'b.json'
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--brackets', str(out)],
                       capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d['study'] == 'brackets' and d['rig']['unset'] == 2 and 'provenance' in d


# ====================================================================== the page
def test_brackets_page(win, app, tmp_path):
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.viewport3d import LineDraw, TriDraw
    win.store.set_project(sample_project('ur5e'))
    win.ctx.analysis.wait(10)
    p = win.store.project
    p.analysis.studies['brackets'] = {'samples': 4}
    win.open_page('brackets')
    pump(app, 0.05)
    page = win.pages['brackets']
    assert page.c_sensor.count() == 2
    page.run()
    win.ctx.analysis.wait(120)
    pump(app, 0.05)
    assert page.doc is not None and page.b_apply.isEnabled()
    n_undo = win.store.undo.count()
    page.apply()                                        # sets both suggested types: one undo step
    assert win.store.undo.count() == n_undo + 1
    assert [s.mount.bracket for s in win.store.project.variant.sensors] == ['flange', 'tslot2020']
    win.ctx.analysis.wait(180)
    pump(app, 0.05)
    assert page.doc['rig']['brackets'] == 2 and page.table.topLevelItemCount() == 2
    assert page.head.figure.text() != '—'
    items = page.overlays(win.viewport)
    assert any(isinstance(i, TriDraw) for i in items) and any(isinstance(i, LineDraw) for i in items)
    paths = page.export_stl(str(tmp_path / 'wrist.stl'))
    assert paths and os.path.getsize(paths[0]) > 84 and os.path.exists(paths[1])
    page._opt('standoff_mm', 25.0)                      # an option: one undo step
    assert win.store.project.analysis.studies['brackets']['sensors'][page.sel_uid]['standoff_mm'] == 25.0
    win.store.undo.undo()
    assert 'sensors' not in (win.store.project.analysis.studies.get('brackets') or {})
    win.store.undo.undo()
    assert [s.mount.bracket for s in win.store.project.variant.sensors] == ['', '']
    win.ctx.analysis.wait(60)
