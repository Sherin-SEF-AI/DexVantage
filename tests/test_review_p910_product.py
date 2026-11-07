"""Product review of Phases 9 and 10 (the new samples, the Menagerie catalogue, the library, tolerances, experiments,
the optimiser, requirements, exports, Rerun, the API, plugins, diff and merge, citations, the Assistant and the MCP
server, run as a robotics research team would, with bad inputs): the fixes it made.

- core/factors.apply refuses a value that is not a finite number ('high', NaN, infinity), an unknown library device
  and a stream that is not [width, height], in a sentence (it raised a raw ValueError, or stored NaN or a model id
  that does not exist); the API's dv.open and dv.sample raise DexVantageError like every other API call; rig.save to
  a folder writes the unpacked form (it wrote a sibling file named after the folder).
- --reproduce of a file that names no study says so (it said 'cannot re-run "this result"', twice);
  --optimise-settings '[1]' says it must be an object (it looked for a file named "[1]").
- A bound that does not settle a requirement has no margin (the table showed a margin that read like a failure) and
  its sentence rounds the figure like the table.
- A Sobol (Saltelli) experiment with too few base samples says its indices are rough (estimates outside [0, 1] were
  shown without a word).
- The MCP server and the assistant's open_project take every sample (aloha, panda-bimanual, g1 were refused), and a
  missing file is "There is no file at …"; the tolerance Monte Carlo, experiments, the optimiser and the design agent
  are long work for the assistant (an approved plan runs them; a chat question ran them for minutes).
- The bimanual Panda sample's Diffusion Policy takes the two wrist views (it opened with "3 of 2 cameras assigned:
  too many").
- The Menagerie import compiles the model on the dialog's worker thread (0.14 s on the UI thread for the Koch arm,
  seconds for a humanoid) and a catalogue import is named as the catalogue names the robot ("Alexander Koch v1.1 …",
  not "low_cost_robot_arm").
- Opening an analysis page makes the Analysis dock tall enough to read the page under its headline; empty selectors
  say why they are empty (a blank box before); the VLA page's empty warnings read "No warnings." (it read "None.").
- A mount tolerance the user set is no longer labelled "(default)" in the Inspector and on the Sensors page; the 3D
  view's header says "1 sensor" (it said "1 sensors"); the assistant's overview counts moving joints as the Outliner
  does (it said 44 joints for the G1 where the window says 28); the bracket export's refusal names the menu.
"""
import math
import os
import time

import numpy as np
import pytest

from conftest import pump
from dexvantage.core import factors as F
from dexvantage.core import jobs
from dexvantage.core.project import Project, Sensor


def _proj():
    p = Project('Review cell')
    p.variant.sensors.append(Sensor(name='Wrist', model_id='intel-realsense-d405'))
    return p


# ============================================================================ factors and the API
@pytest.mark.parametrize('value, words', [('high', 'takes a number'), (float('nan'), 'finite number'),
                                          (float('inf'), 'finite number'), (None, 'takes a number')])
def test_factor_values_that_are_not_numbers_are_refused(value, words):
    p = _proj()
    with pytest.raises(F.FactorError, match=words):
        F.apply(p, {'sensor:Wrist.z': value})
    with pytest.raises(F.FactorError, match=words):
        F.apply(p, {'sensor:Wrist.dpitch': value})
    q = F.apply(p, {'sensor:Wrist.z': np.float64(0.4), 'sensor:Wrist.pitch': 10, 'sensor:Wrist.dx': '0.25'})
    assert F.get(q, 'sensor:Wrist.z') == pytest.approx(0.4) and F.get(q, 'sensor:Wrist.pitch') == pytest.approx(10)
    assert F.get(q, 'sensor:Wrist.x') == pytest.approx(0.25)


def test_unknown_device_and_bad_stream_are_refused():
    p = _proj()
    with pytest.raises(F.FactorError, match='no device with the id "nope-device"'):
        F.apply(p, {'sensor:Wrist.model': 'nope-device'})
    assert F.get(F.apply(p, {'sensor:Wrist.model': 'intel-realsense-d435'}), 'sensor:Wrist.model') == \
        'intel-realsense-d435'
    assert F.get(F.apply(p, {'sensor:Wrist.model': ''}), 'sensor:Wrist.model') == ''
    with pytest.raises(F.FactorError, match=r'\[width, height\]'):
        F.apply(p, {'sensor:Wrist.stream': ['wide', 'tall']})
    with pytest.raises(F.FactorError, match='mapping'):
        F.apply(p, [('sensor:Wrist.z', 1.0)])


def test_api_errors_are_one_type_and_save_to_a_folder_unpacks(tmp_path):
    import dexvantage as dv
    with pytest.raises(dv.DexVantageError, match='no sample named "nope"'):
        dv.sample('nope')
    with pytest.raises(dv.DexVantageError, match='There is no file'):
        dv.open(str(tmp_path / 'missing.dexv'))
    rig = dv.new('Saved cell')
    folder = tmp_path / 'cell'
    folder.mkdir()
    out = rig.save(str(folder))
    assert out == str(folder) and (folder / 'project.json').is_file()
    assert not (tmp_path / 'cell.dexv').exists()
    assert dv.open(str(folder)).name == 'Saved cell'
    with pytest.raises(dv.DexVantageError, match='Could not write'):
        rig.save(str(tmp_path / 'no-such-folder' / 'x.dexv'))


# ============================================================================ batch words
def test_reproduce_of_a_file_without_a_study_says_so(tmp_path):
    from dexvantage.core import reproduce as RP
    f = tmp_path / 'empty.json'
    f.write_text('{}')
    rep = RP.reproduce_file(str(f), _proj(), [])
    lines = RP.lines(rep)
    assert 'names no study' in lines[0] and 'this result' not in ' '.join(lines)
    assert sum('names no study' in x for x in lines) == 1


def test_optimise_settings_that_are_json_but_not_an_object():
    from dexvantage.core.optimise import search
    with pytest.raises(jobs.StudyError, match='must be a JSON object'):
        search.read_settings('[1]')
    with pytest.raises(jobs.StudyError, match='could not read'):
        search.read_settings('settings-file-that-is-not-there.json')
    assert search.read_settings('{"budget": 8}') == {'budget': 8}


def test_an_undecided_bound_has_no_margin_and_rounds_like_the_table():
    from dexvantage.core import requirements as RQ
    status, margin, why = RQ.decide(64.72434567, 'at_most', '<=', 2.0)
    assert status == 'not_evaluated' and margin is None
    assert 'at most 64.72' in why and '64.7243' not in why
    assert RQ.decide(1.5, 'at_most', '<=', 2.0)[:2] == ('pass', pytest.approx(0.5))


# ============================================================================ experiments: rough Sobol indices
@jobs.study('_rv910_noise', 'Review noise toy', quick=False)
def _noise(project, ctx):
    s = project.variant.sensors[0]
    x, z = (float(v) for v in s.mount.pose.position[[0, 2]])
    return {'v': math.sin(9e3 * x) * math.cos(7e3 * z) + 0.3 * math.sin(5e3 * x * z)}


@pytest.fixture
def _noise_metric():
    from dexvantage.core import metrics as M
    M.metric('_rv910.v', 'Review noise', '', 'higher', study='_rv910_noise')(lambda d: d['v'])
    yield
    M.METRICS.pop('_rv910.v', None)


def test_a_small_saltelli_design_says_its_indices_are_rough(_noise_metric):
    from dexvantage.core import experiments as EX
    d = {'factors': [{'path': 'sensor:Wrist.x', 'lo': 0.0, 'hi': 0.3}, {'path': 'sensor:Wrist.z', 'lo': 0.0,
                                                                        'hi': 0.3}],
         'responses': ['_rv910.v'], 'design': {'method': 'saltelli', 'n': 4, 'seed': 2},
         'analysis': {'bootstrap': 100}}
    doc = EX.run(_proj(), d)
    so = doc['analysis']['_rv910.v']['sobol']
    est = so['S1'] + so['ST']
    widest = max(c[1] - c[0] for c in so['S1_ci'] + so['ST_ci'])
    assert min(est) < 0 or max(est) > 1 or widest > 0.5                 # the case the note is for
    note = next(n for n in doc['notes'] if 'Sobol indices are rough' in n)
    assert 'N = 4' in note and 'sampling error' in note


def teardown_module(_m):
    jobs.STUDIES.pop('_rv910_noise', None)


# ============================================================================ the assistant and the MCP server
def test_every_sample_opens_through_the_mcp_server_and_the_assistant(tmp_path):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    from dexvantage.core.robot.samples import SAMPLES
    from dexvantage.mcp import service
    tool = next(t for t in service.MCP_TOOLS if t.name == 'open_project')
    assert set(tool.schema['properties']['sample']['enum']) == set(SAMPLES)
    s = FileSession()
    r = s.open(path=str(tmp_path / 'gone.dexv'))
    assert r['error'].startswith('There is no file at gone.dexv')
    r = s.open()
    assert 'panda-bimanual' in r['error'] and 'g1' in r['error']
    for sid in ('tolerance', 'experiments', 'optimise', 'design_agent'):
        assert sid in T.LONG


def test_mcp_server_accepts_the_new_samples():
    import subprocess
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    r = subprocess.run([sys.executable, '-m', 'dexvantage.mcp', '--help'], cwd=root, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0
    assert 'aloha' in r.stdout and 'panda-bimanual' in r.stdout and 'g1' in r.stdout


# ============================================================================ the bimanual Panda sample's policy
def test_panda_sample_policy_takes_its_two_wrist_views():
    from dexvantage.core import policies as PO
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('panda-bimanual')
    pol = p.policies[0]
    names = {s.uid: s.name for s in p.variant.sensors}
    assert [names[u] for u in pol.cameras] == ['Wrist D405, left (sample placement)',
                                               'Wrist D405, right (sample placement)']
    n, _src = PO.resolve(p, pol.uid)['n_cameras']
    assert n == 2 and len(PO.policy_cameras(p, pol)) == n


# ============================================================================ the window
def test_opening_a_page_makes_the_analysis_dock_readable(app, win):
    win.resize(1680, 1000)
    pump(app, 0.1)
    bd = win.docks['bottom']
    win.resizeDocks([bd], [180], __import__('PyQt6.QtCore', fromlist=['Qt']).Qt.Orientation.Vertical)
    pump(app, 0.1)
    assert bd.height() < 300
    win.open_page('coverage')
    pump(app, 0.1)
    assert bd.height() >= 300
    h = bd.height()
    win.open_page('depth')                                    # a dock the user made tall stays as it is
    pump(app, 0.1)
    assert bd.height() == h


def test_empty_selectors_say_why(app, win):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('so100'))
    win.ctx.analysis.wait(30)
    pump(app, 0.05)
    lt = win.pages['lighting']
    assert lt.c_cam.count() == 0 and lt.c_cam.placeholderText()
    assert lt.obj_box.count() == 0 and 'task object' in lt.obj_box.placeholderText()
    assert win.pages['cables'].cable_box.placeholderText()
    assert win.pages['handeye_solve'].session_box.placeholderText()
    assert win.pages['audit'].ds_box.placeholderText()
    assert win.pages['time_sync'].clock_list.placeholderText()
    vla = win.pages['vla']
    vla.show_result({'warnings': [], 'policies': [], 'cameras': [], 'headline': {}})
    assert vla.warn_label.text() in ('No warnings.', '')


def _wait(app, cond, seconds=20):
    t = time.time()
    while not cond() and time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.005)
    return cond()


def test_menagerie_import_runs_off_the_ui_thread_and_keeps_the_catalogue_name(app, win, tmp_path):
    from test_menagerie import FakeGitHub

    from dexvantage.ui.dialogs.menagerie import MenagerieDialog
    d = MenagerieDialog(win, opener=FakeGitHub(), dest=str(tmp_path), background=True)
    d.robot_name = 'Fake Robotics Fake Arm'
    d.load_index()
    assert _wait(app, lambda: d.view.count() == 2)
    d.select_model('fake_arm')
    d.fetch_selected()
    assert _wait(app, lambda: d.dl is not None)
    d.accept_box.setChecked(True)
    n = win.store.undo.count()
    assert d.import_selected() is False                       # started: the model compiles on the worker
    assert not d.import_btn.isEnabled()                      # no second import while it runs
    assert _wait(app, lambda: d.robot is not None)
    p = win.store.project
    assert [r.name for r in p.robots] == ['Fake Robotics Fake Arm']
    assert win.store.undo.count() == n + 1
    win.store.undo.undo()
    assert not win.store.project.robots
    d.deleteLater()


def test_catalogue_dialog_names_the_robot_after_the_entry(win, tmp_path):
    from test_menagerie import FakeGitHub

    from dexvantage.core.robot import catalogue as CAT
    from dexvantage.ui.dialogs import robot_catalogue as RC
    d = RC.import_entry(win, 'the-robot-studio-so-arm100', opener=FakeGitHub(), dest=str(tmp_path),
                        background=False, run=False)
    assert d.robot_name == CAT.get('the-robot-studio-so-arm100').title
    d.deleteLater()



def test_a_set_mount_tolerance_is_not_labelled_default(app, win):
    from PyQt6.QtWidgets import QLabel

    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('so100'))
    win.ctx.analysis.wait(30)
    s = win.store.project.variant.sensors[0]
    win.store.select(('sensor', s.uid))
    pump(app, 0.05)

    def shown():
        return [lab.text() for w in (win.inspector, win.pages['sensors']) for lab in w.findChildren(QLabel)
                if '± ' in lab.text() and ' mm · ± ' in lab.text()]
    win.pages['sensors'].refresh()
    assert shown() and all(t.endswith('(default)') for t in shown())
    assert win.pages['tolerance'].set_tolerance(s.uid, 'position_mm', 3.0)
    win.pages['sensors'].refresh()
    pump(app, 0.05)
    assert any(t.startswith('± 3 mm') for t in shown())
    assert not any('(default)' in t for t in shown() if t.startswith('± 3 mm'))
    win.store.undo.undo()


def test_overview_counts_moving_joints_and_the_header_counts_sensors(app, win):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    from dexvantage.core.jobs import MOVING_JOINTS
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    s = FileSession(project=p)
    rb = T.call(s, 'get_overview', {})['robots'][0]
    assert rb['joints'] == sum(j.get('type') in MOVING_JOINTS for j in p.robots[0].joints)
    p.variant.sensors[1:] = []
    win.store.set_project(p)
    win.ctx.analysis.wait(30)
    pump(app, 0.05)
    hud = win.scene.hud_label(win.viewport)
    assert hud.endswith('· 1 sensor')
