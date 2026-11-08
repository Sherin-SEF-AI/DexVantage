"""scripts/benchmarks.py: the machine description names no host or user, the children run with throw-away settings
and no API keys, and docs/performance.md is made from the result files with each §16 budget's verdict."""
import getpass
import json
import os
import re
import socket
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import benchmarks as B  # noqa: E402


def results(grasp_s=(90.0, 240.0), ui_ms=(40.0, 130.0)):
    env = {'machine': {'cpu': 'Test CPU 9000', 'logical_cpus': 8, 'physical_cores': 4, 'ram_gb': 16.0,
                       'os': 'Test OS 1', 'kernel': 'Linux 6.0', 'machine': 'x86_64',
                       'nvidia_smi': [{'name': 'Test GPU', 'driver': '1.0', 'memory': '8 GiB'}]},
           'software': {'python': '3.14.0', 'implementation': 'CPython', 'packages': {'numpy': '2.0'},
                        'gl': {'renderer': 'Test GPU/PCIe', 'how': 'EGL device 0', 'software': False}}}
    grasp = {'budget_s': B.GRASP_BUDGET_S, 'grasps': 2000, 'rows': [
        {'sample': name, 'robot': 'arm', 'cameras': 2, 'grasps': 2000, 'feasible': 1000, 'study_s': t,
         'grasps_per_s': 2000 / t, 'timings': {'ik_seconds': 10.0, 'occlusion_seconds': 50.0,
                                               'sensitivity_seconds': 20.0},
         'maps_built': 5000, 'maps_per_s': 150.0, 'cube_n': 512, 'renderer': 'Test GPU/PCIe',
         'run': {'peak_mb': 900.0, 'load_before': 1.0, 'load_after': 2.0},
         'within_budget': t <= B.GRASP_BUDGET_S} for name, t in zip(('ur5e', 'aloha'), grasp_s)]}
    ui = {'budget_ms': 100.0, 'samples': [
        {'sample': name, 'studies': [{'study': 'coverage', 'seconds': 0.4, 'refused': ''}], 'skipped': {},
         'stall_ms': 50.0, 'stalls': [{'ms': ms, 'during': 'coverage: the study running', 'where': 'x.py:1 f'}],
         'rows': [{'what': 'open the sample: build and show the main window (start-up)', 'max_ms': 900.0,
                   'thread': 'start'},
                  {'what': 'coverage: the result arrives (show_result)', 'max_ms': ms - 10, 'thread': 'UI'}]}
        for name, ms in zip(('ur5e', 'aloha'), ui_ms)]}
    return {'environment': env, 'grasp': grasp, 'ui': ui, '_date': '2026-11-08',
            'import': {'runs': 7, 'import_ms_median': 0.1, 'api_ms_median': 80.0, 'sample_ms_median': 300.0,
                       'process_ms_median': 500.0, 'interpreter_ms_median': 15.0, 'modules_import': 80,
                       'modules_api': 200, 'modules_sample': 300, 'imports_qt_on_import': False},
            'startup': {'rows': [{'sample': 'blank', 'first_paint_s_median': 1.9, 'first_paint_s_max': 2.0,
                                  'interpreter_s_median': 0.04, 'app_s_median': 0.3, 'peak_mb': 280.0}]},
            'validate': {'command': 'dexvantage --validate out/validation', 'run': {'wall_s': 390.0,
                         'peak_mb': 5300.0, 'load_before': 1.0, 'load_after': 1.0},
                         'summary': {'checks': 232, 'passed': 232, 'seconds_in_suite': 380.0, 'models': 37}}}


def test_the_machine_description_names_no_host_or_user():
    text = json.dumps(B.machine())
    host = socket.gethostname()
    if len(host) >= 3:
        assert not re.search(rf'\b{re.escape(host)}\b', text)
    user = getpass.getuser()
    if len(user) >= 3:
        assert not re.search(rf'\b{re.escape(user)}\b', text)
    assert '/home/' not in text and '/Users/' not in text


def test_children_get_throw_away_folders_and_no_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fake-test-key')
    env = B.child_env(str(tmp_path), 'EGL device 0')
    assert 'ANTHROPIC_API_KEY' not in env
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'TMPDIR'):
        assert env[k].startswith(str(tmp_path))
    assert env['QT_QPA_PLATFORM'] == 'offscreen' and env['DEXVANTAGE_EGL'] == 'EGL device 0'


def test_the_page_states_each_budget_and_its_verdict():
    text = B.render_doc(results())
    assert 'Test CPU 9000' in text and 'Test GPU/PCIe' in text and 'Measured on 2026-11-08' in text
    assert re.search(r'2,000-grasp study, ur5e: at most 180 s .*\| 90\.0 s \| pass \|', text)
    assert re.search(r'2,000-grasp study, aloha: at most 180 s .*\| 240\.0 s \| FAIL \|', text)
    assert re.search(r'No UI-thread step over 100 ms, ur5e .*\| pass \|', text)
    assert re.search(r'No UI-thread step over 100 ms, aloha .*130\.0 ms.*\| FAIL \|', text)
    assert 'Study (s)' in text and 'Peak memory (MB)' in text and 'How to reproduce' in text
    assert 'scripts/benchmarks.py' in text and 'Known limits' in text
    assert '232 of 232 checks passed' in text and '[validation.md](validation.md)' in text
    assert '/home/' not in text


def test_window_start_up_is_reported_not_counted_against_the_budget():
    text = B.render_doc(results(ui_ms=(40.0, 60.0)))
    assert re.search(r'No UI-thread step over 100 ms, aloha .*\| pass \|', text)
    assert '900' in text                          # building the window is listed


def test_importtime_tree_is_read_as_the_root_s_direct_children():
    err = '\n'.join(['import time: self [us] | cumulative | imported package',
                     'import time:       10 |         10 |   codecs',
                     'import time:       20 |         30 | encodings',
                     'import time:      100 |        100 |     numpy.core',
                     'import time:     1000 |       1100 |   numpy',
                     'import time:       50 |         50 |   json',
                     'import time:       10 |       1160 | dexvantage.api'])
    got = B.importtime_children(err, 'dexvantage.api')
    assert [g['module'] for g in got] == ['numpy', 'json'] and got[0]['cumulative_ms'] == 1.1


def test_a_stack_is_described_without_absolute_paths():
    s = B.describe_stack(sys._getframe())
    assert not os.path.isabs(s.split(':')[0]) and '/home/' not in s


def test_a_dexvantage_frame_is_named_even_through_an_unnormalised_path():
    fake = os.path.join(ROOT, 'scripts', '..', 'dexvantage', 'fake_module.py')
    ns: dict = {}
    exec(compile('import sys\nframe = sys._getframe()\n', fake, 'exec'), ns)
    assert B.describe_stack(ns['frame']).startswith('dexvantage/fake_module.py:2 <module>')


def test_a_stack_names_the_ui_code_that_called_the_core():
    ns: dict = {}
    core = os.path.join(ROOT, 'dexvantage', 'core', 'fake_core.py')
    ui = os.path.join(ROOT, 'dexvantage', 'ui', 'fake_page.py')
    exec(compile('import sys\ndef a():\n    return b()\ndef b():\n    return c()\ndef c():\n    return d()\n'
                 'def d():\n    return sys._getframe()\n', core, 'exec'), ns)
    exec(compile('def page():\n    return a()\n', ui, 'exec'), ns)
    s = B.describe_stack(ns['page']())
    assert s.startswith('dexvantage/core/fake_core.py:9 d < dexvantage/core/fake_core.py:7 c')
    assert ' … < dexvantage/ui/fake_page.py:2 page' in s


def test_a_stall_over_budget_is_explained_from_what_was_measured():
    res = results(ui_ms=(40.0, 300.0))
    s = res['ui']['samples'][1]
    s['stalls'] = [{'ms': 300.0, 'during': 'motion_blur: the study running', 'where': 'outside', 'ui_cpu_ms': 20.0,
                    'gc_ms': 0, 'gc_generations': []},
                   {'ms': 200.0, 'during': 'coverage: the study running', 'where': 'outside', 'ui_cpu_ms': 190.0,
                    'gc_ms': 150.0, 'gc_generations': [2]},
                   {'ms': 150.0, 'during': 'cables: the study running', 'where': 'dexvantage/core/cables.py:1 f',
                    'ui_cpu_ms': 149.0, 'gc_ms': 0, 'gc_generations': []}]
    text = B.render_doc(res)
    assert 'The UI thread used 20 ms of processor time in it' in text
    assert "Python's cycle collector (generation 2) ran for 150 ms of it" in text
    assert 'The UI thread was in `dexvantage/core/cables.py:1 f`' in text
