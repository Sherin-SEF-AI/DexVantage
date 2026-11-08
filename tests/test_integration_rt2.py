"""Research tools part 2, integration: the hooks between the tracks (the publication kit's statistics, cells and runs
tables; the tracker logging cluster runs; the experiment best run's bound in the tracker; the assistant's and MCP's
tool registry; the report's 'all' leaving out the heavy research studies; the exact occlusion cube-map key)."""
import json
import os
import subprocess
import sys

import pytest

from dexvantage.core import jobs, tracking as TR
from dexvantage.core.project import Project, blank_project

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM = 'sensor:Third-person camera (sample placement)'

# the tools the assistant registry holds after research tools part 2 (the MCP server adds its own six)
RT2_TOOLS = {'get_runs', 'get_statistics', 'get_campaign', 'get_sim2real', 'get_methods_paragraph', 'get_cells',
             'get_cluster_job', 'get_rig_cost', 'check_robot_file'}
# Phase 11 added get_project_diff (what changed against another project file or variant)
N_TOOLS = 34
MCP_ONLY = {'dexvantage_status', 'get_history', 'open_project', 'redo_change', 'run_batch', 'save_project'}
N_MCP_TOOLS = N_TOOLS + len(MCP_ONLY)


def _env():
    env = dict(os.environ)
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    return env


def _cli(args, cwd):
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, timeout=600,
                          env=_env(), cwd=cwd)


@pytest.fixture
def saved(tmp_path):
    p = blank_project('Cell for the kit')
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    return Project.load(path)


def _stats_doc(project):
    from dexvantage.core.studies import stats as SS
    st = SS.settings_of({'source': 'rates', 'rates': [{'condition': 'Policy A', 'successes': 31, 'trials': 50},
                                                      {'condition': 'Policy B', 'successes': 42, 'trials': 50}]})
    return {'study': 'stats', **SS.analyse(project, st)}


def _cells_doc():
    return {'study': 'cells', 'reference': 0,
            'cells': [{'name': 'Lab A'}, {'name': 'Lab B'}, {'name': 'Lab C'}],
            'metrics': {'rows': [{'id': 'coverage.covered', 'title': 'Task volume seen', 'unit': '%',
                                  'values': [96.66, 94.5, None], 'bounds': [None, 'at_least', None]}]},
            'gap': {'pairs': [{'b': 'Lab B', 'total': 0.2134}, {'b': 'Lab C', 'total': 0.5}]}}


# ====================================================================== the publication kit
def test_paper_kit_writes_statistics_cells_and_runs_tables_that_trace(tmp_path, saved):
    from dexvantage.core import publication as PB
    saved.analysis.studies['tracking'] = {'enabled': True}
    rig = jobs.run_study('rig', saved.snapshot())
    TR.log('rig', rig, saved, interface='api', tags=['kit'])
    results = {'rig': rig, 'stats': _stats_doc(saved), 'cells': _cells_doc()}
    kit = PB.build(saved, {'results': results, 'studies': []})
    parts = kit['parts']
    assert {'statistics', 'cells', 'runs'} <= set(parts), sorted(parts)
    assert kit['checks']['tables']['problems'] == [], kit['checks']['tables']['problems']
    assert all(not v for v in kit['checks']['latex'].values()), kit['checks']['latex']
    st = parts['statistics']
    assert 'Policy B against Policy A' in st and 'Fisher' in st and r'\dvnum{62}' in st and r'\dvnum{84}' in st
    assert r'\dvnum{22}' in st                                     # the risk difference in percentage points
    assert r'$\geq$ \dvnum{94.5}' in parts['cells'] and 'not stated' in parts['cells']
    assert r'\dvnum{0.213}' in parts['cells'] and '(reference)' in parts['cells']
    assert 'Cell for the kit' not in parts['runs'] and 'finished' in parts['runs']
    assert kit['checks']['tables']['numbers'] >= 10
    meth = parts['methods']
    for t in ('statistics', 'cells', 'runs'):
        assert f'Table~\\ref{{tab:{t}}}' in meth
    main = PB.main_tex(kit['_st'], saved, parts, [], {})
    for name in ('statistics', 'cells', 'runs'):
        assert f'\\input{{tables/{name}}}' in main
    # a wrong number in a table is caught by the table check
    bad = dict(parts)
    bad['statistics'] = st.replace(r'\dvnum{84}', r'\dvnum{85}', 1)
    assert PB.verify_tables({'statistics': bad['statistics']}, kit['_source'])['problems']


def test_paper_kit_without_research_results_has_no_research_tables(saved):
    from dexvantage.core import publication as PB
    kit = PB.build(saved, {'results': {'rig': jobs.run_study('rig', saved.snapshot())}, 'studies': []})
    assert not {'statistics', 'cells', 'runs'} & set(kit['parts'])


def test_report_all_leaves_out_the_heavy_research_studies():
    from dexvantage.core.exporters import report as R
    assert {'cells', 'stats', 'sim2real'} <= set(R.META_STUDIES)


# ====================================================================== the tracker
def test_tracker_keeps_the_experiment_best_runs_bound_and_settled_flag():
    from dexvantage.core.tracking import record as RC
    out = {}
    RC._experiment({'baseline': {}, 'best': {'response': 'coverage.covered', 'run': 3, 'value': 90.0, 'unit': '%',
                                             'better': 'higher', 'bound': 'at_least', 'settled': False}}, out)
    row = out['coverage.covered@best']
    assert row['bound'] == 'at_least' and row['settled'] is False and row['value'] == 90.0


def test_cluster_runs_are_logged_in_the_run_store(tmp_path):
    design = tmp_path / 'design.json'
    design.write_text(json.dumps({'factors': [{'path': CAM + '.pitch', 'lo': -5, 'hi': 5}],
                                  'responses': ['coverage.covered'], 'design': {'method': 'lhs', 'n': 2, 'seed': 1},
                                  'analysis': {'bootstrap': 50}}))
    r = _cli(['--sample', 'so100', '--experiment', str(design), '--jobs', '2', '--experiment-out',
              str(tmp_path / 'e.json'), '--track', str(tmp_path / 'runs'), '--track-tag', 'cluster'], str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert 'logged run' in r.stdout and 'Run store runs: 1 run' in r.stdout
    rows = TR.RunStore(str(tmp_path / 'runs')).runs()
    assert len(rows) == 1 and rows[0]['study'] == 'experiments' and rows[0]['kind'] == 'experiment'
    assert rows[0]['tags'] == ['cluster'] and rows[0]['status'] == 'finished'
    full = TR.RunStore(str(tmp_path / 'runs')).get(rows[0]['run_id'])
    assert 'coverage.covered@best' in full['metrics']
    r = _cli(['--sample', 'so100', '--experiment', str(design), '--jobs', '0'], str(tmp_path))
    assert r.returncode == 2 and 'Traceback' not in r.stderr and len(r.stderr.strip().splitlines()) == 1


def test_dv_cluster_logs_when_the_projects_tracking_is_on(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    rig.save(str(tmp_path / 's.dexv'))
    rig = dv.open(str(tmp_path / 's.dexv'))
    rig.project.analysis.studies['tracking'] = {'enabled': True}
    design = {'factors': [{'path': CAM + '.pitch', 'lo': -3, 'hi': 3}], 'responses': ['coverage.covered'],
              'design': {'method': 'lhs', 'n': 2, 'seed': 3}, 'analysis': {'bootstrap': 50}}
    res = dv.cluster.experiment(rig, design, jobs=1)
    assert res.study == 'experiments'
    runs = dv.runs(rig)
    assert len(runs) == 1 and list(runs['study']) == ['experiments']


# ====================================================================== the assistant and MCP
def test_tool_registry_has_every_research_tool_and_the_count_is_known():
    from dexvantage.assistant import tools as T
    from dexvantage.mcp.server import Router, Server
    assert RT2_TOOLS | {'get_project_diff'} <= set(T.TOOLS)
    assert len(T.TOOLS) == N_TOOLS, sorted(T.TOOLS)
    srv = Server(Router(mode='file'))
    tools = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})['result']['tools']
    assert len(tools) == N_MCP_TOOLS and {t['name'] for t in tools} == set(T.TOOLS) | MCP_ONLY
    by = {t['name']: t for t in tools}
    for name in RT2_TOOLS | {'get_project_diff'}:
        assert by[name]['annotations']['readOnlyHint'] is True, name


def test_check_robot_file_tool_lists_references_and_refuses_plainly(tmp_path):
    from dexvantage.assistant import tools as T
    urdf = tmp_path / 'arm.urdf'
    urdf.write_text('<robot name="arm"><link name="base"><visual><geometry><mesh filename="meshes/base.stl"/>'
                    '</geometry></visual></link></robot>')
    out = T.call(None, 'check_robot_file', {'path': str(urdf)})
    assert out['kind'] == 'urdf' and out['references'] == 1 and out['found'] == 0
    assert out['missing_required'][0]['ref'].endswith('base.stl') and str(tmp_path) not in json.dumps(out)
    err = T.call(None, 'check_robot_file', {'path': str(tmp_path / 'nope.urdf')})
    assert 'error' in err and 'Traceback' not in err['error']
    assert 'error' in T.call(None, 'check_robot_file', {})


def test_every_new_page_has_a_guide():
    from dexvantage.assistant import page_guide as G
    for pid in ('runs', 'statistics', 'campaign', 'sim2real', 'publication', 'cells'):
        assert G.GUIDES[pid].purpose


# ====================================================================== occlusion cache key
def test_cube_cache_keys_the_origin_exactly():
    from dexvantage.core import occlusion as OC
    a = OC.CubeCache.key('d', (0.1, 0.2, 0.3), 64, 0.005, 'numpy')
    b = OC.CubeCache.key('d', (0.1 + 2e-5, 0.2, 0.3), 64, 0.005, 'numpy')
    assert a != b and a == OC.CubeCache.key('d', [0.1, 0.2, 0.3], 64, 0.005, 'numpy')


# ====================================================================== API surface
def test_every_research_tool_is_reachable_from_the_headless_api():
    code = ('import sys, dexvantage as dv\n'
            'for n in ("runs", "track", "run_store", "stats", "campaign", "cells", "cluster", "publication", '
            '"robot_io", "prices", "price"):\n'
            '    getattr(dv, n)\n'
            'assert hasattr(dv.Rig, "sim2real") and hasattr(dv.Rig, "gap_dataset")\n'
            'assert not any(m.startswith("PyQt6") for m in sys.modules), [m for m in sys.modules if "PyQt6" in m]\n')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr


# ====================================================================== the window logs while the Runs page is hidden
def test_window_logs_runs_while_the_runs_page_is_hidden(app, win, tmp_path):
    from conftest import pump
    p = blank_project('Hidden runs page')
    p.analysis.studies['tracking'] = {'enabled': True}
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    win.store.set_project(Project.load(path))
    page = win.pages.get('runs')                       # .get: the page stays held while hidden (pages/_gate.py)
    win.open_page('coverage')
    assert not page.isVisible()
    win.ctx.analysis.run('rig')
    for _ in range(6):
        win.ctx.analysis.wait(120)
        page.wait(60)
        pump(app, 0.05)
    rows = TR.store_for(win.store.project).runs()
    assert [r['study'] for r in rows] == ['rig'] and rows[0]['interface'] == 'app'


def test_batch_stats_campaign_cells_and_plan_steps_are_logged_with_track(tmp_path):
    from dexvantage.core.robot.samples import sample_project
    sample_project('so100').save(str(tmp_path / 'cell.dexv'))
    (tmp_path / 'rates.json').write_text(json.dumps({
        'schema': 'dexvantage.stats/1', 'source': 'rates',
        'rates': [{'condition': 'A', 'successes': 14, 'trials': 20}, {'condition': 'B', 'successes': 19,
                                                                       'trials': 20}]}))
    (tmp_path / 'camp.json').write_text(json.dumps({'schema': 'dexvantage.campaign_spec/1', 'operators': ['Ana']}))
    (tmp_path / 'plan.json').write_text(json.dumps({'name': 'p', 'steps': [{'study': 'rig', 'out': 'rig.json'}]}))
    cwd = str(tmp_path)
    for args in (['--stats', 'rates.json'], ['--campaign', 'camp.json'],
                 ['--cells', 'sample:ur5e', '--cells-out', 'cells.json', '--cells-settings',
                  '{"metrics": ["rig.camera_count"], "variants": false}'],
                 ['--run-plan', 'plan.json', 'planout']):
        r = _cli(['cell.dexv', *args, '--track', 'runs'], cwd)
        assert r.returncode in (0, 1), (args, r.stderr)
        assert 'logged run' in r.stdout + r.stderr, (args, r.stdout, r.stderr)
    r = _cli(['cell.dexv', '--stats', 'nowhere.json', '--track', 'runs'], cwd)
    assert r.returncode == 2 and 'Traceback' not in r.stderr
    rows = TR.RunStore(str(tmp_path / 'runs')).runs()
    assert [x['study'] for x in rows] == ['stats', 'campaign', 'cells', 'rig', 'stats']
    assert rows[-1]['status'] == 'failed' and rows[-1]['reason']


# ====================================================================== the LeRobot reader's decoder retry
def test_frames_the_threaded_decoder_drops_are_decoded_again_single_threaded(tmp_path, monkeypatch):
    # a full run at this integration saw "2 frame(s) … could not be decoded" from the frame-threaded AV1 decoder on a
    # file that decodes cleanly alone: frames it silently does not return are tried once more without threads
    import numpy as np
    av = pytest.importorskip('av', reason='av (the lerobot extra) is not installed')
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    from dexvantage.core import datasets as DS
    from dexvantage.core.datasets import lerobot as L
    from dexvantage.validation import checks_formats as CF
    root = str(tmp_path / 'ds')
    L.write_lerobot(root, CF.lerobot_reference_episodes(np.random.default_rng(7)), 'v2.1', 10, 'reference_arm',
                    CF.LEROBOT_EXTRA)
    ds = DS.open_dataset(root)
    cam = next(iter(ds.load(ds.episodes()[0].index).cameras.values()))
    clean = [i for i, _ in cam.frames()]
    real_open, calls = av.open, []

    class Dropping:                                 # a container whose threaded decoder ends after two frames
        def __init__(self, c):
            self.c = c

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self.c.__exit__(*a)

        def __getattr__(self, k):
            return getattr(self.c, k)

        def decode(self, s):
            calls.append(s.thread_type)
            for k, fr in enumerate(self.c.decode(s)):
                if str(s.thread_type).upper().endswith('AUTO') and k == 2:
                    return
                yield fr
    monkeypatch.setattr(av, 'open', lambda *a, **k: Dropping(real_open(*a, **k).__enter__()))
    got = [i for i, _ in cam.frames()]
    assert got == clean and len(calls) == 2 and 'NONE' in str(calls[1]).upper()


def test_a_decoder_that_fails_after_every_seek_is_read_from_the_start(tmp_path, monkeypatch):
    # the final gate's own full run (2026-11-08) saw "Invalid data found when processing input" escape both the
    # threaded and the single-threaded pass on a file that decodes cleanly alone: a third pass reads from the start
    # with no seek
    import numpy as np
    av = pytest.importorskip('av', reason='av (the lerobot extra) is not installed')
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    from dexvantage.core import datasets as DS
    from dexvantage.core.datasets import lerobot as L
    from dexvantage.validation import checks_formats as CF
    root = str(tmp_path / 'ds')
    L.write_lerobot(root, CF.lerobot_reference_episodes(np.random.default_rng(7)), 'v2.1', 10, 'reference_arm',
                    CF.LEROBOT_EXTRA)
    ds = DS.open_dataset(root)
    cam = next(iter(ds.load(ds.episodes()[0].index).cameras.values()))
    clean = [i for i, _ in cam.frames()]
    real_open, calls = av.open, []

    class FailsAfterSeek:
        def __init__(self, c):
            self.c, self.seeked = c, False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self.c.__exit__(*a)

        def __getattr__(self, k):
            return getattr(self.c, k)

        def seek(self, *a, **k):
            self.seeked = True
            return self.c.seek(*a, **k)

        def decode(self, s):
            calls.append((str(s.thread_type), self.seeked))
            if self.seeked:
                raise av.error.InvalidDataError(1094995529, 'Invalid data found when processing input',
                                                'avcodec_send_packet()')
            yield from self.c.decode(s)
    monkeypatch.setattr(av, 'open', lambda *a, **k: FailsAfterSeek(real_open(*a, **k).__enter__()))
    got = [i for i, _ in cam.frames()]
    assert got == clean
    assert len(calls) == 3 and calls[-1][1] is False and 'NONE' in calls[-1][0].upper()
