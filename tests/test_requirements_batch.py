"""Batch mode's readiness gate, reproduction and plans (dexvantage/batch_gate.py, core/reproduce.py, core/run_plan.py):
--check exit codes 0 / 1 / 2 with a table and JSON with provenance; --reproduce round trips on two studies and the
scorecard, and a deliberately perturbed result caught; --run-plan in order with one summary."""
import json
import os

import pytest

from dexvantage.app import main
from dexvantage.core import provenance as PV
from dexvantage.core.project import Sensor, blank_project


@pytest.fixture
def proj(tmp_path):
    p = blank_project('Gate cell')
    p.variant.sensors.append(Sensor(name='Wrist', kind='depth', model_id='intel-realsense-d405', slot=0))
    p.variant.sensors.append(Sensor(name='Top', kind='camera', model_id='', slot=1))
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    return path


def _req(tmp_path, items, name='req.yaml'):
    import yaml
    path = tmp_path / name
    path.write_text(yaml.safe_dump({'requirements': items}), encoding='utf-8')
    return str(path)


def test_check_pass_writes_the_table_and_json_with_provenance(tmp_path, proj, capsys):
    req = _req(tmp_path, [{'name': 'At most two cameras', 'metric': 'rig.camera_count', 'comparison': '<=',
                           'threshold': 2}])
    out = str(tmp_path / 'card.json')
    assert main([proj, '--check', req, out]) == 0
    text = capsys.readouterr().out
    assert 'At most two cameras' in text and 'Verdict: ready' in text and 'Scorecard → ' in text
    doc = json.load(open(out))
    assert doc['schema'] == 'dexvantage.requirements/1' and doc['verdict'] == 'pass'
    prov = doc['provenance']
    assert prov['interface'] == 'batch' and prov['command'][:4] == ['dexvantage', 'cell.dexv', '--check', 'req.yaml']
    assert prov['inputs'][0]['option'] == '--check' and prov['inputs'][0]['sha256']
    assert PV.verify(doc)['ok'] and str(tmp_path) not in open(out).read()


def test_check_fail_and_not_evaluated_exit_1(tmp_path, proj, capsys):
    req = _req(tmp_path, [{'metric': 'rig.camera_count', 'comparison': '>=', 'threshold': 3}])
    assert main([proj, '--check', req, str(tmp_path / 'a.json')]) == 1
    assert 'fail' in capsys.readouterr().out
    req = _req(tmp_path, [{'metric': 'bandwidth.margin', 'comparison': '>=', 'threshold': 0}], 'ne.yaml')
    assert main([proj, '--check', req, str(tmp_path / 'b.json')]) == 1
    text = capsys.readouterr().out
    assert 'not evaluated' in text and 'no host controller' in text


@pytest.mark.parametrize('content, words', [
    ('requirements: [{metric: nope.x, threshold: 1}]', 'does not have'),
    ('requirements: [{metric: rig.camera_count, threshold: many}]', 'not a number'),
    ('requirements: [{metric: rig.camera_count', 'not valid YAML'),
    ('requirements: []', 'names no requirement')])
def test_check_errors_exit_2_with_a_sentence(tmp_path, proj, capsys, content, words):
    path = tmp_path / 'bad.yaml'
    path.write_text(content, encoding='utf-8')
    assert main([proj, '--check', str(path), str(tmp_path / 'x.json')]) == 2
    err = capsys.readouterr().err
    assert words in err and 'Traceback' not in err
    assert not (tmp_path / 'x.json').exists()


def test_check_needs_a_project_and_a_readable_file(tmp_path, proj, capsys):
    assert main(['--check', 'defaults', str(tmp_path / 'x.json')]) == 2
    assert 'needs a project file' in capsys.readouterr().err
    assert main([proj, '--check', str(tmp_path / 'none.yaml')]) == 2
    assert 'could not be read' in capsys.readouterr().err


def _perturb(path, rehash):
    doc = json.load(open(path))
    leaf = doc
    keys = []
    # the first non-zero float of the result (not provenance, not a time): changed by one part in a million
    def find(o, trail):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ('provenance', 'seconds', 'settings'):
                    continue
                r = find(v, trail + [k])
                if r:
                    return r
        elif isinstance(o, list):
            for i, v in enumerate(o):
                r = find(v, trail + [i])
                if r:
                    return r
        elif isinstance(o, float) and o != 0:
            return trail
        return None
    keys = find(doc, [])
    assert keys
    for k in keys[:-1]:
        leaf = leaf[k]
    leaf[keys[-1]] *= 1 + 1e-6
    if rehash:
        doc['provenance']['result_sha256'] = PV.result_digest(doc)
    with open(path, 'w') as fh:
        json.dump(doc, fh)
    return '.'.join(str(k) if isinstance(k, str) else f'[{k}]' for k in keys).replace('.[', '[')


@pytest.mark.parametrize('flag', ['--bandwidth', '--time-sync'])
def test_reproduce_round_trip_and_a_perturbed_result(tmp_path, capsys, flag):
    out = str(tmp_path / 'r.json')
    assert main(['--sample', 'ur5e', flag, out]) == 0
    assert main(['--reproduce', out]) == 0                  # the sample its command names
    text = capsys.readouterr().out
    assert 'Reproduces exactly' in text and 'the ur5e sample' in text
    rep_path = str(tmp_path / 'rep.json')
    path = _perturb(out, rehash=True)                       # an honest-looking file with one number changed
    assert main(['--reproduce', out, '--reproduce-out', rep_path]) == 1
    text = capsys.readouterr().out
    assert 'Does not reproduce' in text and f'mismatch {path}' in text.replace('\n', ' ')
    rep = json.load(open(rep_path))
    assert rep['verdict'] == 'does_not_reproduce' and [m['path'] for m in rep['mismatches']] == [path]
    assert rep['integrity']['intact'] is True
    _perturb(out, rehash=False)                             # changed after writing: the hash says so too
    assert main(['--reproduce', out]) == 1
    assert 'changed after DexVantage wrote it' in capsys.readouterr().out


def test_reproduce_a_scorecard_with_the_requirements_it_records(tmp_path, proj, capsys):
    req = _req(tmp_path, [{'metric': 'rig.camera_count', 'comparison': '<=', 'threshold': 2},
                          {'metric': 'rig.mass_g', 'comparison': '<=', 'threshold': 500}])
    out = str(tmp_path / 'card.json')
    assert main([proj, '--check', req, out]) in (0, 1)
    os.remove(req)                                         # the scorecard records its requirements
    assert main([proj, '--reproduce', out]) == 0
    assert 'Reproduces exactly' in capsys.readouterr().out
    assert main(['--reproduce', out]) == 0                 # the project file it names, next to the result
    assert 'cell.dexv next to the result' in capsys.readouterr().out


def test_reproduce_cannot_is_exit_2(tmp_path, capsys):
    p = tmp_path / 'r.json'
    p.write_text(json.dumps({'schema': 'dexvantage.nothing/1', 'x': 1}))
    assert main(['--reproduce', str(p)]) == 2
    assert 'Cannot reproduce' in capsys.readouterr().out
    assert main(['--reproduce', str(tmp_path / 'none.json')]) == 2
    p.write_text('not json')
    assert main(['--reproduce', str(p)]) == 2


def test_run_plan_in_order_with_settings_and_one_summary(tmp_path, capsys):
    first = str(tmp_path / 'first.json')
    assert main(['--sample', 'ur5e', '--rig-inventory', first]) == 0
    plan = {'name': 'nightly', 'steps': [
        {'study': 'bandwidth', 'settings': {'margin_warn': 0.5}, 'out': 'bw.json'},
        {'check': [{'metric': 'rig.camera_count', 'comparison': '<=', 'threshold': 1}], 'out': 'card.json'},
        {'reproduce': 'first.json'},
        {'study': 'rig'}]}
    pp = tmp_path / 'plan.json'
    pp.write_text(json.dumps(plan))
    out_dir = str(tmp_path / 'run')
    assert main(['--sample', 'ur5e', '--run-plan', str(pp), out_dir]) == 1      # the check fails: 2 cameras
    s = json.load(open(os.path.join(out_dir, 'plan-summary.json')))
    assert s['schema'] == 'dexvantage.run_plan/1' and [r['kind'] for r in s['steps']] == \
        ['study', 'check', 'reproduce', 'study']
    assert [r['status'] for r in s['steps']] == ['done', 'fail', 'reproduces', 'done'] and s['exit'] == 1
    assert s['steps'][3]['out'] == 'step04-rig.json' and s['steps'][3]['metrics']['rig.camera_count']['value'] == 2
    bw = json.load(open(os.path.join(out_dir, 'bw.json')))
    assert bw['settings']['margin_warn'] == 0.5 and bw['provenance']['plan'] == {'plan': 'nightly', 'version': 1,
                                                                                'step': 0}
    assert PV.verify(s)['ok']
    capsys.readouterr()
    # the summary reproduces step by step, each with its own settings
    assert main(['--reproduce', os.path.join(out_dir, 'plan-summary.json')]) == 0
    assert 'All 3 step results reproduce' in capsys.readouterr().out


def test_run_plan_errors(tmp_path, capsys):
    pp = tmp_path / 'plan.json'
    pp.write_text(json.dumps([{'study': 'nope'}, {'export': 'e', 'formats': 'nope'}, {'study': 'rig'}]))
    assert main(['--sample', 'so100', '--run-plan', str(pp)]) == 2
    s = json.load(open(tmp_path / 'plan-summary.json'))
    assert [r['status'] for r in s['steps']] == ['error', 'error', 'done']
    assert 'There is no study "nope"' in s['steps'][0]['message'] and 'nope' in s['steps'][1]['message']
    pp.write_text(json.dumps({'steps': [{'study': 'rig', 'check': 'defaults'}]}))
    assert main(['--sample', 'so100', '--run-plan', str(pp)]) == 2
    assert 'exactly one of' in capsys.readouterr().err
    pp.write_text('[]')
    assert main(['--sample', 'so100', '--run-plan', str(pp)]) == 2
    assert 'has no steps' in capsys.readouterr().err
