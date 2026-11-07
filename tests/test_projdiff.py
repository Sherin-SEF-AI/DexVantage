"""Project diff and three-way merge (core/projdiff.py) and their batch options (--diff, --merge)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import projdiff as D
from dexvantage.core.frames import quat_from_axis_angle, quat_multiply
from dexvantage.core.project import Requirement, Sensor, TaskObject
from dexvantage.core.robot.samples import sample_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _run(*args, cwd=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd or ROOT, timeout=120)


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


def _edited(base):
    """A copy with seven known edits, and those edits as (section, op, what the text names)."""
    p = base.snapshot()
    wrist, top = p.variant.sensors[0], p.variant.sensors[1]
    wrist.mount.pose.position = wrist.mount.pose.position + np.array([0.012, -0.003, 0.0])
    top.mount.pose.quaternion = quat_multiply(top.mount.pose.quaternion, quat_from_axis_angle([0, 0, 1],
                                                                                              np.radians(5.0)))
    top.model_id = 'intel-realsense-d455'
    p.variant.sensors.append(Sensor(name='Side camera'))
    p.workspace.objects.append(TaskObject(name='Mug'))
    p.requirements.append(Requirement(name='Cameras', metric='rig.camera_count', op='<=', threshold=3))
    p.analysis.studies['coverage'] = {'voxel_mm': 3.0}
    return p, [('sensors', 'changed', 'Wrist D405'), ('sensors', 'changed', 'Overhead D435'),
               ('sensors', 'changed', 'Overhead D435'), ('sensors', 'added', 'Side camera'),
               ('workspace', 'added', 'Mug'), ('requirements', 'added', 'Cameras'),
               ('settings', 'changed', 'coverage')]


def test_diff_lists_exactly_the_edits(ur5e):
    edited, edits = _edited(ur5e)
    d = D.diff(ur5e, edited)
    got = sorted((c['section'], c['op'], c['item']) for c in d['changes'])
    assert got == sorted(edits)
    assert not d['identical'] and sum(d['counts'].values()) == len(edits)
    move = next(c for c in d['changes'] if c['item'] == 'Wrist D405')
    assert move['delta']['shift_mm'] == pytest.approx([12.0, -3.0, 0.0], abs=1e-9)
    assert move['delta']['distance_mm'] == pytest.approx(np.hypot(12, 3), abs=1e-9)
    assert move['delta']['turn_deg'] == pytest.approx(0.0, abs=1e-9)
    turn = next(c for c in d['changes'] if c['item'] == 'Overhead D435' and 'delta' in c)
    assert turn['delta']['turn_deg'] == pytest.approx(5.0, abs=1e-9)
    assert turn['delta']['distance_mm'] == pytest.approx(0.0, abs=1e-9)
    dev = next(c for c in d['changes'] if c['field'] == 'model')
    assert dev['before'] == 'intel-realsense-d435' and dev['after'] == 'intel-realsense-d455'
    json.dumps(d, allow_nan=False)                          # JSON-able


def test_a_project_against_itself_and_a_saved_copy(ur5e, tmp_path):
    assert D.diff(ur5e, ur5e.snapshot())['identical']
    path = str(tmp_path / 'u.dexv')
    ur5e.snapshot().save(path)
    d = D.diff(ur5e, path)                                  # a path works as well as a Project
    assert d['identical'] and d['b']['file'] == 'u.dexv'


def test_sensors_match_by_uid_then_by_name(ur5e):
    a = ur5e.snapshot()
    b = ur5e.snapshot()
    b.variant.sensors[0].uid = 'feedfacecafe'               # a re-created sensor with the same name: matched by name
    b.variant.sensors[1].name = 'Top camera'                # renamed, same uid
    d = D.diff(a, b)
    assert [(c['op'], c['field']) for c in d['changes']] == [('changed', 'name')]
    assert 'renamed Top camera' in d['changes'][0]['text']


def test_robots_and_files(ur5e):
    b = ur5e.snapshot()
    r = b.robots[0]
    r.base.position = r.base.position + np.array([0.0, 0.0, 0.1])
    j = dict(r.joints[0])
    j['upper'] = float(j.get('upper') or 0) - 0.1
    r.joints = [j] + r.joints[1:]
    name = sorted(b.sources)[0]
    b.sources = dict(b.sources, **{name: b'changed'})
    b.meshes = {'extra': np.zeros((3, 3))}
    d = D.diff(ur5e, b)
    texts = [c['text'] for c in d['changes']]
    assert any('base moved 100.0 mm' in t for t in texts)
    assert any(c['kind'] == 'Joint' and c['field'] == 'upper' for c in d['changes'])
    assert {(c['kind'], c['op']) for c in d['changes'] if c['section'] == 'files'} == {('Model file', 'changed'),
                                                                                       ('Mesh', 'added')}


def test_text_and_html(ur5e):
    edited, _ = _edited(ur5e)
    d = D.diff(ur5e, edited)
    t = D.to_text(d)
    assert '7 changes' in t and 'Wrist D405: mount pose moved 12.4 mm' in t
    h = D.to_html(d)
    assert h.startswith('<!doctype html>') and h.count('<tr class=') == 7 and '<script' not in h


def test_metric_deltas_from_results():
    from dexvantage.core import metrics as MX
    MX.load_metrics()
    mid = 'test.projdiff.figure'
    MX.metric(mid, 'Test figure', '%', 'higher', study='test_projdiff_study')(lambda doc: doc['v'])
    try:
        rows = D.metric_deltas({'test_projdiff_study': {'v': 90.0}}, {'test_projdiff_study': {'v': 95.5}}, [mid])
        assert rows[0]['delta'] == pytest.approx(5.5) and rows[0]['verdict'] == 'better'
        rows = D.metric_deltas({'test_projdiff_study': {'v': 90.0}}, {'test_projdiff_study': {}}, [mid])
        assert rows[0]['delta'] is None and rows[0]['status_b'] == 'not stated'
        assert D.metric_deltas({}, {'test_projdiff_study': {'v': 1}}, [mid]) == []
    finally:
        MX.METRICS.pop(mid, None)


# ====================================================================== merge
def _sides(base, ours, theirs):
    o, t = base.snapshot(), base.snapshot()
    ours(o)
    theirs(t)
    return o, t


def test_merge_disjoint_edits_gives_their_union(ur5e):
    def ours(p):
        p.variant.sensors[0].mount.pose.position = p.variant.sensors[0].mount.pose.position + [0.01, 0, 0]
        p.analysis.studies['coverage'] = {'voxel_mm': 4.0}

    def theirs(p):
        p.variant.sensors[1].model_id = 'intel-realsense-d455'
        p.variant.sensors.append(Sensor(name='Side'))
        p.add_log('edit', 'their edit')
    o, t = _sides(ur5e, ours, theirs)
    r = D.merge3(ur5e, o, t)
    d = D.diff(ur5e, r.project)
    assert sorted((c['section'], c['op'], c['item']) for c in d['changes']) == sorted([
        ('sensors', 'changed', 'Wrist D405'), ('settings', 'changed', 'coverage'),
        ('sensors', 'changed', 'Overhead D435'), ('sensors', 'added', 'Side')])
    assert r.project.log[-2]['text'] == 'their edit' and r.project.log[-1]['kind'] == 'merge'
    assert r.from_ours and r.from_theirs and r.project.sources.keys() == ur5e.sources.keys()


def test_merge_same_edit_on_both_sides(ur5e):
    def both(p):
        p.variant.sensors[0].model_id = 'x'
    o, t = _sides(ur5e, both, both)
    assert len(D.diff(ur5e, D.merge3(ur5e, o, t).project)['changes']) == 1


@pytest.mark.parametrize('case', ['value', 'removed-changed', 'pose', 'dangling', 'files'])
def test_merge_conflicts_are_refused_with_a_sentence(ur5e, case):
    s0 = ur5e.variant.sensors[0]

    def dev(v):
        return lambda p: setattr(p.variant.sensors[0], 'model_id', v)

    def move(dx):
        def f(p):
            s = p.variant.sensors[1]
            s.mount.pose.position = s.mount.pose.position + [dx, 0, 0]
        return f

    def drop(p):
        p.variant.sensors = p.variant.sensors[1:]

    def add_on_arm(p):
        s = Sensor(name='On the arm')
        s.mount.robot_uid, s.mount.parent_frame = s0.mount.robot_uid, s0.mount.parent_frame
        p.variant.sensors.append(s)

    def remove_arm(p):
        p.robots = []
        p.variant.sensors = [s for s in p.variant.sensors if not s.mount.robot_uid]

    def src(v):
        return lambda p: setattr(p, 'sources', dict(p.sources, **{sorted(p.sources)[0]: v}))
    ours, theirs = {'value': (dev('a'), dev('b')), 'removed-changed': (drop, dev('c')),
                    'pose': (move(0.01), move(0.02)), 'dangling': (add_on_arm, remove_arm),
                    'files': (src(b'one'), src(b'two'))}[case]
    o, t = _sides(ur5e, ours, theirs)
    with pytest.raises(D.MergeConflict) as e:
        D.merge3(ur5e, o, t)
    assert e.value.conflicts
    msg = str(e.value)
    assert msg.startswith('The two edits cannot be merged') and 'Traceback' not in msg and msg.endswith('.')


# ====================================================================== batch
def test_batch_diff_exit_codes_and_outputs(ur5e, tmp_path):
    a, b = str(tmp_path / 'a.dexv'), str(tmp_path / 'b.dexv')
    ur5e.snapshot().save(a)
    edited, edits = _edited(ur5e)
    edited.save(b)
    r = _run(a, '--diff', a)
    assert r.returncode == 0 and 'No differences.' in r.stdout, r.stderr
    out, page = tmp_path / 'd.json', tmp_path / 'd.html'
    r = _run(a, '--diff', b, str(out), '--diff-html', str(page))
    assert r.returncode == 1, r.stderr
    assert f'{len(edits)} changes' in r.stdout
    doc = json.loads(out.read_text())
    assert doc['schema'] == 'dexvantage.diff/1' and len(doc['changes']) == len(edits)
    assert doc['provenance']['interface'] == 'batch' and str(tmp_path) not in out.read_text()
    assert page.read_text().startswith('<!doctype html>')
    r = _run(a, '--diff', str(tmp_path / 'missing.dexv'))
    assert r.returncode == 2 and 'Traceback' not in r.stderr and 'There is no file' in r.stderr
    r = _run(a, '--diff-html', str(page))
    assert r.returncode == 2 and 'go with --diff' in r.stderr


def test_batch_merge(ur5e, tmp_path):
    base, ours, theirs, out = (str(tmp_path / n) for n in ('base.dexv', 'ours.dexv', 'theirs.dexv', 'm.dexv'))
    ur5e.snapshot().save(base)
    o, t = _sides(ur5e, lambda p: setattr(p.variant.sensors[0], 'model_id', 'a'),
                  lambda p: setattr(p.variant.sensors[1], 'model_id', 'b'))
    o.save(ours)
    t.save(theirs)
    r = _run(ours, '--merge', base, theirs, out)
    assert r.returncode == 0 and os.path.isfile(out), r.stderr
    assert len(D.diff(base, out)['changes']) == 2
    t.variant.sensors[0].model_id = 'c'
    t.save(theirs)
    os.remove(out)
    r = _run(ours, '--merge', base, theirs, out)
    assert r.returncode == 1 and not os.path.exists(out)
    assert 'cannot be merged' in r.stderr and 'Traceback' not in r.stderr
