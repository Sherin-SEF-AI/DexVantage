"""Product review of Phases 4, 5 and 8 (run as a researcher would, with bad inputs): the fixes it made.

- A StudyError subclass (DatasetError, PolicyError, ...) raised in a study reaches the page as its own sentence, not
  "stopped with an unexpected error (DatasetError)".
- Damaged datasets are explained: a damaged meta/info.json, a damaged Parquet file, a bag folder with a damaged file,
  a bag with no messages (it audited as "ready" before), one damaged episode among good ones (the audit goes on and
  drops that episode instead of stopping with a traceback).
- A recording whose joints do not match the project's robot no longer crashes the audit (plan against recording).
- A long status-bar message no longer widens the main window (it grew to several thousand pixels).
- --dataset alone is a batch refusal (it opened the window); an unexpected study error in batch is one sentence.
- The plan-against-recording preview names sensors, not their uids; the silhouette page says when its mount is
  applied (and offers Apply again after an undo).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import pump

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


@pytest.fixture(scope='module')
def campaign(tmp_path_factory):
    """Two small SO-100 LeRobot v2.1 episodes (one Parquet file each), every fault on."""
    from render_support import require
    require()                               # renders MuJoCo scenes
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import episodes as SE
    root = str(tmp_path_factory.mktemp('rv458') / 'so100')
    SE.generate(sample_project('so100'), root, {'episodes': 2, 'frames': 60, 'max_width': 160, 'seed': 5,
                                                'version': 'v2.1'})
    return root


def _copy(src, dst):
    import shutil
    shutil.copytree(src, dst)
    return str(dst)


def test_not_a_dataset_names_the_formats_in_words(tmp_path):
    from dexvantage.core.datasets import DatasetError, open_dataset
    (tmp_path / 'readme.txt').write_text('x')
    with pytest.raises(DatasetError, match='not a dataset format') as e:
        open_dataset(str(tmp_path))
    assert 'LeRobot' in str(e.value) and 'ROS 2 bags' in str(e.value) and 'rosbag2-sqlite' not in str(e.value)


def test_damaged_lerobot_files_are_explained(tmp_path, campaign):
    from dexvantage.core.datasets import DatasetError, open_dataset
    bad = _copy(campaign, tmp_path / 'badjson')
    with open(os.path.join(bad, 'meta', 'info.json'), 'w') as fh:
        fh.write('{"codebase_version": ')
    with pytest.raises(DatasetError, match='cannot be read as JSON'):
        open_dataset(bad)
    bad = _copy(campaign, tmp_path / 'badpq')
    f = os.path.join(bad, 'data', 'chunk-000', 'episode_000001.parquet')
    data = open(f, 'rb').read()
    open(f, 'wb').write(data[:len(data) // 3])
    with open_dataset(bad) as ds:
        ds.load(0)
        with pytest.raises(DatasetError, match='damaged') as e:
            ds.load(1)
    assert bad not in str(e.value)                       # the relative path, not the user's folder


def test_damaged_and_empty_bags(tmp_path):
    from dexvantage.core.audit import campaign as AC
    from dexvantage.core.datasets import DatasetError, open_dataset
    from dexvantage.core.jobs import StudyError
    d = tmp_path / 'junk'
    d.mkdir()
    (d / 'bag.db3').write_bytes(os.urandom(4000))
    (d / 'metadata.yaml').write_text('rosbag2_bagfile_information:\n  version: 5\n')
    with open_dataset(str(d)) as ds, pytest.raises(DatasetError, match='cannot be read'):
        ds.load(0)
    m = tmp_path / 'cut.mcap'
    m.write_bytes(b'\x89MCAP0\r\n' + os.urandom(500))
    with pytest.raises(StudyError, match='No episode of cut.mcap can be audited'):
        AC.audit_dataset(str(m), None)                   # it read as "ready, 1 of 1 episodes" before


def test_one_damaged_episode_is_dropped_not_fatal(tmp_path, campaign):
    from dexvantage.core.audit import campaign as AC
    from dexvantage.core.robot.samples import sample_project
    bad = _copy(campaign, tmp_path / 'mix')
    f = os.path.join(bad, 'data', 'chunk-000', 'episode_000001.parquet')
    open(f, 'wb').write(open(f, 'rb').read()[:900])
    doc = AC.audit_dataset(bad, sample_project('so100'), settings={'mount_source': 'none'})
    by = {e['index']: e for e in doc['episodes']}
    assert by[1]['verdict'] == 'drop' and by[1]['flags'] == ['unreadable'] and 'damaged' in by[1]['error']
    assert by[0]['cameras'] and not by[0].get('error')
    row = next(r for r in doc['requirements'] if r['id'] == 'readable')
    assert row['episodes_failed'] == [1] and not row['pass']
    assert row['checked'] == len(doc['episodes']) and row['failed'] == 1   # "n − 1 of n", not "0 of 1"
    from dexvantage.core.audit import report as RP
    out = RP.export_fiftyone(doc, bad, str(tmp_path / 'fo'), None, 'clips')
    assert out['samples'] == len(by[0]['cameras'])       # the unreadable episode has no media


def test_recording_of_another_robot_is_compared_not_crashed(campaign):
    from dexvantage.core.audit import campaign as AC
    from dexvantage.core.robot.samples import sample_project
    doc = AC.audit_dataset(campaign, sample_project('dexhand'), settings={'mount_source': 'none'})
    assert len(doc['episodes']) == 2
    assert any('do not match' in n for n in doc['plan_notes'])


def test_studyerror_subclass_reaches_the_page_as_its_sentence(app, win, tmp_path):
    from dexvantage.core.studies import audit as AU
    (tmp_path / 'x.txt').write_text('x')
    AU.use_dataset(win.store.project, str(tmp_path), AU.SID)
    page = win.pages['audit']
    win.ctx.analysis.run(AU.SID)
    win.ctx.analysis.wait(30)
    pump(app, 0.3)
    text = page.head.explain.text()
    assert 'not a dataset format' in text and 'unexpected' not in text, text


def test_long_message_does_not_widen_the_window(app, win):
    w0 = win.width()
    win.msg('a long message ' * 400, 2000)
    pump(app, 0.2)
    assert win.width() == w0 and win.minimumSizeHint().width() < 1400
    assert win.msg_label.text().startswith('a long message') and win.msg_label.toolTip()


def test_plan_preview_names_the_sensor(win):
    from dexvantage.core.robot.samples import sample_project
    win.store.set_project(sample_project('so100'))
    page = win.pages['plan_vs_recording']
    s = win.store.project.variant.sensors[0]
    page.doc = {'cameras': [{'sensor_uid': s.uid, 'resolution': {'verdict': 'scaled', 'recorded': [640, 360]}}]}
    page.preview()
    assert s.name in win.msg_label.text() and s.uid not in win.msg_label.text()


def test_silhouette_headline_follows_apply_and_undo(win):
    from dexvantage.core.frames import Pose
    from dexvantage.core.robot.samples import sample_project
    w = win
    w.store.set_project(sample_project('so100'))
    s = w.store.project.variant.sensors[1]           # the third-person camera on its tripod
    target = Pose.from_matrix(s.mount.pose.matrix() @ np.r_[np.c_[np.eye(3), [0.004, 0, 0]], [[0, 0, 0, 1]]])
    page = w.pages['silhouette']
    page.doc = {'apply': {'sensor_uid': s.uid, 'pose': target.to_dict(), 'episode': 2, 'change_mm': 4.0,
                          'change_deg': 0.0}, 'advice': 'Apply writes the recovered mount.'}
    page._apply_state()
    assert page.b_apply.isEnabled() and page.head.explain.text() == 'Apply writes the recovered mount.'
    assert page.apply()
    assert 'Applied' in page.head.explain.text() and not page.b_apply.isEnabled()
    w.store.undo.undo()
    assert page.head.explain.text() == 'Apply writes the recovered mount.' and page.b_apply.isEnabled()


def test_dataset_alone_is_refused_in_batch(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--dataset', str(tmp_path)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and '--dataset goes with' in r.stderr and 'Traceback' not in r.stderr


def test_unexpected_batch_error_is_one_sentence(capsys, monkeypatch):
    from dexvantage import batch
    monkeypatch.delenv('DEXVANTAGE_DEBUG', raising=False)
    try:
        raise KeyError('x')
    except KeyError as e:
        code = batch._unexpected('Dataset audit', e)
    err = capsys.readouterr().err
    assert code == 2 and 'Traceback' not in err and 'unexpected error (KeyError' in err
    path = err.rsplit('the details are in ', 1)[1].strip().rstrip('.')
    assert os.path.isfile(path) and 'Traceback' in open(path).read()
    os.remove(path)
