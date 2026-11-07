"""File ▸ Compare with…, Save as folder (for Git)… and Open folder… (ui/versioning.py)."""
import json
import os
import time

from dexvantage.core.project import Sensor
from dexvantage.ui import versioning as V


def _wait(app, cond, timeout=20.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_file_menu_has_the_commands(win):
    texts = [a.text() for a in win.m_file.actions()]
    for t in ('Open folder…', 'Save as folder (for Git)…', 'Compare with…'):
        assert t in texts
    for aid in ('open_folder', 'save_folder', 'compare'):
        assert win.actions_by_id[aid].isEnabled()


def test_compare_window_shows_the_changes(win, app, tmp_path):
    other = win.store.project.snapshot()
    other.variant.sensors.append(Sensor(name='Side camera'))
    other.notes = 'edited'
    path = str(tmp_path / 'other.dexv')
    other.save(path)
    w = V.compare(win, path)
    assert _wait(app, lambda: w.diff is not None)
    assert w.head.figure.text() == '2'
    assert w.table.rowCount() == 2
    items = {w.table.item(r, 2).text() for r in range(w.table.rowCount())}
    assert 'Sensor Side camera' in items
    assert any(c['op'] == 'removed' for c in w.diff['changes'])     # the other file is before, the open one after
    w.section.setCurrentIndex(w.section.findData('sensors'))
    assert w.table.rowCount() == 1
    first = w.diff
    w.swap()                                                # the open project first: the sensor is added
    assert _wait(app, lambda: w.diff is not first and not win._jobs)
    assert any(c['op'] == 'added' for c in w.diff['changes'])
    out = w.export_json(str(tmp_path / 'd.json'))
    doc = json.load(open(out))
    assert doc['schema'] == 'dexvantage.diff/1' and 'provenance' in doc
    assert w.export_html(str(tmp_path / 'd.html')).endswith('.html')
    ids = w.metric_ids()                                    # the metrics of the studies run on the open project
    started = w.compare_figures()
    assert started == bool(ids)
    if ids:
        assert _wait(app, lambda: w.diff.get('metrics') and not win._jobs, 60)
        assert w.metrics.rowCount() == len(ids) and not w.metrics.isHidden()
    w.close()


def test_compare_with_a_bad_file_says_why(win, app, tmp_path):
    bad = tmp_path / 'bad.dexv'
    bad.write_text('not a zip')
    w = V.compare(win, str(bad))
    assert _wait(app, lambda: 'zip' in w.head.explain.text())
    assert 'zip' in w.head.explain.text() and 'Traceback' not in w.head.explain.text()
    w.close()


def test_save_and_open_folder(win, app, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(V.QMessageBox, 'warning', lambda *a, **k: warned.append(a[2]))
    f = str(tmp_path / 'cell')
    assert V.save_folder(win, f)
    assert os.path.isfile(os.path.join(f + '.dexv.d', 'project.json'))
    assert win.store.project.path == f + '.dexv.d' and win.store.undo.isClean()
    assert V.open_folder(win, f + '.dexv.d')
    assert win.store.project.path == f + '.dexv.d'
    assert not V.open_folder(win, str(tmp_path))              # no project.json: a warning, nothing opened
    assert warned and 'no project.json' in warned[-1]
