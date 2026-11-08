"""The definition of done (spec §18), driven offscreen through the window as docs/user-guide.md's "Ten-minute tour"
writes it: `dexvantage --sample aloha`, then see which grasps lose fingertip visibility and why, move a camera and see
the score change, check depth validity, VLA token coverage and bandwidth, export the cameras to MuJoCo and LeRobot,
and export a printable bracket, all within ten minutes of compute. Each step uses the page's own controls (Run, Apply,
the Sensors page's fields, the export dialog), and the figures asserted are the ones the guide prints (its worked
example `tour` in scripts/guide_examples.py produces them through the API)."""
import json
import math
import os
import time

import numpy as np
from conftest import pump

from dexvantage.core.frames import rpy_from_quat


def _wait(app, win, timeout=400.0):
    assert win.ctx.analysis.wait(timeout)
    t0 = time.time()
    while getattr(win, '_jobs', None) and time.time() - t0 < timeout:
        pump(app, 0.05)
    pump(app, 0.1)


def _texts(page) -> str:
    """Every text the page's headline shows (figure, status, sentence, tooltip)."""
    h = page.head
    return ' '.join(x for x in (h.figure.text(), h.status.text.text(), h.explain.text(), h.toolTip(),
                                h.figure.toolTip()) if x)


def _sensor(win, name):
    return next(s for s in win.store.project.variant.sensors if s.name == name)


def test_ten_minute_tour(app, win, tmp_path):
    t_start = time.time()
    # 1. dexvantage --sample aloha (File ▸ Open sample ▸ ALOHA 2 bimanual with its four cameras)
    assert win.open_sample('aloha')
    _wait(app, win)
    p = win.store.project
    assert len(p.robots) == 2 and [s.name for s in p.variant.sensors] == [
        'cam_high (overhead)', 'cam_low (worms-eye)', 'cam_left_wrist', 'cam_right_wrist']

    # 2. Analysis ▸ Grasp self-occlusion ▸ Run: which grasps lose fingertip visibility, and why
    win.open_page('grasp')
    page = win.pages['grasp']
    pump(app, 0.05)
    assert page.set_box.currentData()                        # the left arm's tabletop grasps, the first set
    page.run()
    _wait(app, win)
    s = page.doc['summary']
    assert (s['grasps'], s['feasible'], s['infeasible']) == (100, 86, {'out of reach': 14})
    assert page.head.figure.text() == '100.0 [95.7–100]' and page.head.status.text.text() == 'well seen'
    cams = {page.cams.topLevelItem(i).text(0): [page.cams.topLevelItem(i).text(c) for c in range(9)]
            for i in range(page.cams.topLevelItemCount())}
    assert set(cams) == {'cam_high (overhead)', 'cam_low (worms-eye)', 'cam_left_wrist', 'cam_right_wrist'}
    hi, lw, rw = cams['cam_high (overhead)'], cams['cam_left_wrist'], cams['cam_right_wrist']
    assert '39' in hi[7] and '46' in hi[8] and '16' in hi[5]           # behind own finger, hidden by scene, seen
    assert '55' in rw[6]                                               # outside its image
    assert '61' in lw[1] and '56' in lw[1]                             # each fingertip in the left wrist camera
    assert page.grasps.topLevelItemCount() == 100 and page.gallery_grid.count() >= 2
    worst = min(g['score'] for g in page.doc['gallery'])
    assert round(100 * worst) == 56
    i = page.dec['feasible'].nonzero()[0][0]
    page.show_grasp(int(i))                                            # click a grasp: the arm takes its pose
    pump(app, 0.05)
    from dexvantage.core.robot.model import get_model
    rid = p.robots[0].uid
    home = get_model(win.store.project, rid).home
    assert not np.allclose(win.store.q[rid], home)
    win.pages['robot'].home_all()                                      # Robot page ▸ Home all: back to home
    pump(app, 0.05)
    assert np.allclose(win.store.q[rid], home)

    # 3. move a camera and see the score change: cam_left_wrist pitched 10° further on the Sensors page
    cam = _sensor(win, 'cam_left_wrist')
    win.store.select(('sensor', cam.uid))
    win.open_page('sensors')
    sp = win.pages['sensors']
    pump(app, 0.1)
    box = sp.rpy_boxes[1]
    assert round(box.value()) == -65
    box.set_value(-75.0)
    box.committed.emit(-75.0)
    pump(app, 0.1)
    assert 'Run again' in page.stale_note.text()
    win.open_page('grasp')
    page.run()
    _wait(app, win)
    assert page.head.figure.text() == '4.7 [1–9]' and page.head.status.text.text() == 'blind at contact'
    assert page.doc['summary']['meets'] == 4
    win.store.undo.undo()                                               # Ctrl+Z puts the camera back
    cam = _sensor(win, 'cam_left_wrist')
    assert round(math.degrees(rpy_from_quat(cam.mount.pose.quaternion)[1])) == -65

    # 4. depth validity
    win.open_page('depth')
    dp = win.pages['depth']
    dp.run()
    _wait(app, win)
    assert dp.head.figure.text() == '100.0' and 'Per sensor: cam_high (overhead)' in _texts(dp)

    # 5. VLA token coverage: none without a task object, then a box added from the Workspace menu
    win.open_page('vla')
    vp = win.pages['vla']
    vp.run()
    _wait(app, win)
    assert vp.head.status.text.text() == 'no task object'
    win.add_object()
    vp.run()
    _wait(app, win)
    assert (vp.head.figure.text(), vp.head.status.text.text()) == ('3', 'too few tokens')
    assert [o.name for o in win.store.project.workspace.objects] == ['Box 1']

    # 6. bandwidth: no hosts, then Apply adds the suggested host and plugs every camera in
    win.open_page('bandwidth')
    bp = win.pages['bandwidth']
    bp.run()
    _wait(app, win)
    assert (bp.head.figure.text(), bp.head.status.text.text()) == ('—', 'no hosts')
    assert round(bp.doc['headline']['total_payload_mbps'], 1) == 1563.0
    bp.apply()
    _wait(app, win)
    assert (bp.head.figure.text(), bp.head.status.text.text()) == ('46', 'within budget')
    assert len(win.store.project.rig['hosts']) == 1

    # 7. export the cameras to MuJoCo and LeRobot (File ▸ Export…)
    from dexvantage.ui.dialogs import export as DX
    d = DX.open_dialog(win)
    try:
        d.set_chosen(['mjcf', 'lerobot'])
        d.folder.setText(str(tmp_path / 'aloha-export'))
        assert d.start()
        _wait(app, win)
        assert d.hint.text() == '2 of 2 formats written; 38 files.'
        man = json.load(open(tmp_path / 'aloha-export' / 'export_manifest.json', encoding='utf-8'))
        assert {k: r['status'] for k, r in man['formats'].items()} == {'mjcf': 'ok', 'lerobot': 'ok'}
        names = {os.path.basename(f['path'] if isinstance(f, dict) else f) for f in man['files']}
        assert any(n.endswith('.xml') for n in names) and any('lerobot' in n or n.endswith('.json') for n in names)
    finally:
        d.close()

    # 8. a printable bracket (Analysis ▸ Brackets: Apply the suggested types, Run, Export STL)
    win.open_page('brackets')
    kp = win.pages['brackets']
    kp.run()
    _wait(app, win)
    assert kp.head.status.text.text() == 'check' and kp.b_apply.isEnabled()
    kp.apply()
    _wait(app, win)
    assert kp.head.figure.text() == '0/2' and kp.head.status.text.text() == 'check'
    kinds = {s.name: s.mount.bracket for s in win.store.project.variant.sensors}
    assert kinds == {'cam_high (overhead)': 'tslot2020', 'cam_low (worms-eye)': 'tslot2020', 'cam_left_wrist': '',
                     'cam_right_wrist': ''}
    hi_cam = _sensor(win, 'cam_high (overhead)')
    k = kp.c_sensor.findData(hi_cam.uid)
    kp.c_sensor.setCurrentIndex(k)
    pump(app, 0.05)
    paths = kp.export_stl(str(tmp_path / 'cam_high_clamp.stl'))
    assert paths and paths[0].endswith('.stl') and os.path.getsize(paths[0]) > 84
    with open(paths[0], 'rb') as fh:                                    # binary STL: 80-byte header, then a count
        head = fh.read(84)
    n_tri = int.from_bytes(head[80:84], 'little')
    assert n_tri > 1000 and os.path.getsize(paths[0]) == 84 + 50 * n_tri
    side = json.load(open(paths[1], encoding='utf-8'))
    assert 'provenance' in side
    assert time.time() - t_start < 600                                  # the ten minutes, compute included
