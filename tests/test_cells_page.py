"""The Cells page (ui/pages/cells.py, File ▸ Compare cells…): the command after Compare with…, settings as undo
steps, the run in the background with the headline and tables, Preview (nothing changed), Apply as one undo step,
Export of the three files with provenance, and the UI-thread budget of its steps."""
import json
import math
import time

import numpy as np

from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import checks_cells as CC

from conftest import pump

WRIST = 'Wrist camera (sample placement)'


def _run(app, win, page, timeout=90):
    page.run()
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(app, 0.05)
        if not win.ctx.analysis.busy('cells') and page.doc is not None and page.doc.get('study') == 'cells':
            break
    pump(app, 0.1)


def test_menu_command_after_compare_with(app, win):
    acts = win.menus['File'].actions()
    a = win.actions_by_id['page_cells']
    assert a.text() == 'Compare cells…' and a in acts
    assert acts.index(a) == acts.index(win.actions_by_id['compare']) + 1


def test_cells_page_settings_run_preview_apply_export(app, win, tmp_path):
    p = sample_project('so100')
    turned = CC.turn_camera(p, WRIST, (0.0, math.radians(15), 0.0), (0.003, 0.0, 0.0))
    other = str(tmp_path / 'bench.dexv')
    p.save(other)                                           # the reference cell: the sample as planned
    turned.name = 'My cell'
    win.store.set_project(turned)
    win.ctx.analysis.wait(10)
    win.open_page('cells')
    page = win.pages['cells']
    pump(app, 0.05)
    assert page.head.state['word'] == 'one cell' and not page.b_run.isEnabled()
    # settings: each one undo step
    n0 = win.store.undo.count()
    t0 = time.perf_counter()
    page.add_files([other])
    dt_add = time.perf_counter() - t0
    assert win.store.undo.count() == n0 + 1 and page.b_run.isEnabled()
    assert win.store.project.analysis.studies['cells']['cells'] == [{'path': other}]
    page.add_sample('ur5e')
    win.store.undo.undo()
    assert win.store.project.analysis.studies['cells']['cells'] == [{'path': other}]
    page._set('metrics', ['rig.camera_count'], 'figures')
    page._set('reference', 1, 'reference cell')
    pump(app, 0.05)
    assert page.cell_tab.topLevelItemCount() == 2 and page.ref_box.currentIndex() == 1
    _run(app, win, page)
    doc = page.doc
    assert doc and doc['reference'] == 1 and page.head.state['word'].endswith('gap')
    assert page.gap_tab.topLevelItemCount() == 1 and page.match_tab.topLevelItemCount() == 2
    assert page.pool_tab.topLevelItemCount() == 1 and page.met_tab.topLevelItemCount() == 1
    assert page.drift_tab.topLevelItemCount() >= 1
    sg = doc['suggestion']
    assert sg['kind'] == 'match_mount' and sg['sensor'] == WRIST and page.b_apply.isEnabled()
    ctx = page.page_context()
    assert ctx['study'] == 'cells' and ctx['has_result'] and ctx['suggestion_ready']
    # Preview: the 3D view draws the pose, nothing changes
    before = win.store.project.variant.to_dict()
    page.preview()
    assert page.overlays(win.viewport) and win.store.project.variant.to_dict() == before
    # Apply: one undo step that puts the camera at the reference's pose
    n1 = win.store.undo.count()
    t0 = time.perf_counter()
    page.apply()
    dt_apply = time.perf_counter() - t0
    assert win.store.undo.count() == n1 + 1
    s = win.store.project.variant.sensor(sg['sensor_uid'])
    assert np.allclose(s.mount.pose.position, sg['pose']['position'])
    win.store.undo.undo()
    assert win.store.project.variant.to_dict() == before
    # Export: the three files into a folder, with provenance
    files = page.export(str(tmp_path / 'out'))
    assert [f.rsplit('/', 1)[-1] for f in files] == ['cells.json', 'cells_pooling.json', 'cells_comparison.html']
    pool = json.loads((tmp_path / 'out' / 'cells_pooling.json').read_text())
    assert pool['provenance'] and pool['cells'][0]['name'] == 'My cell'
    assert str(tmp_path) not in (tmp_path / 'out' / 'cells.json').read_text()
    assert dt_add < 0.1 and dt_apply < 0.1, (dt_add, dt_apply)


def test_as_built_variant_apply(app, win):
    p = sample_project('ur5e')
    copies = [CC.calibrated_copy(p, 'Wrist D405', (0, 0, 3.0), name=f'B{k}') for k in range(3)]
    paths = []
    import tempfile
    import shutil
    folder = tempfile.mkdtemp(prefix='dexvantage-test-cells-')
    try:
        for k, c in enumerate(copies):
            paths.append(f'{folder}/b{k}.dexv')
            c.save(paths[-1])
        p.analysis.studies['cells'] = {'cells': [{'path': x} for x in paths], 'metrics': ['rig.camera_count']}
        win.store.set_project(p)
        win.ctx.analysis.wait(10)
        win.open_page('cells')
        page = win.pages['cells']
        _run(app, win, page)
        sg = page.doc['suggestion']
        assert sg['kind'] == 'as_built_variant' and page.b_apply.isEnabled()
        page.preview()
        assert len(page.overlays(win.viewport)) == 1
        n = len(win.store.project.variants)
        active = win.store.project.active_uid
        page.apply()
        assert len(win.store.project.variants) == n + 1 and win.store.project.active_uid == active
        v = win.store.project.variants[-1]
        assert v.name == sg['name'] and {s.uid for s in v.sensors}.isdisjoint(
            {s.uid for s in win.store.project.variant.sensors})
        win.store.undo.undo()
        assert len(win.store.project.variants) == n
    finally:
        shutil.rmtree(folder, ignore_errors=True)
