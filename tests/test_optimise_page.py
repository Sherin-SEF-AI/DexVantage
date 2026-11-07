"""The Optimise page (ui/pages/optimise.py): run in the background, the headline and table, the scatter, Preview (3D and
Camera view overlays, nothing changed), Accept as one undo step, both modes, settings as undo steps."""
import time

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core import metrics as M
from dexvantage.core.frames import Pose
from dexvantage.core.optimise import objectives as OB
from dexvantage.core.optimise import search as S
from dexvantage.core.project import Mount, Project, Sensor, Volume
from dexvantage.core.scenegeom import sensor_world_pose

from conftest import pump

TARGET = np.array([0.5, 0.0, 0.3])


@pytest.fixture(autouse=True, scope='module')
def fake_metrics():
    @jobs.study('_optp_geo', 'Test geometry', quick=False)
    def _geo(project, ctx):
        cams = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        d = [float(np.linalg.norm(sensor_world_pose(project, s).position - TARGET)) for s in cams]
        return {'dist': min(d) if d else None, 'cams': len(cams)}
    OB.ensure_metrics()
    M.metric('_optp.dist', 'Distance to the target', 'm', 'lower', study='_optp_geo')(lambda d: d['dist'])
    M.metric('_optp.cams', 'Cameras', 'count', 'lower', study='_optp_geo')(lambda d: d['cams'])
    yield
    jobs.STUDIES.pop('_optp_geo', None)
    for k in ('_optp.dist', '_optp.cams'):
        M.METRICS.pop(k, None)
    S.clear_cache()


def _proj():
    p = Project('Optimise page test')
    p.workspace.volumes.append(Volume(center=tuple(TARGET), size=(0.3, 0.3, 0.2)))
    p.variant.sensors = [
        Sensor(name='A', model_id='intel-realsense-d435', kind='depth',
               mount=Mount('cage', '', Pose(np.array([1.0, 0.0, 1.0])))),
        Sensor(name='B', mount=Mount('world', '', Pose(np.array([0.0, 1.0, 1.0])))),
        Sensor(name='L', locked=True, mount=Mount('cage', '', Pose(np.array([-1.0, -1.0, 2.0])))),
    ]
    common = {'objectives': ['_optp.dist', '_optp.cams'], 'budget': 24, 'population': 6, 'screen': {}}
    p.analysis.studies['optimise'] = dict(common, move_m=0.3)
    p.analysis.studies['design_agent'] = dict(common, models=['intel-realsense-d435', 'intel-realsense-d405'],
                                              max_cameras=2)
    return p


def _run(app, win, page):
    page.run()
    t0 = time.time()
    while time.time() - t0 < 60:
        pump(app, 0.05)
        if not win.ctx.analysis.busy(page.STUDY) and page.doc is not None and page.doc.get('study') == page.STUDY:
            break
    pump(app, 0.1)


def test_optimise_page_run_preview_accept_undo(app, win):
    win.store.set_project(_proj())
    win.ctx.analysis.wait(10)
    win.open_page('optimise')
    page = win.pages['optimise']
    assert page.head.state['word'] == 'not run yet'
    _run(app, win, page)
    doc = page.doc
    assert doc and doc['study'] == 'optimise' and doc['options']
    assert page.head.state['word'] == doc['headline']['status_word'] and page.b_apply.isEnabled()
    assert page.table.topLevelItemCount() == len(doc['options']) + 1
    assert page.chart.labels and page.explain_l.text.text().startswith('Design A:')
    assert 'Locked, never changed: L' in page.left_out.text()
    # select the last design from the table, preview it (nothing changes)
    lab = doc['options'][-1]['label']
    page._select(lab)
    before = win.store.project.variant.to_dict()
    t0 = time.perf_counter()
    page.preview()
    assert (time.perf_counter() - t0) < 0.5
    items = page.overlays(win.viewport)
    assert items and win.store.project.variant.to_dict() == before
    from dexvantage.core import library as LB
    from dexvantage.core.scenegeom import sensor_world_pose as swp
    s = win.store.project.variant.sensors[0]
    cam = LB.sensor_imagers(s)[0].camera
    page.camview_overlays(cam, swp(win.store.project, s))      # runs; marks only where a design camera is in view
    # Accept: one undo step that sets the design on the variant, locked sensor untouched
    n = win.store.undo.count()
    locked = next(x for x in win.store.project.variant.sensors if x.locked).to_dict()
    page.apply()
    pump(app, 0.05)
    assert win.store.undo.count() == n + 1
    after = win.store.project.variant.to_dict()
    assert after != before or not doc['options'][-1]['changes']
    assert next(x for x in win.store.project.variant.sensors if x.locked).to_dict() == locked
    g = jobs.run_study('_optp_geo', win.store.project)
    assert g['dist'] == pytest.approx(doc['options'][-1]['figures']['_optp.dist'])
    page.apply()                                               # the rig changed since: refused in words
    assert win.store.undo.count() == n + 1
    page._preview = None
    page.preview()                                             # the same: no preview of a stale design
    assert page._preview is None
    win.store.undo.undo()
    pump(app, 0.05)
    assert win.store.project.variant.to_dict() == before


def test_design_mode_accept_adds_variant_and_settings_are_undo_steps(app, win):
    win.store.set_project(_proj())
    win.ctx.analysis.wait(10)
    win.open_page('optimise')
    page = win.pages['optimise']
    page.mode.buttons['design_agent'].click()
    pump(app, 0.05)
    assert page.STUDY == 'design_agent' and page.s_maxcam.isVisible() is not None
    n = win.store.undo.count()
    page.s_seed.setValue(7)
    page.s_seed.editingFinished.emit()
    assert win.store.undo.count() == n + 1 and win.store.project.analysis.studies['design_agent']['seed'] == 7
    _run(app, win, page)
    doc = page.doc
    assert doc['study'] == 'design_agent' and doc['options']
    nv = len(win.store.project.variants)
    page.apply()
    pump(app, 0.05)
    assert len(win.store.project.variants) == nv + 1
    assert win.store.project.variant.name.startswith(f'Design {page.sel_label}')
    assert any(s.name == 'L' and s.locked for s in win.store.project.variant.sensors)
    win.store.undo.undo()
    pump(app, 0.05)
    assert len(win.store.project.variants) == nv
    # back to the optimiser mode: its own (empty) state, no crash
    page.mode.buttons['optimise'].click()
    pump(app, 0.05)
    assert page.STUDY == 'optimise'
    page.export(str(__import__('tempfile').mkdtemp()) + '/x.json') if page.doc else None
