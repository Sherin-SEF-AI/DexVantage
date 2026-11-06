"""Ported from Vantage tests/test_navigation.py (the orbit, keyboard and navigation-control tests)."""
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from conftest import pump
from dexvantage.ui.navigation import EL_MAX, EL_MIN, orbit_about
from dexvantage.ui.viewport3d import OrbitCam

NOMOD = Qt.KeyboardModifier.NoModifier
LEFT = Qt.MouseButton.LeftButton


def _mouse(vp, kind, pos, button=LEFT, buttons=None):
    t = {'press': QEvent.Type.MouseButtonPress, 'move': QEvent.Type.MouseMove,
         'release': QEvent.Type.MouseButtonRelease}[kind]
    b = Qt.MouseButton.NoButton if kind == 'move' else button
    bs = buttons if buttons is not None else (button if kind != 'release' else Qt.MouseButton.NoButton)
    pos = QPointF(pos)
    QApplication.sendEvent(vp, QMouseEvent(t, pos, QPointF(vp.mapToGlobal(pos)), b, bs, NOMOD))


def _drag(app, vp, a, b, steps=6):
    a, b = QPointF(a), QPointF(b)
    _mouse(vp, 'press', a)
    for k in range(1, steps + 1):
        _mouse(vp, 'move', a + (b - a) * (k / steps), buttons=LEFT)
        pump(app, 0.01)
    _mouse(vp, 'release', b)
    pump(app, 0.03)


def _key(vp, key, mods=NOMOD):
    QApplication.sendEvent(vp, QKeyEvent(QEvent.Type.KeyPress, key, mods))


def _settle(app, vp):
    for _ in range(60):
        if vp._anim is None:
            return
        pump(app, 0.03)


@pytest.mark.parametrize('daz,de', [(30, 0), (0, 25), (-70, -20), (15, 80)])
def test_orbit_about_a_point_keeps_it_where_it_is_in_the_view(daz, de):
    cam = OrbitCam()
    cam.target, cam.dist, cam.az, cam.el = np.array([0.4, 0.1, 0.3]), 2.0, 40.0, 25.0
    P = np.array([0.6, 0.3, 0.5])
    before = cam.view() @ np.r_[P, 1.0]
    el = cam.el
    orbit_about(cam, P, daz, de)
    assert cam.view() @ np.r_[P, 1.0] == pytest.approx(before, abs=1e-9)
    assert cam.el == pytest.approx(min(max(el + de, EL_MIN), EL_MAX), abs=1e-6)


def test_keyboard_views_orbit_pan_and_zoom(app, win):
    vp = win.viewport
    vp.setFocus()
    cam = vp.cam
    _key(vp, Qt.Key.Key_7)
    _settle(app, vp)
    assert cam.el == pytest.approx(89.99, abs=0.05) and cam.ortho
    _key(vp, Qt.Key.Key_0)
    _settle(app, vp)
    assert not cam.ortho
    az = cam.az
    _key(vp, Qt.Key.Key_Left)
    assert cam.az == pytest.approx(az + 15.0)
    dist = cam.dist
    _key(vp, Qt.Key.Key_Plus)
    assert cam.dist == pytest.approx(dist * 0.8)


def test_navigation_control_orbits_zooms_and_frames(app, win):
    vp = win.viewport
    nav = vp.nav
    c = nav.centre()
    vp.set_view('persp')
    az = vp.cam.az
    _drag(app, vp, c, c + QPointF(-60, 0))
    assert vp.cam.az == pytest.approx(az + 60 * 0.4)
    btn = {k: nav.button_centre(i) for i, (k, _, _) in enumerate(nav.BUTTONS)}
    dist = vp.cam.dist
    _drag(app, vp, btn['zoom'], btn['zoom'] + QPointF(0, -80))
    assert vp.cam.dist < dist * 0.6
    QTest.mouseClick(vp, LEFT, NOMOD, btn['frame'].toPoint())
    _settle(app, vp)
    lo, hi, _ = vp.provider.home_box(vp)
    assert vp.cam.target == pytest.approx((np.asarray(lo) + np.asarray(hi)) / 2, abs=1e-6)


def test_zoom_reaches_robot_scale(app, win):
    vp = win.viewport
    for _ in range(60):
        vp._zoom_by(0.5)
    assert vp.cam.dist == pytest.approx(0.03)          # close enough to read a fingertip


def test_scale_bar_reads_millimetres_close_up(app, win):
    from PyQt6.QtGui import QImage, QPainter
    vp = win.viewport
    vp.cam.dist = 0.05
    img = QImage(vp.width(), vp.height(), QImage.Format.Format_ARGB32)
    p = QPainter(img)
    texts = []
    orig = p.drawText
    p.drawText = lambda *a: (texts.append(a[-1]), orig(*a))
    vp._paint_scale(p)
    p.end()
    assert any(str(t).endswith(' mm') for t in texts)
