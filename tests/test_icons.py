"""Ported from Vantage tests/test_icons.py: every icon is well-formed single-colour markup and renders."""
import xml.etree.ElementTree as ET

import pytest

from dexvantage.ui.icon_data import ICONS

ROBOT = ['robot-arm', 'gripper', 'depth', 'tactile', 'task-volume', 'grasp', 'joint', 'teleop']
ALLOWED_TAGS = {'path', 'line', 'rect', 'circle', 'ellipse', 'polyline', 'polygon'}
WRAP = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" '
        'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


def test_robot_icons_present():
    assert not [n for n in ROBOT if n not in ICONS]


@pytest.mark.parametrize('name', sorted(ICONS))
def test_markup_rules(name):
    inner = ICONS[name]
    root = ET.fromstring(f'<g xmlns="http://www.w3.org/2000/svg">{inner}</g>')
    assert list(root), name
    for el in root:
        assert el.tag.rsplit('}', 1)[-1] in ALLOWED_TAGS, name
        for attr in ('fill', 'stroke'):
            if attr in el.attrib:
                assert el.attrib[attr] in ('currentColor', 'none'), name
    assert '#' not in inner


@pytest.mark.parametrize('name', sorted(ICONS))
def test_parses_and_renders(name):
    from PyQt6.QtCore import QByteArray, QRectF, Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer
    r = QSvgRenderer(QByteArray(WRAP.format(inner=ICONS[name]).replace('currentColor', '#f0efec').encode()))
    assert r.isValid(), name
    img = QImage(16, 16, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    r.render(p, QRectF(0, 0, 16, 16))
    p.end()
    assert any(img.pixelColor(x, y).alpha() for y in range(16) for x in range(16)), name
