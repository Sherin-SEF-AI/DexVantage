"""Render the app icon (resources/icons/dexvantage.png) from the robot-arm glyph on the dark base token.

    QT_QPA_PLATFORM=offscreen python scripts/make_icon.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PyQt6.QtCore import QByteArray, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402
from PyQt6.QtSvg import QSvgRenderer  # noqa: E402

app = QGuiApplication(sys.argv[:1])
from dexvantage.ui import icons  # noqa: E402
from dexvantage.ui.theme import TOKENS, CAMERA  # noqa: E402

N = 256
img = QImage(N, N, QImage.Format.Format_ARGB32_Premultiplied)
img.fill(Qt.GlobalColor.transparent)
p = QPainter(img)
p.setRenderHint(QPainter.RenderHint.Antialiasing)
p.setPen(Qt.PenStyle.NoPen)
p.setBrush(QColor(TOKENS['dark']['base']))
p.drawRoundedRect(QRectF(8, 8, N - 16, N - 16), 48, 48)
glyph = QSvgRenderer(QByteArray(icons.svg('robot-arm', TOKENS['dark']['ink'], 1.1).encode()))
glyph.render(p, QRectF(40, 40, N - 80, N - 80))
p.setBrush(QColor(CAMERA['dark'][0]))
p.drawEllipse(QRectF(N - 92, 52, 26, 26))
p.end()
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, '..', 'dexvantage', 'resources', 'icons', 'dexvantage.png')
img.save(out)
print(out)
