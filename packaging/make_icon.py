"""Write the DexVantage application icon for the installers: packaging/icons/dexvantage.ico (Windows, 16-256 px) and
packaging/icons/dexvantage.icns (macOS, 16-1024 px), drawn at each size as scripts/make_icon.py draws
dexvantage/resources/icons/dexvantage.png (the window icon and the Linux menu entry): the robot-arm glyph on the dark
base token with a camera dot. Both containers hold PNG images, which Windows Vista+ and macOS 10.7+ read.

    QT_QPA_PLATFORM=offscreen python packaging/make_icon.py

Ported from Vantage packaging/make_icon.py (the .ico and .icns writers); the drawing is DexVantage's.
"""
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402


def draw(n: int) -> QImage:
    """scripts/make_icon.py's 256 px drawing, scaled to n px."""
    from PyQt6.QtSvg import QSvgRenderer
    from dexvantage.ui import icons
    from dexvantage.ui.theme import CAMERA, TOKENS
    img = QImage(n, n, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(n / 256.0, n / 256.0)
    N = 256
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(TOKENS['dark']['base']))
    p.drawRoundedRect(QRectF(8, 8, N - 16, N - 16), 48, 48)
    glyph = QSvgRenderer(QByteArray(icons.svg('robot-arm', TOKENS['dark']['ink'], 1.1).encode()))
    glyph.render(p, QRectF(40, 40, N - 80, N - 80))
    p.setBrush(QColor(CAMERA['dark'][0]))
    p.drawEllipse(QRectF(N - 92, 52, 26, 26))
    p.end()
    return img


def png(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    buf.close()
    return bytes(ba)


def ico(sizes) -> bytes:
    data = [png(draw(s)) for s in sizes]
    head = struct.pack('<HHH', 0, 1, len(sizes))
    off = 6 + 16 * len(sizes)
    entries = b''
    for s, d in zip(sizes, data):
        entries += struct.pack('<BBBBHHII', s % 256, s % 256, 0, 0, 1, 32, len(d), off)
        off += len(d)
    return head + entries + b''.join(data)


def icns() -> bytes:
    body = b''
    for kind, s in ((b'icp4', 16), (b'icp5', 32), (b'icp6', 64), (b'ic07', 128), (b'ic08', 256), (b'ic09', 512),
                    (b'ic10', 1024)):
        d = png(draw(s))
        body += kind + struct.pack('>I', 8 + len(d)) + d
    return b'icns' + struct.pack('>I', 8 + len(body)) + body


def main(out=None):
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    out = out or os.path.join(ROOT, 'packaging', 'icons')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'dexvantage.ico'), 'wb') as fh:
        fh.write(ico([16, 24, 32, 48, 64, 128, 256]))
    with open(os.path.join(out, 'dexvantage.icns'), 'wb') as fh:
        fh.write(icns())
    print('wrote', os.path.join(out, 'dexvantage.ico'), os.path.join(out, 'dexvantage.icns'))


if __name__ == '__main__':
    main()
