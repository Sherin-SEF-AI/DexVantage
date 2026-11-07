"""A small PDF writer for the report (spec §9.2 reports): text in the standard Type 1 fonts, rules, filled boxes and
bar charts, A4 pages, Flate-compressed content streams. Pure Python (zlib), so the PDF report needs no optional
package.

    doc = PdfDoc(title='...')
    doc.heading('Coverage'); doc.para('text ...'); doc.table([['a', 'b'], ...], widths=[0.3, 0.7])
    doc.bars('Contact visibility', ['wrist', 'overhead'], [91.0, 64.0], unit='%', colors=[...])
    data = doc.tobytes()

The file follows the PDF 1.4 structure of ISO 32000-1:2008 (§7.5 file structure: header, body, cross-reference
table, trailer; §9.6.2.2 the standard 14 fonts, which a reader always has; §7.4.4 FlateDecode). Text is set in
Helvetica, Helvetica-Bold and Courier with WinAnsiEncoding (cp1252); characters outside it are written in words
(≥ as >=, σ as sigma) or, failing that, as their closest unaccented letter, so nothing is dropped silently.

Known limits: line breaking uses approximate glyph widths (Helvetica's widths by character class, at most a few per
cent off), so a line may end a little short of the margin; no hyphenation, images, links, tagged-PDF structure or
embedded fonts; a table cell's text is cut to one line per row with an ellipsis when it is longer than its column.
"""
from __future__ import annotations

import datetime as _dt
import unicodedata
import zlib

PAGE_W, PAGE_H = 595.276, 841.89        # A4 in points (210 × 297 mm)
MARGIN = 54.0
FONTS = {'F1': 'Helvetica', 'F2': 'Helvetica-Bold', 'F3': 'Courier'}
WORDS = {'≥': '>=', '≤': '<=', '−': '-', '≠': '!=', '≈': '~', '→': '->', '←': '<-', '√': 'sqrt', 'σ': 'sigma',
         'π': 'pi', 'θ': 'theta', 'τ': 'tau', 'ρ': 'rho', 'ω': 'omega', 'δ': 'delta', 'Δ': 'Delta', 'λ': 'lambda',
         'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'ε': 'epsilon', 'φ': 'phi', 'ψ': 'psi', 'η': 'eta', 'κ': 'kappa',
         'χ': 'chi', 'Σ': 'Sum', 'ν': 'nu', '⁻': '^-', '⁰': '^0', '¹': '^1', '⁴': '^4', '⁵': '^5', '⁶': '^6',
         '⁷': '^7', '⁸': '^8', '⁹': '^9', '₀': '0', '₁': '1', '₂': '2', '₃': '3', '✓': 'ok', '✕': 'x', '…': '...',
         ' ': ' ', ' ': ' ', ' ': ' ', '▸': '>', '·': '·'}


def to_winansi(text: str) -> bytes:
    """text in cp1252 (WinAnsiEncoding), with WORDS for what it cannot hold and accents dropped as a last resort."""
    out = bytearray()
    for ch in str(text):
        ch = WORDS.get(ch, ch)
        try:
            out += ch.encode('cp1252')
            continue
        except UnicodeEncodeError:
            pass
        base = unicodedata.normalize('NFKD', ch).encode('ascii', 'ignore')
        out += base if base else b'?'
    return bytes(out)


def _width(text: str, size: float, font: str = 'F1') -> float:
    """Approximate width (pt) of text (Helvetica's widths by character class; Courier 0.6 em)."""
    if font == 'F3':
        return 0.6 * size * len(text)
    w = 0.0
    for c in text:
        if c in 'il.,:;\'|!jI':
            w += 0.24
        elif c in 'fjtr ()[]-/':
            w += 0.32
        elif c in 'mwMW':
            w += 0.86
        elif c.isupper():
            w += 0.68
        elif c.isdigit():
            w += 0.556
        else:
            w += 0.54
    return w * size * (1.06 if font == 'F2' else 1.0)


def _esc(b: bytes) -> str:
    out = []
    for x in b:
        c = chr(x)
        if c in '\\()':
            out.append('\\' + c)
        elif 32 <= x < 127:
            out.append(c)
        else:
            out.append(f'\\{x:03o}')
    return ''.join(out)


def _rgb(hexcolor: str) -> tuple:
    h = hexcolor.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class PdfDoc:
    def __init__(self, title: str = '', author: str = 'DexVantage'):
        self.title, self.author = title, author
        self.pages: list[list[str]] = []
        self.y = 0.0
        self._new_page()

    # ------------------------------------------------------------------ layout
    def _new_page(self):
        self.pages.append([])
        self.y = PAGE_H - MARGIN

    @property
    def width(self) -> float:
        return PAGE_W - 2 * MARGIN

    def _need(self, h: float):
        if self.y - h < MARGIN:
            self._new_page()

    def _op(self, s: str):
        self.pages[-1].append(s)

    def text_at(self, x: float, y: float, text: str, size: float = 10.0, font: str = 'F1', color: str = '#0b0b0b'):
        r, g, b = _rgb(color)
        self._op(f'BT /{font} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({_esc(to_winansi(text))}) '
                 'Tj ET')

    def rect(self, x, y, w, h, fill: str | None = '#e1e0d9', stroke: str | None = None, lw: float = 0.5):
        ops = []
        if fill:
            ops.append('{:.3f} {:.3f} {:.3f} rg'.format(*_rgb(fill)))
        if stroke:
            ops.append('{:.3f} {:.3f} {:.3f} RG {:.2f} w'.format(*_rgb(stroke), lw))
        ops.append(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} re ' + ('B' if fill and stroke else 'f' if fill else 'S'))
        self._op(' '.join(ops))

    def line(self, x0, y0, x1, y1, color: str = '#c3c2b7', lw: float = 0.5, dash: bool = False):
        d = '[3 3] 0 d ' if dash else '[] 0 d '
        self._op('{:.3f} {:.3f} {:.3f} RG {:.2f} w '.format(*_rgb(color), lw) + d +
                 f'{x0:.2f} {y0:.2f} m {x1:.2f} {y1:.2f} l S')

    def wrap(self, text: str, size: float, width: float, font: str = 'F1') -> list:
        lines = []
        for para in str(text).split('\n'):
            words, cur = para.split(' '), ''
            for w in words:
                t = (cur + ' ' + w) if cur else w
                if _width(t, size, font) <= width or not cur:
                    cur = t
                else:
                    lines.append(cur)
                    cur = w
            lines.append(cur)
        return lines

    # ------------------------------------------------------------------ blocks
    def heading(self, text: str, size: float = 15.0):
        self._need(size * 2.6 + 48)             # keep a heading with the first lines under it
        self.y -= size * 1.2
        self.text_at(MARGIN, self.y, text, size, 'F2')
        self.y -= size * 0.6

    def para(self, text: str, size: float = 9.5, color: str = '#52514e', font: str = 'F1', indent: float = 0.0):
        for ln in self.wrap(text, size, self.width - indent, font):
            self._need(size * 1.45)
            self.y -= size * 1.45
            self.text_at(MARGIN + indent, self.y, ln, size, font, color)
        self.y -= size * 0.4

    def status(self, word: str, level: str, value: str = ''):
        """A status chip (a coloured box with the STATUS word) and the headline figure next to it."""
        colors = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
        c = colors.get(level, '#898781')
        self._need(24)
        self.y -= 18
        w = _width(word, 9.5, 'F2') + 12
        self.rect(MARGIN, self.y - 4, w, 15, fill=c)
        self.text_at(MARGIN + 6, self.y, word, 9.5, 'F2', '#ffffff' if level != 'warning' else '#0b0b0b')
        if value:
            self.text_at(MARGIN + w + 10, self.y, value, 12, 'F2')
        self.y -= 6

    def table(self, rows: list, widths: list | None = None, size: float = 8.5, header: bool = True):
        if not rows:
            return
        n = max(len(r) for r in rows)
        widths = widths or [1.0 / n] * n
        cols = [w * self.width for w in widths]
        for i, r in enumerate(rows):
            self._need(size * 1.7)
            self.y -= size * 1.6
            x = MARGIN
            for j, cell in enumerate(list(r) + [''] * (n - len(r))):
                t = str(cell if cell is not None else '—')
                lim = cols[j] - 4
                font = 'F2' if header and i == 0 else 'F1'
                if _width(t, size, font) > lim:
                    while t and _width(t + '…', size, font) > lim:
                        t = t[:-1]
                    t += '…'
                self.text_at(x, self.y, t, size, font, '#0b0b0b' if header and i == 0 else '#52514e')
                x += cols[j]
            self.line(MARGIN, self.y - size * 0.45, MARGIN + self.width, self.y - size * 0.45, '#e1e0d9', 0.4)
        self.y -= size * 0.6

    def bars(self, title: str, labels: list, values: list, unit: str = '', colors: list | None = None,
             hline: float | None = None, hline_label: str = ''):
        """A horizontal bar chart (one bar per label, value None drawn as 'not stated')."""
        n = len(labels)
        if not n:
            return
        bh, gap = 11.0, 5.0
        h = 22 + n * (bh + gap) + 14
        self._need(h)
        self.y -= 16
        self.text_at(MARGIN, self.y, title, 9.5, 'F2')
        top = self.y - 8
        lab_w = min(170.0, max(_width(str(lb), 8, 'F1') for lb in labels) + 8)
        x0 = MARGIN + lab_w
        span = self.width - lab_w - 60
        vals = [v for v in values if v is not None]
        vmax = max([abs(v) for v in vals] + [abs(hline) if hline is not None else 0.0, 1e-12])
        for i, (lb, v) in enumerate(zip(labels, values)):
            y = top - (i + 1) * (bh + gap)
            self.text_at(MARGIN, y + 2.5, str(lb), 8, 'F1', '#52514e')
            if v is None:
                self.text_at(x0, y + 2.5, 'not stated', 8, 'F1', '#898781')
                continue
            w = abs(v) / vmax * span
            c = (colors[i % len(colors)] if colors else '#2a78d6')
            self.rect(x0, y, max(w, 0.5), bh, fill=c)
            self.text_at(x0 + w + 4, y + 2.5, f'{v:.4g} {unit}'.strip(), 8, 'F1', '#0b0b0b')
        base = top - n * (bh + gap) - 2
        self.line(x0, base, x0, top, '#c3c2b7', 0.6)
        if hline is not None:
            xh = x0 + abs(hline) / vmax * span
            self.line(xh, base, xh, top, '#898781', 0.8, dash=True)
            if hline_label:
                self.text_at(xh + 2, base - 9, hline_label, 7, 'F1', '#898781')
        self.y = base - 14

    def rule(self):
        self._need(10)
        self.y -= 6
        self.line(MARGIN, self.y, MARGIN + self.width, self.y, '#c3c2b7', 0.6)
        self.y -= 4

    # ------------------------------------------------------------------ the file
    def tobytes(self, when: _dt.datetime | None = None) -> bytes:
        when = when or _dt.datetime.now(_dt.timezone.utc)
        objs: list[bytes] = []

        def add(b: bytes) -> int:
            objs.append(b)
            return len(objs)

        font_ids = {k: add(f'<< /Type /Font /Subtype /Type1 /BaseFont /{v} /Encoding /WinAnsiEncoding >>'.encode())
                    for k, v in FONTS.items()}
        fonts = ' '.join(f'/{k} {i} 0 R' for k, i in font_ids.items())
        pages_id = len(objs) + 1 + 2 * len(self.pages)      # after every page and content object
        kids = []
        total = len(self.pages)
        for k, ops in enumerate(self.pages):
            foot = (f'BT /F1 7.5 Tf 0.537 0.529 0.506 rg {MARGIN:.2f} {MARGIN / 2:.2f} Td '
                    f'({_esc(to_winansi(self.title))} - page {k + 1} of {total}) Tj ET')
            data = zlib.compress(('\n'.join(ops + [foot])).encode('latin-1'))
            cid = add(f'<< /Length {len(data)} /Filter /FlateDecode >>\nstream\n'.encode() + data +
                      b'\nendstream')
            pid = add(f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W:.3f} {PAGE_H:.3f}] '
                      f'/Resources << /Font << {fonts} >> >> /Contents {cid} 0 R >>'.encode())
            kids.append(pid)
        assert add(f'<< /Type /Pages /Kids [{" ".join(f"{p} 0 R" for p in kids)}] /Count {len(kids)} >>'
                   .encode()) == pages_id
        cat = add(f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode())
        stamp = when.strftime("D:%Y%m%d%H%M%SZ")
        info = add(f'<< /Title ({_esc(to_winansi(self.title))}) /Author ({_esc(to_winansi(self.author))}) '
                   f'/Producer (DexVantage) /CreationDate ({stamp}) >>'.encode())
        out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offsets = []
        for i, b in enumerate(objs, 1):
            offsets.append(len(out))
            out += f'{i} 0 obj\n'.encode() + b + b'\nendobj\n'
        xref = len(out)
        out += f'xref\n0 {len(objs) + 1}\n0000000000 65535 f \n'.encode()
        for o in offsets:
            out += f'{o:010d} 00000 n \n'.encode()
        out += f'trailer\n<< /Size {len(objs) + 1} /Root {cat} 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n' \
            .encode()
        return bytes(out)
