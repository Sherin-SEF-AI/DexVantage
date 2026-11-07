"""Publication figures: the charts the analysis pages draw, as vector files for a paper (SVG written here directly,
PDF through Qt's own PDF writer, which DexVantage already depends on).

    spec = curves(series, xr, yr, bands=(), hlines=(), vlines=(), markers=(), xlabel='', ylabel='', title='',
                  reverse_x=False, source='')            the data of a page's curve chart (ui/pages/_curves.py)
    spec = bars(rows, unit='', title='', xlabel='', source='')     rows [(label, value, colour)]: a bar list
    to_svg(spec, width='single', height_mm=None, font_pt=8, palette='tol', title=False, meta=None) -> str
    write(spec, path, **same options) -> {'path', 'format', 'width_mm', 'height_mm', 'series'}  (.svg or .pdf)

A spec is JSON (schema 'dexvantage.figure/1'): {'kind': 'curves' | 'bars', 'title', 'xlabel', 'ylabel', 'xr', 'yr',
'reverse_x', 'series': [{'label', 'x', 'y' (null breaks a line), 'color', 'style': 'line' | 'dash' | 'points'}],
'bands': [{'x0', 'x1', 'color', 'label'}], 'hlines' / 'vlines': [{'value', 'label'}], 'markers': [{'x', 'y', 'color',
'label'}], 'rows': [{'label', 'value', 'color'}] (bars), 'unit', 'source'}. The page's chart builds it from the data
it draws, so the figure shows the same numbers; the SVG carries the spec in its <metadata> so the numbers travel with
the picture.

Publication defaults (each one a default, an option of write()):
- width: one column 88.9 mm or two columns 182 mm, the widths IEEE's author centre gives ("One column width: 3.5
  inches, 88.9 millimeters, or 21 picas"; "Two columns width: 7.16 inches, 182 millimeters, or 43 picas", IEEE Author
  Center, "Resolution and Size", read 2026-11-07); any width in mm can be given instead.
- height: 0.66 of the width for one column, 0.40 for two (DexVantage defaults).
- text: 8 pt axis labels and legend, 7 pt tick labels, Helvetica (Arial, then any sans-serif, where it is missing)
  (DexVantage defaults; IEEE's page states no figure font size).
- lines 1 pt, axes 0.6 pt, a light y grid (DexVantage defaults).
- series colours: Paul Tol's colour-blind-safe "bright" scheme in its order (#4477AA, #EE6677, #228833, #CCBB44,
  #66CCEE, #AA3377; grey #BBBBBB for a seventh and later series, drawn dashed), from Paul Tol's Notes, "Qualitative
  colour schemes", https://sronpersonalpages.nl/~pault/ (page dated 16 August 2026, read 2026-11-07); palette='chart'
  keeps the colours the page draws (camera identity colours).
- no title (the caption carries it; title=True draws the chart's own); axes labelled with their units as the page
  labels them; a legend for two or more series.

Known limits: text widths are estimated (0.56 em per character), not measured from the font, so a long tick or legend
label can sit closer to its neighbour than the layout intends; the PDF needs a Qt GUI platform (offscreen is enough)
and embeds whichever font Qt resolves for Helvetica; only the curve and bar charts have data specs (other page charts
can be saved "as drawn" through Qt's SVG generator in the window, with the screen's colours and sizes); no log axes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from xml.sax.saxutils import escape as _xml_escape

from .jobs import StudyError

SCHEMA = 'dexvantage.figure/1'
WIDTHS_MM = {'single': 88.9, 'double': 182.0}           # IEEE Author Center, "Resolution and Size", read 2026-11-07
HEIGHT_RATIO = {'single': 0.66, 'double': 0.40}          # DexVantage defaults
FONT_PT = 8.0                                            # DexVantage default
TICK_PT = 7.0                                            # DexVantage default
LINE_PT = 1.0
AXIS_PT = 0.6
FONT_FAMILY = 'Helvetica, Arial, sans-serif'
TOL_BRIGHT = ('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377')
TOL_GREY = '#BBBBBB'
INK, INK2, GRID = '#000000', '#444444', '#E2E2E2'
PT_PER_MM = 72.0 / 25.4
CHAR_EM = 0.56


class FigureError(StudyError):
    """A figure spec that cannot be drawn; the message says why."""


# ====================================================================== specs
def _col(c) -> str:
    s = str(c or '').strip()
    if len(s) == 9 and s.startswith('#'):                # #aarrggbb (Qt's HexArgb) -> #rrggbb
        s = '#' + s[3:]
    return s if len(s) == 7 and s.startswith('#') else '#444444'


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def curves(series, xr, yr, bands=(), hlines=(), vlines=(), markers=(), xlabel='', ylabel='', title='',
           reverse_x=False, source='') -> dict:
    """A curves spec from CurveChart-style data: series [(label, xs, ys, colour, style)], bands [(x0, x1, colour,
    label)], hlines / vlines [(value, label)], markers [(x, y, colour, label)]."""
    return {'schema': SCHEMA, 'kind': 'curves', 'title': str(title or ''), 'xlabel': str(xlabel or ''),
            'ylabel': str(ylabel or ''), 'xr': [float(xr[0]), float(xr[1])], 'yr': [float(yr[0]), float(yr[1])],
            'reverse_x': bool(reverse_x), 'source': str(source or ''),
            'series': [{'label': str(s[0]), 'x': [_num(x) for x in s[1]], 'y': [_num(y) for y in s[2]],
                        'color': _col(s[3]), 'style': s[4] if s[4] in ('line', 'dash', 'points') else 'line'}
                       for s in series],
            'bands': [{'x0': float(b[0]), 'x1': float(b[1]), 'color': _col(b[2]), 'label': str(b[3] or '')}
                      for b in bands],
            'hlines': [{'value': float(v), 'label': str(lab or '')} for v, lab in hlines],
            'vlines': [{'value': float(v), 'label': str(lab or '')} for v, lab in vlines],
            'markers': [{'x': float(m[0]), 'y': float(m[1]), 'color': _col(m[2]), 'label': str(m[3] or '')}
                        for m in markers]}


def bars(rows, unit='', title='', xlabel='', source='') -> dict:
    """A bars spec: rows [(label, value, colour)] drawn as horizontal bars from zero."""
    return {'schema': SCHEMA, 'kind': 'bars', 'title': str(title or ''), 'xlabel': str(xlabel or ''),
            'unit': str(unit or ''), 'source': str(source or ''),
            'rows': [{'label': str(r[0]), 'value': _num(r[1]), 'color': _col(r[2])} for r in rows]}


def check_spec(spec) -> dict:
    if not isinstance(spec, dict) or spec.get('kind') not in ('curves', 'bars'):
        raise FigureError('A figure spec is a JSON object with "kind": "curves" or "bars".')
    if spec['kind'] == 'curves':
        for k in ('xr', 'yr'):
            r = spec.get(k)
            if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(_num(v) is not None for v in r)):
                raise FigureError(f'The figure\'s {k} must be two finite numbers [low, high].')
        if not spec.get('series'):
            raise FigureError('The figure has no series to draw.')
        for s in spec['series']:
            if len(s.get('x') or []) != len(s.get('y') or []):
                raise FigureError(f'Series "{s.get("label", "")}" has {len(s.get("x") or [])} x values and '
                                  f'{len(s.get("y") or [])} y values; they must match.')
    elif not spec.get('rows'):
        raise FigureError('The bar figure has no rows to draw.')
    return spec


# ====================================================================== layout helpers
def nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    """Round tick values in [lo, hi], about n of them (steps of 1, 2, 2.5 or 5 × 10^k), as ui/charts.nice_ticks."""
    span = hi - lo
    if not (span > 0) or not math.isfinite(span):
        return [lo]
    mag = 10 ** math.floor(math.log10(span / max(n, 1)))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10, 20) if span / (m * mag) <= n + 1e-9)
    k0 = math.ceil(lo / step - 1e-9)
    out, v = [], k0 * step
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v = (len(out) + k0) * step
    return out


def tick_text(v: float) -> str:
    t = f'{round(v, 6):g}'
    return t.replace('-', '−')                     # a typographic minus


def text_w(s: str, size: float) -> float:
    return len(s) * size * CHAR_EM


def _f(v: float) -> str:
    t = f'{v:.2f}'.rstrip('0').rstrip('.')
    return '0' if t in ('-0', '') else t


def _a(s) -> str:
    return _xml_escape(str(s), {'"': '&quot;'})


def size_mm(width='single', height_mm=None) -> tuple[float, float]:
    if isinstance(width, str):
        if width not in WIDTHS_MM:
            raise FigureError(f'The figure width is "single", "double" or a number of mm, not "{width}".')
        w = WIDTHS_MM[width]
        ratio = HEIGHT_RATIO[width]
    else:
        w = _num(width)
        if w is None or not (10 <= w <= 1000):
            raise FigureError('The figure width must be between 10 and 1000 mm.')
        ratio = HEIGHT_RATIO['single'] if w < 130 else HEIGHT_RATIO['double']
    h = _num(height_mm) if height_mm is not None else w * ratio
    if h is None or not (10 <= h <= 1000):
        raise FigureError('The figure height must be between 10 and 1000 mm.')
    return float(w), float(h)


def series_colours(spec: dict, palette: str) -> list:
    """(colour, style) per series: the palette's colours in order, never cycled (a seventh and later series is grey
    and dashed)."""
    out = []
    for i, s in enumerate(spec.get('series') or []):
        if palette == 'chart':
            out.append((_col(s.get('color')), s.get('style') or 'line'))
        elif i < len(TOL_BRIGHT):
            out.append((TOL_BRIGHT[i], s.get('style') or 'line'))
        else:
            out.append((TOL_GREY, 'dash' if (s.get('style') or 'line') == 'line' else s.get('style')))
    return out


# ====================================================================== SVG
class _Svg:
    def __init__(self, family):
        self.parts = []
        self.family = family

    def text(self, x, y, s, size, anchor='start', colour=INK, rotate=None, baseline=None, cls=''):
        tr = f' transform="rotate({_f(rotate)} {_f(x)} {_f(y)})"' if rotate else ''
        db = f' dominant-baseline="{baseline}"' if baseline else ''
        c = f' class="{cls}"' if cls else ''
        self.parts.append(f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(size)}" text-anchor="{anchor}" '
                          f'fill="{colour}"{db}{tr}{c}>{_xml_escape(str(s))}</text>')

    def line(self, x0, y0, x1, y1, colour, width, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}" stroke="{colour}" '
                          f'stroke-width="{_f(width)}"{d}/>')


def _legend_layout(entries, width, size) -> list:
    """[(row, x)] of each legend entry flowing left to right in rows no wider than width."""
    out, row, x = [], 0, 0.0
    for lab, _c, _s in entries:
        w = 14 + text_w(lab, size) + 8
        if x > 0 and x + w > width:
            row, x = row + 1, 0.0
        out.append((row, x))
        x += w
    return out


def to_svg(spec: dict, width='single', height_mm=None, font_pt: float = FONT_PT, tick_pt: float | None = None,
           palette: str = 'tol', title: bool = False, meta: dict | None = None, font_family: str = FONT_FAMILY,
           grid: bool = True) -> str:
    """The figure as an SVG document (text)."""
    from .. import __version__
    check_spec(spec)
    if palette not in ('tol', 'chart'):
        raise FigureError('The palette is "tol" (colour-blind safe) or "chart" (the colours the page draws).')
    wmm, hmm = size_mm(width, height_mm)
    W, H = wmm * PT_PER_MM, hmm * PT_PER_MM
    fs = float(font_pt)
    ts = float(tick_pt) if tick_pt is not None else max(4.0, fs - (FONT_PT - TICK_PT))
    uid = 'dv' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    S = _Svg(font_family)
    pad = 2.0
    top = pad + (fs * 1.5 if title and spec.get('title') else 0.0)
    if spec['kind'] == 'curves':
        _curves(S, spec, W, H, fs, ts, pad, top, palette, grid, uid, title)
    else:
        _bars(S, spec, W, H, fs, ts, pad, top, palette, grid, title)
    meta_doc = json.dumps({'dexvantage': __version__, 'spec': spec, **(meta or {})}, ensure_ascii=False, sort_keys=True)
    desc = 'Written by DexVantage ' + __version__ + (': ' + spec['source'] if spec.get('source') else '')
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{_f(wmm)}mm" height="{_f(hmm)}mm" '
            f'viewBox="0 0 {_f(W)} {_f(H)}" font-family="{_a(font_family)}">\n'
            f'<title>{_xml_escape(spec.get("title") or spec.get("ylabel") or "DexVantage figure")}</title>\n'
            f'<desc>{_xml_escape(desc)}</desc>\n<metadata id="{uid}-data">{_xml_escape(meta_doc)}'
            f'</metadata>\n<rect width="{_f(W)}" height="{_f(H)}" fill="#FFFFFF"/>\n')
    return head + '\n'.join(S.parts) + '\n</svg>\n'


def _curves(S, spec, W, H, fs, ts, pad, top, palette, grid, uid, title):
    xr, yr = [float(v) for v in spec['xr']], [float(v) for v in spec['yr']]
    if not xr[1] > xr[0]:
        xr[1] = xr[0] + 1.0
    if not yr[1] > yr[0]:
        yr[1] = yr[0] + 1.0
    cols = series_colours(spec, palette)
    entries = [(s['label'], c, st) for s, (c, st) in zip(spec['series'], cols)] if len(spec['series']) > 1 else []
    yt = nice_ticks(yr[0], yr[1])
    xt = nice_ticks(xr[0], xr[1])
    ytw = max([text_w(tick_text(v), ts) for v in yt] + [0.0])
    left = pad + (fs * 1.25 if spec.get('ylabel') else 0.0) + ytw + 4.0
    right = pad + max(3.0, text_w(tick_text(xt[-1]), ts) / 2 if xt else 3.0)
    bottom = pad + (fs * 1.3 if spec.get('xlabel') else 0.0) + ts * 1.25 + 3.5
    lay = _legend_layout(entries, W - left - right, fs) if entries else []
    rows = (max(r for r, _ in lay) + 1) if lay else 0
    top_plot = top + rows * fs * 1.35 + (2.0 if rows else 0.0) + ts * 0.6
    x0, x1, y0, y1 = left, W - right, top_plot, H - bottom
    if x1 - x0 < 20 or y1 - y0 < 20:
        raise FigureError('The figure is too small for its labels: make it larger or the text smaller.')

    def X(x):
        f = (x - xr[0]) / (xr[1] - xr[0])
        return x0 + ((1.0 - f) if spec.get('reverse_x') else f) * (x1 - x0)

    def Y(y):
        return y1 - (y - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
    if title and spec.get('title'):
        S.text(pad, pad + fs, spec['title'], fs * 1.1, cls='title')
    S.parts.append(f'<defs><clipPath id="{uid}-plot"><rect x="{_f(x0)}" y="{_f(y0)}" width="{_f(x1 - x0)}" '
                   f'height="{_f(y1 - y0)}"/></clipPath></defs>')
    if grid:
        for v in yt:
            S.line(x0, Y(v), x1, Y(v), GRID, 0.4)
    S.parts.append(f'<g clip-path="url(#{uid}-plot)">')
    for b in spec.get('bands') or []:
        a, c = sorted((X(b['x0']), X(b['x1'])))
        S.parts.append(f'<rect class="band" x="{_f(a)}" y="{_f(y0)}" width="{_f(c - a)}" height="{_f(y1 - y0)}" '
                       f'fill="{_col(b["color"])}" fill-opacity="0.16"/>')
        if b.get('label'):
            S.text(a + 1.5, y1 - 2.0, b['label'], ts - 0.5, colour=INK2)      # at the foot: clear of the top lines
    for h in spec.get('hlines') or []:
        S.line(x0, Y(h['value']), x1, Y(h['value']), INK2, AXIS_PT, dash='2.5 2')
        if h.get('label'):
            S.text(x1 - 1.5, Y(h['value']) - 1.6, h['label'], ts - 0.5, anchor='end', colour=INK2)
    for v in spec.get('vlines') or []:
        S.line(X(v['value']), y0, X(v['value']), y1, INK2, AXIS_PT)
        if v.get('label'):
            S.text(X(v['value']) + 1.5, y0 + ts, v['label'], ts - 0.5, colour=INK2)
    for s, (c, st) in zip(spec['series'], cols):
        S.parts.append(f'<g class="series" data-label="{_a(s["label"])}">')
        pts = list(zip(s['x'], s['y']))
        if st == 'points':
            for x, y in pts:
                if x is not None and y is not None:
                    S.parts.append(f'<circle cx="{_f(X(x))}" cy="{_f(Y(y))}" r="1.3" fill="{c}"/>')
        else:
            dash = ' stroke-dasharray="3 2"' if st == 'dash' else ''
            run = []
            for x, y in pts + [(None, None)]:
                if x is None or y is None:
                    if len(run) > 1:
                        S.parts.append(f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in run)}" '
                                       f'fill="none" stroke="{c}" stroke-width="{_f(LINE_PT)}" '
                                       f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
                    elif len(run) == 1:
                        S.parts.append(f'<circle cx="{_f(run[0][0])}" cy="{_f(run[0][1])}" r="0.9" fill="{c}"/>')
                    run = []
                else:
                    run.append((X(x), Y(y)))
        S.parts.append('</g>')
    for m in spec.get('markers') or []:
        S.parts.append(f'<circle class="marker" cx="{_f(X(m["x"]))}" cy="{_f(Y(m["y"]))}" r="2.2" '
                       f'fill="{_col(m["color"])}" stroke="#FFFFFF" stroke-width="0.6"/>')
        if m.get('label'):
            S.text(X(m['x']) + 3.2, Y(m['y']) - 3.0, m['label'], ts - 0.5)
    S.parts.append('</g>')
    S.line(x0, y1, x1, y1, INK, AXIS_PT)
    S.line(x0, y0, x0, y1, INK, AXIS_PT)
    for v in yt:
        S.line(x0 - 2.5, Y(v), x0, Y(v), INK, AXIS_PT)
        S.text(x0 - 3.5, Y(v) + ts * 0.35, tick_text(v), ts, anchor='end', cls='tick')
    for v in xt:
        S.line(X(v), y1, X(v), y1 + 2.5, INK, AXIS_PT)
        S.text(X(v), y1 + 3.0 + ts * 0.85, tick_text(v), ts, anchor='middle', cls='tick')
    if spec.get('xlabel'):
        S.text((x0 + x1) / 2, H - pad - fs * 0.15, spec['xlabel'], fs, anchor='middle', cls='axis-label')
    if spec.get('ylabel'):
        cx, cy = pad + fs * 0.85, (y0 + y1) / 2
        S.text(cx, cy, spec['ylabel'], fs, anchor='middle', rotate=-90, cls='axis-label')
    for (lab, c, st), (r, x) in zip(entries, lay):
        yy = top + r * fs * 1.35 + fs * 0.9
        xx = x0 + x
        if st == 'points':
            S.parts.append(f'<circle cx="{_f(xx + 6)}" cy="{_f(yy - fs * 0.3)}" r="1.5" fill="{c}"/>')
        else:
            S.line(xx, yy - fs * 0.3, xx + 12, yy - fs * 0.3, c, LINE_PT * 1.2, dash='3 2' if st == 'dash' else None)
        S.text(xx + 14, yy, lab, fs, cls='legend')


def _bars(S, spec, W, H, fs, ts, pad, top, palette, grid, title):
    rows = spec['rows']
    vals = [r['value'] for r in rows if r.get('value') is not None]
    vmax = max(vals + [0.0])
    vmax = vmax if vmax > 0 else 1.0
    xt = nice_ticks(0.0, vmax)
    if xt[-1] < vmax:
        step = xt[1] - xt[0] if len(xt) > 1 else vmax
        xt.append(round(xt[-1] + step, 10))
    top_v = xt[-1]
    labw = max(text_w(r['label'], fs) for r in rows)
    left = pad + labw + 4.0
    valw = max(text_w(f'{r["value"]:.3g}' if r.get('value') is not None else '—', ts) for r in rows)
    right = pad + valw + 3.0
    xlab = spec.get('xlabel') or spec.get('unit') or ''
    bottom = pad + (fs * 1.3 if xlab else 0.0) + ts * 1.25 + 3.5
    if title and spec.get('title'):
        S.text(pad, pad + fs, spec['title'], fs * 1.1, cls='title')
    x0, x1, y0, y1 = left, W - right, top + ts * 0.6, H - bottom
    if x1 - x0 < 20 or y1 - y0 < 8 * len(rows) * 0.5:
        raise FigureError('The figure is too small for its rows: make it taller or wider.')
    rh = (y1 - y0) / len(rows)

    def X(v):
        return x0 + v / top_v * (x1 - x0)
    if grid:
        for v in xt[1:]:
            S.line(X(v), y0, X(v), y1, GRID, 0.4)
    S.parts.append('<g class="series" data-label="' + _a(spec.get('title') or xlab or 'bars') + '">')
    for i, r in enumerate(rows):
        c = _col(r['color']) if palette == 'chart' else (TOL_BRIGHT[i] if i < len(TOL_BRIGHT) else TOL_GREY)
        yc = y0 + (i + 0.5) * rh
        S.text(x0 - 3.0, yc + fs * 0.35, r['label'], fs, anchor='end', cls='bar-label')
        v = r.get('value')
        if v is not None and v > 0:
            bh = min(rh * 0.62, fs * 1.4)
            S.parts.append(f'<rect class="bar" x="{_f(x0)}" y="{_f(yc - bh / 2)}" width="{_f(X(v) - x0)}" '
                           f'height="{_f(bh)}" fill="{c}"/>')
        S.text((X(v) if v is not None and v > 0 else x0) + 2.0, yc + ts * 0.35,
               f'{v:.3g}' if v is not None else '—', ts, cls='value')
    S.parts.append('</g>')
    S.line(x0, y0, x0, y1, INK, AXIS_PT)
    S.line(x0, y1, x1, y1, INK, AXIS_PT)
    for v in xt:
        S.line(X(v), y1, X(v), y1 + 2.5, INK, AXIS_PT)
        S.text(X(v), y1 + 3.0 + ts * 0.85, tick_text(v), ts, anchor='middle', cls='tick')
    if xlab:
        S.text((x0 + x1) / 2, H - pad - fs * 0.15, xlab, fs, anchor='middle', cls='axis-label')


# ====================================================================== files
def _qt_app():
    from PyQt6.QtGui import QGuiApplication
    app = QGuiApplication.instance()
    if app is None:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QGuiApplication([])
    return app


def svg_to_pdf(svg: str, path: str, width_mm: float, height_mm: float, title: str = ''):
    """Draw an SVG text into a one-page vector PDF of the given size (Qt: QSvgRenderer into QPdfWriter)."""
    try:
        from PyQt6.QtCore import QByteArray, QMarginsF, QRectF, QSizeF
        from PyQt6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
        from PyQt6.QtSvg import QSvgRenderer
    except ImportError:
        raise FigureError('PDF figures need PyQt6 with its QtSvg module; write an SVG instead.') from None
    from .. import __version__
    _app = _qt_app()                                    # noqa: F841 — kept alive while drawing
    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QSizeF(width_mm, height_mm), QPageSize.Unit.Millimeter, '',
                            QPageSize.SizeMatchPolicy.ExactMatch))
    w.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter)
    w.setResolution(1200)
    w.setCreator(f'DexVantage {__version__}')
    w.setTitle(title or 'DexVantage figure')
    r = QSvgRenderer(QByteArray(svg.encode('utf-8')))
    if not r.isValid():
        raise FigureError('The figure could not be drawn into a PDF (Qt could not read its SVG).')
    p = QPainter(w)
    r.render(p, QRectF(0, 0, w.width(), w.height()))
    p.end()


def write(spec: dict, path: str, fmt: str | None = None, **options) -> dict:
    """Write the figure as .svg or .pdf (by fmt, else the file's extension)."""
    fmt = (fmt or os.path.splitext(path)[1].lstrip('.') or 'svg').lower()
    if fmt not in ('svg', 'pdf'):
        raise FigureError(f'Figures are written as SVG or PDF, not {fmt}.')
    svg = to_svg(spec, **options)
    wmm, hmm = size_mm(options.get('width', 'single'), options.get('height_mm'))
    tmp = path + '.part'
    try:
        if fmt == 'svg':
            with open(tmp, 'w', encoding='utf-8') as fh:
                fh.write(svg)
        else:
            svg_to_pdf(svg, tmp, wmm, hmm, spec.get('title') or '')
        os.replace(tmp, path)
    except OSError as e:
        raise FigureError(f'Could not write {os.path.basename(path)}: {e.strerror or e}.') from None
    n = len(spec.get('series') or []) if spec['kind'] == 'curves' else 1
    return {'path': path, 'format': fmt, 'width_mm': wmm, 'height_mm': hmm, 'series': n}


def read_spec(text_or_path: str) -> dict:
    """A figure spec from JSON text or a .json file (FigureError when it cannot be read)."""
    t = text_or_path
    try:
        if not t.lstrip().startswith('{'):
            with open(t, encoding='utf-8') as fh:
                t = fh.read()
        return check_spec(json.loads(t))
    except (OSError, ValueError) as e:
        raise FigureError(f'The figure spec could not be read as JSON ({e}).') from None
