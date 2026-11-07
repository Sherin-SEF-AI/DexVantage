"""Export figure…: the context menu of every page chart (ui/charts.py ChartBase.contextMenuEvent).

A chart that can say what it draws (figure_data(): the curve charts of ui/pages/_curves.py and the bar lists) is
written by core/figures.py as a publication figure, SVG or PDF, one or two columns wide, in the colour-blind-safe
palette, with its data and where it came from (DexVantage's version, the project, the page and its study) in the
file's metadata. Any other chart is saved as drawn on screen through Qt's SVG generator (its screen size and theme
colours, still vector).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import traceback

from PyQt6.QtCore import QRect, QSize
from PyQt6.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox


def page_of(widget):
    """The analysis page (pages.base.StudyPage) a widget sits on, or None."""
    from .pages.base import StudyPage
    w = widget
    while w is not None:
        if isinstance(w, StudyPage):
            return w
        w = w.parentWidget()
    return None


def _window(widget):
    p = page_of(widget)
    return getattr(getattr(p, 'ctx', None), 'window', None) if p is not None else None


def _msg(widget, text, ms=6000):
    win = _window(widget)
    if win is not None:
        win.msg(text, ms)


def _stem(chart) -> str:
    p = page_of(chart)
    base = (getattr(p, 'STUDY', None) or 'figure') if p is not None else 'figure'
    t = getattr(chart, 'title', '') or getattr(chart, 'ylabel', '') or ''
    slug = re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')[:40]
    return f'{base}-{slug}' if slug else base


def meta_for(chart) -> dict:
    """Where the figure came from: DexVantage's version (in core/figures.py), the project, the page and its study,
    the time written (no paths)."""
    p = page_of(chart)
    out = {'written_utc': _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}
    if p is not None:
        out.update(page=p.TITLE, study=p.STUDY)
        try:
            out['project'] = p.ctx.store.project.name
        except AttributeError:
            pass
    return out


def export_chart(chart, width='single', path: str | None = None, palette: str = 'tol') -> str | None:
    """Write the chart's figure (SVG or PDF by the file's extension); the path written, or None."""
    from ..core import figures
    spec = chart.figure_data()
    if spec is None:
        return export_as_drawn(chart, path)
    p = page_of(chart)
    if p is not None and not spec.get('source'):
        spec['source'] = p.TITLE
    if path is None:
        path, flt = QFileDialog.getSaveFileName(chart, 'Export figure', f'{_stem(chart)}.svg',
                                                'SVG figure (*.svg);;PDF figure (*.pdf)')
        if not path:
            return None
        if not os.path.splitext(path)[1]:
            path += '.pdf' if 'PDF' in (flt or '') else '.svg'
    try:
        r = figures.write(spec, path, width=width, palette=palette, meta=meta_for(chart))
    except figures.FigureError as e:
        QMessageBox.warning(chart, 'Could not export the figure', str(e))
        return None
    except Exception as e:                              # noqa: BLE001 — explained, the traceback logged
        traceback.print_exc()
        QMessageBox.warning(chart, 'Could not export the figure',
                            f'The figure could not be written ({type(e).__name__}). Details are in the log.')
        return None
    _msg(chart, f'Exported the figure ({r["width_mm"]:g} × {r["height_mm"]:.0f} mm, {r["format"].upper()}) to '
                f'{os.path.basename(path)}')
    return path


def export_as_drawn(chart, path: str | None = None) -> str | None:
    """The chart as drawn on screen, as SVG (Qt's SVG generator)."""
    try:
        from PyQt6.QtSvg import QSvgGenerator
    except ImportError:
        QMessageBox.warning(chart, 'Could not export the figure', 'Saving a chart as drawn needs the QtSvg module of '
                                                                  'PyQt6, which this installation does not have.')
        return None
    if path is None:
        path, _ = QFileDialog.getSaveFileName(chart, 'Export figure as drawn', f'{_stem(chart)}.svg',
                                              'SVG figure (*.svg)')
        if not path:
            return None
    if not path.lower().endswith('.svg'):
        path += '.svg'
    g = QSvgGenerator()
    g.setFileName(path)
    g.setSize(QSize(chart.width(), chart.height()))
    g.setViewBox(QRect(0, 0, chart.width(), chart.height()))
    p = page_of(chart)
    g.setTitle(getattr(chart, 'title', '') or (p.TITLE if p is not None else 'DexVantage chart'))
    g.setDescription('Saved as drawn by DexVantage: ' + json.dumps(meta_for(chart), ensure_ascii=False))
    chart.render(g)
    if not os.path.isfile(path):
        QMessageBox.warning(chart, 'Could not export the figure', f'{os.path.basename(path)} could not be written: '
                                                                  'choose a folder you can write to.')
        return None
    _msg(chart, f'Saved the chart as drawn to {os.path.basename(path)}')
    return path


def chart_menu(chart, global_pos):
    """Show the chart's context menu at global_pos."""
    m = build_menu(chart)
    m.exec(global_pos)
    return m


def build_menu(chart) -> QMenu:
    """The chart's context menu: Export figure (one column, two columns), Copy figure data; or Export as drawn."""
    m = QMenu(chart)
    spec = None
    try:
        spec = chart.figure_data()
    except Exception:                                   # noqa: BLE001 — a chart that cannot say: offer as drawn
        traceback.print_exc()
    if spec is not None:
        m.addAction('Export figure, one column (88.9 mm)…', lambda: export_chart(chart, 'single'))
        m.addAction('Export figure, two columns (182 mm)…', lambda: export_chart(chart, 'double'))
        m.addSeparator()
        m.addAction('Copy figure data (JSON)', lambda: QApplication.clipboard().setText(
            json.dumps(spec, indent=1, ensure_ascii=False)))
    else:
        m.addAction('Export figure as drawn (SVG)…', lambda: export_as_drawn(chart))
    return m
