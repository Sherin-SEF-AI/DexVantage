"""Publication figures (core/figures.py), --figure, and Export figure… on page charts (ui/figure_export.py)."""
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dexvantage.core import figures as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NS = {'s': 'http://www.w3.org/2000/svg'}


def _spec(n=3):
    x = list(np.linspace(0, 10, 21))
    series = [(f'Camera {i}', x, [None if 4 < v < 5 else 10 * i + v for v in x], '#123456',
               ('line', 'dash', 'points')[i % 3]) for i in range(n)]
    return F.curves(series, (0, 10), (0, 10 * n + 10), bands=[(1, 2, '#ff0000', 'blind')], hlines=[(5, 'default')],
                    vlines=[(3, 'contact')], markers=[(6, 7, '#00ff00', 'min')], xlabel='Distance (cm)',
                    ylabel='Visible (%)', title='Contact visibility', source='test')


@pytest.mark.parametrize('n', [1, 3, 8])
def test_svg_is_well_formed_with_one_group_per_series(n):
    root = ET.fromstring(F.to_svg(_spec(n)))
    groups = root.findall('.//s:g[@class="series"]', NS)
    assert len(groups) == n
    assert [g.get('data-label') for g in groups] == [f'Camera {i}' for i in range(n)]
    assert root.get('width') == '88.9mm' and root.get('height') == f'{88.9 * 0.66:.2f}'.rstrip('0') + 'mm'
    legend = root.findall('.//s:text[@class="legend"]', NS)
    assert len(legend) == (n if n > 1 else 0)               # a legend for two or more series
    assert {t.text for t in root.findall('.//s:text[@class="axis-label"]', NS)} == {'Distance (cm)', 'Visible (%)'}
    meta = json.loads(root.find('s:metadata', NS).text)
    assert meta['spec']['series'][0]['label'] == 'Camera 0'


def test_lines_break_at_gaps_and_palette():
    root = ET.fromstring(F.to_svg(_spec(3)))
    g0 = root.findall('.//s:g[@class="series"]', NS)[0]
    assert len(g0.findall('s:polyline', NS)) == 2           # None breaks the line
    strokes = [g.find('s:polyline', NS).get('stroke') for g in root.findall('.//s:g[@class="series"]', NS)[:2]]
    assert strokes == list(F.TOL_BRIGHT[:2])
    chart = ET.fromstring(F.to_svg(_spec(3), palette='chart'))
    assert chart.findall('.//s:g[@class="series"]', NS)[0].find('s:polyline', NS).get('stroke') == '#123456'
    many = F.series_colours(_spec(8), 'tol')
    assert [c for c, _s in many[:6]] == list(F.TOL_BRIGHT) and many[6] == (F.TOL_GREY, 'dash')


def test_widths_and_errors():
    root = ET.fromstring(F.to_svg(_spec(2), width='double'))
    assert root.get('width') == '182mm'
    assert ET.fromstring(F.to_svg(_spec(2), width=120, height_mm=50)).get('height') == '50mm'
    with pytest.raises(F.FigureError, match='single'):
        F.to_svg(_spec(2), width='triple')
    with pytest.raises(F.FigureError, match='no series'):
        F.to_svg(F.curves([], (0, 1), (0, 1)))
    bad = _spec(1)
    bad['series'][0]['y'] = bad['series'][0]['y'][:-1]
    with pytest.raises(F.FigureError, match='must match'):
        F.to_svg(bad)
    with pytest.raises(F.FigureError, match='too small'):
        F.to_svg(_spec(2), width=12, height_mm=10)


def test_bars():
    spec = F.bars([('Wrist', 0.2, '#ff0000'), ('Top', 0.45, '#00ff00'), ('Side', None, '#0000ff')], unit='m²')
    root = ET.fromstring(F.to_svg(spec))
    assert len(root.findall('.//s:g[@class="series"]', NS)) == 1
    bars = root.findall('.//s:rect[@class="bar"]', NS)
    assert len(bars) == 2
    w = [float(b.get('width')) for b in bars]
    assert w[0] / w[1] == pytest.approx(0.2 / 0.45, rel=1e-3)
    assert [t.text for t in root.findall('.//s:text[@class="value"]', NS)] == ['0.2', '0.45', '—']


def test_write_svg_and_pdf(tmp_path):
    r = F.write(_spec(2), str(tmp_path / 'f.svg'))
    assert r['format'] == 'svg' and r['series'] == 2
    ET.parse(str(tmp_path / 'f.svg'))
    r = F.write(_spec(2), str(tmp_path / 'f.pdf'), width='double')
    data = (tmp_path / 'f.pdf').read_bytes()
    assert data.startswith(b'%PDF')
    m = re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)', data)
    assert float(m.group(1)) == pytest.approx(182 / 25.4 * 72, abs=0.5)
    with pytest.raises(F.FigureError, match='SVG or PDF'):
        F.write(_spec(2), str(tmp_path / 'f.png'))


def test_batch_figure(tmp_path):
    spec = tmp_path / 's.json'
    spec.write_text(json.dumps(_spec(2)))
    out = tmp_path / 'o.svg'
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--figure', str(spec), str(out), '--figure-width',
                        'double'], capture_output=True, text=True, cwd=ROOT, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert ET.parse(str(out)).getroot().get('width') == '182mm'
    spec.write_text('{"kind": "pie"}')
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--figure', str(spec), str(out)], capture_output=True,
                       text=True, cwd=ROOT, timeout=120, env=env)
    assert r.returncode == 2 and '"curves" or "bars"' in r.stderr and 'Traceback' not in r.stderr


# ====================================================================== the page charts
def test_curve_chart_exports_the_data_it_draws(app, tmp_path):
    from dexvantage.ui import figure_export as FX
    from dexvantage.ui.pages._curves import CurveChart
    from dexvantage.ui.theme import THEME
    c = CurveChart()
    c.set_data([('Wrist', [0, 1, 2], [10, None, 30], THEME.color('ink'), 'line'),
                ('Top', [0, 1, 2], [5, 6, 7], '#ff8800', 'dash')], (0, 2), (0, 40),
               bands=[(0.5, 1.0, '#ff0000', 'gap')], hlines=[(20, 'target')], xlabel='Time (s)', ylabel='Error (mm)')
    spec = c.figure_data()
    assert [s['y'] for s in spec['series']] == [[10.0, None, 30.0], [5.0, 6.0, 7.0]]
    assert spec['series'][1]['color'] == '#ff8800' and spec['xlabel'] == 'Time (s)'
    labels = [a.text() for a in FX.build_menu(c).actions() if a.text()]
    assert labels[0].startswith('Export figure, one column') and 'Copy figure data (JSON)' in labels
    path = FX.export_chart(c, 'double', str(tmp_path / 'c.svg'))
    root = ET.parse(path).getroot()
    assert len(root.findall('.//s:g[@class="series"]', NS)) == 2 and root.get('width') == '182mm'
    c.deleteLater()


def test_bar_list_and_charts_without_data(app, tmp_path):
    from dexvantage.ui import figure_export as FX
    from dexvantage.ui.charts import BarList, Histogram
    b = BarList()
    b.set_rows([('u1', 0, 'Wrist', 0.3), ('u2', 1, 'Top', 0.1)])
    spec = b.figure_data()
    assert [r['label'] for r in spec['rows']] == ['Wrist', 'Top']
    assert FX.export_chart(b, 'single', str(tmp_path / 'b.pdf')).endswith('.pdf')
    h = Histogram()
    h.resize(300, 160)
    h.set_data([1, 2, 2, 3])
    assert h.figure_data() is None
    assert [a.text() for a in FX.build_menu(h).actions()] == ['Export figure as drawn (SVG)…']
    p = FX.export_as_drawn(h, str(tmp_path / 'h.svg'))
    ET.parse(p)
    b.deleteLater()
    h.deleteLater()
