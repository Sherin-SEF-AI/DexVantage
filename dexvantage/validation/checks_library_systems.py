"""Data-consistency checks of the library's systems entries: teleop devices, policies and hosts (core/library,
core/teleop_devices.py, core/policies.py). They test the entries, not a model: a failure means a figure was mistyped,
or a maker's own figures disagree and the entry does not say so.

library.teleop_figures — every teleop entry's block is consistent (teleop_devices.entry_problems: kinds, mapping,
leads, a source index and a quote for every stated figure, units in range), and every stated number appears in its
own quote (the source's words copied into the entry), so a value cannot drift from what the source says. Tolerance 0
problems: each one is a data error.

library.policy_token_grid — where an entry states the input size, the ViT patch and the tokens per image (from
separate sources: code or configs and a paper), the patch grid written out again, (w / p)(h / p) / s², must give the
token count (s the pixel-shuffle factor the backbone's config states: 4 for SmolVLM2, 1 otherwise). Tolerance 0
tokens: the counts are integers.

library.stated_totals — where a host entry states both a per-port (per-link, per-lane) figure and a total over the
same ports, count × per-port against the total (the Basler card's "380 MBps per port" and "1.52 GBps total", the
Leopard grabber's "4 x CSI-2 interfaces ... each up to 10 Gbit/s, total 40 Gbit/s"); and the unit conversions the
entries make from the sources' words, redone from the quoted words with exact factors (1 byte = 8 bit; 1 lb =
0.45359237 kg, NIST SP 811). Tolerance 0.005 (relative): converted figures are rounded to three or four significant
digits (453.59 g entered as 454 g is 0.09 %).

Known limits: these checks see only what the entries record; a figure copied wrongly into both the value and its
quote passes, and the makers' figures themselves cannot be checked offline.
"""
from __future__ import annotations

import re

from .suite import Outcome, check

_NUM = re.compile(r'(?<![\w.])(\d+(?:[.,]\d+)?)')
SHUFFLE = {'SmolVLM2': 4}           # pixel-shuffle factor by backbone family word (SmolVLM2 config scale_factor 4)
LB_KG = 0.45359237                  # NIST SP 811, Appendix B.8: pound (avoirdupois) = 0.45359237 kg (exact)


def _numbers(text) -> list:
    return [float(m.group(1).replace(',', '')) for m in _NUM.finditer(str(text or ''))]


def teleop_figure_problems() -> tuple[list, int]:
    """([problem], number of stated figures checked) over every built-in teleop entry."""
    from ..core import library as L
    from ..core import teleop_devices as TD
    problems, n = [], 0
    for e in L.entries('teleop'):
        if e.is_custom:
            continue
        problems += [f'{e.id}: {m}' for m in TD.entry_problems(e.data)]
        t = e.data.get('teleop') or {}
        for k, (_label, _unit, typ) in TD.FIELDS.items():
            fv = t.get(k) or {}
            v = fv.get('value')
            if v is None or typ not in ('int', 'num'):
                continue
            n += 1
            if not any(abs(x - float(v)) <= 1e-9 * max(1.0, abs(v)) for x in _numbers(fv.get('quote'))):
                problems.append(f'{e.id}.{k}: {v:g} does not appear in its quote')
    return problems, n


@check('library.teleop_figures', 'library', 'Teleop device entries: each figure against its source\'s words',
       'teleop entries with an inconsistent block (kind, mapping, leads, a source and a quote for every stated '
       'figure) plus stated numbers that do not appear in their own quote, over every built-in teleop entry (data '
       'consistency)', 'the source\'s words each entry copies into the figure\'s quote, read again here as numbers',
       0, 'problems', refs=('library_teleop_sources',))
def _teleop_figures(ctx):
    from ..core import library as L
    problems, n = teleop_figure_problems()
    es = [e for e in L.entries('teleop') if not e.is_custom]
    return Outcome(float(len(problems)), {'entries': len(es), 'figures_checked': n, 'problems': problems,
                                          'kind': 'data consistency, not a model accuracy figure'},
                   ok=bool(es), note=f'{len(es)} teleop entries, {n} stated figures found in their quotes.')


def token_grid_rows() -> list:
    """[{'entry', 'input', 'patch', 'shuffle', 'grid', 'stated'}] for every policy entry stating all three."""
    from ..core import library as L
    rows = []
    for e in L.entries('policy'):
        p = e.data.get('policy') or {}
        for res_k, tok_k in (('input_resolution', 'tokens_per_image'),
                             ('wrist_input_resolution', 'wrist_tokens_per_image')):
            res = (p.get(res_k) or {}).get('value')
            tok = (p.get(tok_k) or {}).get('value')
            patch = (p.get('patch_px') or {}).get('value')
            if not (res and tok and patch):
                continue
            s = next((v for w, v in SHUFFLE.items() if w in str(e.data.get('family') or '')), 1)
            grid = (int(res[0]) // int(patch)) * (int(res[1]) // int(patch)) // (s * s)
            rows.append({'entry': e.id, 'field': tok_k, 'input': list(res), 'patch': patch, 'shuffle': s,
                         'grid': grid, 'stated': tok})
    return rows


@check('library.policy_token_grid', 'library', 'Policy entries: patch grid against the stated token count',
       'largest difference (tokens) between (w / p)(h / p) / s² of the stated input size and ViT patch and the stated '
       'tokens per image, over every policy entry that states all three (data consistency)',
       'the ViT patch grid (Dosovitskiy et al. 2021) with the backbone\'s stated pixel-shuffle factor, written out '
       'again here', 0, 'tokens', refs=('dosovitskiy2021', 'library_policy_sources'))
def _token_grid(ctx):
    rows = token_grid_rows()
    worst = max((abs(r['grid'] - r['stated']) for r in rows), default=0)
    return Outcome(float(worst), {'rows': rows, 'kind': 'data consistency'}, ok=len(rows) >= 4,
                   note=f'{len(rows)} token counts compared with their patch grids.')


def totals_and_units() -> list:
    """[{'entry', 'what', 'computed', 'entered', 'rel'}]: stated totals against count × per-port, and unit
    conversions redone from the quoted words."""
    from ..core import library as L
    rows = []

    def add(e, what, computed, entered):
        rel = abs(computed - entered) / max(abs(entered), 1e-12)
        rows.append({'entry': e.id, 'what': what, 'computed': computed, 'entered': entered, 'rel': rel})
    for e in L.entries('host'):
        if e.is_custom:
            continue
        for i, g in enumerate((e.data.get('host') or {}).get('ingest') or []):
            c = g.get('count')
            if c and g.get('per_port_mbps') and g.get('aggregate_mbps'):
                add(e, f'ingest[{i}] {g["bus"]}: {c} × {g["per_port_mbps"]:g} against the stated total',
                    c * g['per_port_mbps'], g['aggregate_mbps'])
            if c and g.get('practical_mbps') and g.get('practical_aggregate_mbps'):
                add(e, f'ingest[{i}] {g["bus"]}: {c} × {g["practical_mbps"]:g} (practical) against the stated '
                       'practical total', c * g['practical_mbps'], g['practical_aggregate_mbps'])
            w = str(g.get('what') or '')
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*MBps per port', w):
                add(e, f'ingest[{i}]: "{m.group(0)}" × 8 bit/byte', float(m.group(1)) * 8.0,
                    float(g.get('practical_mbps') or 0.0))
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*GBps total', w):
                add(e, f'ingest[{i}]: "{m.group(0)}" × 8000', float(m.group(1)) * 8000.0,
                    float(g.get('practical_aggregate_mbps') or 0.0))
            for m in re.finditer(r'~(\d+(?:\.\d+)?)\s*Gbit/s netto', w):
                add(e, f'ingest[{i}]: "{m.group(0)}" × 1000', float(m.group(1)) * 1000.0,
                    float(g.get('practical_aggregate_mbps') or 0.0))
    for e in L.entries():
        if e.is_custom or e.data.get('mass_g') is None:
            continue
        m = re.search(r'"Weight ~ ?(\d+(?:\.\d+)?) lb', e.note)
        if m:
            add(e, f'mass: "{m.group(0)[1:]}" × 453.59237 g', float(m.group(1)) * LB_KG * 1000.0,
                float(e.data['mass_g']))
    return rows


@check('library.stated_totals', 'library', 'Host totals and unit conversions against the sources\' words',
       'largest relative difference between count × the stated per-port figure and the stated total over the same '
       'ports, and between each unit conversion an entry makes and the same conversion redone from the quoted words '
       '(data consistency)', 'the arithmetic written out again, with exact unit factors (1 byte = 8 bit; 1 lb = '
       '0.45359237 kg, NIST SP 811)', 0.005, 'relative', refs=('nist_sp811', 'library_host_sources'))
def _totals(ctx):
    rows = totals_and_units()
    worst = max((r['rel'] for r in rows), default=0.0)
    return Outcome(float(worst), {'rows': rows, 'kind': 'data consistency'}, ok=len(rows) >= 4,
                   note=f'{len(rows)} totals and conversions compared.')
