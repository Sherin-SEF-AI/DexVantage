"""Make docs/validation.md from the report of a full validation run (spec §18: the measured agreement with MuJoCo
renders is published in the docs).

    dexvantage --validate out/validation                       # a full run: every model, not --quick
    python scripts/gen_validation_doc.py out/validation/validation.json [--out docs/validation.md]

The page lists every model and every check with what it compares, its independent reference, its tolerance and why,
and the value the run measured; the checks that compare with MuJoCo renders come first, with the agreement figures
they report. Nothing on the page is typed in: every number is read from the report. The report must come from a
full run of the checks registered now (the same ids, every model, not quick, not cancelled); otherwise this script
refuses in one sentence and writes nothing, so the page always describes the current check list
(tests/test_validation_doc.py holds the page to it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)

SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
AGREE_NAMES = {'exact': 'exact ray casting', 'cube_gl': 'cube map (GPU)', 'cube_numpy': 'cube map (NumPy)'}
PERSONAL = re.compile(r'(/home/|/Users/|[A-Za-z]:\\Users\\)')


class GenError(Exception):
    """The report cannot make the page (said in one sentence)."""


def num(x) -> str:
    """A measured value: 4 significant digits, powers of ten written out (1.2 × 10⁻⁵), '—' when missing."""
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return '—'
    if f == 0:
        return '0'
    if f == int(f) and abs(f) < 1e12:
        return f'{int(f):,}'.replace(',', ' ')
    if 1e-3 <= abs(f) < 1e5:
        return f'{f:.4g}'
    m, e = f'{f:.2e}'.split('e')
    m = m.rstrip('0').rstrip('.')
    return f'{m} × 10{str(int(e)).translate(SUP)}'


def pct_of(p: float) -> str:
    """A percentage, with enough decimals that a value short of 100 % is not written 100 %."""
    p = float(p)
    nd = 2
    while nd < 6 and round(p, nd) == 100.0 and p < 100.0:
        nd += 1
    return f'{p:.{nd}f} %'


def unit(u: str) -> str:
    u = (u or '').strip()
    return '' if u in ('(unitless)', 'unitless', '') else u


def pct(x) -> str:
    return '—' if x is None else pct_of(float(x) * 100)


def cell(s: str) -> str:
    """Text safe inside a Markdown table cell."""
    return str(s).replace('|', '\\|').replace('\n', ' ')


def current_ids() -> list:
    from dexvantage.validation import suite
    return [s.id for s in suite.specs()]


def check_report(rep: dict, ids: list | None = None):
    """Refuse a report that is not a full run of the checks registered now."""
    if not isinstance(rep, dict) or not str(rep.get('schema', '')).startswith('dexvantage.validation/'):
        raise GenError('That file is not a DexVantage validation report (validation.json from dexvantage --validate).')
    st = rep.get('settings') or {}
    if st.get('quick') or st.get('cheap_only') or st.get('models') or rep.get('cancelled'):
        raise GenError('The report is from a partial run (quick, cheap checks only, some models or cancelled); '
                       'the page needs a full run: dexvantage --validate out/validation.')
    have = [c['id'] for m in rep.get('models', []) for c in m.get('checks', [])]
    ids = current_ids() if ids is None else ids
    missing = [i for i in ids if i not in have]
    extra = [i for i in have if i not in ids]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f'{len(missing)} registered check(s) are not in it ({", ".join(missing[:5])}'
                         f'{", …" if len(missing) > 5 else ""})')
        if extra:
            parts.append(f'{len(extra)} of its checks are no longer registered ({", ".join(extra[:5])}'
                         f'{", …" if len(extra) > 5 else ""})')
        raise GenError('The report does not match the current check list: ' + '; '.join(parts)
                       + '. Run dexvantage --validate out/validation again.')


def agreement_lines(c: dict) -> list:
    """The agreement figures a check reports in its detail (read, never computed here), as short phrases."""
    d = c.get('detail') or {}
    out = []
    if isinstance(d.get('agreement'), (int, float)):
        s = f'agreement {pct(d["agreement"])}'
        if isinstance(d.get('iou'), (int, float)):
            s += f', IoU {num(d["iou"])}'
        out.append(s)
    bk = d.get('by_kind')
    if isinstance(bk, dict):
        out.append('by case: ' + ', '.join(f'{k} {pct(v.get("agreement"))} of {num(v.get("n"))} points'
                                           for k, v in bk.items() if isinstance(v, dict)))
    why = d.get('why')
    if isinstance(why, dict):
        out.append('disagreements: ' + ', '.join(f'{num(v)} {k}' for k, v in why.items()))
    if isinstance(d.get('occlusion_share'), (int, float)):
        out.append(f'occlusion disagreements away from an edge {pct(d["occlusion_share"])} of the points')
    for k, v in d.items():                         # another sample's figures, reported beside the check's
        if isinstance(v, dict) and isinstance(v.get('agreement'), (int, float)) and k != 'grasp_level':
            out.append(f'{k}: agreement {pct(v["agreement"])}'
                       + (f', IoU {num(v["iou"])}' if isinstance(v.get('iou'), (int, float)) else '')
                       + ' (reported, not held to the floor)')
    sens = d.get('sensors')
    if isinstance(sens, dict):
        for k, v in sens.items():
            if isinstance(v, dict) and 'voxel_agreement' in v:
                out.append(f'{k}: voxels {pct(v.get("voxel_agreement"))}, on the surfaces '
                           f'{pct(v.get("surface_agreement"))}')
    for k, v in d.items():
        if k.startswith('agree') and k.endswith('_pct') and isinstance(v, (int, float)):
            what = k[len('agree_'):-len('_pct')]
            out.append(f'{AGREE_NAMES.get(what, what.replace("_", " "))} {pct_of(v)} of the pixels agree')
    for k in ('worst_px_by_camera', 'worst_px_by_lens'):
        if isinstance(d.get(k), dict):
            out.append('worst by camera: ' + ', '.join(f'{a} {num(b)} px' for a, b in d[k].items()))
    if isinstance(d.get('by_lens'), dict):
        out.append('worst by lens model: ' + ', '.join(f'{a} {num(b.get("worst_px"))} px' for a, b in
                                                       d['by_lens'].items() if isinstance(b, dict)))
    if isinstance(d.get('pixels'), (int, float)) and isinstance(d.get('differ'), (int, float)):
        out.append(f'{num(d["differ"])} of {num(d["pixels"])} pixels differ')
    return out


def is_mujoco(c: dict) -> bool:
    """A check whose reference is a MuJoCo render (segmentation, depth or colour images)."""
    return 'mujoco_renderer' in (c.get('references') or [])


def renderer_of(rep: dict) -> str:
    for m in rep.get('models', []):
        for c in m.get('checks', []):
            r = (c.get('detail') or {}).get('renderer')
            if isinstance(r, str) and r:
                return r
    return ''


def render(rep: dict) -> str:
    env = rep.get('environment') or {}
    sm = rep.get('summary') or {}
    models = rep.get('models', [])
    checks = [c for m in models for c in m.get('checks', [])]
    refs_used = []
    for c in checks:
        refs_used += [r for r in c.get('references') or [] if r not in refs_used]
    date = (rep.get('created_utc') or '')[:10]
    rend = renderer_of(rep)
    L = ['# Validation', '',
         '<!-- Made by scripts/gen_validation_doc.py from out/validation/validation.json; do not edit by hand. -->', '',
         f'Every model DexVantage computes is compared with an independent reference: a closed form written out again, '
         f'another library, a published table, brute-force ray casting, Monte Carlo, or MuJoCo renders. This page is '
         f'made from the report of a full run (`dexvantage --validate out/validation`, {date} UTC, seed '
         f'{(rep.get("settings") or {}).get("seed")}): **{num(sm.get("passed"))} of {num(sm.get("checks"))} checks '
         f'pass** over {len(models)} models in {num(round(float(rep.get("seconds") or 0)))} s. A check passes when its '
         f'error is finite and at most its tolerance; each tolerance is stated with why it is that size.', '',
         'The run: ' + ', '.join(f'{k} {v}' for k, v in env.items() if v and k != 'platform')
         + (f' on {env["platform"]}' if env.get('platform') else '')
         + (f'; MuJoCo renders by {rend} (EGL)' if rend else '') + '.', '',
         f'Frame and units: {rep.get("frame", "")}.', '', f'Limits: {rep.get("limits", "")}', '']
    if sm.get('failed'):
        L += ['**Failing checks:** ' + ', '.join(f'`{i}`' for i in sm['failed']) + '.', '']

    # ------------------------------------------------------------------ MuJoCo
    mj = [c for c in checks if is_mujoco(c)]
    L += ['## Agreement with MuJoCo renders', '',
          'The checks whose reference is a MuJoCo render: MuJoCo\'s own OpenGL renderer (segmentation, depth and '
          'colour images through EGL) of the same scene, cameras and joint states, against what DexVantage predicts '
          'without rendering (spec §9.1, §18). Measured agreement is reported as it came out, including where it is '
          'lower.', '',
          '| Check | Compared | Measured | Tolerance | Result |', '|---|---|---|---|---|']
    for c in mj:
        u = unit(c.get('unit'))
        L.append(f'| `{c["id"]}` | {cell(c["title"])} | {num(c.get("error"))} {cell(u)} | {num(c.get("tolerance"))} '
                 f'{cell(u)} | {"pass" if c.get("passed") else "FAIL"} |')
    L.append('')
    for c in mj:
        lines = agreement_lines(c)
        if lines:
            L.append(f'- **`{c["id"]}`**: ' + '; '.join(lines) + '.')
    L += ['', 'Each check\'s full description, reference and the reason for its tolerance are in its model\'s section '
          'below.', '']

    # ------------------------------------------------------------------ every model
    L += ['## Every check, by model', '']
    for m in models:
        cs = m.get('checks', [])
        L += [f'### {m["title"]}', '',
              f'Model `{m["id"]}` ({m.get("modules", "")}): {num(m.get("passed"))} of {len(cs)} checks pass.', '',
              '| Check | Title | Measured | Tolerance | Unit | Result |', '|---|---|---|---|---|---|']
        for c in cs:
            L.append(f'| `{c["id"]}` | {cell(c["title"])} | {num(c.get("error"))} | {num(c.get("tolerance"))} | '
                     f'{cell(unit(c.get("unit")))} | {"pass" if c.get("passed") else "FAIL"} |')
        L.append('')
        for c in cs:
            refs = ', '.join(f'[{r}](#ref-{r})' for r in c.get('references') or [])
            item = [f'- **`{c["id"]}`** — {c["title"]}. *Compared:* {c.get("quantity", "")}. '
                    f'*Reference:* {c.get("reference", "")}' + (f' ({refs})' if refs else '') + '.']
            if c.get('note'):
                item.append(f'  *Tolerance and why:* {c["note"]}')
            if c.get('exception'):
                item.append(f'  *Failed with:* {c["exception"]}')
            L += item
        L.append('')

    # ------------------------------------------------------------------ references
    texts = rep.get('references') or {}
    L += ['## References', '']
    for r in refs_used:
        L.append(f'- <a id="ref-{r}"></a>**[{r}]** {texts.get(r, "")}')
    L += ['', '## How to regenerate', '', '```', 'dexvantage --validate out/validation',
          'python scripts/gen_validation_doc.py out/validation/validation.json', '```', '',
          'The script refuses a report that is not a full run of the checks registered now, so a new or removed '
          'check needs a new run before this page can be made (`tests/test_validation_doc.py` fails until then).', '']
    text = '\n'.join(L)
    if PERSONAL.search(text):
        raise GenError('The page would hold a personal path; it was not written.')
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('report', nargs='?', default=os.path.join(ROOT, 'out', 'validation', 'validation.json'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'validation.md'))
    a = ap.parse_args(argv)
    try:
        try:
            with open(a.report, encoding='utf-8') as fh:
                rep = json.load(fh)
        except OSError:
            raise GenError(f'There is no report at {os.path.basename(a.report)}: run dexvantage --validate '
                           'out/validation first.') from None
        except json.JSONDecodeError:
            raise GenError(f'{os.path.basename(a.report)} is not valid JSON.') from None
        check_report(rep)
        text = render(rep)
    except GenError as e:
        print(f'gen_validation_doc: {e}', file=sys.stderr)
        return 1
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f'wrote {os.path.relpath(a.out, ROOT)}: {sum(len(m.get("checks", [])) for m in rep["models"])} checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
