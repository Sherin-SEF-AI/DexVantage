"""The number post-check: every figure in an answer should come from somewhere the app can see (a tool result, the
page on screen, the user's own words, the memory notes). After each answer the loop lists the numbers it cannot find
there, and the dock shows them under the answer ("Not found in what the tools returned: 87.5 %"), so a figure the
model made up is flagged to the user, never passed off as the app's.

How a number is matched (deterministic, no model involved):
- The answer's numbers are read with their written precision: "87.5" has one decimal, so any source value within
  half a unit of that last place (87.45 to 87.55) supports it; "88" is supported by 87.5 to 88.5.
- A source value also supports its common restatements: as a percentage (× 100, a fraction 0.875 shown as 87.5 %),
  in milli- or kilo- units (× 1000, ÷ 1000: 0.0042 m as 4.2 mm), in degrees from radians (× 180/π), and with the sign
  dropped (a delta of −0.004 written as "lowered by 4 mm").
- Not checked (a documented default, NUMBER_FLOOR): integers from 0 to 10 written without a decimal point and
  without a unit after them (counts, step numbers, list items: "3 cameras", "step 2"); a small integer with a unit or
  a percent or degree sign ("5 mm", "8 ms", "2 %", "3°") is a figure and is checked. Numbers that are part of a
  name are not figures and are skipped: a digit run joined to letters or to a hyphenated word (UR5e, SO-100,
  claude-opus-5-5, v2.1, §7.3, P4F2A1C), dates written YYYY-MM-DD, times hh:mm, and numbers inside inline code
  (`...`) or a URL.

Known limits: a figure derived by arithmetic the model did itself (a sum, a difference, a ratio) is flagged even when
it is right; the dock says "check these figures", not "these are wrong". A number written in words ("twelve") is not
read. Matching is by value, so a right number cited for the wrong quantity is not caught.

New in DexVantage (Vantage relies on the system prompt alone)."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

NUMBER_FLOOR = 10               # integers 0..NUMBER_FLOOR without a decimal point are not checked (a default)
SCALES = (1.0, 100.0, 0.01, 1000.0, 0.001, 1e6, 1e-6, 180.0 / math.pi, math.pi / 180.0)

# a number: optional sign (ASCII or the minus sign), digits with optional thousands groups (comma, thin or narrow
# no-break space, or a plain space between groups of three), optional decimals, optional exponent (e-5, × 10⁻⁵)
_NUM = re.compile(r'(?<![\w.])([+\-−]?)((?:\d{1,3}(?:[,   ]\d{3})+|\d+)(?:\.\d+)?)(?:[eE]([+\-−]?\d+))?')
_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
_POW = re.compile(r'^\s*[×x]\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)')
_CODE = re.compile(r'`[^`\n]*`|https?://\S+|\b\d{4}-\d{2}-\d{2}(?:T[\d:]+Z?)?\b|\b\d{1,2}:\d{2}(?::\d{2})?\b')
_UNIT_SUFFIX = re.compile(r'^(mm|cm|m|km|µm|um|nm|px|ms|µs|us|ns|s|min|h|hz|khz|mhz|ghz|kg|g|w|mw|kw|v|a|ma|mbps|gbps|'
                          r'kbps|mb|gb|kb|tb|b|deg|rad|mp|mpx|fps|k|m²|m³|mm²|lux|lx|nit|nits|db)(?![a-z])', re.I)


@dataclass(frozen=True)
class Num:
    value: float
    text: str
    decimals: int
    start: int = 0
    unit: bool = False                  # a unit, % or ° follows: a figure even when it is a small integer


def _mask_spans(text: str) -> str:
    """Blank out inline code, URLs, ISO dates and clock times (same length, so offsets stay)."""
    return _CODE.sub(lambda m: ' ' * len(m.group(0)), text)


def numbers_in(text: str, names_skipped: bool = True) -> list:
    """The numbers in a piece of text, with their written precision."""
    if not isinstance(text, str) or not text:
        return []
    t = _mask_spans(text) if names_skipped else text
    out = []
    for m in _NUM.finditer(t):
        sign, body, exp = m.group(1), m.group(2), m.group(3)
        s, e = m.start(), m.end()
        if names_skipped:
            before = t[s - 1] if s > 0 else ''
            # joined to a word: SO-100, claude-opus-5-5, §7.3, x86, P4F2
            joined = before == '-' and not sign and s > 1 and t[s - 2].isalnum()
            if before.isalpha() or (before and before in '§_#') or joined:
                continue
            if sign == '-' and s > 0 and t[s - 1].isalnum():
                continue
            after = t[e:e + 8]
            if after[:1].isalpha() and not _UNIT_SUFFIX.match(after):
                continue                                   # UR5e, 3D, 2nd
            if after[:1] == '-' and after[1:2].isalpha():
                continue                                   # 5-axis, 6-DoF
            if after[:1] == '.' and after[1:2].isdigit():
                continue                                   # a version 2.1.3 (the regex stopped at the second dot)
            line_start = t.rfind('\n', 0, s) + 1
            if t[line_start:s].strip() == '' and after[:2] in ('. ', ') ') and '.' not in body:
                continue                                   # a numbered list item "1. "
        clean = body.replace(',', '').replace(' ', '').replace(' ', '').replace(' ', '')
        try:
            v = float(clean)
        except ValueError:
            continue
        dec = len(clean.split('.', 1)[1]) if '.' in clean else 0
        p = 0
        if exp:
            p = int(exp.replace('−', '-'))
        else:
            pm = _POW.match(t[e:e + 12])
            if pm:
                p = int(pm.group(1).translate(_SUP))
        if p:
            if abs(p) > 300:                               # an id such as 3e512f…, not a figure
                continue
            v *= 10.0 ** p
            dec -= p
        if sign in ('-', '−'):
            v = -v
        tail = t[e:e + 10].lstrip(' \u00a0\u202f\u2009')
        has_unit = tail[:1] in ('%', '°', '‰') or bool(_UNIT_SUFFIX.match(tail))
        out.append(Num(v, text[s:e], dec, s, has_unit))
    return out


def _source_values(sources) -> list:
    vals = []
    for src in sources or ():
        if src is None:
            continue
        if isinstance(src, (int, float)) and not isinstance(src, bool):
            vals.append(float(src))
            continue
        if not isinstance(src, str):
            try:
                src = json.dumps(src, default=str)
            except (TypeError, ValueError):
                src = str(src)
        vals += [n.value for n in numbers_in(src, names_skipped=False)]
    return [v for v in vals if math.isfinite(v)]


def supported(n: Num, values) -> bool:
    tol = 0.5 * 10.0 ** (-n.decimals)
    a = abs(n.value)
    for v in values:
        for k in SCALES:
            s = abs(v * k)
            if abs(s - a) <= tol * (1 + 1e-9) + 1e-12 * max(1.0, s):
                return True
    return False


def checked(n: Num) -> bool:
    """Is this number a figure the check looks at (not a small count written as a bare integer without a unit)?"""
    return n.unit or not (n.decimals <= 0 and float(n.value).is_integer() and 0 <= abs(n.value) <= NUMBER_FLOOR
                and 'e' not in n.text.lower())


def unsupported(answer: str, sources) -> dict:
    """{'checked': how many figures the answer has, 'unsupported': [{'text', 'value'}], 'ok': bool}."""
    values = _source_values(sources)
    nums = [n for n in numbers_in(answer) if checked(n)]
    bad, seen = [], set()
    for n in nums:
        if not supported(n, values) and n.text not in seen:
            seen.add(n.text)
            bad.append({'text': n.text.strip(), 'value': n.value})
    return {'checked': len(nums), 'unsupported': bad, 'ok': not bad}


def warning_text(res: dict) -> str:
    """The sentence under an answer whose figures were not all found ('' when they were)."""
    bad = res.get('unsupported') or []
    if not bad:
        return ''
    shown = ', '.join(b['text'] for b in bad[:6]) + (f' and {len(bad) - 6} more' if len(bad) > 6 else '')
    return (f'Not found in what the tools returned or the page shows: {shown}. The assistant may have worked '
            'these out or made them up: check them before you rely on them.')
