"""Joint values as the UI shows them: revolute and continuous joints in degrees, prismatic joints in millimetres.

Stored values stay SI (radians, metres; docs/DEVELOPING.md "Frames and units"); these helpers only convert for
display and back. A figure the file does not state (None) is shown as "—".
"""
from __future__ import annotations

import math

from ..core.mathutil import fmt_num

NOT_STATED = 'not stated in the model file'


def is_linear(jtype: str) -> bool:
    return jtype == 'prismatic'


def unit(jtype: str) -> str:
    return 'mm' if is_linear(jtype) else '°'


def to_display(v, jtype: str):
    """radians -> degrees, metres -> millimetres (None and ±inf pass through)."""
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return v
    return v * 1000.0 if is_linear(jtype) else math.degrees(v)


def from_display(v: float, jtype: str) -> float:
    return float(v) / 1000.0 if is_linear(jtype) else math.radians(float(v))


def fmt(v, jtype: str, dp: int = 1, with_unit: bool = True) -> str:
    """'−114.6°', '10.0 mm', '—' (not stated), 'none' (no limit)."""
    if v is None:
        return '—'
    d = to_display(v, jtype)
    if not math.isfinite(d):
        return 'none'
    s = fmt_num(d, dp)
    if not with_unit:
        return s
    return f'{s}°' if not is_linear(jtype) else f'{s} mm'


def fmt_rate(v, jtype: str) -> str:
    """A velocity limit: °/s or mm/s."""
    if v is None:
        return '—'
    d = to_display(v, jtype)
    return f'{fmt_num(d, 0 if abs(d) >= 100 else 1)} {"mm/s" if is_linear(jtype) else "°/s"}'


def fmt_effort(v, jtype: str) -> str:
    if v is None:
        return '—'
    return f'{fmt_num(float(v), 1)} {"N" if is_linear(jtype) else "N·m"}'
