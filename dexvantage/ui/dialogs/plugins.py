"""Help ▸ Plugins: every installed DexVantage plugin (dexvantage/plugin_sdk.py), its version, whether it loaded, what it
added and, for one that could not be loaded, why in a plain sentence. Nothing here loads or runs a plugin: the list is
what start-up found."""
from __future__ import annotations

from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from ... import plugin_sdk as PL
from ..widgets import StatusLine, button, hline, label
from .base import VDialog

GUIDE = 'docs/guide/plugins.md'


def rows() -> list:
    """[{'title', 'level', 'status', 'added', 'error', 'problems'}] per plugin, as the dialog shows them."""
    out = []
    for r in PL.loaded():
        d = r.to_dict()
        ver = f' {d["version"]}' if d['version'] else ''
        dist = f' ({d["distribution"]})' if d['distribution'] and d['distribution'] != d['name'] else ''
        out.append({'title': f'{d["name"]}{ver}{dist}', 'level': PL.STATUS_LEVEL[d['status']],
                    'status': d['status_text'], 'added': d['summary'], 'error': d['error'],
                    'problems': list(d['problems'])})
    return out


def intro_text() -> str:
    found = PL.loaded()
    if not PL.enabled():
        return (f'Plugins are off for this session (started with --no-plugins). {len(found)} installed plugin'
                f'{"s" * (len(found) != 1)} found; start DexVantage without --no-plugins to load them.')
    if not found:
        return ('No plugins are installed. A plugin is a Python package that adds studies, metrics, exporters, '
                f'dataset readers, library folders or pages: {GUIDE} says how to write one.')
    n_ok = sum(r.status == 'loaded' for r in found)
    return (f'{n_ok} of {len(found)} installed plugin{"s" * (len(found) != 1)} loaded. Plugins run with the same '
            'rights as DexVantage: install only those you trust.')


class PluginsDialog(VDialog):
    def __init__(self, parent=None):
        super().__init__(parent, 'Plugins', 'installed DexVantage plugins', (620, 420))
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        self.intro = label(intro_text(), wrap=True)
        lay.addWidget(self.intro)
        lay.addWidget(hline())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(8)
        self.items = []
        for r in rows():
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            bl.addWidget(StatusLine(r['level'], f'{r["title"]} · {r["status"]}'))
            bl.addWidget(label(f'Added: {r["added"]}', muted=True, wrap=True))
            if r['error']:
                bl.addWidget(label(r['error'], wrap=True))
            for p in r['problems']:
                bl.addWidget(label(p, muted=True, small=True, wrap=True))
            il.addWidget(box)
            self.items.append(r)
        il.addStretch(1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        self.hint.setText(f'How to write one: {GUIDE}')
        o = button('Close', 'primary')
        o.clicked.connect(self.accept)
        self.buttons.addWidget(o)

    def text(self) -> str:
        """Everything the dialog says, as plain text (tests, the assistant)."""
        parts = [self.intro.text()]
        for r in self.items:
            parts.append(f'{r["title"]} · {r["status"]}. Added: {r["added"]}. {r["error"]} {" ".join(r["problems"])}'
                         .strip())
        return '\n'.join(parts)


def open_dialog(window, modal: bool = True):
    d = PluginsDialog(window)
    if modal:
        d.exec()
    else:
        d.show()
    return d
