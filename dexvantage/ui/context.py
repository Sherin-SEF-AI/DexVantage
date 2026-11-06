"""What every panel shares: the store, the analysis controller, the settings, the window and the view toggles."""
from __future__ import annotations

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

# view overlays: id -> (label, default)
OVERLAYS = {
    'grid': ('Floor grid', True),
    'frames': ('World frame axes', True),
    'volumes': ('Task volumes', True),
    'shadows': ('Shadows and sky occlusion', True),
}


class AppContext(QObject):
    overlays_changed = pyqtSignal()

    def __init__(self, store, settings: QSettings | None = None):
        super().__init__()
        from .store import AnalysisController
        self.store = store
        self.settings = settings or QSettings('DexVantage', 'DexVantage')
        self.analysis = AnalysisController(store)
        self.window = None
        self.overlays = {k: v[1] for k, v in OVERLAYS.items()}

    def overlay(self, key) -> bool:
        return bool(self.overlays.get(key, False))

    def set_overlay(self, key, on: bool):
        if self.overlays.get(key) != bool(on):
            self.overlays[key] = bool(on)
            self.overlays_changed.emit()
