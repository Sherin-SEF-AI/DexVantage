from conftest import pump

from dexvantage.ui.theme import QSS, THEME, TOKENS
from dexvantage.ui.widgets import ScrubSpinBox


def test_scrub_spin_box_shows_its_value(app):
    """Qt 6.12 lays a style-sheeted spin box's line edit out 1 px wide; the widget gives it its width back."""
    b = ScrubSpinBox('X', 'length', 3)
    b.resize(90, 22)
    b.set_value(0.45)
    b.show()
    pump(app, 0.02)
    assert b.lineEdit().width() > 40 and b.lineEdit().text() == '0.450'
    b.hide()


def test_scrub_spin_box_parses_units_and_commits(app):
    b = ScrubSpinBox('X', 'length', 3)
    got = []
    b.committed.connect(got.append)
    b.show()
    b.lineEdit().setText('450 mm')
    b._typed = True
    b._on_finished()
    assert got and abs(got[-1] - 0.45) < 1e-12
    b.hide()


def test_theme_is_vantage_s():
    """The tokens and the style sheet are Vantage's, unchanged (spec §4)."""
    assert TOKENS['dark']['base'] == '#121211' and TOKENS['light']['ink'] == '#0b0b0b'
    assert 'IBM Plex Sans' in QSS
    assert THEME.name in ('dark', 'light')
