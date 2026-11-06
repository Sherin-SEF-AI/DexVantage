"""Analysis pages: one per study, registered here so the main window builds their tabs and menu commands itself.

A page module in this package calls register() at import time:

    from . import register
    from .base import StudyPage

    class GraspPage(StudyPage):
        STUDY = 'grasp_visibility'           # the core.jobs study it shows (None: a page without a study)
        TITLE = 'Grasp self-occlusion'
        def show_result(self, doc): ...      # fill the headline, the chart and the tables from the study's doc

    register('grasp', 'Grasp self-occlusion', GraspPage, menu='Analysis', icon='grasp', action_id='grasp_study',
             workspace='grasp', order=30)

The window imports every module here (load_pages), adds each page as a tab of the Analysis dock (built on first use),
and puts a command in the named menu that opens it. When action_id names a command that already exists (the
disabled "arrives in Phase N" placeholders), that command is enabled and opens the page instead of adding a new one.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from dataclasses import dataclass

PAGES: dict = {}


@dataclass
class PageInfo:
    id: str
    title: str
    cls: type
    menu: str = 'Analysis'          # a top-level menu's title (without &)
    icon: str | None = None
    action_id: str | None = None    # an existing command to take over, else 'page_<id>'
    shortcut: str | None = None
    workspace: str | None = None    # the workspace whose Analysis tab this page is
    order: int = 100


def register(pid, title, cls, menu='Analysis', icon=None, action_id=None, shortcut=None, workspace=None, order=100):
    PAGES[pid] = PageInfo(pid, title, cls, menu, icon, action_id, shortcut, workspace, order)
    return cls


def load_pages() -> list:
    """Import every page module once; the pages in their order."""
    here = os.path.dirname(os.path.abspath(__file__))
    for m in pkgutil.iter_modules([here]):
        if m.name not in ('base',) and not m.name.startswith('_'):
            importlib.import_module(f'{__package__}.{m.name}')
    return sorted(PAGES.values(), key=lambda p: (p.order, p.id))
