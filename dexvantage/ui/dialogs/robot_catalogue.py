"""Library ▸ Robots ▸ <a catalogue robot> ▸ Import…: the Menagerie dialog opened on that robot's model (spec §6, §9.4).

The robot catalogue (core/robot/catalogue.py) names each robot's MuJoCo Menagerie folder and the MJCF file to import.
This dialog is the Robot ▸ Import from MuJoCo Menagerie… dialog (ui/dialogs/menagerie.py) with its list narrowed to
that folder, the download started at once and, once the files are here, the catalogue's file picked in the "Model
file" box. Everything else is the Menagerie dialog's own flow: the folder's own LICENSE is shown, Import stays
disabled until the user ticks "I accept the licence of this model", and the import is one undo step that records
the licence on the robot. Nothing is imported without that consent.

Tests pass an `opener` and `background=False` so nothing touches the network or waits.
"""
from __future__ import annotations

from ...core.robot import catalogue as CAT
from ...core.robot import menagerie as MG
from .menagerie import MenagerieDialog


class CatalogueImportDialog(MenagerieDialog):
    def __init__(self, window, entry, opener=None, dest=None, background=True, parent=None):
        super().__init__(window, opener=opener, dest=dest, background=background, parent=parent)
        self.entry = entry
        self.robot_name = entry.title                    # the robot as the catalogue names it, not its file
        m = CAT.model(entry)
        self.folder = str(m.get('folder') or '')
        self.want_file = str(m.get('file') or '')
        self.title_lbl.setText(f'Import {entry.title}')
        self.sub_lbl.setText('from the robot catalogue, through MuJoCo Menagerie: each model has its own licence')
        self.models = [self.folder]
        lic = m.get('license') or 'not read'
        self.index_note.setText(f'Robot catalogue: {entry.title} is the Menagerie model {self.folder} '
                                f'({self.want_file}). Its licence read at {str(m.get("commit") or "")[:12]}: {lic}. '
                                'The download shows the licence of the version you get; read it before accepting.')
        self._apply_filter('')
        self.select_model(self.folder)

    def _show_download(self, dl: MG.MenagerieDownload):
        super()._show_download(dl)
        if self.dl is dl and self.want_file:
            i = self.file.findText(self.want_file)
            if i >= 0:
                self.file.setCurrentIndex(i)


def import_entry(window, entry_id, opener=None, dest=None, background=True, run=True):
    """Open the import dialog for catalogue entry entry_id and start the download. With run=False the dialog is
    returned without waiting (tests); with run=True it is shown modally and True is returned when a robot was
    imported. A robot without a Menagerie model gets a plain message instead."""
    e = CAT.get(entry_id)
    msg = getattr(window, 'msg', None)
    if e is None:
        if msg:
            msg(f'The robot catalogue has no entry "{entry_id}".', 5000)
        return None if not run else False
    m = CAT.model(e)
    if m.get('source') != 'menagerie' or not m.get('folder'):
        if msg:
            msg(f'{e.title} has no MuJoCo Menagerie model to import. {m.get("note") or ""}'.strip(), 8000)
        return None if not run else False
    d = CatalogueImportDialog(window, e, opener=opener, dest=dest, background=background)
    d.fetch_selected()
    if not run:
        return d
    d.exec()
    return d.robot is not None


def tooltip(entry) -> str:
    """Every figure of a catalogue robot ("—" where no source states it), the model and what was verified, and the
    sources with the date each was read."""
    from ...core.library import NOT_STATED
    lines = [entry.title, f'{CAT.TYPE_LABELS.get(CAT.block(entry).get("type"), "Robot")}'
             + (f' · {CAT.block(entry)["variant"]}' if CAT.block(entry).get('variant') else '')]
    for label, text, src in CAT.describe(entry):
        lines.append(f'{label}: {text}' + (f' ({src})' if src else ''))
    lines.append(f'"—": {NOT_STATED}.')
    for s in entry.sources:
        lines.append(f'Source: {s.get("title") or s.get("url")} — {s.get("url")} (read {s.get("read")})')
    lines.append('Double-click to import it from MuJoCo Menagerie (its licence is shown first).')
    return '\n'.join(lines)
