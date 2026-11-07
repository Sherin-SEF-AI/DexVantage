"""The Publication page (Export ▸ Paper kit…): the methods paragraph, the LaTeX tables and the appendix of the paper
kit (core/publication.py) previewed from the results on screen, the sections and options chosen, and the kit
exported to a folder.

The headline counts the numbers of the methods paragraph that trace to the project, a library entry, a result, the
software versions or the check registry, with a status word: "ready" when every number traces, every \\cite key has
its refs.bib entry (and every entry is cited) and the TeX is balanced, else "check" with the first problem. Refresh
preview (Run) builds the kit in the background from the results the window holds for the design as it is now (no
study runs). Apply keeps the chosen sections and options in the project (project.analysis.studies['publication'],
one undo step), so batch exports and the run plan write the same kit; Preview rebuilds the preview with the choices
without keeping them; Explain says how the numbers are traced. Export… writes the kit (main.tex, methods.tex,
tables/, figures/ as PDF and SVG, refs.bib, appendix.tex, reproduce/run_plan.json, provenance.json) into a folder,
optionally running the studies not computed yet first.

Known limits: those of core/publication.py; the preview shows TeX source, not the typeset page; the figures are
written at export, not previewed; the page has no 3D or camera overlays.
"""
from __future__ import annotations

import os
import traceback

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QGridLayout, QLineEdit, QPlainTextEdit, QWidget

from ...core import publication as PB
from ..theme import mono
from ..widgets import Section, button, label
from . import register
from .base import StudyPage, row


class PublicationPage(StudyPage):
    """Publication kit: a methods paragraph, booktabs tables, figures, refs.bib and a reproducibility appendix written
    from the project and the results on screen. The headline counts the numbers of the methods paragraph that trace
    to the project, a library entry, a result or the software versions (every one must); Export writes the kit into a
    folder with provenance.json listing where each number came from."""
    STUDY = None
    TITLE = 'Publication'
    HEADLINE = 'Methods numbers traced to a source'
    MENU_TITLE = 'Paper kit…'

    def __init__(self, ctx):
        self.kit = None
        self._busy = False
        super().__init__(ctx)
        self.b_run.setText('Refresh preview')
        self.b_run.setToolTip('Build the kit from the results on screen (no study runs)')
        self.b_apply.setToolTip('Keep these sections and options in the project (one undo step)')
        self.b_preview.setToolTip('Rebuild the preview with these choices without keeping them')
        self.b_export.setToolTip('Write the paper kit into a folder (main.tex, tables, figures, refs.bib, provenance)')
        self.b_preview.setEnabled(True)
        ctx.store.changed.connect(lambda *_: self._sync_apply())
        if hasattr(ctx.store, 'project_replaced'):
            ctx.store.project_replaced.connect(self._load_settings)
        self._load_settings()

    # ---------------------------------------------------------------- the body
    def build_body(self, layout):
        sec = Section('Sections')
        self.sec_checks = {}
        grid_w = QWidget()
        grid = QGridLayout(grid_w)                    # three columns: eleven sections in four rows
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(2)
        for k, sid in enumerate(PB.SECTIONS):
            cb = QCheckBox(PB.SECTION_TITLES[sid])
            cb.setChecked(True)
            cb.toggled.connect(self._sync_apply)
            self.sec_checks[sid] = cb
            grid.addWidget(cb, k % 4, k // 4)
        grid.setColumnStretch(3, 1)
        sec.add(grid_w)
        layout.addWidget(sec)
        opt = Section('Options')
        self.c_si = QCheckBox('SI units with siunitx (\\num, \\unit)')
        self.c_single = QCheckBox('One column (88.9 mm)')
        self.c_double = QCheckBox('Two columns (182 mm)')
        self.c_pdf = QCheckBox('PDF figures beside the SVGs')
        self.c_run = QCheckBox('Run the studies not computed yet when exporting')
        self.c_run.setToolTip('Export only: runs every study the window has no result for (long studies take '
                              'their time); the preview never runs a study')
        for cb in (self.c_si, self.c_single, self.c_double, self.c_pdf):
            cb.toggled.connect(self._sync_apply)
        self.bib = QComboBox()
        self.bib.addItem('BibTeX (plain style)', 'bibtex')
        self.bib.addItem('biblatex with biber', 'biblatex')
        self.bib.currentIndexChanged.connect(self._sync_apply)
        self.title = QLineEdit()
        self.title.setPlaceholderText('the project\'s name: sensing rig')
        self.title.textChanged.connect(self._sync_apply)
        self.validation = QLineEdit()
        self.validation.setPlaceholderText('validation.json (optional: the validation status of each model)')
        b_val = button('Browse…', role='ghost', tip='A validation report written by Help ▸ Validate or --validate')
        b_val.clicked.connect(self._browse_validation)
        for w in (self.c_si, self.c_single, self.c_double, self.c_pdf, self.c_run):
            opt.add(w)
        opt.row('Bibliography', self.bib)
        opt.row('Title', self.title)
        opt.row('Validation', self.validation, b_val)
        layout.addWidget(opt)
        pv = Section('Preview')
        self.part = QComboBox()
        self.part.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.part.currentIndexChanged.connect(self._show_part)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono(11))
        self.text.setMinimumHeight(220)
        self.text.setPlaceholderText('Refresh the preview to see the methods paragraph and the tables.')
        self.problems = label('', muted=True, wrap=True)
        pv.add(row(label('Part', muted=True), self.part))
        pv.add(self.text)
        pv.add(self.problems)
        layout.addWidget(pv)

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        widths = [w for w, cb in (('single', self.c_single), ('double', self.c_double)) if cb.isChecked()]
        return {'sections': [s for s, cb in self.sec_checks.items() if cb.isChecked()],
                'siunitx': self.c_si.isChecked(), 'widths': widths or ['single'],
                'pdf_figures': self.c_pdf.isChecked(), 'bibliography': self.bib.currentData() or 'bibtex',
                'title': self.title.text().strip()}

    def _stored(self) -> dict:
        try:
            return PB.stored_settings(PB.settings_of(self.ctx.store.project))
        except PB.PublicationError:
            return PB.stored_settings(PB.settings_of(None))

    def _load_settings(self, *_):
        st = self._stored()
        widgets = list(self.sec_checks.values()) + [self.c_si, self.c_single, self.c_double, self.c_pdf, self.bib,
                                                    self.title]
        for w in widgets:
            w.blockSignals(True)
        for s, cb in self.sec_checks.items():
            cb.setChecked(s in st['sections'])
        self.c_si.setChecked(bool(st['siunitx']))
        self.c_single.setChecked('single' in st['widths'])
        self.c_double.setChecked('double' in st['widths'])
        self.c_pdf.setChecked(bool(st['pdf_figures']))
        self.bib.setCurrentIndex(max(0, self.bib.findData(st['bibliography'])))
        self.title.setText(st.get('title') or '')
        for w in widgets:
            w.blockSignals(False)
        self._sync_apply()

    def _sync_apply(self, *_):
        try:
            differs = PB.stored_settings(PB.settings_of(None, self.settings())) != self._stored()
        except PB.PublicationError:
            differs = False
        self.b_apply.setEnabled(differs)
        self.b_preview.setEnabled(True)

    def _browse_validation(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Validation report', '', 'Validation report (*.json)')
        if path:
            self.validation.setText(path)

    def options(self, for_export: bool = False) -> dict:
        """The kit's options: the chosen settings, the results on screen for the design as it is now, the
        validation report when one is named."""
        o = self.settings()
        an = self.ctx.analysis
        rev = self.ctx.store.revision
        o['results'] = {sid: {k: v for k, v in d.items() if k not in ('revision', 'cached')}
                        for sid, d in dict(getattr(an, 'results', {}) or {}).items()
                        if isinstance(d, dict) and d.get('revision', rev) == rev}
        o['studies'] = 'all' if for_export and self.c_run.isChecked() else []
        if self.validation.text().strip():
            o['validation'] = self.validation.text().strip()
        return o

    # ---------------------------------------------------------------- run (the preview)
    def run(self):
        """Build the kit from the results on screen in the background."""
        try:
            opts = self.options()
        except Exception:                                   # noqa: BLE001 — explained, never a traceback
            traceback.print_exc()
            self.head.set('—', '', 'critical', 'could not build', 'The choices could not be read.')
            return False
        snap = self.ctx.store.project.snapshot()

        def job(progress=None, cancel=None):
            try:
                kit = PB.build(snap, opts)
            except PB.StudyError as e:
                return {'error': str(e)}
            return {'kit': {k: v for k, v in kit.items() if k != '_source'}}
        win = self.ctx.window
        if win is None:
            self._done(job())
            return True
        self._busy = True
        self.note.setText('Building the kit preview…')
        started = win.run_job('publication', 'Paper kit preview', job, self._done, self._failed_job)
        if not started:
            self.note.setText('A kit preview is already being built.')
        return bool(started)

    def _failed_job(self, message):
        self._busy = False
        self.head.set('—', '', 'critical', 'could not build', 'The kit could not be built (details in the log).')

    def _done(self, res):
        self._busy = False
        if 'error' in res:
            self.head.set('—', '', 'critical', 'could not build', res['error'])
            return
        self.kit = res['kit']
        self.doc = PB.public(self.kit)
        self.show_result(self.kit)

    def show_result(self, kit):
        st = kit['status']
        self.head.set(f'{st["traced"]} / {st["numbers"]}', 'numbers', st['level'], st['word'], st['text'])
        names = [k for k in ('methods', 'sensors', 'results', 'scorecard', 'experiments', 'variants', 'appendix')
                 if k in kit['parts']] + ['refs.bib']
        cur = self.part.currentData()
        self.part.blockSignals(True)
        self.part.clear()
        for n in names:
            self.part.addItem(n if n == 'refs.bib' else PB.SECTION_TITLES.get(n, n), n)
        idx = self.part.findData(cur)
        self.part.setCurrentIndex(idx if idx >= 0 else 0)
        self.part.blockSignals(False)
        self._show_part()
        c = kit['checks']
        bits = [f'{len(c["keys"]["used"])} citations, all in refs.bib' if not c['keys']['missing'] else
                f'citations without an entry: {", ".join(c["keys"]["missing"])}',
                f'{c["tables"]["numbers"]} table numbers checked against their sources']
        if kit.get('refused'):
            bits.append(f'{len(kit["refused"])} studies could not run')
        if kit.get('scorecard_missing'):
            bits.append('the scorecard needs ' + ', '.join(kit['scorecard_missing']))
        if not kit['results']:
            bits.append('no study result on screen yet: run the studies to describe them')
        self.problems.setText('; '.join(bits) + '.')
        self.note.setText(f'Kit preview of {len(kit["results"])} result(s); Export… writes it.')

    def _show_part(self, *_):
        if not self.kit:
            return
        n = self.part.currentData()
        self.text.setPlainText(self.kit['bib'] if n == 'refs.bib' else self.kit['parts'].get(n, ''))

    def empty(self):
        self.head.set('—', '', 'warning', 'not built', 'Refresh the preview to build the methods paragraph and the '
                      'tables from the results on screen, or Export… to write the kit.')

    # ---------------------------------------------------------------- Apply, Preview, Explain, Export
    def apply(self):
        st = PB.stored_settings(PB.settings_of(None, self.settings()))

        def fn(p):
            p.analysis.studies[PB.STUDY_ID] = dict(st)
        self.ctx.store.edit('analysis', fn, 'Publication · keep the paper kit settings')
        self._sync_apply()
        if self.ctx.window is not None:
            self.ctx.window.msg('The paper kit settings are kept in the project (one undo step).', 5000)
        return st

    def preview(self):
        return self.run()

    def explain(self):
        if self.ctx.window is not None:
            self.ctx.window.msg('Each number of the methods paragraph is written with the place it is read from (the '
                                'project, a library entry, a result, the versions); the kit resolves every place '
                                'again and checks the number, and each table row names the source of its numbers.',
                                15000)

    def export(self, path=None):
        """Write the kit into a folder (a window job); path: the folder (else asked)."""
        win = self.ctx.window
        if path is None:
            path = QFileDialog.getExistingDirectory(self, 'Paper kit folder', '')
            if not path:
                return None
        try:
            opts = self.options(for_export=True)
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            return None
        if win is None:
            from ...core import exporters as X
            return X.export(self.ctx.store.project.snapshot(), path, ['paper_kit'], {'paper_kit': opts})
        from ..dialogs import export as EXD

        def done(man, error=''):
            if man is None:
                self.note.setText(error)
                return
            row_ = man['formats'].get('paper_kit') or {}
            self.note.setText(row_.get('reason') or (row_.get('notes') or [''])[0])
        EXD.run(win, path, ['paper_kit'], {'paper_kit': opts}, on_done=done)
        self.note.setText(f'Writing the paper kit into {os.path.basename(os.path.normpath(path))}…')
        return path


register('publication', 'Publication', PublicationPage, menu='Export', icon='export', action_id=None, workspace=None,
         order=140)
