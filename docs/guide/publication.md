# Citing your sources and making figures for a paper

A worked example on the UR5e sample. The rules are in [docs/models/publication.md](../models/publication.md).

## 1. A BibTeX file of everything the plan rests on

    dexvantage --sample ur5e --cite ur5e.bib
    185 BibTeX entries → ur5e.bib

The file starts with DexVantage itself, then the sources behind each library entry the project uses, each with the
date it was read:

    @misc{intel-realsense-d405,
      title   = {{RealSense Product Family D400 Series Datasheet, Revision 023, March 2026, …}},
      author  = {{Intel RealSense}},
      url     = {https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf},
      urldate = {2026-11-06},
      note    = {A source of DexVantage library entries intel-realsense-d405, intel-realsense-d435; read 2026-11-06},
    }

then the robot models (the UR5e and 2F-85 folders of MuJoCo Menagerie at commit 98a70c569c4c, each with its licence,
and Menagerie's own citation), π0's paper and code, and the references the validation checks cite. Use
`--cite-references none` for only the project's own sources, `all` for every reference DexVantage holds. With biblatex
the `urldate` fields print as "accessed"; with classic BibTeX the date is in the note.

## 2. A figure from a page

Right-click a chart on an analysis page (the action-chunk error curve, the tactile handoff curve, the audit chart,
the tolerance contributions…): **Export figure, one column (88.9 mm)…** or **two columns (182 mm)…**, then choose
`.svg` or `.pdf`. The figure has the page's data, axis labels and units, 8 pt text and colour-blind-safe colours, on
white, ready for a paper. **Copy figure data (JSON)** puts the spec on the clipboard. Charts that cannot give their
data offer **Export figure as drawn (SVG)…** instead.

Every SVG keeps its data and origin in its metadata; to re-make a figure at another size or in the page's colours:

    dexvantage --figure figure-spec.json fig.pdf --figure-width double --figure-palette chart
    Figure (PDF, 182 × 72.8 mm, 2 series) → fig.pdf

where `figure-spec.json` is the copied JSON (or the `spec` member of the SVG's metadata).
