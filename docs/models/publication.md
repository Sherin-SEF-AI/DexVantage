# Citations and publication figures

Modules: `dexvantage/core/cite.py` (BibTeX), `dexvantage/core/figures.py` (SVG and PDF figures),
`dexvantage/ui/figure_export.py` (Export figure… on page charts), `dexvantage/batch_research.py` (`--cite`,
`--figure`). Validation: `dexvantage/validation/checks_versioning.py` (model `versioning`).

## BibTeX of what a project depends on

`cite.bibtex(project, references='checks')` writes, in groups:

1. **DexVantage** (`@software{dexvantage, …}`) from `CITATION.cff` at the repository root: title, authors, version,
   licence, repository URL. An installed copy without that file uses the package metadata.
2. **Library sources**: every source of every library entry the project names (the sensors of all rig variants,
   policies, the rig's hosts, cables, lights) as `@misc` with `title`, `author` (the maker, kept whole in braces; not
   for policies, whose titles carry the paper's authors), `url`, `urldate` (the date the source was read, from the
   entry) and a note naming the entries it backs. A datasheet several entries cite is one item.
3. **Robot models**: a MuJoCo Menagerie model as `@misc` with the tree URL at the commit its files came from and its
   folder's licence, read from the robot's source record; Menagerie's own citation as its README gives it
   (`menagerie2022github`, README.md of google-deepmind/mujoco_menagerie, read 2026-11-07); other robots with their
   name, licence and import note.
4. **Datasets** on the Hugging Face Hub the project audits, with their revision (local folders are not citations).
5. **References of the validation checks** (`validation/references.py`): those the checks cite (`checks`, default),
   all (`all`) or none (`none`). A reference written `Authors, "Title", venue, year` becomes `author`, `title`,
   `howpublished` and `year`; any other keeps its whole text as the title (its parts are not guessed).

Keys are unique (a repeated key gets `-2`, `-3`…) and use letters, digits and `- _ : .`. Values are UTF-8 (biber
reads them as they are; classic BibTeX needs `\usepackage[utf8]{inputenc}`), TeX's special characters are escaped
except in `url`. `urldate` is a biblatex field; the date is also in `note` for classic styles.

## Figures

A page's curve chart (`ui/pages/_curves.py`) and bar lists (`ui/charts.py`) give their data as a figure spec
(`figure_data()`, schema `dexvantage.figure/1`): the same series, bands, reference lines and markers the page draws.
`figures.to_svg(spec, …)` writes SVG 1.1 directly; `figures.write(spec, 'f.pdf')` draws that SVG into a one-page
vector PDF through Qt (QSvgRenderer into QPdfWriter: Qt is already a dependency, so no extra is needed). The SVG
carries the spec and where it came from (DexVantage's version, the project, the page, the study, the time) in its
`<metadata>`, so the numbers travel with the picture.

Layout, in points (1 pt = 25.4/72 mm): the plot rectangle is the figure less the tick labels (width estimated at
0.56 em per character), the axis labels, the legend rows and a 2 pt pad. Data map linearly:

    X(x) = x0 + (x − xmin)/(xmax − xmin) · (x1 − x0)       (mirrored when the page draws x right to left)
    Y(y) = y1 − (y − ymin)/(ymax − ymin) · (y1 − y0)

Ticks are round values (steps of 1, 2, 2.5 or 5 × 10ᵏ, as the pages use), with a typographic minus.

Publication defaults (each an option; the widths are a venue's figures, cited):

| Setting | Default | Source |
|---|---|---|
| Width | one column 88.9 mm, two columns 182 mm | IEEE Author Center, "Resolution and Size": "One column width: 3.5 inches, 88.9 millimeters, or 21 picas"; "Two columns width: 7.16 inches, 182 millimeters, or 43 picas" (read 2026-11-07) |
| Height | 0.66 × width (one column), 0.40 × width (two) | DexVantage default |
| Text | 8 pt labels and legend, 7 pt ticks, Helvetica (Arial, sans-serif) | DexVantage default (the IEEE page states no figure font size) |
| Lines | 1 pt data, 0.6 pt axes, light y grid | DexVantage default |
| Series colours | Paul Tol's "bright" scheme in order: #4477AA, #EE6677, #228833, #CCBB44, #66CCEE, #AA3377; grey #BBBBBB dashed from the seventh series (never cycled) | Paul Tol's Notes, "Qualitative colour schemes", https://sronpersonalpages.nl/~pault/ (page dated 16 August 2026, read 2026-11-07) |
| Title | none (the caption carries it) | DexVantage default |

`palette='chart'` keeps the page's own colours (the camera identity colours). A chart without a data spec (the
other page charts) can be saved "as drawn" with Qt's SVG generator: vector, but with the screen's size and theme.

## Validation (measured 2026-11-07)

| Check | Reference | Tolerance (why) | Measured |
|---|---|---|---|
| `versioning.bibtex_parse` | a brace-counting BibTeX parser written in the check; the source URLs read from the library's JSON files | 0 entries | 179 entries on the built-in sample parse, 0 repeated keys, all 4 library source URLs present |
| `versioning.svg_data_recovery` | the source arrays; the axis map fitted from the SVG's tick labels alone (xml.etree) | 10⁻³ of the axis range (coordinates are written to 0.01 pt, about 5 × 10⁻⁵ of a 150 pt plot) | 4.7 × 10⁻⁵ of range; 3 series; width 88.9mm |
| `versioning.pdf_page_size` | 182 × 72.8 mm converted (1 in = 25.4 mm = 72 pt) | 0.5 pt (Qt writes whole points) | 0.36 pt (MediaBox 516 × 206 pt for 515.9 × 206.4) |

## Known limits

Text widths are estimated, not measured from the font, so a long tick or legend label can sit closer to its
neighbour than intended; band labels sit at the foot of the plot and line labels at its top, and can overlap each
other where they meet. The PDF page is rounded to whole points by Qt (up to 0.5 pt, 0.18 mm) and embeds whichever
font Qt resolves for Helvetica. No log axes. Only curve charts and bar lists give data specs; other charts are saved
as drawn. The reference parser splits authors only at "and" and commas. Menagerie commits are read from the robot's
source note and licence text; a robot imported some other way with no URL there is cited by its name and note only.
