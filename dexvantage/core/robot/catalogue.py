"""The robot catalogue: arms, hands, grippers, humanoids and mobile manipulators with their makers' figures and the
official model each one imports from (spec §6 "Robots, grippers and hands", §9.4).

One library entry of kind 'robot' per robot (dexvantage/resources/library/robot/<id>.json, schema
dexvantage/resources/library.schema.json, the `robot` block). An entry holds

- the maker's figures, each `{value, source, quote}` with `source` an index into the entry's `sources` (URL or
  document and the date read): `dof`, `reach_mm`, `payload_kg`, `repeatability_mm` (the ± half-width), `mass_kg`,
  `control_hz`, `interfaces`, `joint_ranges` (per joint min and max in the `unit` the maker uses). A figure no source
  read states is null (shown as "—", "not stated by the maker"); nothing is filled in from memory or from the model
  file. `dof` is an integer, or [low, high] when the maker states a range for configurable versions (the Unitree G1
  EDU: "23 - 43").
- `model`: where the official model comes from. For MuJoCo Menagerie, the folder name (which must be one of
  menagerie.OFFLINE_MODELS), the MJCF file the catalogue imports, the commit its LICENSE was read at and the licence
  as menagerie.license_name() recognises it. Menagerie licences differ per model; the import shows the downloaded
  LICENSE again and asks for consent (ui/dialogs/robot_catalogue.py, through the Menagerie dialog).
- `verified`: what DexVantage measured when the model was downloaded once at that commit and imported through
  core/robot/importer.py: the joints in q (hinge and slide joints, as RobotModel counts them), actuators, links, free
  joints, and the DoF compared with the maker's (`dof_rule`: 'joints' = joints in q minus `excluded_joints`, e.g. the
  finger joints of an arm file that includes its gripper; 'actuators' = MuJoCo actuators, for linkages and
  tendon-coupled hands whose passive joints the maker does not count). A model that did not import says so in plain
  words (`imported: false`, `note`). `dof_differs` explains, in the maker's and the model's words, a DoF that differs
  from the maker's.

API
    entries(kind=None) -> [library.Entry]          the catalogue (kind: a ROBOT_TYPES value)
    get(entry_id) -> library.Entry | None
    figure(entry, key) -> (value, source dict | None, quote)
    folders() -> {menagerie folder: entry id}
    measured_dof(verified) -> int | None           the DoF a verified record gives under its rule
    dof_problems(entry) -> [str]                   data-consistency messages ([] = consistent)
    import_entry(entry, opener=None, dest=None, accepted_on=None, progress=None) -> (Robot, sources, download)
                                                    the Menagerie download and import (the caller shows the licence)

Known limits: the figures are what each maker publishes, so they describe the product, not the model file (Menagerie
models are "simplified" descriptions; joint limits come from the file at import, never from the catalogue); makers
state DoF, reach and payload under their own conditions (payload at full or mid reach, repeatability per ISO 9283 or
not), recorded in each quote; a product the maker no longer sells is read from its last datasheet (archived copy,
named in the source); `verified` is a record of one download at one commit, so a later Menagerie commit may differ
until the catalogue is checked again.
"""
from __future__ import annotations

from .. import library as L

ROBOT_TYPES = ('arm', 'bimanual', 'hand', 'gripper', 'humanoid', 'mobile_manipulator')
TYPE_LABELS = {'arm': 'Arm', 'bimanual': 'Bimanual', 'hand': 'Hand', 'gripper': 'Gripper', 'humanoid': 'Humanoid',
               'mobile_manipulator': 'Mobile manipulator'}
FIGURES = ('dof', 'reach_mm', 'payload_kg', 'repeatability_mm', 'mass_kg', 'control_hz', 'interfaces',
           'joint_ranges')
FIGURE_LABELS = {'dof': ('Degrees of freedom', ''), 'reach_mm': ('Reach', 'mm'), 'payload_kg': ('Payload', 'kg'),
                 'repeatability_mm': ('Repeatability', '± mm'), 'mass_kg': ('Mass', 'kg'),
                 'control_hz': ('Control rate', 'Hz'), 'interfaces': ('Interfaces', ''),
                 'joint_ranges': ('Joint ranges', '')}
DOF_RULES = ('joints', 'actuators')


class CatalogueError(ValueError):
    """A catalogue operation DexVantage cannot do; the message says why in plain words."""


def entries(kind: str | None = None) -> list:
    """The catalogue's entries (library kind 'robot'), sorted by maker and name; kind filters by robot type."""
    es = L.entries('robot')
    if kind is not None:
        if kind not in ROBOT_TYPES:
            raise CatalogueError(f'There is no robot type "{kind}". The types are: {", ".join(ROBOT_TYPES)}.')
        es = [e for e in es if (e.data.get('robot') or {}).get('type') == kind]
    return es


def get(entry_id: str):
    e = L.get(entry_id)
    return e if e is not None and e.kind == 'robot' else None


def block(entry) -> dict:
    return (entry.data.get('robot') or {}) if entry is not None else {}


def figure(entry, key: str):
    """(value, the source it comes from or None, the source's words) of one figure; (None, None, '') when no source
    states it."""
    f = block(entry).get(key)
    if not isinstance(f, dict):
        return None, None, ''
    src = f.get('source')
    srcs = entry.sources
    s = srcs[src] if isinstance(src, int) and 0 <= src < len(srcs) else None
    return f.get('value'), s, str(f.get('quote') or '')


def model(entry) -> dict:
    return dict(block(entry).get('model') or {})


def verified(entry) -> dict:
    return dict(block(entry).get('verified') or {})


def folders() -> dict:
    """{Menagerie folder: entry id} of the entries whose model comes from MuJoCo Menagerie."""
    out = {}
    for e in entries():
        m = model(e)
        if m.get('source') == 'menagerie' and m.get('folder'):
            out[m['folder']] = e.id
    return out


def measured_dof(v: dict):
    """The DoF a verified record gives: joints in q minus the excluded joints ('joints'), or the actuators
    ('actuators'); None when the model did not import or the record lacks the count."""
    if not v or not v.get('imported'):
        return None
    rule = v.get('dof_rule', 'joints')
    if rule == 'actuators':
        n = v.get('actuators')
        return int(n) if isinstance(n, int) else None
    n = v.get('joints_in_q')
    return int(n) - len(v.get('excluded_joints') or []) if isinstance(n, int) else None


def dof_matches(stated, measured) -> bool | None:
    """True / False when both are known (a stated [low, high] range holds the measured DoF), None otherwise."""
    if stated is None or measured is None:
        return None
    if isinstance(stated, list) and len(stated) == 2:
        return int(stated[0]) <= measured <= int(stated[1])
    return int(stated) == int(measured)


def dof_problems(entry) -> list:
    """Data-consistency messages for one entry, in plain words ([] when consistent): the verified record's DoF must
    follow from its own counts, and must equal the maker's DoF where the maker states one, unless `dof_differs`
    explains the difference."""
    out = []
    b = block(entry)
    v = verified(entry)
    m = model(entry)
    if not v:
        return [f'{entry.id}: has no verified record.']
    if m.get('commit') and v.get('commit') and m['commit'] != v['commit']:
        out.append(f'{entry.id}: the model commit ({m["commit"][:12]}) is not the commit it was verified at '
                   f'({v["commit"][:12]}).')
    if m.get('file') and v.get('file') and m['file'] != v['file']:
        out.append(f'{entry.id}: the catalogue imports {m["file"]} but {v["file"]} was verified.')
    if not v.get('imported'):
        if not str(v.get('note') or '').strip():
            out.append(f'{entry.id}: the model did not import and the entry does not say why.')
        return out
    rule = v.get('dof_rule', 'joints')
    if rule not in DOF_RULES:
        out.append(f'{entry.id}: unknown DoF rule "{rule}".')
        return out
    names = v.get('joint_names')
    ex = v.get('excluded_joints') or []
    if isinstance(names, list) and any(j not in names for j in ex):
        out.append(f'{entry.id}: excluded joints {[j for j in ex if j not in names]} are not joints of the model.')
    if isinstance(names, list) and len(names) != v.get('joints_in_q'):
        out.append(f'{entry.id}: {len(names)} joint names recorded for {v.get("joints_in_q")} joints in q.')
    meas = measured_dof(v)
    if meas is None:
        out.append(f'{entry.id}: the verified record has no count for its DoF rule "{rule}".')
    elif v.get('dof') != meas:
        out.append(f'{entry.id}: verified DoF {v.get("dof")} does not follow from its counts ({meas} by "{rule}").')
    stated = (b.get('dof') or {}).get('value')
    ok = dof_matches(stated, meas)
    if ok is False and not str(v.get('dof_differs') or '').strip():
        out.append(f'{entry.id}: the maker states {stated} DoF, the model has {meas} ("{rule}"), and the entry does '
                   'not explain the difference.')
    return out


def import_entry(entry, opener=None, dest=None, accepted_on=None, progress=None):
    """(Robot, sources, MenagerieDownload): download the entry's Menagerie folder (menagerie.download) and import the
    catalogue's file from it (menagerie.import_download). The licence must have been shown and accepted by the
    caller; the UI does this through the Menagerie dialog. Raises CatalogueError / MenagerieError in plain words."""
    from . import menagerie as MG
    if isinstance(entry, str):
        e = get(entry)
        if e is None:
            raise CatalogueError(f'The robot catalogue has no entry "{entry}".')
        entry = e
    m = model(entry)
    if m.get('source') != 'menagerie' or not m.get('folder'):
        raise CatalogueError(f'{entry.title} has no MuJoCo Menagerie model to import: '
                             f'{m.get("note") or "see the entry note"}')
    dl = MG.download(m['folder'], dest, opener, progress)
    main = m.get('file') if m.get('file') in dl.files else None
    robot, sources = MG.import_download(dl, main, accepted_on)
    robot.source.note = ' '.join(x for x in (robot.source.note, f'Robot catalogue entry {entry.id}.') if x)
    return robot, sources, dl


def summary(entry) -> str:
    """One short line: type, DoF, reach and payload as stated ("—" where not stated)."""
    b = block(entry)
    parts = [TYPE_LABELS.get(b.get('type'), '')]
    dof = figure(entry, 'dof')[0]
    parts.append(f'{dof[0]}–{dof[1]} DoF' if isinstance(dof, list) else (f'{dof} DoF' if dof is not None else '— DoF'))
    r = figure(entry, 'reach_mm')[0]
    if r is not None:
        parts.append(f'{r:g} mm')
    p = figure(entry, 'payload_kg')[0]
    if p is not None:
        parts.append(f'{p:g} kg')
    return ' · '.join(x for x in parts if x)


def _fmt(key, value) -> str:
    if value is None:
        return '—'
    if key == 'interfaces' and isinstance(value, list):
        return ', '.join(map(str, value))
    if key == 'joint_ranges' and isinstance(value, list):
        return '; '.join(f'{j.get("joint")} {j.get("min"):g} to {j.get("max"):g}' for j in value
                         if isinstance(j, dict) and j.get('min') is not None and j.get('max') is not None)
    if key == 'dof' and isinstance(value, list):
        return f'{value[0]}–{value[1]}'
    unit = FIGURE_LABELS.get(key, ('', ''))[1]
    return f'{value:g} {unit}'.strip() if isinstance(value, (int, float)) else str(value)


def describe(entry) -> list:
    """[(label, text, source title or '')] of every figure, for tooltips and the Inspector; then the model and what
    was verified."""
    rows = []
    for k in FIGURES:
        v, s, q = figure(entry, k)
        text = _fmt(k, v)
        if k == 'joint_ranges' and v is not None:
            text += f' {block(entry)["joint_ranges"].get("unit", "")}'.rstrip()
        rows.append((FIGURE_LABELS[k][0], text, (s or {}).get('title', '') if v is not None else ''))
    m, v = model(entry), verified(entry)
    if m.get('source') == 'menagerie':
        rows.append(('Model', f'MuJoCo Menagerie {m.get("folder")}/{m.get("file") or ""} at '
                              f'{str(m.get("commit") or "")[:12]}, {m.get("license") or "licence not read"}', ''))
    elif m:
        rows.append(('Model', str(m.get('note') or m.get('source') or ''), ''))
    if v.get('imported'):
        rows.append(('Verified', f'{v.get("joints_in_q")} joints in q, {v.get("actuators")} actuators, '
                                 f'{v.get("links")} links on {v.get("date")}; DoF {v.get("dof")} by '
                                 f'{v.get("dof_rule", "joints")}', ''))
    elif v:
        rows.append(('Verified', f'did not import: {v.get("note") or ""}', ''))
    return rows
