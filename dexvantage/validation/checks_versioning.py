"""Checks of the research tools: the project diff and merge (core/projdiff.py), the unpacked folder form
(core/projfolder.py), the BibTeX writer (core/cite.py) and the publication figures (core/figures.py), each against a
reference built independently here.

- versioning.diff_lists_edits: the built-in sample and a copy with 24 seeded random edits from nine kinds (a mount
  moved, a mount turned, a device changed, a sensor added, a volume removed, an object renamed, a study setting set,
  a requirement added, a robot base moved). The reference is the list of edits as applied here; the error counts the
  edits the diff misses plus the changes it reports that were not made. Tolerance 0: a change is listed or not.
- versioning.pose_deltas: the shift (mm) and turn (degrees) the diff reports for the moved and turned mounts against
  the shift applied here (metres × 1000) and the angle of the rotation applied (axis-angle, written out here with
  SciPy's Rotation as the reference for the angle). Tolerance 10⁻⁶ mm or degree: float64 arithmetic on metre-sized
  numbers carries about 10⁻¹³; the bound leaves room for a quaternion renormalised on load.
- versioning.merge_cases: eight three-way merge cases with their expected outcome written out here (disjoint edits
  merge to the union, checked by diffing the result against base: exactly the two sides' edits; the same edit on
  both sides merges; a value changed differently, a removed item changed on the other side, a sensor whose robot the
  other side removed, two edits of one mount conflict). The error counts the cases whose outcome differs. Tolerance 0.
- versioning.unpack_round_trip: the sample with a mesh array and a nested model file written as a folder, read back
  and written again: every file's bytes equal (byte stability), and the project's content hash
  (core/provenance.project_digest, an independent canonical hash) equal before and after, and equal through
  pack → .dexv → load. The error counts differing files and hashes. Tolerance 0.
- versioning.bibtex_parse: the sample's BibTeX read by a brace-counting parser written here: every entry parses, keys
  are unique, and every source URL of the library entries the project names (collected here from the library files)
  is in some entry's url field. The error counts failures. Tolerance 0.
- versioning.svg_data_recovery: a figure of known curves written as SVG, read with xml.etree (the standard library's
  XML parser), the axis map fitted from the tick labels' positions and values alone, every polyline vertex mapped
  back to data and compared with the source. Error: the largest difference as a fraction of the axis range.
  Tolerance 10⁻³: coordinates are written to 0.01 pt on a plot about 150 pt tall (5 × 10⁻⁵ of the range per
  coordinate), so 10⁻³ is twenty times that. The series count and the declared width (88.9 mm) must match too.
- versioning.pdf_page_size: the same figure as PDF two columns wide; the page's MediaBox (ISO 32000-1 §7.7.3.3) read
  from the file's bytes against 182 × 72.8 mm in points. Tolerance 0.5 pt: Qt writes whole points.
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from .suite import Outcome, check

# ====================================================================== diff
def _edit(p, rng, k, applied, touched) -> bool:
    """Apply edit kind k to p (a project copy) on an item no edit touched yet; record (section, op, target, what is
    needed to check it). False when no such item is left."""
    from ..core.frames import quat_from_axis_angle, quat_multiply
    from ..core.project import Requirement, Sensor

    def pick(items):
        free = [x for x in items if x.uid not in touched]
        return free[int(rng.integers(len(free)))] if free else None
    v = p.variant
    if k in (0, 1, 2):
        s = pick(v.sensors)
        if s is None:
            return False
        if k == 0:                                          # move a mount
            d = rng.uniform(-0.05, 0.05, 3)
            s.mount.pose.position = s.mount.pose.position + d
            info = {'shift_mm': d * 1000.0, 'turn_deg': 0.0}
        elif k == 1:                                        # turn a mount
            ang = float(rng.uniform(0.5, 20.0))
            s.mount.pose.quaternion = quat_multiply(s.mount.pose.quaternion,
                                                    quat_from_axis_angle(rng.normal(size=3), math.radians(ang)))
            info = {'shift_mm': np.zeros(3), 'turn_deg': ang}
        else:                                               # change a device
            s.model_id = f'device-{rng.integers(1 << 30)}'
            info = None
        applied.append(('sensors', 'changed', s.uid, info))
        touched.add(s.uid)
    elif k == 3:                                            # add a sensor
        s = Sensor(name=f'Added {rng.integers(1 << 30)}')
        v.sensors.append(s)
        applied.append(('sensors', 'added', s.uid, None))
        touched.add(s.uid)
    elif k == 4:                                            # remove a volume
        x = pick(p.workspace.volumes)
        if x is None:
            return False
        p.workspace.volumes.remove(x)
        applied.append(('workspace', 'removed', x.uid, None))
        touched.add(x.uid)
    elif k == 5:                                            # rename an object
        o = pick(p.workspace.objects)
        if o is None:
            return False
        o.name = f'Renamed {rng.integers(1 << 30)}'
        applied.append(('workspace', 'changed', o.uid, None))
        touched.add(o.uid)
    elif k == 6:                                            # set a study setting
        key = f'k{rng.integers(1 << 30)}'
        p.analysis.studies.setdefault('coverage', {})[key] = float(rng.uniform(1, 9))
        applied.append(('settings', 'changed', key, None))
    elif k == 7:                                            # add a requirement
        r = Requirement(name=f'Req {rng.integers(1 << 30)}', metric='rig.camera_count', threshold=2)
        p.requirements.append(r)
        applied.append(('requirements', 'added', r.uid, None))
    else:                                                   # move a robot base
        r = pick(p.robots)
        if r is None:
            return False
        d = rng.uniform(-0.1, 0.1, 3)
        r.base.position = r.base.position + d
        applied.append(('robots', 'changed', r.uid, {'shift_mm': d * 1000.0, 'turn_deg': 0.0}))
        touched.add(r.uid)
    return True


def _edited_safe(ctx, n=24):
    """The sample, a copy with n seeded edits (one per item), and the edits as applied."""
    rng = ctx.rng('versioning.diff_lists_edits')
    base = ctx.project.snapshot()
    edited = base.snapshot()
    applied, touched = [], set()
    while len(applied) < n:
        _edit(edited, rng, int(rng.integers(9)), applied, touched)
    return base, edited, applied


def _match(changes, applied):
    """(missed, extra): the applied edits no change reports, and the reported changes no edit explains."""
    def ident(c):
        p = c['path']
        m = re.findall(r'\[([^\]]+)\]', p)
        return m[-1] if m else p.rsplit('.', 1)[-1]
    left = list(changes)
    missed = 0
    for sec, op, tgt, _info in applied:
        hit = [c for c in left if c['section'] == sec and c['op'] == op and (ident(c) == tgt or tgt in c['path'])]
        if not hit:
            missed += 1
        else:
            left.remove(hit[0])
    return missed, len(left)


@check('versioning.diff_lists_edits', 'versioning', 'Project diff: exactly the edits made',
       'edits missed plus changes reported that were not made, over 24 seeded random edits of nine kinds',
       'the list of edits applied, kept as they are applied (written here)', 0, 'changes', refs=('git_textconv',))
def _diff_lists_edits(ctx):
    from ..core import projdiff as D
    base, edited, applied = _edited_safe(ctx)
    d = D.diff(base, edited)
    missed, extra = _match(d['changes'], applied)
    same = D.diff(base, base.snapshot())
    return Outcome(missed + extra + (0 if same['identical'] else len(same['changes'])),
                   {'edits': len(applied), 'reported': len(d['changes']), 'missed': missed, 'extra': extra,
                    'kinds': sorted({a[0] + ' ' + a[1] for a in applied}), 'self_diff_identical': same['identical']})


@check('versioning.pose_deltas', 'versioning', 'Project diff: mount shifts and turns',
       'largest difference between the reported shift (mm, per axis) or turn (degrees) and the applied one',
       'the shift applied (metres × 1000) and the angle of the applied rotation (SciPy Rotation.magnitude)',
       1e-6, 'mm or °', refs=('scipy_rotation',))
def _pose_deltas(ctx):
    from scipy.spatial.transform import Rotation

    from ..core import projdiff as D
    base, edited, applied = _edited_safe(ctx)
    d = D.diff(base, edited)
    worst, n = 0.0, 0
    for sec, op, tgt, info in applied:
        if info is None:
            continue
        c = next(c for c in d['changes'] if c['section'] == sec and tgt in c['path'] and 'delta' in c)
        worst = max(worst, float(np.max(np.abs(np.asarray(c['delta']['shift_mm']) - info['shift_mm']))))
        qa = np.asarray(c['before']['quaternion'])
        qb = np.asarray(c['after']['quaternion'])
        ang = math.degrees((Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).magnitude())
        worst = max(worst, abs(c['delta']['turn_deg'] - info['turn_deg']), abs(ang - info['turn_deg']))
        n += 1
    return Outcome(worst, {'poses_compared': n}, ok=n > 0)


# ====================================================================== merge
def _merge_cases(ctx):
    from ..core.project import Sensor
    b = ctx.project.snapshot()
    s0, s1 = b.variant.sensors[0], b.variant.sensors[1]

    def mv(p, uid, dx):
        s = p.variant.sensor(uid)
        s.mount.pose.position = s.mount.pose.position + np.array([dx, 0, 0])

    def dev(p, uid, mid):
        p.variant.sensor(uid).model_id = mid

    def add(p, name):
        p.variant.sensors.append(Sensor(name=name))

    def drop(p, uid):
        p.variant.sensors = [s for s in p.variant.sensors if s.uid != uid]

    def setting(p, k, v):
        p.analysis.studies.setdefault('coverage', {})[k] = v

    def rm_robot(p, uid):                               # the robot and the sensors mounted on it
        p.robots = [r for r in p.robots if r.uid != uid]
        p.variant.sensors = [s for s in p.variant.sensors if s.mount.robot_uid != uid]

    def add_on(p, ruid):
        s = Sensor(name='On the arm')
        s.mount.robot_uid, s.mount.parent_frame = ruid, s0.mount.parent_frame
        p.variant.sensors.append(s)
    robot_of_s0 = s0.mount.robot_uid
    assert robot_of_s0, 'the built-in sample\'s first sensor rides on a robot'
    # (name, ours edits, theirs edits, expected: 'merge' with the number of changes from base, or 'conflict')
    return b, [
        ('disjoint sensors', [lambda p: mv(p, s0.uid, 0.01)], [lambda p: dev(p, s1.uid, 'x-dev')], ('merge', 2)),
        ('additions on both sides', [lambda p: add(p, 'Ours')], [lambda p: add(p, 'Theirs')], ('merge', 2)),
        ('settings and a mount', [lambda p: setting(p, 'a', 1.0)], [lambda p: mv(p, s1.uid, -0.02)], ('merge', 2)),
        ('the same edit', [lambda p: dev(p, s0.uid, 'same')], [lambda p: dev(p, s0.uid, 'same')], ('merge', 1)),
        ('one value, two answers', [lambda p: dev(p, s0.uid, 'a')], [lambda p: dev(p, s0.uid, 'b')], ('conflict',)),
        ('removed and changed', [lambda p: drop(p, s0.uid)], [lambda p: mv(p, s0.uid, 0.01)], ('conflict',)),
        ('one mount, two moves', [lambda p: mv(p, s1.uid, 0.01)], [lambda p: mv(p, s1.uid, 0.02)], ('conflict',)),
        ('a sensor added on a robot the other side removed', [lambda p: add_on(p, robot_of_s0)],
         [lambda p: rm_robot(p, robot_of_s0)], ('conflict',)),
    ]


@check('versioning.merge_cases', 'versioning', 'Three-way merge: merges and conflicts',
       'merge cases whose outcome differs from the expected one (a merge, with the expected number of changes from '
       'the base, or a refusal)', 'eight cases with their outcome written out here', 0, 'cases',
       refs=('git_textconv',))
def _merge(ctx):
    from ..core import projdiff as D
    base, cases = _merge_cases(ctx)
    wrong, detail = 0, []
    for name, eo, et, exp in cases:
        o, t = base.snapshot(), base.snapshot()
        for f in eo:
            f(o)
        for f in et:
            f(t)
        try:
            r = D.merge3(base, o, t)
            got = ('merge', len(D.diff(base, r.project)['changes']))
        except D.MergeConflict as e:
            got = ('conflict',)
            if not e.conflicts:
                got = ('conflict without a list',)
        ok = got == exp
        wrong += not ok
        detail.append({'case': name, 'expected': list(exp), 'got': list(got), 'ok': ok})
    return Outcome(wrong, {'cases': detail})


# ====================================================================== unpacked form
def _with_files(project):
    p = project.snapshot()
    p.meshes = dict(p.meshes, **{'object/mug verts': np.arange(36, dtype=np.float32).reshape(12, 3),
                                 'faces': np.arange(12, dtype=np.int32).reshape(4, 3)})
    p.sources = dict(p.sources, **{'abc123/assets/part.stl': b'solid part\nendsolid part\n'})
    return p


@check('versioning.unpack_round_trip', 'versioning', 'Unpacked folder: byte-stable round trip',
       'files whose bytes differ between two unpackings, plus content hashes that differ (folder, .dexv)',
       'the first unpacking\'s bytes; core/provenance.project_digest of the original', 0, 'files',
       refs=('json_schema', 'zip_appnote'))
def _unpack(ctx):
    from ..core import projfolder as PF, provenance as PV
    from ..core.project import Project
    p = _with_files(ctx.project)
    a = os.path.join(ctx.tmp(), 'a.dexv.d')
    b = os.path.join(ctx.tmp(), 'b.dexv.d')
    z = os.path.join(ctx.tmp(), 'round.dexv')
    PF.save_unpacked(p, a)
    q = Project.load(a)
    PF.save_unpacked(q, b)

    def files(root):
        out = {}
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                path = os.path.join(dp, f)
                with open(path, 'rb') as fh:
                    out[os.path.relpath(path, root)] = fh.read()
        return out
    fa, fb = files(a), files(b)
    differ = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    h0 = PV.project_digest(p)['sha256']
    h1 = PV.project_digest(q)['sha256']
    PF.pack(a, z)
    h2 = PV.project_digest(Project.load(z))['sha256']
    bad_hash = (h0 != h1) + (h0 != h2)
    return Outcome(len(differ) + bad_hash, {'files': len(fa), 'differing': differ[:10], 'hash_folder_equal': h0 == h1,
                                            'hash_dexv_equal': h0 == h2})


# ====================================================================== BibTeX
def parse_bibtex(text: str) -> list:
    """[(type, key, {field: value})] by brace counting (an escaped brace \\{ does not count)."""
    out = []
    i = 0
    while True:
        i = text.find('@', i)
        if i < 0:
            return out
        m = re.match(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text[i:])
        if not m:
            raise ValueError(f'no entry head at {i}')
        j = i + m.end()
        depth, k = 1, j
        while depth:
            c = text[k]
            if c == '\\':
                k += 2
                continue
            depth += (c == '{') - (c == '}')
            k += 1
        body = text[j:k - 1]
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{', body):
            if fm.start() and body[:fm.start()].count('{') - body[:fm.start()].count('\\{') != \
                    body[:fm.start()].count('}') - body[:fm.start()].count('\\}'):
                continue                                    # inside another field's value
            s = fm.end()
            d, e = 1, s
            while d:
                c = body[e]
                if c == '\\':
                    e += 2
                    continue
                d += (c == '{') - (c == '}')
                e += 1
            fields[fm.group(1).lower()] = body[s:e - 1]
        out.append((m.group(1).lower(), m.group(2), fields))
        i = k


@check('versioning.bibtex_parse', 'versioning', 'BibTeX: parses, unique keys, every source cited',
       'entries that do not parse, repeated keys and library source URLs no entry carries',
       'a brace-counting BibTeX parser written here; the URLs read from the library entries\' JSON files here', 0,
       'entries', refs=('bibtex_btxdoc',))
def _bibtex(ctx):
    import json as _json

    from ..core import cite, library as L
    p = ctx.project
    text = cite.bibtex(p)
    try:
        ents = parse_bibtex(text)
    except (ValueError, IndexError) as e:
        return Outcome(1.0, {'parse_error': str(e)})
    keys = [k for _t, k, _f in ents]
    dup = len(keys) - len({k.lower() for k in keys})
    urls = {f.get('url') for _t, _k, f in ents if f.get('url')}
    ids = {s.model_id for v in p.variants for s in v.sensors if s.model_id} | {x.library_id for x in p.policies}
    want = set()
    for mid in ids:
        e = L.get(mid)
        if e is None or not e.path:
            continue
        with open(e.path, encoding='utf-8') as fh:
            want |= {s['url'] for s in _json.load(fh).get('sources') or [] if s.get('url')}
    missing = sorted(want - urls)
    heads = len(re.findall(r'^@', text, re.M))
    return Outcome(dup + len(missing) + abs(heads - len(ents)),
                   {'entries': len(ents), 'duplicate_keys': dup, 'missing_urls': missing[:5],
                    'library_urls': len(want)})


# ====================================================================== figures
def _figure_spec():
    from ..core import figures as F
    x = np.linspace(0.0, 12.0, 49)
    s = [('a', x, 40 + 30 * np.sin(x / 2.0), '#000000', 'line'), ('b', x, 55 - 3 * x, '#000000', 'dash'),
         ('c', x[::4], 20 + x[::4], '#000000', 'points')]
    return F.curves(s, (0.0, 12.0), (0.0, 100.0), xlabel='Distance (cm)', ylabel='Visible (%)'), s


@check('versioning.svg_data_recovery', 'versioning', 'SVG figure: data recovered from the drawing',
       'largest difference between a vertex mapped back through the tick labels and its data value, as a fraction '
       'of the axis range', 'the source arrays; the axis map fitted here from the SVG\'s tick labels alone', 1e-3,
       'of range', refs=('svg11', 'ieee_figure_size', 'tol_colours'))
def _svg(ctx):
    from ..core import figures as F
    spec, src = _figure_spec()
    root = ET.fromstring(F.to_svg(spec))
    ns = {'s': 'http://www.w3.org/2000/svg'}
    ticks = root.findall('.//s:text[@class="tick"]', ns)
    yt = [(float(t.get('y')) - 0.35 * float(t.get('font-size')), float(t.text.replace('−', '-')))
          for t in ticks if t.get('text-anchor') == 'end']
    xt = [(float(t.get('x')), float(t.text.replace('−', '-'))) for t in ticks if t.get('text-anchor') == 'middle']
    ay = np.polyfit([v for _p, v in yt], [p for p, _v in yt], 1)
    ax = np.polyfit([v for _p, v in xt], [p for p, _v in xt], 1)
    groups = root.findall('.//s:g[@class="series"]', ns)
    worst = 0.0
    for g, (_lab, xs, ys, _c, st) in zip(groups, src):
        if st == 'points':
            pts = [(float(c.get('cx')), float(c.get('cy'))) for c in g.findall('s:circle', ns)]
        else:
            pts = [tuple(map(float, q.split(','))) for pl in g.findall('s:polyline', ns)
                   for q in pl.get('points').split()]
        dx = [(px - ax[1]) / ax[0] for px, _py in pts]
        dy = [(py - ay[1]) / ay[0] for _px, py in pts]
        if len(pts) != len(xs):
            return Outcome(1.0, {'series': _lab, 'vertices': len(pts), 'data': len(xs)})
        worst = max(worst, float(np.max(np.abs(np.asarray(dx) - xs))) / 12.0,
                    float(np.max(np.abs(np.asarray(dy) - ys))) / 100.0)
    ok = len(groups) == 3 and root.get('width') == '88.9mm'
    return Outcome(worst, {'series': len(groups), 'width': root.get('width'), 'y_ticks': len(yt), 'x_ticks': len(xt)},
                   ok=ok)


@check('versioning.pdf_page_size', 'versioning', 'PDF figure: page size',
       'largest difference between the PDF MediaBox and the requested two-column size, in points',
       '182 × 72.8 mm converted here (1 in = 25.4 mm = 72 pt)', 0.5, 'pt', refs=('pdf_32000', 'ieee_figure_size'))
def _pdf(ctx):
    from ..core import figures as F
    spec, _src = _figure_spec()
    path = os.path.join(ctx.tmp(), 'fig.pdf')
    r = F.write(spec, path, width='double')
    with open(path, 'rb') as fh:
        data = fh.read()
    m = re.search(rb'/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', data)
    if not m or not data.startswith(b'%PDF'):
        return Outcome(float('inf'), {'pdf': data[:8].decode('latin-1')})
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    w, h = 182.0 / 25.4 * 72.0, r['height_mm'] / 25.4 * 72.0
    return Outcome(max(abs((x1 - x0) - w), abs((y1 - y0) - h)), {'mediabox_pt': [x1 - x0, y1 - y0],
                                                                  'requested_pt': [w, h]})
