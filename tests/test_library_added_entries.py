"""The camera, depth and tactile entries added on 2026-11-07 (docs/models/library.md, "Entries"), the schema fields
they brought (focal_length_mm, sensor_size_mm, sensor_diagonal_mm, optical_format, figures_conflict, tactile
area_mm2) and the data-consistency checks of validation/checks_library.py.

Every new entry must parse, match the schema, cite a dated https source with what it covers, load through the library
and appear in the Library dock with its sources in the tooltip. The data-consistency bounds are tested on cameras
built here with known projections (a pinhole, an equidistant fisheye, a pincushion lens that must break them), and
then on the shipped entries.
"""
from __future__ import annotations

import glob
import json
import math
import os

import pytest

from conftest import pump

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dexvantage', 'resources')
LIB = os.path.join(RES, 'library')

NEW = {
    'camera': ('e-con-see3cam-37cugm', 'e-con-see3cam-37cug', 'e-con-see3cam-27cug', 'e-con-see3cam-10cug',
               'e-con-sturdecam29', 'e-con-sturdecam57', 'e-con-nilecam56', 'e-con-sturdecam39',
               'arducam-b0452-ov9281', 'arducam-b0429-ar0234', 'basler-dart-daa1920-160uc',
               'basler-ace2-a2a1920-160ucbas', 'flir-blackfly-s-bfs-u3-04s2m-c', 'flir-blackfly-s-bfs-u3-16s2c-cs',
               'flir-firefly-dl-ffy-u3-16s2m-s-dl', 'allied-vision-alvium-1800-u-240c',
               'allied-vision-alvium-1800-c-158c', 'leopard-li-ar0234csc-gmsl2-060h',
               'leopard-li-ar0234csc-gmsl2-120h', 'luxonis-oak-1', 'luxonis-oak-1-ov9782', 'luxonis-oak-1-w-ov9782',
               'stereolabs-zed-x-one-gs', 'stereolabs-zed-x-one-gs-narrow'),
    'depth': ('intel-realsense-d435f', 'intel-realsense-d435if', 'intel-realsense-d455f', 'intel-realsense-d456',
              'intel-realsense-d436', 'orbbec-gemini-335l', 'orbbec-gemini-336l', 'orbbec-gemini-2-l',
              'orbbec-femto-mega', 'luxonis-oak-d-s2', 'luxonis-oak-d-w', 'luxonis-oak-d-pro-w'),
    'tactile': ('meta-digit-360', 'xela-uspa-44', 'xela-uspa-46', 'paxini-px6ax-gen3-dp-s2015-elite',
                'paxini-px6ax-gen3-ip-m2324-core', 'anyskin', 'reskin', 'contactile-papillarray'),
}
ALL_NEW = [(k, i) for k, ids in NEW.items() for i in ids]


def _load(kind, mid):
    with open(os.path.join(LIB, kind, mid + '.json'), encoding='utf-8') as f:
        return json.load(f)


def _sensor(mid):
    from dexvantage.core.project import Sensor
    return Sensor(model_id=mid)


# ------------------------------------------------------------------------------------------------ the entries
def test_how_many_were_added():
    """The task: at least 15 more cameras; depth and tactile entries beside them."""
    assert len(NEW['camera']) >= 15 and len(NEW['depth']) >= 10 and len(NEW['tactile']) >= 6
    for kind, mid in ALL_NEW:
        assert os.path.isfile(os.path.join(LIB, kind, mid + '.json')), mid


def test_ids_are_unique_across_every_kind():
    ids = {}
    for path in glob.glob(os.path.join(LIB, '*', '*.json')):
        with open(path, encoding='utf-8') as f:
            mid = json.load(f)['id']
        assert mid not in ids, f'{mid} is in both {ids.get(mid)} and {path}'
        ids[mid] = path


@pytest.mark.parametrize('kind,mid', ALL_NEW)
def test_new_entry_matches_the_schema_and_cites_dated_sources(kind, mid):
    from dexvantage.core import library
    d = _load(kind, mid)
    assert d['id'] == mid and d['kind'] == kind
    assert library.validate(d) == []
    assert library.validate_minimal(d) == []
    for s in d['sources']:
        assert s['url'].startswith('https://') and s['read'] == '2026-11-07' and s['title'] and s['covers'], s
    note = d['note']
    for key, word in (('power_w', 'power'), ('mass_g', 'mass'), ('size_mm', 'size')):
        if d.get(key) is None:
            assert word in note.lower(), f'{key} is null but the note does not say why'
    for im in d.get('imagers') or []:
        if not im.get('pose_stated', True):
            assert 'ASSUMPTION' in note


@pytest.mark.parametrize('kind,mid', [(k, m) for k, m in ALL_NEW if k in ('camera', 'depth')])
def test_new_entry_builds_imagers(kind, mid):
    """Each camera and depth entry resolves to its own imagers (never the labelled default); a stated FOV comes out
    as stated at the native image size."""
    from dexvantage.core import library
    library.reload()
    d = _load(kind, mid)
    ims = library.sensor_imagers(_sensor(mid))
    assert [i.name for i in ims] == [im['name'] for im in d['imagers']]
    assert not any(i.is_default for i in ims)
    for got, im in zip(ims, d['imagers']):
        fov = got.camera.fov_deg()
        if im.get('hfov_deg') is None:
            assert 'default' in got.label
        elif (got.camera.width, got.camera.height) == tuple(im['resolution']):
            assert abs(fov['h'] - im['hfov_deg']) < 0.5, (mid, im['name'], fov)
            if im.get('vfov_deg') is not None:
                assert abs(fov['v'] - im['vfov_deg']) < 0.5


@pytest.mark.parametrize('mid', NEW['tactile'])
def test_new_tactile_entry_is_a_contact_sensor(mid):
    from dexvantage.core import library
    library.reload()
    d = _load('tactile', mid)
    assert library.sensor_imagers(_sensor(mid)) == []
    t = d['tactile']
    assert t['sensing'] and (t['pad_mm'] or t.get('area_mm2') or 'pad_mm is null' in d['note'])


def test_figures_the_sources_state():
    """A few figures against the sources as read (the note quotes them)."""
    zx = _load('camera', 'stereolabs-zed-x-one-gs')['imagers'][0]
    assert (zx['hfov_deg'], zx['vfov_deg'], zx['dfov_deg'], zx['focal_length_mm'], zx['f_number']) == \
        (110.0, 80.0, 120.0, 2.2, 2.2)
    d436 = _load('depth', 'intel-realsense-d436')
    rgb = next(im for im in d436['imagers'] if im['name'] == 'RGB')
    assert rgb['shutter'] == 'global' and (rgb['hfov_deg'], rgb['vfov_deg'], rgb['dfov_deg']) == (90.0, 65.0, 98.0)
    assert d436['depth']['baseline_m'] == 0.05 and d436['depth']['depth_start_mm'] == -4.2
    g = _load('depth', 'orbbec-gemini-335l')
    assert g['depth']['baseline_m'] == 0.095 and g['mass_g'] == 133.0
    rgb = next(im for im in g['imagers'] if im['name'] == 'RGB')
    assert rgb['pose_stated'] and rgb['pose']['position'] == [0.0, -0.02375, 0.0] and rgb['shutter'] == 'global'
    assert _load('depth', 'intel-realsense-d455f')['depth']['depth_start_mm'] == -5.05
    assert _load('tactile', 'reskin')['tactile']['pad_mm'] == [20.0, 20.0]
    assert _load('tactile', 'xela-uspa-44')['tactile']['taxels'] == 16
    alv = _load('camera', 'allied-vision-alvium-1800-u-240c')['imagers'][0]
    assert alv['sensor_size_mm'] == [6.7, 4.2] and alv['pixel_um'] == 3.45


def test_schema_refuses_bad_new_fields():
    from dexvantage.core import library
    d = _load('camera', 'stereolabs-zed-x-one-gs')
    d['imagers'][0]['focal_length_mm'] = -1.0
    assert library.validate(d)
    d = _load('camera', 'stereolabs-zed-x-one-gs')
    d['imagers'][0]['sensor_size_mm'] = [1.0]
    assert library.validate(d)
    d = _load('tactile', 'meta-digit-360')
    d['tactile']['area_mm2'] = 'large'
    assert library.validate(d)


# ------------------------------------------------------------------------------------------------ Library dock
def test_every_new_entry_is_listed_in_the_library_dock(win, app):
    from dexvantage.core import library
    from dexvantage.ui.rigedit import entry_summary, entry_tooltip
    library.reload()
    win.library.refill()
    pump(app, 0.02)
    for kind, mid in ALL_NEW:
        it = win.library.item_for(mid)
        assert it is not None, mid
        tip = it.toolTip(0)
        assert 'Source:' in tip and '(read 2026-11-07)' in tip, mid
        e = library.get(mid)
        assert entry_tooltip(e) == tip.split('\nDouble-click')[0]
        assert it.text(1) == entry_summary(e)
    win.library.search.setText('xela')
    assert win.library.item_for('xela-uspa-44') is not None and win.library.item_for('luxonis-oak-1') is None
    win.library.search.setText('')


def test_a_new_camera_is_added_to_the_rig_from_the_dock(win, app):
    win.library.refill()
    n = len(win.store.project.variant.sensors)
    s = win.library.add_device('stereolabs-zed-x-one-gs')
    assert s is not None and s.model_id == 'stereolabs-zed-x-one-gs'
    assert len(win.store.project.variant.sensors) == n + 1
    win.store.undo.undo()
    assert len(win.store.project.variant.sensors) == n


# ------------------------------------------------------------------------------------------------ the checks
def _fov(W, H, project):
    """Full H, V, D angles in degrees of a centred lens whose field angle at image radius r (px) is project(r)."""
    return tuple(2 * math.degrees(project(r)) for r in (W / 2, H / 2, math.hypot(W, H) / 2))


@pytest.mark.parametrize('W,H', [(1920, 1200), (1280, 800), (1600, 1300), (800, 1280)])
def test_fov_bounds_hold_for_pinhole_and_fisheye_lenses(W, H):
    """Angles computed here from lenses whose field angle θ(r) is concave in the image radius - a pinhole
    (θ = atan(r/f)), the equidistant (r/f, the bounds' limiting case) and stereographic (2 atan(r/2f)) fisheyes and a
    barrel-distorted pinhole (r_d = r (1 + k r²), k < 0) - satisfy every bound; lenses whose field angle grows faster
    than linearly - the equisolid fisheye 2 asin(r/2f) and θ = (r/f)² / 2 (strong pincushion) - break them, which is
    why the check does not count fisheye imagers."""
    from dexvantage.validation.checks_library import fov_violations
    f = 0.6 * max(W, H)
    lenses = {'pinhole': lambda r: math.atan(r / f), 'equidistant': lambda r: r / f,
              'stereographic': lambda r: 2 * math.atan(r / (2 * f)),
              'barrel': lambda r: math.atan(_undistort(r / f, -0.1))}
    for name, proj in lenses.items():
        h, v, d = _fov(W, H, proj)
        worst = max(x for _, x in fov_violations(W, H, h, v, d))
        assert worst <= 1e-9, (name, worst)
    for proj in (lambda r: 0.5 * (r / f) ** 2, lambda r: 2 * math.asin(r / (2 * f))):
        h, v, d = _fov(W, H, proj)
        assert max(x for _, x in fov_violations(W, H, h, v, d)) > 1.0


def _undistort(xd, k):
    """Solve x (1 + k x²) = xd for the undistorted normalised radius x (Newton, from x = xd)."""
    x = xd
    for _ in range(100):
        x -= (x * (1 + k * x * x) - xd) / (1 + 3 * k * x * x)
    return x


def test_the_shipped_entries_pass_the_data_consistency_checks():
    from dexvantage.validation import checks_library as C
    from dexvantage.validation.suite import CHECKS

    class Ctx:
        quick = True
    specs = {s.id: s for s in CHECKS if s.id.startswith('library.')}
    # these three are this file's; other library checks (policy token grids, teleop figures, stated totals) are others'
    assert set(specs) >= {'library.sensor_geometry', 'library.fov_shape', 'library.fov_focal'}
    for cid, fn in (('library.sensor_geometry', C._sensor_geometry), ('library.fov_shape', C._fov_shape),
                    ('library.fov_focal', C._fov_focal)):
        o = fn(Ctx())
        assert o.ok and o.error <= specs[cid].tolerance, (cid, o.error, o.detail)
    shape = C._fov_shape(Ctx()).detail
    # a declared conflict is a real one: each declaring imager breaks a bound by more than the rounding allowance
    assert shape['declared_conflicts'] and all(r['violation_deg'] > 1.0 for r in shape['declared_conflicts'])
    assert shape['cases_checked'] > 100


def test_a_mistyped_figure_is_caught(monkeypatch):
    """A pixel pitch typed 10 x too large, or a FOV typed for the wrong axis, fails the checks."""
    from dexvantage.core import library
    from dexvantage.validation import checks_library as C

    class Ctx:
        quick = True
    library.reload()
    e = library.get('allied-vision-alvium-1800-u-240c')
    monkeypatch.setitem(e.data['imagers'][0], 'pixel_um', 34.5)
    assert C._sensor_geometry(Ctx()).error > 0.5
    monkeypatch.setitem(e.data['imagers'][0], 'pixel_um', 3.45)
    z = library.get('stereolabs-zed-x-one-gs')
    monkeypatch.setitem(z.data['imagers'][0], 'vfov_deg', 110.0)
    monkeypatch.setitem(z.data['imagers'][0], 'hfov_deg', 80.0)
    o = C._fov_shape(Ctx())
    assert o.error > C.FOV_TOL_DEG and any(r['entry'] == 'stereolabs-zed-x-one-gs' for r in o.detail['over_tolerance'])
    monkeypatch.setitem(z.data['imagers'][0], 'hfov_deg', 110.0)
    monkeypatch.setitem(z.data['imagers'][0], 'vfov_deg', 80.0)
    base = C._fov_focal(Ctx()).error
    monkeypatch.setitem(z.data['imagers'][0], 'focal_length_mm', 22.0)
    assert C._fov_focal(Ctx()).error == base                     # a longer focal length only narrows the pinhole
    monkeypatch.setitem(z.data['imagers'][0], 'focal_length_mm', 1.0)
    assert C._fov_focal(Ctx()).error > C.FOV_TOL_DEG
