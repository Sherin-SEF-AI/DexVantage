"""Structural checks of the built-in camera, tactile and host library entries (spec §6, §1.1).

Every file in dexvantage/resources/library/{camera,tactile,host}/ must parse, match the library schema, be named after
its id and live in the folder of its kind, cite at least one dated source with a URL, keep its figures finite and
non-negative, and say in its note why a figure is missing. The figures themselves cannot be tested here (they come
from the makers' pages and datasheets); these tests catch the ways an entry can lose its provenance or contradict
itself, and check that the library builds a camera model for every camera entry.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pytest

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dexvantage', 'resources')
LIB = os.path.join(RES, 'library')
KINDS = ('camera', 'tactile', 'host')
FILES = [(k, f) for k in KINDS if os.path.isdir(os.path.join(LIB, k))
         for f in sorted(os.listdir(os.path.join(LIB, k))) if f.endswith('.json')]
CAMERAS = [f for k, f in FILES if k == 'camera']
DATE = re.compile(r'^(20[0-9]{2})-([01][0-9])-([0-3][0-9])$')
# the optical frame (z forward, x right, y down) of a camera looking along body +x (REP-103 body frame)
OPTICAL_Q = np.array([-0.5, 0.5, -0.5, 0.5])

# the devices this part of the library covers (spec §6: cameras, palm-embeddable micro cameras, tactile, hosts)
EXPECTED = {
    'camera': {'e-con-see3cam-24cug', 'e-con-see3cam-20cug', 'e-con-sturdecam25-cuoagx',
               'leopard-li-ar0234cs-gmsl2-owl', 'leopard-li-ov2312-gmsl2-110h', 'basler-dart-daa1440-220um',
               'flir-blackfly-s-bfs-u3-16s2m-cs', 'omnivision-ovm6948', 'ams-osram-naneyec'},
    'tactile': {'gelsight-mini', 'meta-digit'},
    'host': {'usb-3.2-gen1-host', 'usb-3.2-gen2-host', 'usb-3.2-gen2x2-host', 'leopard-li-jag-adp-gmsl2-8ch',
             'nvidia-jetson-agx-orin-64gb', 'nvidia-jetson-agx-orin-devkit', 'nvidia-jetson-orin-nano-8gb',
             'nvidia-jetson-orin-nano-super-devkit'},
}


def _load(kind, name):
    with open(os.path.join(LIB, kind, name), encoding='utf-8') as f:
        return json.load(f)


def _numbers(node, path=''):
    """Every numeric leaf (not bool) with its path."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _numbers(v, f'{path}.{k}' if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _numbers(v, f'{path}[{i}]')


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _sensor(model_id):
    from dexvantage.core.project import Sensor
    return Sensor(model_id=model_id)


# ------------------------------------------------------------------------------------------------ coverage
@pytest.mark.parametrize('kind', KINDS)
def test_expected_devices_present(kind):
    ids = {f[:-5] for k, f in FILES if k == kind}
    assert EXPECTED[kind] <= ids, f'missing {kind} entries: {sorted(EXPECTED[kind] - ids)}'


def test_camera_coverage():
    """§6: global-shutter USB and GMSL cameras, at least one fisheye, and palm-embeddable micro cameras."""
    cams = [_load('camera', f) for f in CAMERAS]
    gs = [d for d in cams if any(im.get('shutter') == 'global' for im in d['imagers'])]
    assert any('USB' in (d.get('interface') or '') for d in gs)
    assert any('GMSL' in (d.get('interface') or '') for d in gs)
    assert any(im.get('lens_model') == 'kb' for d in cams for im in d['imagers'])
    micro = [d for d in cams if d.get('size_mm') and max(d['size_mm']) < 5.0]
    footprint = [d for d in cams if d.get('size_mm') is None and 'footprint' in d['note']]
    assert micro and len(micro) + len(footprint) >= 2


# ------------------------------------------------------------------------------------------------ structure
@pytest.mark.parametrize('kind,name', FILES)
def test_parses_id_matches_filename_and_folder(kind, name):
    d = _load(kind, name)
    assert isinstance(d, dict)
    assert d.get('id') == name[:-5]
    assert d.get('kind') == kind
    for key in ('maker', 'name', 'note'):
        assert isinstance(d.get(key), str) and d[key].strip(), key


@pytest.mark.parametrize('kind,name', FILES)
def test_matches_the_schema(kind, name):
    jsonschema = pytest.importorskip('jsonschema')
    with open(os.path.join(RES, 'library.schema.json'), encoding='utf-8') as f:
        schema = json.load(f)
    errors = [f'{list(e.absolute_path)}: {e.message}'
              for e in jsonschema.Draft202012Validator(schema).iter_errors(_load(kind, name))]
    assert not errors, errors


@pytest.mark.parametrize('kind,name', FILES)
def test_sources_have_url_title_read_date_and_coverage(kind, name):
    srcs = _load(kind, name).get('sources')
    assert isinstance(srcs, list) and srcs, 'at least one source'
    for s in srcs:
        assert isinstance(s.get('url'), str) and s['url'].startswith('https://'), s
        m = DATE.match(s.get('read') or '')
        assert m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31, s
        assert isinstance(s.get('title'), str) and s['title'].strip(), s
        assert isinstance(s.get('covers'), str) and s['covers'].strip(), f'source {s["url"]} says what it covers'


@pytest.mark.parametrize('kind,name', FILES)
def test_figures_are_finite_and_non_negative(kind, name):
    d = _load(kind, name)
    for path, v in _numbers(d):
        assert math.isfinite(v), path
        if 'quaternion' not in path and 'position' not in path:
            assert v >= 0, path


@pytest.mark.parametrize('kind,name', FILES)
def test_no_contact_details_or_personal_paths(kind, name):
    for s in _strings(_load(kind, name)):
        assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}', s), s
        assert '/home/' not in s and '/Users/' not in s, s


@pytest.mark.parametrize('kind,name', [(k, f) for k, f in FILES if k in ('camera', 'tactile')])
def test_missing_figures_are_explained(kind, name):
    """A device figure left null is accounted for in the note (not stated, a bound, only in a drawing ...)."""
    d = _load(kind, name)
    note = d['note'].lower()
    for key, word in (('power_w', 'power'), ('mass_g', 'mass'), ('size_mm', 'size')):
        if d.get(key) is None:
            assert word in note, f'{key} is null but the note does not say why'


# ------------------------------------------------------------------------------------------------ cameras
@pytest.mark.parametrize('name', CAMERAS)
def test_camera_imagers_are_consistent(name):
    d = _load('camera', name)
    ims = d['imagers']
    assert ims, 'a camera entry has at least one imager'
    names = [im['name'] for im in ims]
    assert len(set(names)) == len(names)
    for im in ims:
        assert im['shutter'] in ('global', 'rolling', None)
        if len(ims) == 1:
            # the body frame's origin is the only imager's optical centre, looking along body +x
            assert im['pose_stated'] is True
            assert np.allclose(im['pose']['position'], 0.0)
            q = np.asarray(im['pose']['quaternion'], float)
            assert np.isclose(abs(q @ OPTICAL_Q), 1.0, atol=1e-12)
        fovs = [im.get(k) for k in ('hfov_deg', 'vfov_deg', 'dfov_deg') if im.get(k) is not None]
        if im.get('lens_model') in ('pinhole', 'brown'):
            assert all(f < 180.0 for f in fovs), 'a pinhole lens cannot see 180 degrees'
        if im.get('lens_model') == 'kb':
            # the fisheye model only where the maker states a fisheye field of view
            assert fovs and max(fovs) >= 180.0
        if im.get('hfov_deg') is not None and im.get('vfov_deg') is not None:
            assert im['vfov_deg'] < im['hfov_deg'] or im['resolution'][1] >= im['resolution'][0]
        if im.get('dfov_deg') is not None:
            assert im['dfov_deg'] >= max(im.get('hfov_deg') or 0, im.get('vfov_deg') or 0)
    for s in d['streams']:
        assert s['imager'] in names, f'stream names an unknown imager {s["imager"]!r}'
        im = ims[names.index(s['imager'])]
        if im.get('resolution'):
            assert s['width'] <= im['resolution'][0] and s['height'] <= im['resolution'][1]
        assert s['fps'] and all(f > 0 for f in s['fps'])


@pytest.mark.parametrize('name', CAMERAS)
def test_library_builds_a_camera_model(name):
    """The loader turns every camera entry into imagers; a stated FOV is reproduced, a missing one is a labelled
    default."""
    from dexvantage.core import library
    library.reload()
    d = _load('camera', name)
    ims = library.sensor_imagers(_sensor(d['id']))
    assert [i.name for i in ims] == [im['name'] for im in d['imagers']]
    for got, im in zip(ims, d['imagers']):
        fov = got.camera.fov_deg()
        if im.get('hfov_deg') is None and im.get('dfov_deg') is not None:
            # only a diagonal stated: a square-pixel pinhole shows it across the image diagonal (labelled derived)
            assert 'derived from the stated diagonal' in got.label
            c = got.camera
            half = math.hypot(c.width, c.height) / 2
            assert abs(2 * math.degrees(math.atan(half / c.fx)) - im['dfov_deg']) < 1e-6 and c.fx == c.fy
        elif im.get('hfov_deg') is None:
            assert 'default' in got.label
        else:
            native = im['resolution']
            if (got.camera.width, got.camera.height) == tuple(native):
                assert abs(fov['h'] - im['hfov_deg']) < 0.5
                if im.get('vfov_deg') is not None:
                    assert abs(fov['v'] - im['vfov_deg']) < 0.5
        if im.get('lens_model') == 'kb':
            assert got.camera.model == 'kb'
            # rays at the stated half-angles land on the image edges (pixel centres at integers: edge at W - 0.5)
            h = math.radians(im['hfov_deg'] / 2)
            uv, ok = got.camera.project(np.array([[math.sin(h), 0.0, math.cos(h)]]))
            assert ok[0] and abs(uv[0, 0] - (got.camera.width - 0.5)) < 0.5
            if im.get('vfov_deg') is not None:
                v = math.radians(im['vfov_deg'] / 2)
                uv, ok = got.camera.project(np.array([[0.0, math.sin(v), math.cos(v)]]))
                assert ok[0] and abs(uv[0, 1] - (got.camera.height - 0.5)) < 0.5


# ------------------------------------------------------------------------------------------------ tactile
@pytest.mark.parametrize('name', [f for k, f in FILES if k == 'tactile'])
def test_tactile_entries_are_contact_sensors(name):
    from dexvantage.core import library
    library.reload()
    d = _load('tactile', name)
    t = d['tactile']
    assert isinstance(t, dict)
    assert t.get('sensing')
    assert t.get('pad_mm') is not None and len(t['pad_mm']) == 2
    assert not d['imagers'], 'a tactile pad is not a workspace camera'
    assert library.sensor_imagers(_sensor(d['id'])) == []
    if t.get('camera_resolution') is not None:
        sizes = {(s['width'], s['height']) for s in d['streams']}
        assert not sizes or tuple(t['camera_resolution']) in sizes


# ------------------------------------------------------------------------------------------------ hosts
@pytest.mark.parametrize('name', [f for k, f in FILES if k == 'host'])
def test_host_entries(name):
    d = _load('host', name)
    h = d['host']
    assert isinstance(h, dict) and h.get('type') in ('usb_controller', 'pcie_card', 'gmsl_deser', 'compute')
    assert not d['imagers'] and not d['streams']
    # a practical figure only when a source states one; none of these sources does
    assert h.get('practical_mbps') is None
    if h['type'] == 'usb_controller':
        m = re.search(r'\((\d+) Gbit/s\)', d['name'])
        assert m and h['stated_mbps'] == 1000.0 * int(m.group(1))
        assert 'practical' in d['note'] and 'measure' in d['note'].lower()
    if h['type'] == 'gmsl_deser':
        assert h.get('gmsl_links')
