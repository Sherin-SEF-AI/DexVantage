"""Structural checks of the built-in depth-camera library entries other than Intel RealSense (spec §6, §1.1).

Every file in dexvantage/resources/library/depth/ (except intel-*) must parse, be named after its id, cite at least one
source with a URL and the date it was read, and carry no figure without a source. The figures themselves cannot be
tested here (they come from the makers' pages); these tests catch the ways an entry can lose its provenance or
contradict itself.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pytest

DEPTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dexvantage', 'resources',
                         'library', 'depth')
FILES = sorted(f for f in os.listdir(DEPTH_DIR) if f.endswith('.json') and not f.startswith('intel-'))
DATE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
# the optical frame (z forward, x right, y down) of a camera looking along body +x (REP-103 body frame)
OPTICAL_Q = np.array([-0.5, 0.5, -0.5, 0.5])

# the devices this part of the library is meant to cover (spec §6)
EXPECTED = {
    'stereolabs-zed-2i', 'stereolabs-zed-mini', 'stereolabs-zed-x', 'stereolabs-zed-x-mini',
    'luxonis-oak-d', 'luxonis-oak-d-pro', 'luxonis-oak-d-sr', 'luxonis-oak-d-lite',
    'orbbec-femto-bolt', 'orbbec-gemini-2', 'orbbec-gemini-335', 'orbbec-gemini-336',
}


def _load(name):
    with open(os.path.join(DEPTH_DIR, name), encoding='utf-8') as f:
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


def test_expected_devices_present():
    ids = {f[:-5] for f in FILES}
    assert EXPECTED <= ids, f'missing entries: {sorted(EXPECTED - ids)}'


@pytest.mark.parametrize('name', FILES)
def test_parses_and_id_matches_filename(name):
    d = _load(name)
    assert isinstance(d, dict)
    assert d.get('id') == name[:-5]
    assert d.get('kind') == 'depth'
    for key in ('maker', 'name', 'note'):
        assert isinstance(d.get(key), str) and d[key].strip(), key


@pytest.mark.parametrize('name', FILES)
def test_sources_have_url_and_read_date(name):
    srcs = _load(name).get('sources')
    assert isinstance(srcs, list) and srcs, 'at least one source'
    for s in srcs:
        assert isinstance(s.get('url'), str) and s['url'].startswith('https://'), s
        assert DATE.match(s.get('read') or ''), s
        assert isinstance(s.get('covers'), str) and s['covers'].strip(), f'source {s["url"]} says what it covers'


@pytest.mark.parametrize('name', FILES)
def test_no_number_without_a_source(name):
    """Structural: an entry carrying any figure must cite a dated source, and every figure is finite."""
    d = _load(name)
    nums = list(_numbers({k: v for k, v in d.items() if k not in ('sources',)}))
    if nums:
        assert any(s.get('url') and DATE.match(s.get('read') or '') for s in d.get('sources') or [])
    for path, v in nums:
        assert math.isfinite(v), path


@pytest.mark.parametrize('name', FILES)
def test_schema_valid(name):
    from dexvantage.core import library
    assert library.validate(_load(name)) == []


@pytest.mark.parametrize('name', FILES)
def test_internal_consistency(name):
    """Names referenced by depth and streams exist; poses are optical frames looking along body x; the reference
    imager sits at the body origin; assumed poses are explained in the note."""
    d = _load(name)
    names = [im['name'] for im in d['imagers']]
    assert len(set(names)) == len(names)
    dp = d.get('depth') or {}
    ref = dp.get('reference_imager')
    assert ref in names
    for n in dp.get('stereo_pair') or []:
        assert n in names
    lower = {n.lower() for n in names} | {'depth', 'rgb'}
    for s in d.get('streams') or []:
        assert s['imager'].lower() in lower, s['imager']
        assert s['fps'] and all(f > 0 for f in s['fps'])
    for im in d['imagers']:
        q = np.asarray(im['pose']['quaternion'], float)
        assert abs(np.linalg.norm(q) - 1) < 1e-9
        assert np.allclose(q, OPTICAL_Q) or np.allclose(q, -OPTICAL_Q), im['name']
        if im['name'] == ref:
            assert np.allclose(im['pose']['position'], 0.0)
            assert im['pose_stated'] is True
    # the right camera of a stereo pair sits one stated baseline along -y
    if dp.get('stereo_pair') and dp.get('baseline_m'):
        right = next(im for im in d['imagers'] if im['name'] == dp['stereo_pair'][1])
        assert np.allclose(right['pose']['position'], [0.0, -dp['baseline_m'], 0.0], atol=1e-9)
    assumed = [im for im in d['imagers'] if not im.get('pose_stated', True)]
    pj = d.get('projector')
    if pj and not pj.get('pose_stated', False):
        assumed.append(pj)
    if assumed:
        assert 'ASSUMPTION' in d['note'], 'an assumed pose is labelled in the note'


@pytest.mark.parametrize('name', FILES)
def test_min_z_rows_unique_and_plausible(name):
    rows = (_load(name).get('depth') or {}).get('min_z') or []
    keys = [(r['width'], r['height']) for r in rows]
    assert len(keys) == len(set(keys))
    for r in rows:
        assert 0 < r['m'] < 5


def test_loader_builds_imagers_for_every_entry(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    from dexvantage.core import library
    from dexvantage.core.project import Sensor
    library.reload()
    for name in FILES:
        mid = name[:-5]
        assert library.get(mid) is not None, mid
        ims = library.sensor_imagers(Sensor(model_id=mid))
        assert ims and all(not getattr(i, 'is_default', False) for i in ims), mid
        for im in ims:
            fov = im.camera.fov_deg()
            assert 0 < fov['h'] < 180 and 0 < fov['v'] < 180, (mid, im.name, fov)
