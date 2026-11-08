"""The hardware library (core/library): the schema against every built-in file, the loader, custom devices in a
throw-away XDG_DATA_HOME, and the imagers, projector and depth figures a project sensor gets."""
import copy
import glob
import json
import math
import os
import pickle

import numpy as np
import pytest

from dexvantage.core import library as L
from dexvantage.core.frames import BODY_TO_OPTICAL, quat_to_matrix
from dexvantage.core.project import Sensor

BUILTIN_FILES = sorted(glob.glob(os.path.join(L.BUILTIN_DIR, '**', '*.json'), recursive=True))
INTEL = ('d405', 'd415', 'd435', 'd435i', 'd455', 'd457')


@pytest.fixture(autouse=True)
def fresh_library(tmp_path, monkeypatch):
    """Each test gets its own user library folder and a freshly read library."""
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    L.reload()
    yield
    monkeypatch.undo()
    L.reload()


def _custom(**kw):
    d = {'id': 'my-test-camera', 'kind': 'camera', 'maker': 'Test maker', 'name': 'Bench camera',
         'sources': [{'url': 'measured on the bench', 'title': 'own measurement', 'read': '2026-11-06',
                      'covers': 'FOV, resolution'}],
         'note': 'A test device; nothing else stated.',
         'imagers': [{'name': 'Main', 'role': 'mono', 'hfov_deg': 70.0, 'vfov_deg': None, 'resolution': [800, 600],
                      'lens_model': 'pinhole', 'shutter': 'global'}]}
    d.update(kw)
    return d


# ---------------------------------------------------------------- the schema and the built-in files
def test_schema_is_valid_draft_2020_12():
    jsonschema = pytest.importorskip('jsonschema', reason='jsonschema (the test extra) is not installed')
    s = L.schema()
    assert s['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
    jsonschema.Draft202012Validator.check_schema(s)
    assert set(s['required']) == {'id', 'kind', 'maker', 'name', 'sources', 'note'}


def test_there_are_built_in_files():
    assert len(BUILTIN_FILES) >= len(INTEL)


@pytest.mark.parametrize('path', BUILTIN_FILES, ids=lambda p: os.path.relpath(p, L.BUILTIN_DIR))
def test_every_built_in_file_matches_the_schema(path):
    """Every file in the folder (any kind, whoever added it), with jsonschema and with the fallback check."""
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    assert L.validate(d) == [], path
    assert L.validate_minimal(d) == [], path
    # its folder is its kind and its file name its id, so the folder is browsable
    assert os.path.basename(os.path.dirname(path)) == d['kind']
    assert os.path.splitext(os.path.basename(path))[0] == d['id']
    for s in d['sources']:
        assert s['url'] and s['read']


@pytest.mark.parametrize('path', BUILTIN_FILES, ids=lambda p: os.path.relpath(p, L.BUILTIN_DIR))
def test_built_in_files_hold_no_personal_paths(path):
    text = open(path, encoding='utf-8').read()
    for bad in ('/' + 'home/', '/' + 'Users/', '/' + 'tmp/', 'C:' + '\\' + 'Users'):   # split: a repo grep stays clean
        assert bad not in text


def test_the_loader_skipped_no_built_in_file():
    assert [p for p in L.problems() if 'built-in' in p] == []
    ids = {e.id for e in L.entries()}
    for f in BUILTIN_FILES:
        assert os.path.splitext(os.path.basename(f))[0] in ids


def test_schema_refuses_what_it_must():
    d = _custom()
    assert L.validate(d) == []
    for broken, word in ((dict(d, sources=[]), 'sources'), (dict(d, kind='lidar'), 'kind'),
                         (dict(d, sources=[{'url': 'x', 'read': 'yesterday'}]), 'read'),
                         (dict(d, power_w='about 2 W'), 'power_w'),
                         ({k: v for k, v in d.items() if k != 'note'}, 'note')):
        for fn in (L.validate, L.validate_minimal):
            msgs = fn(broken)
            assert msgs and any(word in m for m in msgs), (fn.__name__, word, msgs)
    # figures may be null: "not stated by the maker"
    ok = dict(d, power_w=None, mass_g=None, size_mm=None, interface=None)
    assert L.validate(ok) == [] and L.validate_minimal(ok) == []


# ---------------------------------------------------------------- the loader
def test_get_and_entries_and_kind_filter():
    d435 = L.get('intel-realsense-d435')
    assert d435 is not None and d435.kind == 'depth' and d435.maker == 'Intel RealSense' and not d435.is_custom
    assert d435.data['id'] == 'intel-realsense-d435' and 'D435' in d435.name
    assert L.get('no-such-device') is None and L.get('') is None
    depth = L.entries('depth')
    assert {f'intel-realsense-{m}' for m in INTEL} <= {e.id for e in depth}
    assert all(e.kind == 'depth' for e in depth)
    assert len(L.entries()) == sum(len(L.entries(k)) for k in L.KINDS)
    for k in L.KINDS:
        assert all(e.kind == k for e in L.entries(k))
    with pytest.raises(L.LibraryError):
        L.entries('lidar')
    # entries are sorted by kind, maker, name
    es = L.entries()
    keys = [(L.KINDS.index(e.kind), e.maker.lower(), e.name.lower()) for e in es]
    assert keys == sorted(keys)


def test_entry_imager_lookup():
    e = L.get('intel-realsense-d435')
    assert e.imager('IR left')['role'] == 'ir_left'
    assert e.imager('ir_left')['name'] == 'IR left'                # by role too
    assert e.imager('depth')['name'] == 'IR left'                  # the reference imager
    assert e.imager('rgb')['name'] == 'RGB'
    assert e.imager('thermal') is None
    assert {(s['width'], s['height']) for s in e.streams('depth')} >= {(1280, 720), (848, 480)}


def test_custom_add_remove_round_trip():
    d = _custom()
    e = L.add_custom(d)
    assert e.is_custom and e.id == 'my-test-camera' and L.get('my-test-camera') is not None
    path = os.path.join(L.user_dir(), 'camera', 'my-test-camera.json')
    assert os.path.isfile(path) and e.path == path
    assert path.startswith(os.environ['XDG_DATA_HOME'])            # the throw-away folder, never the real one
    with open(path, encoding='utf-8') as fh:
        assert json.load(fh) == d
    assert 'my-test-camera' in {x.id for x in L.entries('camera')}
    # survives a reload (read from disk), replacing keeps one file
    L.reload()
    assert L.get('my-test-camera').is_custom
    L.add_custom(_custom(name='Bench camera, revised'))
    assert L.get('my-test-camera').name == 'Bench camera, revised'
    assert len(glob.glob(os.path.join(L.user_dir(), '**', '*.json'), recursive=True)) == 1
    # a kind change moves the file
    L.add_custom(_custom(kind='tactile', imagers=[], tactile={'pad_mm': None, 'camera_resolution': None,
                                                              'fps': None, 'sensing': 'optical', 'taxels': None}))
    assert not os.path.exists(path) and os.path.isfile(os.path.join(L.user_dir(), 'tactile', 'my-test-camera.json'))
    assert L.remove_custom('my-test-camera') is True
    assert L.get('my-test-camera') is None and L.remove_custom('my-test-camera') is False


def test_custom_refusals_are_plain():
    with pytest.raises(L.LibraryError, match='not saved'):
        L.add_custom(_custom(sources=[]))
    with pytest.raises(L.LibraryError, match='built-in'):
        L.add_custom(_custom(id='intel-realsense-d435', kind='depth'))
    with pytest.raises(L.LibraryError, match='built-in'):
        L.remove_custom('intel-realsense-d435')
    assert L.get('intel-realsense-d435') is not None
    with pytest.raises(L.LibraryError):
        L.add_custom('not a dict')


def test_a_bad_file_is_skipped_never_crashes():
    folder = os.path.join(L.user_dir(), 'camera')
    os.makedirs(folder)
    with open(os.path.join(folder, 'broken.json'), 'w') as fh:
        fh.write('{ not json')
    with open(os.path.join(folder, 'nosource.json'), 'w') as fh:
        json.dump(_custom(id='nosource', sources=[]), fh)
    with open(os.path.join(folder, 'clash.json'), 'w') as fh:
        json.dump(_custom(id='intel-realsense-d405', kind='depth'), fh)
    with open(os.path.join(folder, 'good.json'), 'w') as fh:
        json.dump(_custom(id='good'), fh)
    L.reload()
    probs = L.problems()
    assert len(probs) == 3 and all(p.startswith('Skipped your library') for p in probs)
    assert any('not a readable JSON file' in p for p in probs)
    assert any('does not match the library schema' in p for p in probs)
    assert any('already used by the built-in library' in p for p in probs)
    assert L.get('good').is_custom and L.get('nosource') is None
    assert not L.get('intel-realsense-d405').is_custom


def test_minimal_validation_without_jsonschema(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_jsonschema(name, *a, **k):
        if name == 'jsonschema':
            raise ImportError('not installed (test)')
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', no_jsonschema)
    assert L.validate(_custom()) == []
    assert L.validate(_custom(kind='lidar'))
    L.reload()
    assert L.get('intel-realsense-d435') is not None and L.problems() == []


# ---------------------------------------------------------------- what the figures say (from the datasheet)
def test_intel_figures_as_read():
    """A few figures, as read off Intel's D400 datasheet (Tables 3-49, 4-12, 3-52): a typo in a file shows here."""
    d = {m: L.get(f'intel-realsense-{m}').data for m in INTEL}
    assert [d[m]['depth']['baseline_m'] for m in INTEL] == [0.018, 0.055, 0.050, 0.050, 0.095, 0.095]
    minz = {(r['width'], r['height']): r['m'] for r in d['d435']['depth']['min_z']}
    assert minz[(1280, 720)] == 0.28 and minz[(424, 240)] == 0.105
    assert d['d405']['projector'] is None and d['d405']['depth']['technology'] == 'passive_stereo'
    assert d['d435']['mass_g'] == 75.0 and d['d457']['interface'].startswith('GMSL2')
    assert d['d415']['imagers'][0]['shutter'] == 'rolling' and d['d435']['imagers'][0]['shutter'] == 'global'
    assert d['d435']['imagers'][2]['shutter'] == 'rolling' and d['d455']['imagers'][2]['shutter'] == 'global'
    for m in INTEL:
        e = d[m]
        assert e['depth']['reference_imager'] == 'IR left'
        assert e['depth']['subpixel_px'] is None                    # not stated by the maker
        assert all(im['pixel_um'] is None for im in e['imagers'])  # not stated by the maker
        rgb = next(im for im in e['imagers'] if im['role'] == 'rgb')
        # the RGB offset is stated only where the RGB stream comes from the left imager itself (D405)
        assert rgb['pose_stated'] is (m == 'd405')
        if not rgb['pose_stated']:
            assert 'ASSUMPTION' in e['note']


# ---------------------------------------------------------------- sensors -> imagers
def test_d435_imagers_are_a_baseline_apart_along_minus_y():
    s = Sensor(name='Overhead', kind='depth', model_id='intel-realsense-d435')
    ims = L.sensor_imagers(s)
    by = {im.name: im for im in ims}
    assert {'IR left', 'IR right', 'RGB'} == set(by)
    left, right = by['IR left'], by['IR right']
    assert (left.role, right.role) == ('ir_left', 'ir_right')
    b = L.depth_spec(s)['baseline_m']
    assert b == pytest.approx(0.050)
    d = right.pose_in_body.position - left.pose_in_body.position
    np.testing.assert_allclose(d, [0.0, -b, 0.0], atol=1e-12)
    np.testing.assert_allclose(left.pose_in_body.position, 0.0, atol=1e-12)    # the body origin is the left imager
    # both optical frames look along body x (optical z), x right (body -y), y down (body -z)
    for im in (left, right):
        R = quat_to_matrix(im.pose_in_body.quaternion)
        np.testing.assert_allclose(R, BODY_TO_OPTICAL.T, atol=1e-12)
        assert im.pose_stated and not im.is_default
    # the cameras are the stated depth FOV at the largest depth stream
    cam = left.camera
    assert (cam.width, cam.height) == (1280, 720)
    f = cam.fov_deg()
    assert f['h'] == pytest.approx(87.0, abs=1e-6) and f['v'] == pytest.approx(58.0, abs=1e-6)
    assert 'Depth FOV HD' in left.label
    # unpacks as the contract's 4-tuple
    name, role, pose, camera = left
    assert name == 'IR left' and camera is cam
    # the RGB offset is not stated: flagged and said in the label
    assert not by['RGB'].pose_stated and 'assumption' in by['RGB'].label


def test_mode_choice_and_derived_fov_is_labelled():
    s = Sensor(kind='depth', model_id='intel-realsense-d435')
    ims = {im.name: im for im in L.sensor_imagers(s, (640, 480))}
    assert ims['IR left'].mode == (640, 480)
    assert ims['IR left'].camera.fov_deg()['h'] == pytest.approx(75.0, abs=1e-6)       # the stated VGA depth FOV
    rgb = ims['RGB']
    assert rgb.mode == (640, 480) and 'centred crop' in rgb.label and 'assumption' in rgb.label
    # a mode no imager streams: each keeps its largest stream
    ims = {im.name: im for im in L.sensor_imagers(s, (333, 222))}
    assert ims['IR left'].mode == (1280, 720) and ims['RGB'].mode == (1920, 1080)
    # per-imager modes
    ims = {im.name: im for im in L.sensor_imagers(s, {'RGB': (1280, 720)})}
    assert ims['RGB'].mode == (1280, 720) and ims['IR left'].mode == (1280, 720)
    # same aspect as the stated native FOV: the same FOV
    assert ims['RGB'].camera.fov_deg()['h'] == pytest.approx(69.0, abs=1e-6)


def test_every_intel_entry_gives_cameras():
    for m in INTEL:
        ims = L.sensor_imagers(Sensor(kind='depth', model_id=f'intel-realsense-{m}'))
        assert [im.role for im in ims] == ['ir_left', 'ir_right', 'rgb']
        for im in ims:
            f = im.camera.fov_deg()
            assert 30 < f['h'] < 120 and 20 < f['v'] < 100
        p = L.projector(Sensor(model_id=f'intel-realsense-{m}'))
        if m == 'd405':
            assert p is None
        else:
            assert p['hfov_deg'] > 0 and p['pose_stated'] is False and hasattr(p['pose'], 'matrix')


def test_default_imager_is_labelled():
    for s in (Sensor(name='Wrist camera'), Sensor(model_id='gone-from-the-library')):
        ims = L.sensor_imagers(s)
        assert len(ims) == 1
        im = ims[0]
        assert im.is_default and 'default' in im.label.lower() and 'no device chosen' in im.label
        assert 'no device chosen' in im.name
        cam = im.camera
        assert (cam.width, cam.height) == L.DEFAULT_RESOLUTION
        f = cam.fov_deg()
        assert f['h'] == pytest.approx(90.0, abs=1e-6) and f['v'] == pytest.approx(60.0, abs=1e-6)
    assert 'not in the library' in L.sensor_imagers(Sensor(model_id='gone-from-the-library'))[0].label
    assert L.depth_spec(Sensor()) is None and L.projector(Sensor()) is None


def test_calibration_overrides_intrinsics_and_pose():
    K = [[640.0, 0.0, 639.5], [0.0, 641.0, 359.5], [0.0, 0.0, 1.0]]
    s = Sensor(kind='depth', model_id='intel-realsense-d435', calibration={'imagers': {
        'IR right': {'K': K, 'D': [0.01, -0.02, 0.0, 0.0, 0.0], 'width': 1280, 'height': 720, 'model': 'brown',
                     'pose': {'position': [0.0, -0.0502, 0.0], 'quaternion': [-0.5, 0.5, -0.5, 0.5]}}}})
    ims = {im.name: im for im in L.sensor_imagers(s)}
    r = ims['IR right']
    assert r.calibrated and r.camera.model == 'brown' and r.camera.fx == pytest.approx(640.0)
    assert r.pose_in_body.position[1] == pytest.approx(-0.0502)
    assert not ims['IR left'].calibrated
    # the calibration scales to a stream of the same shape
    r = {im.name: im for im in L.sensor_imagers(s, (640, 360))}['IR right']
    assert r.camera.width == 640 and r.camera.fx == pytest.approx(320.0)
    # a flat calibration applies to the first imager
    s2 = Sensor(model_id='intel-realsense-d435', calibration={'K': K, 'D': [], 'width': 1280, 'height': 720,
                                                              'model': 'pinhole'})
    assert L.sensor_imagers(s2)[0].calibrated
    bad = Sensor(model_id='intel-realsense-d435', calibration={'K': [[1, 2], [3, 4]], 'width': 1280, 'height': 720})
    with pytest.raises(L.LibraryError, match='calibration'):
        L.sensor_imagers(bad)


def test_depth_spec_and_min_z():
    s = Sensor(kind='depth', model_id='intel-realsense-d405')
    dp = L.depth_spec(s)
    assert dp['model_id'] == 'intel-realsense-d405' and dp['stereo_pair'] == ['IR left', 'IR right']
    assert L.min_z(s, 848, 480) == pytest.approx(0.07)
    assert L.min_z(s, 640, 480) is None                             # "-" in the datasheet's table
    assert L.min_z(s, 99, 99) is None
    dp['baseline_m'] = 1.0                                          # a copy: the library is untouched
    assert L.depth_spec(s)['baseline_m'] == 0.018


def test_imager_copies_keep_labels():
    im = L.sensor_imagers(Sensor())[0]
    for c in (copy.copy(im), copy.deepcopy(im), pickle.loads(pickle.dumps(im))):
        assert c.is_default and c.label == im.label and c.mode == im.mode and c.name == im.name


def test_ur5e_sample_sensors_have_their_entries():
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('ur5e')
    by = {s.name: s for s in p.variant.sensors}
    assert by['Wrist D405'].model_id == 'intel-realsense-d405'
    assert by['Overhead D435'].model_id == 'intel-realsense-d435'
    for s in p.variant.sensors:
        assert L.get(s.model_id) is not None and not L.sensor_imagers(s)[0].is_default
    # every model_id a sample names exists in the library
    so = sample_project('so100')
    for s in so.variant.sensors:
        assert not s.model_id or L.get(s.model_id) is not None


def test_entry_numbers_are_finite():
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float):
            assert math.isfinite(x)
    for e in L.entries():
        walk(e.data)


def test_entry_title_says_a_shared_word_once():
    from dexvantage.core import library
    e = library.Entry({'id': 'x', 'kind': 'depth', 'maker': 'Intel RealSense', 'name': 'RealSense Depth Camera D405'})
    assert e.title == 'Intel RealSense Depth Camera D405'
    assert library.Entry({'id': 'y', 'kind': 'camera', 'maker': 'Basler', 'name': 'dart'}).title == 'Basler dart'
    assert library.Entry({'id': 'z', 'kind': 'camera', 'maker': 'Luxonis', 'name': 'Luxonis OAK-D'}).title == \
        'Luxonis OAK-D'
    assert library.get('intel-realsense-d435').title == 'Intel RealSense Depth Camera D435'
