"""Phase 2 review, truthfulness lens: the library's figures against the makers' documents, the metric of each maker's
depth-error figure, and what the Depth study does with a figure that is not an RMS.

Every figure below was typed again from the maker's own document, read 2026-11-06 for this review (not copied from
the library files): Intel RealSense D400 series datasheet rev 023 (337029-017) Tables 3-12 to 3-20, 3-49, 3-51 to
3-54, 4-12, 4-15, 4-17, 7-7 to 7-10; D457 datasheet rev 006 (337029-011) Tables 3-19, 3-20, 4-4, 4-7, 7-3;
Stereolabs ZED 2i (v1.5), ZED Mini (rev 1.3) and ZED X / X Mini (rev 1.6) datasheets; Luxonis docs and store pages;
Orbbec Gemini 2 (V1.7) and Gemini 330 series (V1.8) datasheets and the Femto Bolt documentation. URLs are in each
entry's sources."""
import json
import math

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core import library as L
from dexvantage.core.frames import Pose
from dexvantage.core.project import Mount, Sensor, TaskObject, Volume, blank_project
from dexvantage.core.robot.samples import look_at

METRIC = {   # what each maker's depth-error figure measures, from the wording of its document
    'intel-realsense-d405': 'rms', 'intel-realsense-d415': 'rms', 'intel-realsense-d435': 'rms',
    'intel-realsense-d435i': 'rms', 'intel-realsense-d455': 'rms', 'intel-realsense-d457': 'rms',
    'orbbec-gemini-2': 'rms', 'orbbec-gemini-335': 'rms', 'orbbec-gemini-336': 'rms',     # "root mean square" / RMS
    'orbbec-femto-bolt': 'std_dev',                                                     # "≤ 17 mm" std. dev.
    'luxonis-oak-d': 'median_abs', 'luxonis-oak-d-pro': 'median_abs', 'luxonis-oak-d-lite': 'median_abs',
    'stereolabs-zed-2i': 'unspecified', 'stereolabs-zed-2i-narrow': 'unspecified',
    'stereolabs-zed-mini': 'unspecified', 'stereolabs-zed-x': 'unspecified', 'stereolabs-zed-x-narrow': 'unspecified',
    'stereolabs-zed-x-mini': 'unspecified', 'stereolabs-zed-x-mini-narrow': 'unspecified',
}


def test_every_depth_error_figure_names_its_metric():
    seen = set()
    for e in L.entries():
        rms = (e.data.get('depth') or {}).get('rms_error')
        if not rms or e.id not in METRIC:
            continue
        assert rms.get('metric') == METRIC[e.id], e.id
        seen.add(e.id)
    assert seen == set(METRIC)
    assert L.get('luxonis-oak-d-sr').data['depth']['rms_error'] is None        # the maker states no figure


@pytest.mark.parametrize('mid, power_w, mass_g, size_mm', [
    ('intel-realsense-d405', 1.482, 58.0, (23.0, 42.0, 42.0)),       # Table 7-10 "All Components"; Table 3-54
    ('intel-realsense-d415', 2.96197, 64.0, (20.0, 99.0, 23.0)),     # Table 7-7; Table 3-51
    ('intel-realsense-d435', 3.40297, 75.0, (25.0, 90.0, 25.0)),     # Table 7-8 (D435x column); Table 3-52
    ('intel-realsense-d455', 3.46147, 116.0, (26.0, 124.0, 29.0)),   # Table 7-9; Table 3-53
    ('intel-realsense-d457', 3.72, 145.0, (36.0, 124.0, 29.0)),      # D457 Table 7-3; Table 3-20 (approximate)
    ('stereolabs-zed-2i', 1.9, 229.0, (43.1, 175.3, 30.3)),          # "Typical consumption: 1.9W"; "229g"
    ('stereolabs-zed-x', 1.46, 239.0, (36.7, 163.4, 31.8)),          # "1.46W (0.122A 12V)"; "239g"
    ('stereolabs-zed-x-mini', 1.46, 151.0, (36.7, 93.6, 31.8)),
    ('orbbec-gemini-2', 2.5, 97.0, (30.0, 90.0, 25.0)),              # "Max average < 2.5W"; "97g ± 3g"
    ('orbbec-gemini-336', 3.0, 99.0, (30.7, 90.0, 25.0)),            # "Average < 3.0W"; "97g / 99g"
    ('orbbec-femto-bolt', 4.35, 348.0, (65.0, 115.0, 40.0)),         # "4.35W average"; documentation 348 g
])
def test_power_mass_size_as_stated(mid, power_w, mass_g, size_mm):
    d = L.get(mid).data
    assert d['power_w'] == pytest.approx(power_w, abs=1e-9)
    assert d['mass_g'] == pytest.approx(mass_g)
    assert tuple(d['size_mm']) == pytest.approx(size_mm)


def test_depth_fov_is_not_the_rgb_fov():
    """The depth FOV (overlap of the stereo pair, Table 3-49) and the colour camera's FOV are different figures; the
    library keeps them apart (D435: depth HD 87 × 58, colour 69 × 42; D455: depth 87 × 58, colour 90 × 65)."""
    for mid, depth, rgb in (('intel-realsense-d435', (87, 58, 95), (69, 42, 77)),
                            ('intel-realsense-d455', (87, 58, 95), (90, 65, 98)),
                            ('intel-realsense-d415', (65, 40, 72), (69.4, 42.5, 77))):
        e = L.get(mid)
        dp = e.data['depth']['depth_fov']
        assert (dp['h'], dp['v'], dp['d']) == depth
        im = e.imager('rgb')
        assert (im['hfov_deg'], im['vfov_deg'], im['dfov_deg']) == rgb
        # the IR imagers carry the sensor's own FOV (Tables 3-12/3-13/3-14), the depth FOV only as a 16:9 mode
        ir = e.imager('IR left')
        assert ir['hfov_deg'] != dp['h'] and any(tuple(m['aspect']) == (16, 9) and m['hfov_deg'] == dp['h']
                                                 for m in ir['fov_modes'])


def test_min_z_rows_belong_to_their_resolution():
    # Gemini 335/336, Table 4-6-1 at disparity range 128 (the default), and the 256 column kept apart
    e = L.get('orbbec-gemini-335')
    mz = {(r['width'], r['height']): r['m'] for r in e.data['depth']['min_z']}
    assert mz == {(1280, 800): 0.26, (640, 400): 0.26, (424, 266): 0.26, (1280, 720): 0.26, (848, 480): 0.18,
                  (640, 360): 0.14, (480, 270): 0.11, (424, 240): 0.1, (640, 480): 0.17, (848, 100): 0.18}
    assert {r['m'] for r in e.data['depth']['min_z_disparity_256']} == {0.25}
    # Luxonis OAK-D: "~40cm (400P OR 800P + extended)", "~80cm (800P)" (docs page): 400P without extended is 40 cm
    mz = {(r['width'], r['height']): r['m'] for r in L.get('luxonis-oak-d').data['depth']['min_z']}
    assert mz == {(1280, 800): 0.8, (640, 400): 0.4}


def test_corrected_notes():
    d405 = L.get('intel-realsense-d405').data
    assert d405['depth']['range_text'] == 'Range 7 cm to 1 m (varies with lighting conditions)'
    assert 'does not say "over"' in d405['note']
    sr = L.get('luxonis-oak-d-sr').data['note']
    # the SR has no datasheet: its note must not quote one for the interface
    assert 'No datasheet PDF is published' in sr and 'gen2 10gbps' not in sr


def _one_sensor_project(model_id):
    p = blank_project()
    p.workspace.volumes = [Volume(name='Table', center=(0.45, 0.0, 0.08), size=(0.2, 0.2, 0.12))]
    p.workspace.objects = [TaskObject(name='Block', shape='box', size=(0.05, 0.05, 0.05),
                                      pose=Pose(np.array([0.45, 0.02, 0.025])))]
    p.variant.sensors.append(Sensor(name='D', slot=0, model_id=model_id, kind='depth',
                                    mount=Mount('cage', '', look_at((0.45, -0.9, 0.6), (0.45, 0.0, 0.05)))))
    p.analysis.voxel_mm = p.analysis.refine_mm = 10.0
    p.analysis.studies['depth_validity'] = {'backend': 'numpy', 'cube_n': 64}
    return p


@pytest.mark.parametrize('mid, metric, equiv', [('intel-realsense-d435', 'rms', True),
                                                ('luxonis-oak-d', 'median_abs', False),
                                                ('stereolabs-zed-mini', 'unspecified', False)])
def test_depth_study_matches_sigma_d_only_to_an_rms_figure(mid, metric, equiv):
    p = _one_sensor_project(mid)
    doc = jobs.run_study('depth_validity', p)
    json.dumps(doc)
    mk = doc['sensors'][0]['maker_rms']
    assert mk['metric'] == metric
    rms = L.get(mid).data['depth']['rms_error']
    assert mk['sigma_m'] == pytest.approx(rms['pct'] / 100 * rms['at_m'])
    assert mk['model_sigma_m'] is not None                                 # the model is still shown at that z
    assert (mk['sigma_d_equiv_px'] is not None) == equiv
    if equiv:
        rec = doc['sensors'][0]
        assert mk['sigma_d_equiv_px'] == pytest.approx(mk['sigma_m'] * rec['f_px'] * rec['baseline_m']
                                                       / rms['at_m'] ** 2)


def test_entry_without_metric_counts_as_unspecified(monkeypatch):
    e = L.get('intel-realsense-d435')
    data = json.loads(json.dumps(e.data))
    del data['depth']['rms_error']['metric']
    monkeypatch.setattr(e, 'data', data)
    doc = jobs.run_study('depth_validity', _one_sensor_project('intel-realsense-d435'))
    mk = doc['sensors'][0]['maker_rms']
    assert mk['metric'] == 'unspecified' and mk['sigma_d_equiv_px'] is None


def test_depth_page_says_what_the_maker_figure_is(app):
    from dexvantage.ui.pages import depth as DP
    assert DP.METRIC_WORDS['median_abs'] == 'median abs.' and 'not an RMS' in DP.METRIC_TIPS['median_abs']
    assert set(DP.METRIC_WORDS) == {'rms', 'median_abs', 'std_dev', 'unspecified'}


def test_pinhole_fov_closed_form_from_intrinsics():
    """The check optics.from_fov now also compares the stated angles with the pinhole closed form from fx, fy, cx, cy
    (independent of CameraModel.fov_deg); the same here for the library's D435 IR imager at 848 × 480 (16:9 mode)."""
    ims = L.sensor_imagers(Sensor(name='x', kind='depth', model_id='intel-realsense-d435'), (848, 480))
    cam = next(i for i in ims if i.name == 'IR left').camera
    h = math.degrees(math.atan((cam.cx + 0.5) / cam.fx) + math.atan((848 - 0.5 - cam.cx) / cam.fx))
    v = math.degrees(math.atan((cam.cy + 0.5) / cam.fy) + math.atan((480 - 0.5 - cam.cy) / cam.fy))
    assert h == pytest.approx(87.0, abs=1e-9) and v == pytest.approx(58.0, abs=1e-9)
