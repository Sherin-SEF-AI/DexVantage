"""Regression tests for the truthfulness review of Phases 4, 5 and 8: figures a source states that were left null,
words that claimed more than a bound supports, a headline that named the wrong unit, and validation checks whose
reference or comparison was looser than the model needs."""
import math

import numpy as np
import pytest

from dexvantage.core import library as LB
from dexvantage.core import materials as MT
from dexvantage.core.studies import groundtruth as GT
from dexvantage.core.studies import motion_blur as MB
from dexvantage.validation import checks_audit as CA
from dexvantage.validation import checks_lighting as CL
from dexvantage.validation import suite


# ====================================================================== light library: masses the pages state
def test_light_masses_are_the_stated_ones():
    kl, air, ring = (LB.get(i) for i in ('elgato-key-light', 'elgato-key-light-air-mk2', 'elgato-ring-light'))
    assert kl.data['mass_g'] == 1300.0 and '1.3 kg' in kl.data['note']
    assert air.data['mass_g'] == 735.0 and '735 g' in air.data['note']
    assert ring.data['mass_g'] == 1070.0
    for e in (kl, air):
        assert 'mass on that page' not in e.data['note']          # the earlier, wrong "not stated"


def test_projector_power_is_said_to_be_the_laser_maximum():
    for i in ('intel-realsense-d415', 'intel-realsense-d435', 'intel-realsense-d435i', 'intel-realsense-d455'):
        what = LB.get(i).data['projector']['optical_power_what']
        assert 'Table 4-23' in what and 'maximum' in what and '30 mW steps' in what


# ====================================================================== ground truth: the depth sentence's unit
def _conf(a, n):
    return {'tp': int(a * n), 'fp': 0, 'fn': n - int(a * n), 'tn': 0, 'n': n, 'agreement': a, 'iou': a}


def test_groundtruth_advice_names_voxels_and_the_surfaces():
    sens = [{'name': 'Overhead D435', 'compared_on': 'voxels', 'voxels': _conf(0.9966, 480000),
             'surfaces': {**_conf(0.916, 118192), 'inside_volumes': _conf(0.996, 30000)}}]
    depth = {'sensors': sens, 'totals': _conf(0.9966, 480000)}
    s = {'status': 'good'}
    text = GT.advice(s, None, depth, {})
    assert '480,000 task-volume voxels agree' in text and 'surface pixels agree' not in text
    assert 'lowest is Overhead D435: 91.6 % of 118,192 pixels (99.6 % inside the task volume)' in text
    sens[0]['compared_on'] = 'surfaces'
    assert 'rendered surface pixels agree' in GT.advice(s, None, depth, {})


# ====================================================================== motion blur: a bound is not a finding
def test_blur_word_over_a_bound_says_may():
    assert MB._word('serious', True) == 'may be blurred' and MB._word('warning', True) == 'may blur'
    assert MB._word('good', True) == 'sharp'                        # a bound within the limit is a finding
    assert MB._word('serious', False) == 'blurred'


# ====================================================================== checks: tighter references
def test_white_furnace_by_quadrature():
    for a in (0.02, 0.3, 0.9):
        for td in (0.0, 60.0, 89.0):
            assert abs(CL._furnace_quad(MT, a, math.radians(td)) - 1.0) < 1e-8
    s = suite.ValidationSettings()
    s.quick = True
    o = CL._white_furnace(suite.Ctx(None, None, None, s))
    assert o.error < 1e-6 and o.ok and o.detail['monte_carlo_band'] > 0
    spec = next(c for c in suite.CHECKS if c.id == 'lighting.white_furnace')
    assert spec.tolerance == pytest.approx(1e-6)


def test_plan_check_compares_rotation_vectors():
    """The recovered slip (0.19, 1.47, −0.06)° and the truth (0, 1.5, 0)° have sizes 0.020° apart but are 0.20° apart
    as vectors: the check's figure is the vector difference, under a tolerance with a stated reason."""
    got, want = np.array([0.19132868, 1.46671931, -0.06162971]), np.array([0.0, 1.5, 0.0])
    assert abs(np.linalg.norm(got) - np.linalg.norm(want)) < 0.02 < 0.2 < np.linalg.norm(got - want)
    assert np.linalg.norm(got - want) <= CA.PLAN_TOL_DEG
    doc = ' '.join(CA.__doc__.split())
    assert '0.31°' in doc and 'difference of the two rotation vectors' in doc
    assert '±12 ms' not in doc and '16.9 ms' in doc
