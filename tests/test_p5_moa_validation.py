"""P5.4a: MOA segmented V-I curve validation."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_models_pscad import SegmentedMOAResistor


@pytest.mark.validation
def test_moa_breakpoints_exact_current():
    points = [(0.0, 0.0), (100e3, 1.0), (200e3, 10.0), (300e3, 100.0), (400e3, 1000.0)]
    moa = SegmentedMOAResistor.from_breakpoints("MOA", points, add_zero_point=False)
    for vk, ik in points:
        actual = moa.get_current_exact(vk)
        np.testing.assert_allclose(actual, ik, rtol=1e-9, atol=1e-12)


@pytest.mark.validation
def test_moa_negative_voltage_symmetry():
    points = [(0.0, 0.0), (100.0, 1e-3), (200.0, 0.1), (300.0, 10.0)]
    moa = SegmentedMOAResistor.from_breakpoints("MOA", points, add_zero_point=False)
    for v in [50.0, 150.0, 250.0, 350.0]:
        i_pos = moa.get_current_exact(v)
        i_neg = moa.get_current_exact(-v)
        np.testing.assert_allclose(i_neg, -i_pos, rtol=1e-9, atol=1e-12)


@pytest.mark.validation
def test_moa_norton_equivalent_finite():
    points = [(0.0, 0.0), (100.0, 1e-3), (200.0, 0.1), (300.0, 10.0)]
    moa = SegmentedMOAResistor.from_breakpoints("MOA", points, add_zero_point=False)
    for v in [0.0, 50.0, 150.0, 250.0]:
        g, i_eq = moa.get_norton_equivalent(v)
        assert g > 0
        assert np.isfinite(g)
        assert np.isfinite(i_eq)


@pytest.mark.validation
def test_moa_segment_switching():
    points = [(0.0, 0.0), (100.0, 1e-3), (200.0, 0.1), (300.0, 10.0)]
    moa = SegmentedMOAResistor.from_breakpoints("MOA", points, add_zero_point=False)
    assert moa.current_segment_index == 0
    changed, new_idx = moa.check_segment(250.0)
    assert changed and new_idx >= 2
    assert moa.update_segment(250.0)
    assert moa.current_segment_index == new_idx
