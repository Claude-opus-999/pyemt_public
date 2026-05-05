"""P5.4b: LPM insulator flashover validation — state machine and P0 regression."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_models_pscad import InsulatorFlashoverLPM, LPMConfig
from emtp_solver_v3 import EMTPSolver


@pytest.mark.validation
def test_lpm_no_flashover_below_threshold():
    lpm = InsulatorFlashoverLPM("gap", LPMConfig(
        gap_length=1.0, k=1e-6, E0=600.0, R_open=1e9, R_arc=1.0,
    ))
    changed = lpm.update(voltage_V=300e3, dt=1e-6, current_A=0.0, time=0.0)
    assert not changed
    assert not lpm.is_flashed_over
    assert lpm.leader_length < lpm.config.gap_length


@pytest.mark.validation
def test_lpm_flashover_with_high_voltage():
    lpm = InsulatorFlashoverLPM("gap", LPMConfig(
        gap_length=0.1, k=1e-4, E0=1.0, R_open=1e9, R_arc=0.5,
    ))
    for step in range(200):
        lpm.update(voltage_V=2e6, dt=1e-6, current_A=0.0, time=step * 1e-6)
    assert lpm.is_flashed_over
    assert lpm.leader_length >= lpm.config.gap_length


@pytest.mark.validation
def test_lpm_flashover_sets_branch_to_arc_resistance():
    s = EMTPSolver(dt=1e-7, finish_time=3e-6, verbose=False)
    s.add_VS("Vs", 1, 0, 2.0e6)
    lpm = s.add_insulator_LPM("GAP", 1, 0, gap_length=0.05, E0=1.0, k=1e-2,
                               R_arc=0.5, R_open=1e9)
    s.run()
    assert lpm.is_flashed_over is True
    br = s.branches["GAP"]
    assert br.is_closed is True
    np.testing.assert_allclose(br.value, 0.5, rtol=1e-12)


@pytest.mark.validation
def test_lpm_solver_voltage_equals_gap_voltage():
    """P0 regression: LPM internal peak voltage = actual gap voltage."""
    s = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
    s.add_VS("Vs", 1, 0, 500e3)
    lpm = s.add_insulator_LPM("GAP", 1, 0, gap_length=0.5, E0=600.0, k=1e-5)
    s.add_voltage_probe("V_gap", 1, 0)
    s.run()

    actual = s.get_voltage_probe("V_gap", "V")
    internal = np.array(lpm.voltage_history) * 1e3  # kV -> V
    np.testing.assert_allclose(
        np.max(np.abs(internal)), np.max(np.abs(actual)), rtol=1e-6, atol=1e-4)
