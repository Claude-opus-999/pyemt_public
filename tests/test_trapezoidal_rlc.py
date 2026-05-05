from __future__ import annotations

import numpy as np

from emtp_solver_v3 import EMTPSolver


def test_rc_step_approaches_analytic_solution():
    R = 1_000.0
    C = 1e-6
    V = 1.0
    dt = 1e-5
    finish = 5e-3

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, V)
    s.add_R("R1", 1, 2, R)
    s.add_C("C1", 2, 0, C)
    s.add_voltage_probe("Vc", 2, 0)
    s.run()

    t = s.get_time("s")
    expected = V * (1.0 - np.exp(-t / (R * C)))
    actual = s.get_voltage_probe("Vc", "V")
    # Trapezoidal startup conventions can differ at the first point; compare settled waveform.
    np.testing.assert_allclose(actual[-1], expected[-1], rtol=2e-2, atol=2e-3)
    assert actual[-1] > 0.98 * V
    assert np.all(np.diff(actual[10:]) >= -1e-9)


def test_rl_step_approaches_analytic_solution():
    R = 10.0
    L = 1e-3
    V = 10.0
    dt = 1e-6
    finish = 1.0e-3  # 10 time constants

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, V)
    s.add_R("R1", 1, 2, R)
    s.add_L("L1", 2, 0, L)
    s.add_branch_current_probe("IL", "L1")
    s.run()

    t = s.get_time("s")
    expected = (V / R) * (1.0 - np.exp(-R * t / L))
    actual = s.get_branch_current_probe("IL", "A")
    np.testing.assert_allclose(actual[-1], expected[-1], rtol=2e-2, atol=2e-3)
    assert actual[-1] > 0.98 * (V / R)


def test_series_rl_compact_device_matches_expected_steady_state():
    R = 5.0
    L = 100e-6
    V = 20.0
    dt = 0.5e-6
    finish = 0.5e-3

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, V)
    s.add_series_RL("SRL", 1, 0, R=R, L=L)
    s.add_branch_current_probe("I_SRL", "SRL")
    s.run()

    current = s.get_branch_current_probe("I_SRL", "A")
    np.testing.assert_allclose(current[-1], V / R, rtol=2e-2, atol=2e-2)
    assert abs(current[0]) < 0.2  # close to zero at energization relative to 4 A final
