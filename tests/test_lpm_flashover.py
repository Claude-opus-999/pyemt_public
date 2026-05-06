from __future__ import annotations

import numpy as np

from emtp import EMTPSolver


class SpyLPM:
    """Small test double that verifies solver -> LPM argument semantics."""

    def __init__(self):
        self.calls = []
        self.is_flashed_over = False
        self.R_current = 1e9
        self.G_current = 1e-9

    def update(self, voltage_V, dt, current_A=0.0, time=0.0):
        self.calls.append(
            {
                "voltage_V": float(voltage_V),
                "dt": float(dt),
                "current_A": float(current_A),
                "time": float(time),
            }
        )
        return False

    def reset(self):
        self.calls.clear()
        self.is_flashed_over = False
        self.R_current = 1e9
        self.G_current = 1e-9


def test_solver_passes_branch_voltage_dt_and_time_to_lpm_update():
    dt = 1e-6
    s = EMTPSolver(dt=dt, finish_time=2e-6, verbose=False)
    s.add_VS("Vs", 1, 0, 1234.0)
    s.add_insulator_LPM("GAP", 1, 0, gap_length=1.0, E0=600.0, k=1e-6)

    spy = SpyLPM()
    s._lpm_elements["GAP"] = spy
    s.run()

    assert spy.calls, "LPM update should be called during the nonlinear resolve check"
    first = spy.calls[0]
    np.testing.assert_allclose(first["voltage_V"], 1234.0, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(first["dt"], dt, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(first["time"], 0.0, rtol=0.0, atol=1e-18)


def test_real_lpm_flashover_switches_branch_to_arc_resistance():
    # Deliberately aggressive LPM settings produce a deterministic flashover in a tiny circuit.
    s = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
    s.add_VS("Vs", 1, 0, 2.0e6)
    lpm = s.add_insulator_LPM(
        "GAP", 1, 0,
        gap_length=0.1,
        E0=1.0,
        k=1e-2,
        R_arc=0.5,
        R_open=1e9,
    )
    s.run()

    assert lpm.is_flashed_over is True
    assert s.branches["GAP"].is_closed is True
    np.testing.assert_allclose(s.branches["GAP"].value, 0.5, rtol=0.0, atol=1e-12)
    assert s.get_solver_statistics().get("G_rebuilds", 0) >= 2
