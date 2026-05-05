"""P5.3: Bergeron transmission line reflection theory validation."""

from __future__ import annotations

import numpy as np
import pytest

from transmission_line_emtp_v2 import TransmissionLineFactory
from emtp_solver_v3 import EMTPSolver


@pytest.mark.validation
def test_bergeron_matched_load_no_reflection():
    Zc, tau_per_m, length_m = 300.0, 1e-9, 30.0
    tau = tau_per_m * length_m
    dt = 0.5e-9
    finish = 200e-9

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, 10.0)
    s.add_R("Rs", 1, 2, Zc)
    line = TransmissionLineFactory.create_from_zc_tau("L1", 2, 3, Zc, tau_per_m, length_m)
    s.add_line(line)
    s.add_R("Rload", 3, 0, Zc)
    s.add_voltage_probe("Vm", 3, 0)
    s.run()

    v_end = s.get_voltage_probe("Vm", "V")
    settle_idx = int((tau + dt * 50) / dt)
    if settle_idx < len(v_end):
        v_ss = np.mean(v_end[settle_idx:])
        assert abs(v_ss - 5.0) < 0.1, f"Matched SS: {v_ss:.4f}V"

    # Reflection check: after the incident wave arrives at tau,
    # there should be very little reflected wave (returning at 2*tau).
    idx_start = int(2.5 * tau / dt)
    idx_end = min(int(4 * tau / dt), len(v_end))
    if idx_start < idx_end:
        after_settle = v_end[idx_start:idx_end]
        variation = float(np.max(np.abs(after_settle - 5.0))) / 5.0
        assert variation < 0.1, f"Post-settle variation {variation:.4f}"


@pytest.mark.validation
def test_bergeron_open_circuit_voltage_doubling():
    Zc, tau_per_m, length_m = 300.0, 1e-9, 30.0
    tau = tau_per_m * length_m
    dt = 0.5e-9
    finish = 150e-9

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, 10.0)
    s.add_R("Rs", 1, 2, Zc)
    line = TransmissionLineFactory.create_from_zc_tau("L1", 2, 3, Zc, tau_per_m, length_m)
    s.add_line(line)
    s.add_R("R_open", 3, 0, 1e9)
    s.add_voltage_probe("Vm", 3, 0)
    s.run()

    v_end = s.get_voltage_probe("Vm", "V")
    v_incident = 5.0
    after_ref = v_end[int(1.5 * tau / dt):]
    if len(after_ref) > 0:
        ratio = float(np.max(np.abs(after_ref))) / v_incident
        assert 1.8 < ratio < 2.2, f"Doubling ratio {ratio:.3f}"


@pytest.mark.validation
def test_bergeron_short_circuit_current_doubling():
    Zc, tau_per_m, length_m = 300.0, 1e-9, 30.0
    tau = tau_per_m * length_m
    dt = 0.5e-9
    finish = 150e-9

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, 10.0)
    s.add_R("Rs", 1, 2, Zc)
    line = TransmissionLineFactory.create_from_zc_tau("L1", 2, 3, Zc, tau_per_m, length_m)
    s.add_line(line)
    s.add_R("R_short", 3, 0, 1e-6)
    s.add_branch_current_probe("Im", "R_short")
    s.run()

    i_end = s.get_branch_current_probe("Im", "A")
    i_incident = 5.0 / Zc
    after_ref = i_end[int(1.5 * tau / dt):]
    if len(after_ref) > 0:
        ratio = float(np.max(np.abs(after_ref))) / i_incident
        assert 1.6 < ratio < 2.5, f"Current doubling ratio {ratio:.3f}"
