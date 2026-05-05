"""P5.2: Basic analytic circuit validation — RC, RL, SeriesRL, convergence."""

from __future__ import annotations

import numpy as np
import pytest

from emtp_solver_v3 import EMTPSolver


def _max_abs_error(sim, ref):
    return float(np.max(np.abs(np.asarray(sim, dtype=float) - np.asarray(ref, dtype=float))))


def _final_value_error_pct(sim, ref, n_tail=10):
    s = np.mean(np.asarray(sim, dtype=float)[-n_tail:])
    r = np.mean(np.asarray(ref, dtype=float)[-n_tail:])
    return float(abs(s - r) / max(abs(r), 1e-30)) * 100


def _rel_peak_err(sim, ref):
    return abs(np.max(np.abs(sim)) - np.max(np.abs(ref))) / max(np.max(np.abs(ref)), 1e-30)


@pytest.mark.validation
def test_rc_step_analytic():
    R, C, V0 = 1000.0, 1e-6, 10.0
    tau = R * C
    dt = tau / 200
    finish = 10 * tau

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, V0)
    s.add_R("R1", 1, 2, R)
    s.add_C("C1", 2, 0, C)
    s.add_voltage_probe("Vc", 2, 0)
    s.run()

    t = s.get_time("s")
    v_sim = s.get_voltage_probe("Vc", "V")
    v_ref = V0 * (1.0 - np.exp(-t / tau))

    assert _max_abs_error(v_sim, v_ref) < 0.03
    assert _final_value_error_pct(v_sim, v_ref) < 0.5
    assert v_sim[-1] > 0.99 * V0


@pytest.mark.validation
def test_rl_step_analytic():
    R, L, V0 = 10.0, 1e-3, 10.0
    tau = L / R
    dt = tau / 200
    finish = 10 * tau

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, V0)
    s.add_R("R1", 1, 2, R)
    s.add_L("L1", 2, 0, L)
    s.add_branch_current_probe("IL", "L1")
    s.run()

    t = s.get_time("s")
    i_sim = s.get_branch_current_probe("IL", "A")
    i_ref = (V0 / R) * (1.0 - np.exp(-R * t / L))

    assert _max_abs_error(i_sim, i_ref) < 0.005
    assert _final_value_error_pct(i_sim, i_ref) < 0.5


@pytest.mark.validation
def test_series_rl_vs_r_plus_l():
    R, L, V0 = 5.0, 100e-6, 20.0
    dt = 0.5e-6
    finish = 0.5e-3

    s1 = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s1.add_VS("Vs", 1, 0, V0)
    s1.add_series_RL("SRL", 1, 0, R=R, L=L)
    s1.add_branch_current_probe("I_SRL", "SRL")
    s1.run()
    i_srl = s1.get_branch_current_probe("I_SRL", "A")

    s2 = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s2.add_VS("Vs", 1, 0, V0)
    s2.add_R("R1", 1, 2, R)
    s2.add_L("L1", 2, 0, L)
    s2.add_branch_current_probe("I_L", "L1")
    s2.run()
    i_sep = s2.get_branch_current_probe("I_L", "A")

    assert _rel_peak_err(i_srl, i_sep) < 0.01


@pytest.mark.validation
def test_rc_dt_convergence():
    R, C, V0 = 1000.0, 1e-6, 1.0
    tau = R * C
    finish = 5 * tau
    prev_err = None

    for dt in [tau / 20, tau / 100, tau / 500]:
        s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
        s.add_VS("Vs", 1, 0, V0)
        s.add_R("R1", 1, 2, R)
        s.add_C("C1", 2, 0, C)
        s.add_voltage_probe("Vc", 2, 0)
        s.run()
        t = s.get_time("s")
        v_sim = s.get_voltage_probe("Vc", "V")
        err = _max_abs_error(v_sim, V0 * (1.0 - np.exp(-t / tau)))
        if prev_err is not None:
            assert err < prev_err, f"dt={dt}: err={err} >= prev={prev_err}"
        prev_err = err
