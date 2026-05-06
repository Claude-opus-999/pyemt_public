"""PR0: Waveform regression — key metrics for representative cases.

These tests record scalar invariants (max, min, final value) rather than
full waveforms, so they tolerate minor floating-point changes from
refactoring while still catching real regressions.
"""

import numpy as np
import pytest
from emtp import EMTPSolver


def _run_rc_step():
    s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
    s.add_VS("VS", 1, 0, 1.0)
    s.add_R("R1", 1, 2, 100.0)
    s.add_C("C1", 2, 0, 1e-6)
    s.add_voltage_probe("Vc", 2, 0)
    s.run()
    return s


def _run_rl_step():
    s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
    s.add_VS("VS", 1, 0, 1.0)
    s.add_R("R1", 1, 2, 10.0)
    s.add_L("L1", 2, 0, 10e-3)
    s.add_branch_current_probe("IL", "L1")
    s.run()
    return s


def _run_rc_switch():
    s = EMTPSolver(dt=1e-6, finish_time=200e-6, verbose=False)
    s.add_VS("VS", 1, 0, 1.0)
    s.add_SW("SW1", 1, 2, t_close=50e-6, t_open=150e-6)
    s.add_R("R1", 2, 3, 100.0)
    s.add_C("C1", 3, 0, 1e-6)
    s.add_voltage_probe("Vc", 3, 0)
    s.run()
    return s


# =========================================================================
# RC step
# =========================================================================

class TestRCStepRegression:
    def test_vc_starts_near_zero(self):
        s = _run_rc_step()
        v = s.get_voltage_probe("Vc", "V")
        assert abs(v[0]) < 0.01, f"Vc[0] = {v[0]:.6f}, expected near 0"

    def test_vc_ends_near_one(self):
        s = _run_rc_step()
        v = s.get_voltage_probe("Vc", "V")
        assert 0.55 < v[-1] < 0.70, f"Vc[-1] = {v[-1]:.6f}"

    def test_vc_monotonic_increasing(self):
        s = _run_rc_step()
        v = s.get_voltage_probe("Vc", "V")
        assert np.all(np.diff(v) >= 0), "Vc should be monotonic increasing"

    def test_vc_max_eq_final(self):
        s = _run_rc_step()
        v = s.get_voltage_probe("Vc", "V")
        assert v[-1] == pytest.approx(v.max(), rel=1e-9)

    def test_time_span_correct(self):
        s = _run_rc_step()
        t = s.get_time("s")
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(100e-6)

    def test_step_count(self):
        s = _run_rc_step()
        assert s.step_count == 101  # 0 to 100e-6 inclusive with dt=1e-6


# =========================================================================
# RL step
# =========================================================================

class TestRLStepRegression:
    def test_il_starts_near_zero(self):
        s = _run_rl_step()
        i = s.get_branch_current_probe("IL", "A")
        # Inductor trapezoidal discretization gives ~5e-5 at t=0 for these params
        assert abs(i[0]) < 1e-4, f"IL[0] = {i[0]:.6g}"

    def test_il_ends_positive(self):
        s = _run_rl_step()
        i = s.get_branch_current_probe("IL", "A")
        # τ = L/R = 1ms, 100us is ~10% of τ
        assert i[-1] > 0.005, f"IL[-1] = {i[-1]:.6f}"

    def test_il_monotonic_increasing(self):
        s = _run_rl_step()
        i = s.get_branch_current_probe("IL", "A")
        assert np.all(np.diff(i) >= -1e-15), "IL should be monotonic"


# =========================================================================
# RLC switch
# =========================================================================

class TestRCSwitchRegression:
    def test_vc_before_close_near_zero(self):
        s = _run_rc_switch()
        v = s.get_voltage_probe("Vc", "V")
        assert v[30] < 0.01, f"Vc before close = {v[30]:.6f}"

    def test_vc_after_close_positive(self):
        s = _run_rc_switch()
        v = s.get_voltage_probe("Vc", "V")
        # at t=100us (50us after switch closes at 50us)
        assert v[100] > 0.1, f"Vc at t=100us = {v[100]:.6f}"


# =========================================================================
# Minimal Bergeron line (if available)
# =========================================================================

class TestBergeronRegression:
    def test_bergeron_matched_loads_and_runs(self):
        """Bergeron matched case produces finite probes."""
        from emtp.config.loader import load_case_config
        from emtp.builders.solver_builder import build_solver_from_config

        CASE = PROJECT_ROOT / "cases" / "templates" / "bergeron_matched.json"
        if not CASE.exists():
            pytest.skip("bergeron_matched.json not found")

        config = load_case_config(CASE)
        s = build_solver_from_config(config)
        s.run()
        v = s.get_voltage_probe("V_recv", "V")
        assert len(v) > 0
        assert np.all(np.isfinite(v))

        # Delay should produce initially zero at receive end
        # (line is matched, source is behind the line impedance)
        assert abs(v[0]) < 1e-3, f"V_recv[0] = {v[0]:.6g}"


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
