"""PR4: RHSEngine tests — wrapper existence and plan invalidation."""

import numpy as np
import pytest
from emtp import EMTPSolver


class TestRHSEngineExistence:
    def test_engine_present_after_init(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        assert s.rhs_engine is not None

    def test_engine_build_returns_valid_rhs(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r1", 1, 0, 100.0)
        s.run()  # solver.run() calls engine.build() internally
        assert s.step_count > 0

    def test_engine_build_with_current_source(self):
        s = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False)
        s.add_IS("isrc", 1, 0, lambda t: 1.0)
        s.add_R("r", 1, 0, 50.0)
        s.add_voltage_probe("V1", 1, 0)
        s.run()
        v = s.get_voltage_probe("V1", "V")
        # 1A * 50Ω = 50V with a DC current source and resistive load
        assert abs(v[0]) > 10, f"V1[0] = {v[0]:.1f}, expected > 10"
        assert abs(v[-1]) > 10, f"V1[-1] = {v[-1]:.1f}"

    def test_engine_invalidate_plan(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        assert s.rhs_engine is not None
        # plan invalidation should not raise
        s.rhs_engine.invalidate_plan()

    def test_rhs_engine_with_pre_sampled_sources(self):
        s = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False,
                       pre_sample_sources=True)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 1, 0, 100.0)
        s.add_voltage_probe("V1", 1, 0)
        s.run()
        v = s.get_voltage_probe("V1", "V")
        assert len(v) > 0

    def test_pre_sample_on_vs_constant_gives_same_result_as_off(self):
        """pre_sample_sources=True/False produce identical results for DC VS."""
        s_on = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False,
                          pre_sample_sources=True)
        s_on.add_VS("vs", 1, 0, 1.0)
        s_on.add_R("r", 2, 0, 100.0)
        s_on.add_C("c", 1, 2, 1e-6)
        s_on.add_voltage_probe("Vc", 2, 0)
        s_on.run()

        s_off = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False,
                           pre_sample_sources=False)
        s_off.add_VS("vs", 1, 0, 1.0)
        s_off.add_R("r", 2, 0, 100.0)
        s_off.add_C("c", 1, 2, 1e-6)
        s_off.add_voltage_probe("Vc", 2, 0)
        s_off.run()

        assert np.allclose(s_on.get_voltage_probe("Vc", "V"),
                           s_off.get_voltage_probe("Vc", "V"), atol=1e-12)
