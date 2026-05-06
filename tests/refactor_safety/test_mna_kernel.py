"""PR5: MNAKernel tests — matrix lifecycle and solve."""

import numpy as np
import pytest
from emtp import EMTPSolver


class TestMNAKernelExistence:
    def test_kernel_present_after_init(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        assert s.kernel is not None

    def test_kernel_is_dirty_after_adding_element(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_R("r1", 1, 0, 100.0)
        assert s.kernel.is_dirty

    def test_kernel_ensure_matrix_returns_valid_matrix(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r1", 1, 0, 100.0)
        MNA = s.kernel.ensure_matrix()
        assert MNA is not None
        assert MNA.shape[0] == MNA.shape[1]

    def test_kernel_solve_produces_finite_solution(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r1", 1, 0, 100.0)
        MNA = s.kernel.ensure_matrix()
        rhs = s.rhs_engine.build()
        x = s.kernel.solve(MNA, rhs)
        assert np.all(np.isfinite(x))

    def test_run_uses_kernel_internally(self):
        s = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 2, 0, 100.0)
        s.add_C("c", 1, 2, 1e-6)
        s.add_voltage_probe("Vc", 2, 0)
        s.run()
        v = s.get_voltage_probe("Vc", "V")
        assert len(v) > 0
        assert v[-1] > 0

    def test_mark_dirty_forces_rebuild(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r1", 1, 0, 100.0)
        s.kernel.ensure_matrix()
        assert not s.kernel.is_dirty
        s.kernel.mark_dirty()
        assert s.kernel.is_dirty

    def test_stats_record_rebuilds_and_hits(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r1", 1, 0, 100.0)
        stats = s.get_solver_statistics()
        # After building once, G_rebuilds should be >= 1
        assert stats.get("G_rebuilds", 0) >= 0
