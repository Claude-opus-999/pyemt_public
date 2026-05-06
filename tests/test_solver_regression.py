import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emtp import EMTPSolver, NodeIndexer
from emtp.sparse_solver import SparseLinearSolver
from emtp.types import ValidationReport, ValidationIssue


class SolverRegressionTests(unittest.TestCase):
    def test_run_uses_integer_steps_and_includes_finish_time(self):
        solver = EMTPSolver(dt=1e-4, finish_time=1e-3, verbose=False)

        solver.run()

        self.assertEqual(len(solver.time_array), 11)
        self.assertLess(abs(solver.time_array[-1] - 1e-3), 1e-15)

    def test_resistor_branch_current_probe_does_not_need_full_history(self):
        solver = EMTPSolver(dt=1e-6, finish_time=2e-6, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_branch_current_probe("I_R1", "R1")

        solver.run()

        np.testing.assert_allclose(solver.get_probe("I_R1", unit="A"), 1.0)
        with self.assertRaises(RuntimeError):
            solver.get_branch_current("R1")

    def test_capacitor_probe_includes_parallel_damping_current(self):
        solver = EMTPSolver(dt=1.0, finish_time=0.0, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_capacitor("C1", 1, 0, 1.0, Rp=1.0)
        solver.add_branch_current_probe("I_C1", "C1")

        solver.run()

        np.testing.assert_allclose(solver.get_probe("I_C1", unit="A"), [3.0])

    def test_run_resets_dynamic_state(self):
        solver = EMTPSolver(dt=1.0, finish_time=2.0, verbose=False)
        solver.add_capacitor("C1", 1, 0, 1.0)
        solver.add_current_source("I1", 0, 1, 1.0)

        solver.run()
        first = solver.get_node_voltage(1)
        solver.run()
        second = solver.get_node_voltage(1)

        np.testing.assert_allclose(first, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(second, first)

    def test_timed_switch_events_are_consumed_once(self):
        solver = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_switch(
            "S1", 1, 0,
            t_close=1e-6,
            t_open=2e-6,
            R_closed=1e-3,
            R_open=1e9,
        )

        solver.run()

        self.assertFalse(solver.branches["S1"].is_closed)
        self.assertEqual(solver.get_solver_statistics().get("G_rebuilds"), 3)

    def test_voltage_probe_does_not_create_unknown_named_node(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)

        with self.assertRaises(ValueError):
            solver.add_voltage_probe("bad", "typo_node")

        self.assertEqual(solver.num_nodes, 0)

    def test_record_all_node_voltages_can_be_disabled(self):
        solver = EMTPSolver(
            dt=1e-6,
            finish_time=0.0,
            verbose=False,
            record_all_node_voltages=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_voltage_probe("V_1", 1, 0)

        solver.run()

        np.testing.assert_allclose(solver.get_probe("V_1"), [1.0])
        with self.assertRaises(RuntimeError):
            solver.get_node_voltage(1)

    def test_validate_circuit_reports_floating_current_source_node(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_current_source("I1", 0, 1, 1.0)

        with self.assertRaises(RuntimeError):
            solver.run()

    def test_duplicate_device_names_are_rejected(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_resistor("X1", 1, 0, 1.0)

        with self.assertRaises(ValueError):
            solver.add_current_source("X1", 0, 1, 1.0)

    def test_duplicate_probe_names_are_rejected(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_voltage_probe("P1", 1, 0)

        with self.assertRaises(ValueError):
            solver.add_branch_current_probe("P1", "R1")

    def test_result_api_requires_run(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)

        with self.assertRaises(RuntimeError):
            solver.get_time()

    def test_voltage_source_self_loop_is_rejected(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)

        with self.assertRaises(ValueError):
            solver.add_voltage_source("Vbad", 1, 1, 1.0)

    # ------------------------------------------------------------------
    # v3.1 新增: ValidationReport / validate_circuit 增强
    # ------------------------------------------------------------------

    def test_validate_circuit_strict_mode_raises_on_errors(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_current_source("I1", 0, 1, 1.0)

        with self.assertRaises(RuntimeError):
            solver.validate_circuit(strict=True)

    def test_validate_circuit_non_strict_returns_report_with_errors(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_current_source("I1", 0, 1, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.has_errors)
        self.assertGreater(len(report.errors()), 0)
        self.assertIn("E013", [i.code for i in report.errors()])

    def test_validate_circuit_returns_report_on_clean_circuit(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertFalse(report.has_errors)

    def test_large_integer_node_warning(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_resistor("R1", 100000, 0, 1.0)
        solver.add_voltage_source("V1", 100000, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertTrue(report.has_warnings)
        warning_codes = [i.code for i in report.warnings()]
        self.assertIn("W001", warning_codes)

    def test_record_all_node_voltages_memory_info(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-3,
            record_all_node_voltages=True, verbose=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        info_codes = [i.code for i in report.issues if i.severity == "info"]
        self.assertIn("I001", info_codes)

    def test_max_result_memory_mb_warning(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-3,
            record_all_node_voltages=True,
            max_result_memory_mb=0.001,  # 1 KB, very low
            verbose=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertTrue(report.has_warnings)
        self.assertIn("W002", [i.code for i in report.warnings()])

    def test_max_result_memory_mb_none_no_warning(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-3,
            record_all_node_voltages=True,
            max_result_memory_mb=None,
            verbose=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertNotIn("W002", [i.code for i in report.warnings()])

    def test_estimate_result_memory_bytes_positive(self):
        solver = EMTPSolver(dt=1e-6, finish_time=1e-3, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        mem = solver.estimate_result_memory_bytes()
        self.assertGreater(mem, 0)

    def test_estimate_result_memory_no_node_voltage_history(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-3,
            record_all_node_voltages=False, verbose=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        mem_full = EMTPSolver(
            dt=1e-6, finish_time=1e-3,
            record_all_node_voltages=True, verbose=False,
        )
        mem_full.add_voltage_source("V1", 1, 0, 1.0)
        mem_full.add_resistor("R1", 1, 0, 1.0)

        self.assertLess(
            solver.estimate_result_memory_bytes(),
            mem_full.estimate_result_memory_bytes(),
        )

    def test_validation_issue_structure(self):
        issue = ValidationIssue(
            severity="error", code="E001",
            message="test message",
            related_nodes=[1, 2],
            related_branches=["B1"],
        )
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.code, "E001")
        self.assertEqual(issue.related_nodes, [1, 2])

    def test_validation_report_bool(self):
        report = ValidationReport(issues=[
            ValidationIssue(severity="error", code="E001", message="bad"),
        ])
        self.assertFalse(bool(report))
        self.assertTrue(report.has_errors)

        clean = ValidationReport(issues=[
            ValidationIssue(severity="warning", code="W001", message="hint"),
        ])
        self.assertTrue(bool(clean))
        self.assertFalse(clean.has_errors)
        self.assertTrue(clean.has_warnings)

    def test_switch_event_ceil_alignment(self):
        """开关事件对齐到 ceil(t_event / dt) * dt 的规则。"""
        solver = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("Rload", 1, 0, 1e3)
        solver.add_switch(
            "S1", 1, 0,
            t_close=1.5e-6,  # 不在时间网格上
            t_open=3.7e-6,   # 不在时间网格上
            R_closed=1e-3, R_open=1e9,
        )

        solver.run()

        stats = solver.get_solver_statistics()
        # close at ceil(1.5e-6/1e-6)*1e-6 = 2e-6 → step 2
        # open  at ceil(3.7e-6/1e-6)*1e-6 = 4e-6 → step 4
        # Each event triggers a G rebuild → 2 rebuilds (initial + 2 = 3)
        self.assertEqual(stats.get("G_rebuilds"), 3)

    # ------------------------------------------------------------------
    # Phase 1 new tests: PR-2 validation strict, PR-3 vs loop, PR-1 ULM
    # ------------------------------------------------------------------

    def test_validate_dt_zero_strict_raises(self):
        solver = EMTPSolver(dt=0, finish_time=1e-3, verbose=False)
        with self.assertRaises(RuntimeError):
            solver.validate_circuit(strict=True)

    def test_validate_dt_zero_strict_false_returns_error(self):
        solver = EMTPSolver(dt=0, finish_time=1e-3, verbose=False)
        report = solver.validate_circuit(strict=False)
        self.assertTrue(report.has_errors)
        self.assertIn("E001", [i.code for i in report.errors()])

    def test_validate_dt_invalid_does_not_call_memory_estimation(self):
        """dt <= 0 should not proceed to estimate_result_memory_bytes()."""
        solver = EMTPSolver(dt=0, finish_time=1e-3, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        report = solver.validate_circuit(strict=False)
        # Only E001 (dt <= 0) should be present; no memory/estimate issues
        error_codes = [i.code for i in report.issues]
        self.assertIn("E001", error_codes)
        # dt <= 0 should block further checks
        self.assertNotIn("I001", error_codes)

    def test_validate_finish_time_negative_early_return(self):
        solver = EMTPSolver(dt=1e-6, finish_time=-1.0, verbose=False)
        report = solver.validate_circuit(strict=False)
        self.assertTrue(report.has_errors)
        self.assertIn("E002", [i.code for i in report.errors()])

    def test_validate_empty_circuit_strict_applied(self):
        """Empty circuit with no errors should not raise even in strict mode."""
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        report = solver.validate_circuit(strict=True)
        self.assertFalse(report.has_errors)

    def test_voltage_source_loop_via_vs_node_set_prevention(self):
        """_vs_node_set prevents node reuse across VS terminals.

        V1(1,2) + V2(2,3) is allowed (series chain through shared node),
        but V3(3,1) is blocked because node 1 is already a pos terminal
        and cannot also be a neg terminal.  The _vs_node_set check is the
        primary loop-prevention mechanism; union-find is a secondary net.
        """
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_VS("V1", 1, 2, 1.0)
        solver.add_VS("V2", 2, 3, 1.0)
        # node 1 is already V1.pos, cannot be V3.neg
        with self.assertRaises(ValueError):
            solver.add_VS("V3", 3, 1, 1.0)

    def test_voltage_source_union_find_else_branch_prevents_false_loop(self):
        """Two independent VS branches with resistors: no loop detected."""
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_VS("V1", 1, 0, 1.0)
        solver.add_VS("V2", 2, 0, 2.0)
        solver.add_resistor("R1", 1, 2, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertFalse(report.has_errors)

    def test_voltage_sources_in_series_no_loop(self):
        """V1(1,2) + V2(2,3) in series is legal, no loop."""
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_VS("V1", 1, 2, 1.0)
        solver.add_VS("V2", 2, 3, 1.0)
        solver.add_resistor("R1", 3, 0, 1.0)
        solver.add_resistor("R2", 1, 0, 1.0)

        report = solver.validate_circuit(strict=False)
        self.assertFalse(report.has_errors)

    # ------------------------------------------------------------------
    # Phase 2 tests: pre-sampling, scalar fast path, voltage history error
    # ------------------------------------------------------------------

    def test_pre_sample_sources_matches_runtime_sampling(self):
        solver_scalar = EMTPSolver(
            dt=1e-6, finish_time=2e-6, verbose=False,
            pre_sample_sources=False,
        )
        solver_scalar.add_voltage_source("V1", 1, 0, 1.0)
        solver_scalar.add_resistor("R1", 1, 0, 1.0)
        solver_scalar.run()

        solver_pre = EMTPSolver(
            dt=1e-6, finish_time=2e-6, verbose=False,
            pre_sample_sources=True,
        )
        solver_pre.add_voltage_source("V1", 1, 0, 1.0)
        solver_pre.add_resistor("R1", 1, 0, 1.0)
        solver_pre.run()

        np.testing.assert_allclose(
            solver_scalar.get_node_voltage(1),
            solver_pre.get_node_voltage(1),
        )

    def test_pre_sample_sources_current_source(self):
        def ramp(t): return float(t * 1e6)

        solver = EMTPSolver(
            dt=1e-6, finish_time=3e-6, verbose=False,
            pre_sample_sources=True,
        )
        solver.add_current_source("I1", 0, 1, ramp)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.run()

        # ramp(t) = t*1e6 A → at t=0,1,2,3 μs: 0,1,2,3 A → V=IR with R=1Ω
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            solver.get_node_voltage(1), expected, atol=1e-15,
        )

    def test_pre_sample_sources_rebuilt_on_second_run(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-6, verbose=False,
            pre_sample_sources=True,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)

        solver.run()
        first = solver.get_node_voltage(1)
        solver.run()
        second = solver.get_node_voltage(1)

        np.testing.assert_allclose(first, second)

    def test_pre_sample_disabled_uses_runtime_calls(self):
        call_log = []

        def traced_func(t):
            call_log.append(t)
            return 1.0

        solver = EMTPSolver(
            dt=1e-6, finish_time=2e-6, verbose=False,
            pre_sample_sources=False,
        )
        solver.add_voltage_source("V1", 1, 0, traced_func)
        solver.add_resistor("R1", 1, 0, 1.0)
        # creation calls voltage_at(0.0) for validation; clear the log
        call_log.clear()
        solver.run()
        # 3 time steps; each calls voltage_at once per RHS build
        self.assertEqual(len(call_log), 3)

    def test_get_node_voltage_raises_when_not_recorded(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=0.0, verbose=False,
            record_all_node_voltages=False,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.run()

        with self.assertRaises(RuntimeError) as ctx:
            solver.get_node_voltage(1)
        self.assertIn("record_all_node_voltages", str(ctx.exception))

    # ------------------------------------------------------------------
    # PR-6: RHSPlan tests
    # ------------------------------------------------------------------

    def test_rhs_plan_matches_legacy_rhs(self):
        solver_legacy = EMTPSolver(
            dt=1e-6, finish_time=3e-6, verbose=False,
            use_rhs_plan=False,
        )
        solver_legacy.add_voltage_source("V1", 1, 0, 1.0)
        solver_legacy.add_resistor("R1", 1, 0, 1.0)
        solver_legacy.add_capacitor("C1", 1, 0, 1e-6)
        solver_legacy.add_current_source("I1", 0, 2, 2.0)
        solver_legacy.add_resistor("R2", 2, 0, 1.0)
        solver_legacy.run()

        solver_plan = EMTPSolver(
            dt=1e-6, finish_time=3e-6, verbose=False,
            use_rhs_plan=True,
        )
        solver_plan.add_voltage_source("V1", 1, 0, 1.0)
        solver_plan.add_resistor("R1", 1, 0, 1.0)
        solver_plan.add_capacitor("C1", 1, 0, 1e-6)
        solver_plan.add_current_source("I1", 0, 2, 2.0)
        solver_plan.add_resistor("R2", 2, 0, 1.0)
        solver_plan.run()

        np.testing.assert_allclose(
            solver_legacy.get_node_voltage(1),
            solver_plan.get_node_voltage(1),
        )
        np.testing.assert_allclose(
            solver_legacy.get_node_voltage(2),
            solver_plan.get_node_voltage(2),
        )

    def test_rhs_plan_rebuilt_after_switch_event(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=5e-6, verbose=False,
            use_rhs_plan=True,
        )
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_switch(
            "S1", 1, 0,
            t_close=2e-6, t_open=4e-6,
            R_closed=1e-3, R_open=1e9,
        )
        solver.run()

        # Switch events should trigger G_rebuilds = 3 (initial + 2 events)
        self.assertEqual(
            solver.get_solver_statistics().get("G_rebuilds"), 3,
        )

    def test_rhs_plan_empty_circuit(self):
        solver = EMTPSolver(
            dt=1e-6, finish_time=1e-6, verbose=False,
            use_rhs_plan=True,
        )
        solver.run()
        self.assertEqual(len(solver.time_array), 2)


class NodeIndexerTests(unittest.TestCase):
    def test_register_sparse_ids_gives_compact_range(self):
        idx = NodeIndexer()
        self.assertEqual(idx.register(1), 0)
        self.assertEqual(idx.register(5), 1)
        self.assertEqual(idx.register(9999), 2)
        self.assertEqual(idx.n, 3)

    def test_to_compact_returns_correct_mapping(self):
        idx = NodeIndexer()
        idx.register(1)
        idx.register(5)
        idx.register(9999)
        self.assertEqual(idx.to_compact(1), 0)
        self.assertEqual(idx.to_compact(5), 1)
        self.assertEqual(idx.to_compact(9999), 2)

    def test_to_external_reverses_mapping(self):
        idx = NodeIndexer()
        idx.register(10)
        idx.register(20)
        self.assertEqual(idx.to_external(0), 10)
        self.assertEqual(idx.to_external(1), 20)

    def test_ground_is_always_compact_gnd(self):
        idx = NodeIndexer()
        self.assertEqual(idx.register(0), NodeIndexer.COMPACT_GND)
        self.assertEqual(idx.to_compact(0), NodeIndexer.COMPACT_GND)
        self.assertEqual(idx.n, 0)

    def test_repeat_register_is_idempotent(self):
        idx = NodeIndexer()
        self.assertEqual(idx.register(3), 0)
        self.assertEqual(idx.register(3), 0)
        self.assertEqual(idx.register(3), 0)
        self.assertEqual(idx.n, 1)

    def test_freeze_blocks_new_nodes(self):
        idx = NodeIndexer()
        idx.register(1)
        idx.freeze()
        idx.register(1)  # existing node, ok
        with self.assertRaises(RuntimeError):
            idx.register(2)

    def test_frozen_register_does_not_mutate(self):
        idx = NodeIndexer()
        idx.register(1)
        idx.freeze()
        try:
            idx.register(2)
        except RuntimeError:
            pass
        self.assertEqual(idx.n, 1)
        self.assertNotIn(2, idx.externals)

    def test_externals_returns_ordered_copy(self):
        idx = NodeIndexer()
        idx.register(7)
        idx.register(3)
        self.assertEqual(idx.externals, [7, 3])
        idx.externals.append(99)  # must not mutate internals
        self.assertEqual(idx.n, 2)

    def test_to_compact_unknown_raises_keyerror(self):
        idx = NodeIndexer()
        idx.register(1)
        with self.assertRaises(KeyError):
            idx.to_compact(99)

    def test_integration_with_solver_after_add_devices(self):
        solver = EMTPSolver(dt=1e-6, finish_time=0.0, verbose=False)
        solver.add_resistor("R1", 1, 0, 100.0)
        solver.add_resistor("R2", 5, 1, 200.0)
        solver.add_resistor("R3", 9999, 0, 300.0)
        self.assertEqual(solver._indexer.n, 3)
        self.assertEqual(solver._indexer.to_compact(1), 0)
        self.assertEqual(solver._indexer.to_compact(5), 1)
        self.assertEqual(solver._indexer.to_compact(9999), 2)


class SparseLinearSolverTests(unittest.TestCase):
    def test_solve_positive_definite(self):
        import scipy.sparse as sp
        A = sp.csc_matrix(np.array([[4., 1.], [1., 3.]]))
        b = np.array([1., 2.])
        solver = SparseLinearSolver()
        x = solver.solve(A, b, matrix_id=0, n_compact=2)
        np.testing.assert_allclose(x, np.linalg.solve(A.toarray(), b))

    def test_singular_without_regularization_raises(self):
        import scipy.sparse as sp
        A = sp.csc_matrix(np.array([[1., 2.], [2., 4.]]))
        b = np.array([1., 1.])
        solver = SparseLinearSolver(allow_singular_regularization=False)
        with self.assertRaises(RuntimeError):
            solver.solve(A, b, matrix_id=0, n_compact=2)

    def test_same_matrix_id_reuses_factorization(self):
        import scipy.sparse as sp
        A = sp.csc_matrix(np.array([[4., 1.], [1., 3.]]))
        b1 = np.array([1., 2.])
        b2 = np.array([3., 4.])
        solver = SparseLinearSolver()
        # first call factors
        x1 = solver.solve(A, b1, matrix_id=0, n_compact=2)
        # second call with same id should reuse
        x2 = solver.solve(A, b2, matrix_id=0, n_compact=2)
        np.testing.assert_allclose(x1, np.linalg.solve(A.toarray(), b1))
        np.testing.assert_allclose(x2, np.linalg.solve(A.toarray(), b2))

    def test_different_matrix_id_refactors(self):
        import scipy.sparse as sp
        A1 = sp.csc_matrix(np.array([[4., 1.], [1., 3.]]))
        A2 = sp.csc_matrix(np.array([[5., 0.], [0., 2.]]))
        b = np.array([1., 1.])
        solver = SparseLinearSolver()
        x1 = solver.solve(A1, b, matrix_id=0, n_compact=2)
        x2 = solver.solve(A2, b, matrix_id=1, n_compact=2)
        np.testing.assert_allclose(x1, np.linalg.solve(A1.toarray(), b))
        np.testing.assert_allclose(x2, np.linalg.solve(A2.toarray(), b))

    def test_singular_with_regularization_returns_result(self):
        import scipy.sparse as sp
        A = sp.csc_matrix(np.array([[1., 2.], [2., 4.]]))
        b = np.array([1., 1.])
        solver = SparseLinearSolver(allow_singular_regularization=True)
        x = solver.solve(A, b, matrix_id=0, n_compact=2)
        self.assertEqual(x.shape, (2,))

    def test_invalidate_forces_refactor(self):
        import scipy.sparse as sp
        A = sp.csc_matrix(np.array([[4., 1.], [1., 3.]]))
        b = np.array([1., 2.])
        solver = SparseLinearSolver()
        x1 = solver.solve(A, b, matrix_id=0, n_compact=2)
        solver.invalidate()
        x2 = solver.solve(A, b, matrix_id=0, n_compact=2)
        np.testing.assert_allclose(x1, x2)  # same result, but refactored


if __name__ == "__main__":
    unittest.main()
