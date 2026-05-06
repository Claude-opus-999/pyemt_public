"""PR0: Public API contract — every method and attribute external code relies on.

These tests enforce that refactoring PRs (PR2-PR7) do not break the
documented public API surface of EMTPSolver.  If a test here fails,
the refactoring PR must either restore the API or explicitly update
this contract with a clear migration note.
"""

import pytest
from emtp import EMTPSolver


# =========================================================================
# Method existence
# =========================================================================

PUBLIC_METHODS = [
    # Node management
    "node", "node_name", "bind_node", "alias_node",

    # Elements
    "add_R", "add_resistor",
    "add_L", "add_inductor",
    "add_C", "add_capacitor",
    "add_series_RL",
    "add_SW", "add_switch",
    "add_IS", "add_current_source",
    "add_lightning_IS", "add_lightning_current_source",
    "add_standard_twoexpf_IS",
    "add_standard_double_exponential_current_source",
    "add_VS", "add_voltage_source",

    # Lines
    "add_bergeron_line",
    "add_ulm_line",
    "add_ULM_line",
    "add_line",
    "compile_transmission_lines",
    "get_transmission_line",

    # Transformers
    "add_UMEC_transformer",
    "get_transformer_info",

    # Nonlinear
    "add_MOA_from_file",
    "add_insulator_LPM", "add_lpm_flashover_insulator",
    "get_insulator_leader_length",
    "get_insulator_leader_velocity",
    "get_insulator_voltage",
    "get_insulator_state",
    "get_insulator_info",
    "get_flashover_log",

    # Probes
    "add_voltage_probe",
    "add_branch_current_probe",
    "get_voltage_probe",
    "get_branch_current_probe",
    "get_probe",
    "list_probes",

    # Results
    "get_time",
    "get_node_voltage",
    "get_branch_current",
    "get_branch_voltage",
    "get_source_current",
    "get_vs_current",
    "get_vs_voltage",
    "get_line_current_k", "get_line_current_m",
    "get_line_voltage_k", "get_line_voltage_m",
    "get_line_info",

    # Simulation
    "run", "run_until",
    "reset_dynamic_state",
    "validate_circuit", "validate_probes",

    # Stats
    "get_solver_statistics", "print_solver_statistics",
    "estimate_result_memory_bytes",
    "print_timing_report", "get_timing_report",
    "print_circuit_summary",

    # Snapshot
    "save_snapshot", "load_snapshot",

    # Internal
    "mark_topology_changed",
]


class TestPublicMethodExistence:
    @pytest.mark.parametrize("method_name", PUBLIC_METHODS)
    def test_method_exists(self, method_name):
        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        assert hasattr(solver, method_name), f"Missing: EMTPSolver.{method_name}"
        assert callable(getattr(solver, method_name)), f"Not callable: {method_name}"


# =========================================================================
# Key attributes
# =========================================================================

PUBLIC_ATTRS = [
    "dt", "finish_time", "verbose",
    "time", "step_count",
    "branches", "voltage_probes",
    "transmission_lines", "nodes",
    "seg_helper",
    "_indexer", "_runtime", "_resolve_mgr", "_stepper",
    "_stats",
    # Probe-related internals (accessed by case_runner)
    "_voltage_probe_names", "_branch_current_probe_names",
    "_voltage_probe_data", "_branch_current_probe_data",
    "_voltage_probe_index", "_branch_current_probe_index",
    "_lines_compiled", "num_nodes",
    "current_sources", "voltage_sources",
    "_has_nonlinear",
    "_lpm_elements",
    "_results_valid", "_actual_steps",
    # Config flags
    "pre_sample_sources", "use_rhs_plan",
    "use_multiport_lines", "use_multiport_transformers",
    "record_all_node_voltages",
    "record_branch_history", "record_source_history",
    "record_line_history",
    "ulm_batch_mode", "allow_singular_regularization",
]


class TestPublicAttributeExistence:
    def test_basic_attrs_exist_after_init(self):
        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        for attr in PUBLIC_ATTRS:
            assert hasattr(solver, attr), f"Missing attr: EMTPSolver.{attr}"

    def test_attrs_exist_after_run(self):
        solver = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False)
        solver.add_VS("Vs", 1, 0, 1.0)
        solver.add_R("Rload", 1, 0, 100.0)
        solver.add_voltage_probe("V1", 1, 0)
        solver.run()
        for attr in PUBLIC_ATTRS:
            assert hasattr(solver, attr), f"Missing attr after run: {attr}"


# =========================================================================
# Key call patterns (signatures must not change)
# =========================================================================

class TestPublicCallPatterns:
    def test_add_R_single_node_int(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_R("r", 1, 0, 100.0)
        assert "r" in s.branches

    def test_add_L(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_L("l", 1, 2, 1e-3)

    def test_add_C(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_C("c", 1, 0, 1e-6)

    def test_add_series_RL(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_series_RL("rl", 1, 2, 10.0, 1e-3)

    def test_add_VS_constant(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 5.0)
        assert "VS1" not in s.voltage_sources
        assert "vs" in s.voltage_sources

    def test_add_IS(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_IS("isrc", 1, 0, lambda t: 1.0)

    def test_add_voltage_probe_positional(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 1, 0, 100.0)
        s.add_voltage_probe("V1", 1, 0)
        assert "V1" in s.voltage_probes

    def test_minimal_rc_run_produces_output(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 1, 2, 10.0)
        s.add_C("c", 2, 0, 1e-6)
        s.add_voltage_probe("Vc", 2, 0)
        s.run()
        v = s.get_voltage_probe("Vc", "V")
        assert len(v) > 0
