"""Test the Case/Config layer — load, validate, build, run."""

import json
from pathlib import Path

import numpy as np
import pytest

from emtp.config.loader import load_case_config
from emtp.config.schema import CaseConfig, SimulationOptions
from emtp.config.validator import validate_case_config
from emtp.config.defaults import SUPPORTED_ELEMENTS, SUPPORTED_SOURCES
from emtp.builders.solver_builder import build_solver_from_config
from emtp.builders.element_builder import add_element_to_solver
from emtp.builders.source_builder import add_source_to_solver
from emtp.builders.probe_builder import add_probe_to_solver
from emtp.case_runner import run_case
from emtp.result_bundle import ResultBundle


CASES_DIR = Path(__file__).resolve().parent.parent / "cases" / "templates"


# ---------------------------------------------------------------------------
# Schema & validation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_simulation_options_defaults(self):
        sim = SimulationOptions(dt=1e-6, finish_time=100e-6)
        assert sim.output_stride == 1
        assert sim.pre_sample_sources is True

    def test_case_config_defaults(self):
        sim = SimulationOptions(dt=1e-6, finish_time=100e-6)
        config = CaseConfig(schema_version="0.1.0", case_name="test", simulation=sim)
        assert config.elements == []
        assert config.sources == []

    def test_circular_config_roundtrip(self):
        sim = SimulationOptions(dt=1e-9, finish_time=1e-5)
        config = CaseConfig(schema_version="0.1.0", case_name="roundtrip", simulation=sim)
        # Should not raise
        validate_case_config(config)


class TestValidator:
    def test_rejects_zero_dt(self):
        sim = SimulationOptions(dt=0.0, finish_time=100e-6)
        config = CaseConfig(schema_version="0.1", case_name="x", simulation=sim)
        with pytest.raises(ValueError, match="dt"):
            validate_case_config(config)

    def test_rejects_unknown_element_kind(self):
        sim = SimulationOptions(dt=1e-6, finish_time=100e-6)
        config = CaseConfig(
            schema_version="0.1", case_name="x", simulation=sim,
            elements=[{"kind": "unicorn", "name": "u"}],
        )
        with pytest.raises(ValueError, match="unicorn"):
            validate_case_config(config)

    def test_rejects_duplicate_element_names(self):
        sim = SimulationOptions(dt=1e-6, finish_time=100e-6)
        config = CaseConfig(
            schema_version="0.1", case_name="x", simulation=sim,
            elements=[
                {"kind": "resistor", "name": "R1", "node_from": 1, "node_to": 0, "R": 10},
                {"kind": "resistor", "name": "R1", "node_from": 2, "node_to": 0, "R": 20},
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            validate_case_config(config)

    def test_rejects_zero_stride(self):
        sim = SimulationOptions(dt=1e-6, finish_time=100e-6, output_stride=0)
        config = CaseConfig(schema_version="0.1", case_name="x", simulation=sim)
        with pytest.raises(ValueError, match="output_stride"):
            validate_case_config(config)

    def test_all_supported_kinds_recognized(self):
        for kind in SUPPORTED_ELEMENTS:
            assert isinstance(kind, str)
        for kind in SUPPORTED_SOURCES:
            assert isinstance(kind, str)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_loads_rc_config(self):
        path = CASES_DIR / "rc_step.json"
        config = load_case_config(path)
        assert config.case_name == "rc_step"
        assert config.simulation.dt == 1e-6
        assert len(config.elements) == 2
        assert len(config.sources) == 1
        assert len(config.probes) == 2

    def test_loads_rl_config(self):
        config = load_case_config(CASES_DIR / "rl_step.json")
        assert config.case_name == "rl_step"

    def test_loads_bergeron_config(self):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        assert config.case_name == "bergeron_matched"
        assert any(e["kind"] == "bergeron_line" for e in config.elements)

    def test_defaults_applied(self, tmp_path):
        raw = {
            "schema_version": "0.1",
            "case_name": "minimal",
            "simulation": {"dt": 1e-6, "finish_time": 10e-6},
        }
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps(raw))
        config = load_case_config(p)
        assert config.simulation.pre_sample_sources is True
        assert config.simulation.record_branch_history is True


# ---------------------------------------------------------------------------
# Solver builder
# ---------------------------------------------------------------------------

class TestSolverBuilder:
    def test_builds_rc_solver(self):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        assert solver is not None
        assert "R1" in solver.branches
        assert "C1" in solver.branches
        assert len(solver.voltage_probes) == 1

    def test_builds_bergeron_solver(self):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        solver = build_solver_from_config(config)
        assert "BL1" in solver.transmission_lines
        assert len(solver.voltage_probes) == 2

    def test_elements_have_correct_types(self):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        from emtp.types import ElementType
        assert solver.branches["R1"].element_type == ElementType.RESISTOR
        assert solver.branches["C1"].element_type == ElementType.CAPACITOR


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------

class TestRunCase:
    def test_run_case_rc_success(self):
        result = run_case(CASES_DIR / "rc_step.json")
        assert result.success
        assert result.metrics["total_steps"] > 0
        assert "probe_V_cap_peak_V" in result.metrics
        assert "time_s" in result.waveforms
        assert "V_cap" in result.waveforms

    def test_run_case_rc_waveform_correct_shape(self):
        result = run_case(CASES_DIR / "rc_step.json")
        assert len(result.waveforms["time_s"]) == 101
        assert len(result.waveforms["V_cap"]) == 101

    def test_run_case_with_config_object(self):
        config = load_case_config(CASES_DIR / "rc_step.json")
        result = run_case(config)
        assert result.success

    def test_run_case_rl_success(self):
        result = run_case(CASES_DIR / "rl_step.json")
        assert result.success
        assert result.metrics["total_steps"] == 101

    def test_rc_step_approaches_analytic(self):
        """V_cap should approach 1V exponentially with tau = R*C = 10us."""
        result = run_case(CASES_DIR / "rc_step.json")
        t = np.array(result.waveforms["time_s"])
        v = np.array(result.waveforms["V_cap"])
        # At t = 5*tau = 50us, V should be ~0.993V
        idx_50us = np.searchsorted(t, 50e-6)
        assert v[idx_50us] > 0.98


# ---------------------------------------------------------------------------
# ResultBundle
# ---------------------------------------------------------------------------

class TestResultBundle:
    def test_success_bundle(self):
        bundle = ResultBundle(
            case_name="test", success=True,
            metrics={"a": 1}, waveforms={"x": [1, 2, 3]},
        )
        assert bundle.success
        assert bundle.error is None

    def test_failure_bundle(self):
        bundle = ResultBundle(
            case_name="test", success=False,
            metrics={}, waveforms={}, error="bad config",
        )
        assert not bundle.success
        assert bundle.error == "bad config"
