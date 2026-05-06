"""PR3: ProbeManager tests — registration, indexing, sampling."""

import numpy as np
import pytest
from emtp.probes import ProbeManager, ProbeSpec


class TestProbeManagerRegistration:
    def test_add_voltage_probe(self):
        pm = ProbeManager()
        spec = pm.add_voltage_probe("V1", 1, 0)
        assert spec.name == "V1"
        assert spec.kind == "voltage_diff"
        assert spec.target == (1, 0)
        assert pm.has("V1")
        assert "V1" in pm.voltage_probe_names

    def test_add_branch_current_probe(self):
        pm = ProbeManager()
        spec = pm.add_branch_current_probe("IL", "L1")
        assert spec.name == "IL"
        assert spec.kind == "branch_current"
        assert spec.target == ("L1",)
        assert "IL" in pm.branch_current_probe_names

    def test_duplicate_name_raises(self):
        pm = ProbeManager()
        pm.add_voltage_probe("V1", 1, 0)
        with pytest.raises(ValueError, match="Duplicate"):
            pm.add_voltage_probe("V1", 2, 0)

    def test_mixed_probes_listed_correctly(self):
        pm = ProbeManager()
        pm.add_voltage_probe("Vcap", 2, 0)
        pm.add_branch_current_probe("IL", "L1")
        pm.add_voltage_probe("Vsrc", 1, 0)

        assert pm.voltage_probe_names == ["Vcap", "Vsrc"]
        assert pm.branch_current_probe_names == ["IL"]
        assert set(pm.names) == {"Vcap", "IL", "Vsrc"}

    def test_list_by_kind(self):
        pm = ProbeManager()
        pm.add_voltage_probe("V1", 1, 0)
        pm.add_branch_current_probe("I1", "R1")
        result = pm.list_by_kind()
        assert result["voltage"] == ["V1"]
        assert result["branch_current"] == ["I1"]

    def test_probe_index(self):
        pm = ProbeManager()
        pm.add_voltage_probe("Va", 1, 0)
        pm.add_voltage_probe("Vb", 2, 0)
        idx = pm.probe_index()
        assert idx["voltage"]["Va"] == 0
        assert idx["voltage"]["Vb"] == 1

    def test_get_spec_raises_for_missing(self):
        pm = ProbeManager()
        with pytest.raises(KeyError):
            pm.get_spec("nonexistent")

    def test_has_returns_false_for_missing(self):
        pm = ProbeManager()
        assert not pm.has("nonexistent")


class TestProbeSpecImmutability:
    def test_same_params_equal(self):
        a = ProbeSpec("V1", "voltage_diff", (1, 0))
        b = ProbeSpec("V1", "voltage_diff", (1, 0))
        assert a == b
        assert hash(a) == hash(b)

    def test_different_name_not_equal(self):
        a = ProbeSpec("V1", "voltage_diff", (1, 0))
        b = ProbeSpec("V2", "voltage_diff", (1, 0))
        assert a != b


class TestSolverProbeIntegration:
    def test_solver_probe_manager_has_probes(self):
        from emtp import EMTPSolver
        s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        s.add_VS("Vs", 1, 0, 1.0)
        s.add_R("Rload", 1, 0, 100.0)
        s.add_voltage_probe("V1", 1, 0)
        s.add_branch_current_probe("IR", "Rload")
        assert s.probe_manager.has("V1")
        assert s.probe_manager.has("IR")

    def test_solver_probe_backward_compat(self):
        """Legacy voltage_probes dict still works after PR3 migration."""
        from emtp import EMTPSolver
        s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        s.add_VS("Vs", 1, 0, 1.0)
        s.add_R("Rload", 1, 0, 100.0)
        s.add_voltage_probe("V1", 1, 0)
        assert "V1" in s.voltage_probes
        assert s.voltage_probes["V1"]["node_pos"] == 1
