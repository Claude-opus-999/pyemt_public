"""PR2: Registry consistency tests — shadow state must mirror solver containers."""

import pytest
from emtp import EMTPSolver


class TestRegistryMirrorsSolver:
    def test_registry_has_element_after_add_R(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_R("r1", 1, 0, 100.0)
        assert "r1" in s.registry.elements, "R element not in registry"
        assert s.registry.elements["r1"].kind == "resistor"
        assert s.registry.elements["r1"].nodes == (1, 0)

    def test_registry_has_element_after_add_C(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_C("c1", 1, 0, 1e-6)
        assert "c1" in s.registry.elements
        assert s.registry.elements["c1"].kind == "capacitor"

    def test_registry_has_source_after_add_VS(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_VS("vs", 1, 0, 5.0)
        assert "vs" in s.registry.sources
        assert s.registry.sources["vs"].kind == "voltage"

    def test_registry_has_source_after_add_IS(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_IS("isrc", 1, 0, lambda t: 1.0)
        assert "isrc" in s.registry.sources
        assert s.registry.sources["isrc"].kind == "current"

    def test_registry_has_element_after_add_switch(self):
        s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 2, 0, 100.0)
        s.add_SW("sw1", 1, 2, t_close=10e-6, t_open=80e-6)
        assert "sw1" in s.registry.elements
        assert s.registry.elements["sw1"].kind == "switch"

    def test_registry_topology_version_bumps_on_element(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        v0 = s.registry.topology_version
        s.add_R("r1", 1, 0, 100.0)
        assert s.registry.topology_version > v0

    def test_registry_topology_version_bumps_on_vs(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        v0 = s.registry.topology_version
        s.add_VS("vs", 1, 0, 1.0)
        assert s.registry.topology_version > v0

    def test_registry_numeric_version_bumps_on_is(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        v0 = s.registry.numeric_version
        s.add_IS("isrc", 1, 0, lambda t: 1.0)
        # Current source bumps numeric (not topology)
        assert s.registry.numeric_version > v0

    def test_duplicate_name_raises(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_R("r1", 1, 0, 100.0)
        with pytest.raises(ValueError, match="Duplicate"):
            s.registry.register_element(
                type("Record", (), {"name": "r1", "kind": "resistor",
                 "nodes": (2, 0), "device": None, "metadata": {}})()
            )

    def test_registry_elements_are_readonly_copy(self):
        """The public elements property returns a copy, not a mutable ref."""
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s.add_R("r1", 1, 0, 100.0)
        elems = s.registry.elements
        elems.pop("r1", None)
        assert "r1" in s.registry.elements, "elements property must return a copy"
