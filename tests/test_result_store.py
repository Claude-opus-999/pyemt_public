"""Verify ResultStore allocates, records and finalizes correctly."""

import numpy as np
import pytest

from emtp.results import ResultStore
from emtp.nodes import NodeIndexer


class TestResultStore:
    def test_time_array_is_pre_allocated(self):
        store = ResultStore(n_nodes=2, n_steps=10, record_node_voltage=False)
        assert len(store.time) == 10
        assert store.voltage is None

    def test_voltage_matrix_when_enabled(self):
        store = ResultStore(n_nodes=3, n_steps=5, record_node_voltage=True)
        assert store.voltage.shape == (3, 5)

    def test_vs_current_buffers(self):
        store = ResultStore(n_nodes=1, n_steps=4, vs_names=["vs1", "vs2"])
        assert list(store.vs_current) == ["vs1", "vs2"]
        assert store.vs_current["vs1"].shape == (4,)

    def test_branch_history_disabled_by_default(self):
        store = ResultStore(n_nodes=1, n_steps=5)
        assert store.branch_v == {}
        assert store.branch_i == {}

    def test_branch_history_when_enabled(self):
        store = ResultStore(
            n_nodes=1, n_steps=5,
            record_branch_history=True, branch_names=["r1", "l1"],
        )
        assert store.branch_v["r1"].shape == (5,)
        assert store.branch_i["l1"].shape == (5,)

    def test_probe_storage(self):
        store = ResultStore(
            n_nodes=1, n_steps=5,
            voltage_probe_names=["vp1", "vp2"],
            branch_current_probe_names=["bp1"],
        )
        assert store.voltage_probe_data.shape == (5, 2)
        assert store.branch_current_probe_data.shape == (5, 1)

    def test_record_step_writes_time_and_voltage(self):
        store = ResultStore(n_nodes=2, n_steps=5, record_node_voltage=True)
        V = np.array([1.5, 2.5])
        store.record_step(0, 1e-6, V)
        assert np.isclose(store.time[0], 1e-6)
        assert np.allclose(store.voltage[:, 0], [1.5, 2.5])

    def test_record_step_writes_probes(self):
        store = ResultStore(
            n_nodes=1, n_steps=3, record_node_voltage=False,
            voltage_probe_names=["vp1"],
            branch_current_probe_names=["bp1"],
        )
        store.record_step(
            0, 0.0, np.array([]),
            voltage_probe_values=[3.3],
            branch_current_probe_values=[0.1],
        )
        assert np.isclose(store.voltage_probe_data[0, 0], 3.3)
        assert np.isclose(store.branch_current_probe_data[0, 0], 0.1)

    def test_record_vs_current(self):
        store = ResultStore(n_nodes=1, n_steps=3, vs_names=["vs1"])
        store.record_vs_current(0, "vs1", 0.5)
        store.record_vs_current(2, "vs1", -0.5)
        assert np.isclose(store.vs_current["vs1"][0], 0.5)
        assert np.isclose(store.vs_current["vs1"][2], -0.5)

    def test_record_branch_history(self):
        store = ResultStore(
            n_nodes=1, n_steps=3,
            record_branch_history=True, branch_names=["r1"],
        )
        store.record_branch_history(0, "r1", 1.0, 0.1)
        assert np.isclose(store.branch_v["r1"][0], 1.0)
        assert np.isclose(store.branch_i["r1"][0], 0.1)

    def test_finalize_trims_to_actual_steps(self):
        store = ResultStore(n_nodes=2, n_steps=10, record_node_voltage=True)
        for i in range(5):
            store.record_step(i, i * 0.1, np.array([i, i + 1]))
        indexer = NodeIndexer()
        indexer.register(10)
        indexer.register(20)
        indexer.freeze()
        store.finalize(indexer)
        assert len(store.time) == 5
        assert store.voltage.shape == (2, 5)
        assert 10 in store.voltage_results
        assert store.voltage_results[10].shape == (5,)
