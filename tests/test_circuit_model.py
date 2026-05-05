"""Verify CircuitModel container semantics."""

import pytest

from emtp.circuit import CircuitModel
from emtp.nodes import NodeIndexer


class TestCircuitModel:
    @pytest.fixture
    def model(self):
        return CircuitModel()

    def test_starts_empty(self, model):
        assert model.devices == []
        assert model.branches == {}
        assert model.multiport_devices == []
        assert model.current_sources == {}
        assert model.voltage_sources == {}
        assert model.transmission_lines == {}
        assert model.transformers == {}
        assert model.is_empty()

    def test_update_nodes(self, model):
        model.update_nodes(1, 2, 5)
        assert model.num_nodes == 5
        assert model.indexer.n == 3  # 1, 2, 5 registered

    def test_update_nodes_list(self, model):
        model.update_nodes([3, 7, 1])
        assert model.num_nodes == 7
        assert model.indexer.n == 3

    def test_is_not_empty_after_adding_branch(self, model):
        model.branches["r1"] = None  # simplified
        assert not model.is_empty()

    def test_indexer_is_shared(self, model):
        assert isinstance(model.indexer, NodeIndexer)
        model.update_nodes(10)
        assert model.indexer.to_compact(10) == 0
