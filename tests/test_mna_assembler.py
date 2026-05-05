"""Verify MNAAssembler builds correct G and RHS for a basic circuit."""

import numpy as np
import pytest

from emtp.nodes import NodeIndexer
from emtp.stamping import StampingEngine
from emtp.types import VoltageSource
from emtp.devices import ResistorDevice, CapacitorDevice


class TestMNAAssembler:
    @pytest.fixture
    def indexer(self):
        idx = NodeIndexer()
        idx.register(1)
        idx.register(2)
        idx.freeze()
        return idx

    @pytest.fixture
    def assembler(self, indexer):
        eng = StampingEngine(indexer)
        return __import__(
            "emtp.assembly", fromlist=["MNAAssembler"],
        ).MNAAssembler(eng, indexer)

    def test_build_minimal_G(self, assembler, indexer):
        dev = ResistorDevice("r1", 1, 2, 100.0)
        stamper = assembler.begin_G(m_vs=0)
        assembler.stamp_devices_G(stamper, [dev])
        G = assembler.finish_G(stamper)
        assert G.shape == (2, 2)
        assert G.nnz == 4  # 2×2 fill from Norton stamp

    def test_build_G_with_vs(self, assembler, indexer):
        dev = ResistorDevice("r1", 1, 2, 100.0)
        vs = VoltageSource("vs", 1, 0, lambda t: 5.0)
        stamper = assembler.begin_G(m_vs=1)
        assembler.stamp_devices_G(stamper, [dev])
        assembler.stamp_vs_G(stamper, [vs])
        G = assembler.finish_G(stamper)
        assert G.shape == (3, 3)  # n=2 + m=1

    def test_build_rhs(self, assembler, indexer):
        dev = CapacitorDevice("c1", 1, 2, 1e-6, 1e-6)
        vs = VoltageSource("vs1", 1, 0, lambda t: 10.0)
        n = indexer.n  # 2
        rhs = assembler.new_rhs(n + 1)  # +1 for VS

        assembler.stamp_devices_rhs(rhs, [dev], 1e-6)
        assembler.stamp_vs_rhs(rhs, [vs], 1e-6)

        # VS excitation at row n
        assert np.isclose(rhs[n], 10.0)

    def test_multiport_rhs_stamping(self, assembler, indexer):
        class FakeMP:
            name = "mp1"
            contributes_G = True
            def stamp_G(self, stamper, indexer): pass
            def stamp_rhs(self, rhs, indexer, t):
                rhs[0] += 0.5

        rhs = assembler.new_rhs(indexer.n)
        assembler.stamp_multiport_rhs(rhs, [FakeMP()], 0.0)
        assert np.isclose(rhs[0], 0.5)
