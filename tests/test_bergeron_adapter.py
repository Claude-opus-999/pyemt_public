"""Verify BergeronLineDevice adapter satisfies MultiPortDevice and reproduces
the same G / RHS stamping as the existing manual line stamp in the solver."""

import numpy as np
import pytest

from emtp.devices.multiport import MultiPortDevice
from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper

try:
    from transmission_line_emtp_v2 import BergeronLine
    BERGERON_AVAILABLE = True
except ImportError:
    BERGERON_AVAILABLE = False

try:
    from emtp.lines.bergeron import BergeronLineDevice
except ImportError:
    BergeronLineDevice = None


@pytest.mark.skipif(not BERGERON_AVAILABLE,
                    reason="transmission_line_emtp_v2 not installed")
class TestBergeronLineDevice:
    """Adapter protocol compliance and stamping correctness."""

    @staticmethod
    def _make_line(Zc=300.0, tau=10e-6, dt=1e-6):
        line = BergeronLine("bl", 1, 2, Zc, tau)
        line.initialize(dt)
        return line

    def test_adapter_satisfies_multiport_protocol(self):
        line = self._make_line()
        dev = BergeronLineDevice("adapter", line, 1, 2)
        assert isinstance(dev, MultiPortDevice)

    def test_ports_are_ground_referenced(self):
        line = self._make_line()
        dev = BergeronLineDevice("adapter", line, 5, 7)
        assert dev.ports == ((5, 0), (7, 0))

    def test_stamp_G_matches_manual_convention(self):
        line = self._make_line(Zc=300.0, dt=1e-6)
        dev = BergeronLineDevice("adapter", line, 1, 2)

        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        n = indexer.n
        # -- adapter path
        stamper_adapter = COOStamper(n)
        dev.stamp_G(stamper_adapter, indexer)
        G_adapter = stamper_adapter.tocsc()

        # -- manual path (replicating solver convention)
        stamper_manual = COOStamper(n)
        G_eq = float(line.G_eq)
        stamper_manual.add(0, 0, G_eq)   # node 1
        stamper_manual.add(1, 1, G_eq)   # node 2
        G_manual = stamper_manual.tocsc()

        assert (G_adapter - G_manual).nnz == 0

    def test_stamp_rhs_sign_convention(self):
        line = self._make_line(Zc=300.0, dt=1e-6)
        # Set known history currents
        line.I_hist_k = 0.5
        line.I_hist_m = -0.3

        dev = BergeronLineDevice("adapter", line, 1, 2)

        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        rhs = np.zeros(indexer.n, dtype=np.float64)
        dev.stamp_rhs(rhs, indexer, 0.0)

        # rhs[node_k] -= I_hist_k  → rhs[0] = -0.5
        # rhs[node_m] -= I_hist_m  → rhs[1] = -(-0.3) = +0.3
        assert np.isclose(rhs[0], -0.5)
        assert np.isclose(rhs[1], 0.3)

    def test_update_after_solve_reads_voltages(self):
        line = self._make_line()
        dev = BergeronLineDevice("adapter", line, 1, 2)

        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        V = np.array([10.0, 5.0])  # V1=10, V2=5
        dev.update_after_solve(V, indexer, 0.0)
        assert np.isclose(dev._vk, 10.0)
        assert np.isclose(dev._vm, 5.0)

    def test_reset_state_clears_voltages(self):
        line = self._make_line()
        dev = BergeronLineDevice("adapter", line, 1, 2)
        dev._vk = 100.0
        dev._vm = 50.0
        dev.reset_state()
        assert dev._vk == 0.0 and dev._vm == 0.0

    def test_register_nodes_is_idempotent(self):
        line = self._make_line()
        dev = BergeronLineDevice("adapter", line, 1, 2)

        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        dev.register_nodes(indexer)  # second call should not error
        assert indexer.n == 2
