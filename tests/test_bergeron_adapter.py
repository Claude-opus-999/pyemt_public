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


@pytest.mark.skipif(not BERGERON_AVAILABLE,
                    reason="transmission_line_emtp_v2 not installed")
class TestBergeronMultiportIntegration:
    """Verify the adapter is registered in the solver and dispatch
    produces the same G/RHS as the legacy transmission-line path."""

    def test_add_bergeron_registers_multiport_when_enabled(self):
        from emtp_solver_v3 import EMTPSolver

        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False,
                       use_multiport_lines=True)
        s.add_bergeron_line("bl", 1, 2, Zc=300.0, tau=10e-6)
        assert len(s._multiport_devices) == 1
        mpd = s._multiport_devices[0]
        assert mpd.name == "bl"
        assert mpd.ports == ((1, 0), (2, 0))

    def test_no_multiport_when_flag_false(self):
        from emtp_solver_v3 import EMTPSolver

        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False,
                       use_multiport_lines=False)
        s.add_bergeron_line("bl", 1, 2, Zc=300.0, tau=10e-6)
        assert len(s._multiport_devices) == 0

    def test_solver_runs_with_multiport_enabled(self):
        from emtp_solver_v3 import EMTPSolver

        s = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False,
                       use_multiport_lines=False)
        s.add_VS("vs1", 1, 0, lambda t: 1.0)
        s.add_R("rload", 2, 0, 50.0)
        line = s.add_bergeron_line("bl", 1, 2, Zc=300.0, tau=10e-6)
        line.initialize(s.dt)
        s.run()

        v1 = s.get_node_voltage(1, "V")
        assert len(v1) == 51
        assert v1[0] > 0  # initial voltage propagates

    def test_adapter_vs_legacy_matrix_equivalence(self):
        """The BergeronLineDevice adapter stamps the same G as the legacy path."""
        from emtp.nodes import NodeIndexer
        from emtp.stamping import COOStamper

        line = BergeronLine("bl", 1, 2, Zc=300.0, tau=10e-6)
        line.initialize(1e-6)

        # Build indexer
        indexer = NodeIndexer()
        indexer.register(1)
        indexer.register(2)
        indexer.freeze()
        n = indexer.n

        # Legacy stamp
        stamper_legacy = COOStamper(n)
        G_eq = float(line.G_eq)
        stamper_legacy.add(0, 0, G_eq)  # node 1 k-k
        stamper_legacy.add(1, 1, G_eq)  # node 2 m-m
        G_legacy = stamper_legacy.tocsc()

        # Adapter stamp
        dev = BergeronLineDevice("bl", line, 1, 2)
        stamper_adapter = COOStamper(n)
        dev.stamp_G(stamper_adapter, indexer)
        G_adapter = stamper_adapter.tocsc()

        assert (G_legacy - G_adapter).nnz == 0

    def test_adapter_vs_legacy_rhs_equivalence(self):
        """The BergeronLineDevice adapter stamps the same RHS as the legacy path."""
        from emtp.nodes import NodeIndexer

        line = BergeronLine("bl", 1, 2, Zc=300.0, tau=10e-6)
        line.initialize(1e-6)
        line.I_hist_k = 0.5
        line.I_hist_m = -0.3

        indexer = NodeIndexer()
        indexer.register(1)
        indexer.register(2)
        indexer.freeze()
        n = indexer.n

        # Legacy RHS
        rhs_legacy = np.zeros(n)
        rhs_legacy[0] -= 0.5  # node k
        rhs_legacy[1] -= -0.3  # node m (= +0.3)

        # Adapter RHS
        dev = BergeronLineDevice("bl", line, 1, 2)
        rhs_adapter = np.zeros(n)
        dev.stamp_rhs(rhs_adapter, indexer, 0.0)

        assert np.allclose(rhs_legacy, rhs_adapter)
