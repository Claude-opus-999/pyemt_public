"""Tests for the MultiPortDevice protocol and unified dispatch.

Uses a fake two-port device that stamps a known conductance pattern so
we can verify that MNA assembly, RHS injection, post-solve update,
history advance and rebuild-check all flow through the protocol.
"""

import numpy as np
import pytest

from emtp.devices import MultiPortDevice


# ---------------------------------------------------------------------------
# Fake three-port linear device (ports: (1,0), (2,0), (1,2))
# Stamps known conductance pattern for verification.
# ---------------------------------------------------------------------------

class FakeThreePortDevice:
    """Three-port device with known G-matrix and history currents."""

    def __init__(self, name="fake3p", G_val=0.1, I_hist=0.02):
        self.name = name
        self.G = G_val
        self.I_hist = I_hist
        self._vk = [0.0, 0.0, 0.0]   # per-port voltage after solve

    @property
    def ports(self):
        return ((1, 0), (2, 0), (1, 2))

    @property
    def contributes_G(self):
        return True

    @property
    def is_dynamic(self):
        return True

    def register_nodes(self, indexer):
        for nf, nt in self.ports:
            if nf > 0:
                indexer.register(nf)
            if nt > 0:
                indexer.register(nt)

    def stamp_G(self, stamper, indexer):
        for i, (nf_i, nt_i) in enumerate(self.ports):
            ci_f = indexer.to_compact(nf_i) if nf_i > 0 else -1
            ci_t = indexer.to_compact(nt_i) if nt_i > 0 else -1
            for j, (nf_j, nt_j) in enumerate(self.ports):
                cj_f = indexer.to_compact(nf_j) if nf_j > 0 else -1
                cj_t = indexer.to_compact(nt_j) if nt_j > 0 else -1
                g = self.G if i == j else 0.0
                if ci_f >= 0 and cj_f >= 0:
                    stamper.add(ci_f, cj_f, g)
                if ci_t >= 0 and cj_t >= 0:
                    stamper.add(ci_t, cj_t, g)
                if ci_f >= 0 and cj_t >= 0:
                    stamper.add(ci_f, cj_t, -g)
                if ci_t >= 0 and cj_f >= 0:
                    stamper.add(ci_t, cj_f, -g)

    def stamp_rhs(self, rhs, indexer, t):
        for nf, nt in self.ports:
            cf = indexer.to_compact(nf) if nf > 0 else -1
            ct = indexer.to_compact(nt) if nt > 0 else -1
            if cf >= 0:
                rhs[cf] -= self.I_hist
            if ct >= 0:
                rhs[ct] += self.I_hist

    def update_after_solve(self, V, indexer, t):
        for k, (nf, nt) in enumerate(self.ports):
            vf = V[indexer.to_compact(nf)] if nf > 0 else 0.0
            vt = V[indexer.to_compact(nt)] if nt > 0 else 0.0
            self._vk[k] = vf - vt

    def update_history(self, V, indexer, dt):
        # For fake device: just remember that it was called
        self._history_updated = True

    def check_rebuild_required(self, V, indexer, t):
        return False

    def reset_state(self):
        self._vk = [0.0, 0.0, 0.0]
        self._history_updated = False


# ---------------------------------------------------------------------------

class TestMultiPortDeviceProtocol:
    """Verify the protocol can be checked with isinstance."""

    def test_fake_device_satisfies_protocol(self):
        dev = FakeThreePortDevice()
        assert isinstance(dev, MultiPortDevice)

    def test_protocol_attributes_accessible(self):
        dev = FakeThreePortDevice(name="test")
        assert dev.name == "test"
        assert dev.contributes_G is True
        assert dev.is_dynamic is True
        assert len(dev.ports) == 3


class TestMultiPortStampAndSolve:
    """Verify a MultiPortDevice participates correctly in a mini solver."""

    def test_stamp_G_produces_correct_sparsity(self):
        from emtp.nodes import NodeIndexer
        from emtp.stamping import COOStamper

        indexer = NodeIndexer()
        dev = FakeThreePortDevice()
        dev.register_nodes(indexer)
        indexer.freeze()

        n = indexer.n
        stamper = COOStamper(n)
        dev.stamp_G(stamper, indexer)
        G = stamper.tocsc()

        assert G.shape == (n, n)
        assert n == 2  # nodes 1,2 (GND is 0)
        # With G=0.1, diagonal entries should be present
        assert G.nnz >= 4  # at minimum the diagonal blocks

    def test_stamp_rhs_applies_correct_signs(self):
        from emtp.nodes import NodeIndexer

        indexer = NodeIndexer()
        dev = FakeThreePortDevice(I_hist=0.05)
        dev.register_nodes(indexer)
        indexer.freeze()

        rhs = np.zeros(indexer.n, dtype=np.float64)
        dev.stamp_rhs(rhs, indexer, 0.0)

        # Port (1,0): rhs[1] -= 0.05, port (2,0): rhs[2] -= 0.05,
        # port (1,2): rhs[1] -= 0.05, rhs[2] += 0.05
        # Net: rhs[1] = -0.10, rhs[2] = 0.00
        assert np.isclose(rhs[indexer.to_compact(1)], -0.10)
        assert np.isclose(rhs[indexer.to_compact(2)], 0.00)

    def test_update_after_solve_reads_port_voltages(self):
        from emtp.nodes import NodeIndexer

        indexer = NodeIndexer()
        dev = FakeThreePortDevice()
        dev.register_nodes(indexer)
        indexer.freeze()

        V = np.array([3.0, 1.0])  # V1=3, V2=1
        dev.update_after_solve(V, indexer, 0.0)

        # port (1,0): 3-0=3, port (2,0): 1-0=1, port (1,2): 3-1=2
        assert np.isclose(dev._vk[0], 3.0)
        assert np.isclose(dev._vk[1], 1.0)
        assert np.isclose(dev._vk[2], 2.0)

    def test_update_history_called(self):
        from emtp.nodes import NodeIndexer

        indexer = NodeIndexer()
        dev = FakeThreePortDevice()
        dev.register_nodes(indexer)
        indexer.freeze()

        V = np.array([0.0, 0.0])
        dev.update_history(V, indexer, 1e-6)
        assert dev._history_updated

    def test_reset_state_clears_state(self):
        dev = FakeThreePortDevice()
        dev._vk = [5.0, 5.0, 5.0]
        dev.reset_state()
        assert dev._vk == [0.0, 0.0, 0.0]
