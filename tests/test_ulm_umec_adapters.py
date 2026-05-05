"""Smoke-tests that ULM and UMEC adapters satisfy MultiPortDevice and import
cleanly even when the external modules are unavailable."""

import numpy as np
import pytest

from emtp.devices.multiport import MultiPortDevice


# ---------------------------------------------------------------------------
# ULM adapter smoke
# ---------------------------------------------------------------------------

class TestULMLineDeviceImport:
    def test_adapter_imports(self):
        from emtp.lines.ulm import ULMLineDevice
        assert ULMLineDevice is not None


class TestULMLineDeviceSmoke:
    """Lightweight tests that only need a mock ULM line."""

    @pytest.fixture
    def mock_line(self):
        class MockULM:
            name = "mock"
            nc = 2
            G_eq = np.array([[0.01, 0.0], [0.0, 0.01]])
            I_hist_k = np.array([0.5, 0.5])
            I_hist_m = np.array([-0.3, -0.3])

            def update_state(self, vk, vm, record_history=False):
                self._last_vk = vk
                self._last_vm = vm

            def update_history_sources(self):
                self._history_updated = True

        return MockULM()

    def test_adapter_satisfies_protocol(self, mock_line):
        from emtp.lines.ulm import ULMLineDevice
        dev = ULMLineDevice("mock", mock_line, [1, 2], [3, 4])
        assert isinstance(dev, MultiPortDevice)

    def test_ports_from_nodes(self, mock_line):
        from emtp.lines.ulm import ULMLineDevice
        dev = ULMLineDevice("mock", mock_line, [1, 2], [3, 4])
        assert dev.ports == ((1, 0), (2, 0), (3, 0), (4, 0))

    def test_stamp_G_correct_shape(self, mock_line):
        from emtp.lines.ulm import ULMLineDevice
        from emtp.nodes import NodeIndexer
        from emtp.stamping import COOStamper

        dev = ULMLineDevice("mock", mock_line, [1, 2], [3, 4])
        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        stamper = COOStamper(indexer.n)
        dev.stamp_G(stamper, indexer)
        G = stamper.tocsc()
        assert G.shape == (4, 4)
        assert G.nnz > 0

    def test_stamp_rhs_sign_convention(self, mock_line):
        from emtp.lines.ulm import ULMLineDevice
        from emtp.nodes import NodeIndexer

        dev = ULMLineDevice("mock", mock_line, [1, 2], [3, 4])
        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        rhs = np.zeros(indexer.n, dtype=np.float64)
        dev.stamp_rhs(rhs, indexer, 0.0)
        # rhs[1] -= 0.5, rhs[2] -= 0.5, rhs[3] -= -0.3, rhs[4] -= -0.3
        assert np.allclose(rhs[:2], [-0.5, -0.5])
        assert np.allclose(rhs[2:4], [0.3, 0.3])

    def test_reset_state(self, mock_line):
        from emtp.lines.ulm import ULMLineDevice
        dev = ULMLineDevice("mock", mock_line, [1, 2], [3, 4])
        dev._vk = np.array([10.0, 20.0])
        dev.reset_state()
        assert np.all(dev._vk == 0.0)


# ---------------------------------------------------------------------------
# UMEC adapter smoke
# ---------------------------------------------------------------------------

class TestUMECTransformerDeviceImport:
    def test_adapter_imports(self):
        from emtp.transformers.umec import UMECTransformerDevice
        assert UMECTransformerDevice is not None


class TestUMECTransformerDeviceSmoke:
    """Lightweight tests with a mock UMEC transformer."""

    @pytest.fixture
    def mock_xfmr(self):
        class MockUMEC:
            name = "mock_umec"
            n_phases = 3
            n_windings = 2
            m = 6
            _G = np.eye(6) * 0.01
            _I = np.zeros(6)

            def __init__(self):
                self._ports = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]

            def get_port_nodes(self):
                return self._ports

            def get_norton_equivalent(self):
                return self._G.copy(), self._I.copy()

            def update_history(self, V_ports):
                self._V_updated = V_ports

            def check_saturation(self, V_ports):
                return False, {}

            def reset_state(self):
                self._reset = True

        return MockUMEC()

    def test_adapter_satisfies_protocol(self, mock_xfmr):
        from emtp.transformers.umec import UMECTransformerDevice
        dev = UMECTransformerDevice("mock", mock_xfmr)
        assert isinstance(dev, MultiPortDevice)

    def test_ports_match_transformer(self, mock_xfmr):
        from emtp.transformers.umec import UMECTransformerDevice
        dev = UMECTransformerDevice("mock", mock_xfmr)
        assert len(dev.ports) == 6
        assert dev.ports[0] == (1, 0)

    def test_stamp_G_is_square(self, mock_xfmr):
        from emtp.transformers.umec import UMECTransformerDevice
        from emtp.nodes import NodeIndexer
        from emtp.stamping import COOStamper

        dev = UMECTransformerDevice("mock", mock_xfmr)
        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        stamper = COOStamper(indexer.n)
        dev.stamp_G(stamper, indexer)
        G = stamper.tocsc()
        assert G.shape == (6, 6)

    def test_check_rebuild_no_saturation(self, mock_xfmr):
        from emtp.transformers.umec import UMECTransformerDevice
        from emtp.nodes import NodeIndexer

        dev = UMECTransformerDevice("mock", mock_xfmr)
        indexer = NodeIndexer()
        dev.register_nodes(indexer)
        indexer.freeze()

        V = np.zeros(indexer.n)
        assert dev.check_rebuild_required(V, indexer, 0.0) is False

    def test_reset_state(self, mock_xfmr):
        from emtp.transformers.umec import UMECTransformerDevice
        dev = UMECTransformerDevice("mock", mock_xfmr)
        dev.reset_state()
        assert mock_xfmr._reset
