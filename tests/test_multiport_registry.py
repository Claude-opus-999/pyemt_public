"""Verify the MultiPortDevice registry skeleton in EMTPSolver."""

import numpy as np
import pytest

from emtp.devices.multiport import MultiPortDevice
from emtp_solver_v3 import EMTPSolver


class FakeMPDev:
    """Minimal MultiPortDevice that records which methods were called."""

    def __init__(self, name="fake", dynamic=True):
        self.name = name
        self._dynamic = dynamic
        self.calls = []  # (method_name, args_summary)

    @property
    def ports(self):
        return ((1, 0),)

    @property
    def contributes_G(self):
        return True

    @property
    def is_dynamic(self):
        return self._dynamic

    def register_nodes(self, indexer):
        self.calls.append(("register_nodes", ()))

    def stamp_G(self, stamper, indexer):
        self.calls.append(("stamp_G", ()))

    def stamp_rhs(self, rhs, indexer, t):
        self.calls.append(("stamp_rhs", (t,)))

    def update_after_solve(self, V, indexer, t):
        self.calls.append(("update_after_solve", (t,)))

    def update_history(self, V, indexer, dt):
        self.calls.append(("update_history", (dt,)))

    def check_rebuild_required(self, V, indexer, t):
        return False

    def reset_state(self):
        self.calls.append(("reset_state", ()))


class TestMultiportRegistry:
    def test_registry_starts_empty(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        assert s._multiport_devices == []

    def test_register_adds_device_and_marks_dirty(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        dev = FakeMPDev()
        s._register_multiport_device(dev)
        assert len(s._multiport_devices) == 1
        assert s._multiport_devices[0] is dev
        assert s._stamping.G_dirty

    def test_dispatch_methods_call_device_methods(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        dev = FakeMPDev()
        s._register_multiport_device(dev)

        # register nodes
        s._register_multiport_nodes()
        assert ("register_nodes", ()) in dev.calls

        # stamp G
        from emtp.stamping import COOStamper
        stamper = COOStamper(2)
        s._stamp_multiport_G(stamper)
        assert ("stamp_G", ()) in dev.calls

        # stamp RHS
        rhs = np.zeros(2)
        s._stamp_multiport_rhs(rhs, 1e-6)
        assert ("stamp_rhs", (1e-6,)) in dev.calls

        # update after solve
        V = np.array([1.0, 2.0])
        s._update_multiport_after_solve(V, 2e-6)
        assert ("update_after_solve", (2e-6,)) in dev.calls

        # update history
        s._update_multiport_history(V, 1e-6)
        assert ("update_history", (1e-6,)) in dev.calls

    def test_check_rebuild_returns_false_when_no_changes(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        dev = FakeMPDev()
        s._register_multiport_device(dev)
        V = np.array([0.0, 0.0])
        assert s._check_multiport_rebuild_required(V, 0.0) is False

    def test_check_rebuild_marks_dirty_when_changed(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)

        class ChangingDev(FakeMPDev):
            def check_rebuild_required(self, V, indexer, t):
                return True

        dev = ChangingDev()
        s._register_multiport_device(dev)
        V = np.array([0.0, 0.0])
        assert s._check_multiport_rebuild_required(V, 0.0) is True
        assert s._stamping.G_dirty

    def test_reset_dynamic_state_resets_multiport(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        dev = FakeMPDev()
        s._register_multiport_device(dev)
        s.reset_dynamic_state()
        assert ("reset_state", ()) in dev.calls
