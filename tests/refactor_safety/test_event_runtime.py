"""PR6: EventRuntime tests — existence and step orchestration."""

import numpy as np
import pytest
from emtp import EMTPSolver


class TestEventRuntimeExistence:
    def test_runtime_present_after_init(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        assert s.event_runtime is not None

    def test_runtime_step_runs_without_error(self):
        """event_runtime.step delegates to solver._run_one_step."""
        s = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_R("r", 1, 0, 100.0)
        s.add_voltage_probe("V1", 1, 0)
        s.run()
        v = s.get_voltage_probe("V1", "V")
        assert len(v) > 0

    def test_runtime_with_switch_produces_events(self):
        s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        s.add_VS("vs", 1, 0, 1.0)
        s.add_SW("sw1", 1, 2, t_close=20e-6, t_open=80e-6)
        s.add_R("r", 2, 0, 100.0)
        s.add_voltage_probe("Vr", 2, 0)
        s.run()
        v = s.get_voltage_probe("Vr", "V")
        assert abs(v[10]) < 0.01  # before switch closes
        assert v[50] > 0.1  # after switch closes
