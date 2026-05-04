import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emtp_solver_v3 import EMTPSolver


class SolverRegressionTests(unittest.TestCase):
    def test_run_uses_integer_steps_and_includes_finish_time(self):
        solver = EMTPSolver(dt=1e-4, finish_time=1e-3, verbose=False)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_current_source("I1", 0, 1, 1.0)

        solver.run()

        self.assertEqual(len(solver.time_array), 11)
        self.assertLess(abs(solver.time_array[-1] - 1e-3), 1e-15)

    def test_resistor_branch_current_probe_does_not_need_full_history(self):
        solver = EMTPSolver(dt=1e-6, finish_time=2e-6, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_branch_current_probe("I_R1", "R1")

        solver.run()

        np.testing.assert_allclose(solver.get_probe("I_R1", unit="A"), 1.0)
        with self.assertRaises(RuntimeError):
            solver.get_branch_current("R1")

    def test_capacitor_probe_includes_parallel_damping_current(self):
        solver = EMTPSolver(dt=1.0, finish_time=0.0, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_capacitor("C1", 1, 0, 1.0, Rp=1.0)
        solver.add_branch_current_probe("I_C1", "C1")

        solver.run()

        np.testing.assert_allclose(solver.get_probe("I_C1", unit="A"), [3.0])

    def test_run_resets_dynamic_state(self):
        solver = EMTPSolver(dt=1.0, finish_time=2.0, verbose=False)
        solver.add_capacitor("C1", 1, 0, 1.0)
        solver.add_current_source("I1", 0, 1, 1.0)

        solver.run()
        first = solver.get_node_voltage(1)
        solver.run()
        second = solver.get_node_voltage(1)

        np.testing.assert_allclose(first, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(second, first)

    def test_timed_switch_events_are_consumed_once(self):
        solver = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
        solver.add_voltage_source("V1", 1, 0, 1.0)
        solver.add_resistor("R1", 1, 0, 1.0)
        solver.add_switch(
            "S1", 1, 0,
            t_close=1e-6,
            t_open=2e-6,
            R_closed=1e-3,
            R_open=1e9,
        )

        solver.run()

        self.assertFalse(solver.branches["S1"].is_closed)
        self.assertEqual(solver.get_solver_statistics().get("G_rebuilds"), 3)


if __name__ == "__main__":
    unittest.main()
