from __future__ import annotations

import numpy as np

from emtp import EMTPSolver


def test_resistor_voltage_source_smoke():
    s = EMTPSolver(dt=1e-6, finish_time=5e-6, verbose=False)
    s.add_R("R1", 1, 0, 100.0)
    s.add_VS("V1", 1, 0, 10.0)
    s.add_voltage_probe("V_node1", 1, 0)
    s.add_branch_current_probe("I_R1", "R1")
    s.run()

    np.testing.assert_allclose(s.get_voltage_probe("V_node1", "V"), 10.0, atol=1e-10)
    np.testing.assert_allclose(s.get_branch_current_probe("I_R1", "A"), 0.1, atol=1e-10)


def test_current_source_direction_from_ground_injects_positive_node_voltage():
    s = EMTPSolver(dt=1e-6, finish_time=3e-6, verbose=False, record_source_history=True)
    s.add_R("Rload", 1, 0, 100.0)
    s.add_IS("Iin", 0, 1, 2.0)  # current direction 0 -> 1 injects +2 A into node 1
    s.add_voltage_probe("V1", 1, 0)
    s.run()

    np.testing.assert_allclose(s.get_voltage_probe("V1", "V"), 200.0, atol=1e-9)
    np.testing.assert_allclose(s.get_source_current("Iin"), 2.0, atol=1e-12)
