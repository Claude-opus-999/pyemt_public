from __future__ import annotations

import numpy as np

from emtp import EMTPSolver


def test_timed_switch_closing_rebuilds_matrix_and_energizes_load():
    dt = 1e-6
    s = EMTPSolver(dt=dt, finish_time=8e-6, verbose=False)
    s.add_VS("Vs", 1, 0, 10.0)
    s.add_SW("SW", 1, 2, initially_closed=False, R_closed=1e-6, R_open=1e9, t_close=3e-6)
    s.add_R("Rload", 2, 0, 100.0)
    s.add_voltage_probe("Vload", 2, 0)
    s.run()

    t = s.get_time("s")
    v = s.get_voltage_probe("Vload", "V")
    assert np.max(np.abs(v[t < 3e-6 - 0.5 * dt])) < 1e-4
    assert v[-1] > 9.99
    assert s.get_solver_statistics().get("G_rebuilds", 0) >= 2
