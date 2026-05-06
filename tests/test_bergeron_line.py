from __future__ import annotations

import numpy as np

from transmission_line_emtp_v2 import DelayBuffer, TransmissionLineFactory
from emtp import EMTPSolver


def test_bergeron_factory_parameters_and_delay_buffer():
    dt = 1e-9
    tau_per_m = 3.33333e-9
    length_m = 12.3
    Zc = 270.6

    line = TransmissionLineFactory.create_from_zc_tau("L1", 1, 2, Zc, tau_per_m, length_m)
    line.initialize(dt)
    info = line.get_info()

    np.testing.assert_allclose(info["Zc"], Zc)
    np.testing.assert_allclose(info["tau"], tau_per_m * length_m, rtol=1e-12)
    np.testing.assert_allclose(info["G_eq"], 1.0 / Zc, rtol=1e-12)
    assert info["delay_steps"] == int((tau_per_m * length_m) / dt)

    buf = DelayBuffer.create_for_delay(tau_per_m * length_m, dt)
    for i in range(buf.delay_steps + 3):
        buf.push(1.0 if i >= 1 else 0.0)
    assert buf.get_delayed() > 0.5


def test_bergeron_line_runs_inside_solver_smoke():
    s = EMTPSolver(dt=1e-9, finish_time=10e-9, verbose=False, record_line_history=True)
    s.add_IS("I", 0, 1, 1.0)
    s.add_R("Rk", 1, 0, 300.0)
    s.add_R("Rm", 2, 0, 300.0)
    line = TransmissionLineFactory.create_from_zc_tau(
        "L1", 1, 2, Zc=300.0, tau_per_m=1e-9, length_m=3.0
    )
    s.add_line(line)
    s.run()

    vk = s.get_line_voltage_k("L1", "V")
    ik = s.get_line_current_k("L1", "A")
    assert len(vk) == len(s.get_time("s"))
    assert len(ik) == len(s.get_time("s"))
    assert np.all(np.isfinite(vk))
    assert np.all(np.isfinite(ik))
