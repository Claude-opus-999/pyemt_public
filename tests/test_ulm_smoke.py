from __future__ import annotations

import numpy as np

from ulm_transmission_line_PARA import ULMLine, create_test_fitulm_data


def test_create_test_fitulm_data_is_reproducible_by_default_seed():
    a = create_test_fitulm_data(nf=1, n_poles=2)
    b = create_test_fitulm_data(nf=1, n_poles=2)
    np.testing.assert_allclose(a.yc_d, b.yc_d)
    for ar, br in zip(a.yc_residues, b.yc_residues):
        np.testing.assert_allclose(ar, br)


def test_ulm_line_full_step_smoke():
    line = ULMLine.create_test("ULM1", 1, 2, dt=1e-6, nf=1, n_poles=2, verbose=False)
    line.full_step(1000.0, 0.0)
    ik, im = line.get_currents()
    assert np.isfinite(ik)
    assert np.isfinite(im)
    assert np.isfinite(line.I_hist_k)
    assert np.isfinite(line.I_hist_m)
    assert line.V_k_history.shape[0] == 1
