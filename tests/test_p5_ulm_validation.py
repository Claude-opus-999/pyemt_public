"""P5.5b: ULM frequency-dependent line validation."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from ulm_transmission_line_PARA import ULMLine, ULMModel, create_test_fitulm_data
    ULM_AVAILABLE = True
except ImportError:
    ULM_AVAILABLE = False


@pytest.mark.validation
@pytest.mark.skipif(not ULM_AVAILABLE, reason="ULM module not available")
def test_ulm_seed_reproducibility():
    d1 = create_test_fitulm_data(nf=1, seed=42)
    d2 = create_test_fitulm_data(nf=1, seed=42)
    np.testing.assert_allclose(d1.yc_poles, d2.yc_poles, rtol=1e-12)


@pytest.mark.validation
@pytest.mark.skipif(not ULM_AVAILABLE, reason="ULM module not available")
def test_ulm_different_seeds_different_data():
    d1 = create_test_fitulm_data(nf=1, seed=42)
    d2 = create_test_fitulm_data(nf=1, seed=123)
    # different seeds should produce different residue matrices
    r1 = np.asarray(d1.yc_residues[0]).ravel()
    r2 = np.asarray(d2.yc_residues[0]).ravel()
    diff = np.max(np.abs(r1 - r2))
    assert diff > 0, "Different seeds should produce different FitULMData"


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.skipif(not ULM_AVAILABLE, reason="ULM module not available")
def test_ulm_single_line_finite_outputs():
    from emtp_solver_v3 import EMTPSolver
    dt, finish = 1e-6, 100e-6
    data = create_test_fitulm_data(nf=1, seed=42)
    model = ULMModel(data, line_length=100e3, dt=dt, verbose=False)
    line = ULMLine("ULM1", model, node_k=1, node_m=2)

    s = EMTPSolver(dt=dt, finish_time=finish, verbose=False)
    s.add_VS("Vs", 1, 0, 100e3)
    s.add_R("Rterm", 2, 0, 300.0)
    s.add_line(line)
    s.add_voltage_probe("Vk", 1, 0)
    s.add_voltage_probe("Vm", 2, 0)
    s.run()

    vk = s.get_voltage_probe("Vk", "V")
    vm = s.get_voltage_probe("Vm", "V")
    assert np.all(np.isfinite(vk)), "NaN/inf in Vk"
    assert np.all(np.isfinite(vm)), "NaN/inf in Vm"
