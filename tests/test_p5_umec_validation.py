"""P5.5: UMEC transformer validation — Norton, turn ratio, reset."""

from __future__ import annotations

import numpy as np
import pytest

from umec_transformer import (
    UMECTransformer,
    UMECTransformerData,
    create_umec_transformer_3ph_bank_data,
    create_umec_transformer_3ph_bank_instance,
)


@pytest.mark.validation
def test_umec_norton_matrix_symmetric_finite():
    data = create_umec_transformer_3ph_bank_data(
        name="T1", S_mva=1.0, V1_kV=0.69, V2_kV=35.0,
        nodes=[[(i + 1, 0), (i + 4, 0)] for i in range(3)],
    )
    xfmr = UMECTransformer(data, dt=50e-6, verbose=False)
    G, Ih = xfmr.get_norton_equivalent()
    assert G.shape[0] == G.shape[1]
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(Ih))
    np.testing.assert_allclose(G, G.T, rtol=1e-12, atol=1e-12)


@pytest.mark.validation
def test_umec_reset_clears_state():
    data = create_umec_transformer_3ph_bank_data(
        name="T1", S_mva=1.0, V1_kV=0.69, V2_kV=35.0,
        nodes=[[(i + 1, 0), (i + 4, 0)] for i in range(3)],
    )
    xfmr = UMECTransformer(data, dt=50e-6, verbose=False)
    xfmr.update_history(np.ones(6) * 100.0)
    assert np.linalg.norm(xfmr.I_hist) > 0
    xfmr.reset_state()
    np.testing.assert_allclose(xfmr.I_hist, 0.0, atol=1e-12)


@pytest.mark.validation
def test_umec_instance_factory_returns_transformer():
    xfmr = create_umec_transformer_3ph_bank_instance(
        dt=50e-6, name="T1", S_mva=1.0, V1_kV=10.0, V2_kV=0.4,
        nodes=[[(1, 0), (4, 0)], [(2, 0), (5, 0)], [(3, 0), (6, 0)]],
    )
    assert isinstance(xfmr, UMECTransformer)
    G, Ih = xfmr.get_norton_equivalent()
    assert G.shape[0] == 6


@pytest.mark.validation
def test_umec_legacy_factory_returns_data():
    from umec_transformer import create_umec_transformer_3ph_bank
    data = create_umec_transformer_3ph_bank("T_legacy", 1.0, 0.69, 35.0)
    assert isinstance(data, UMECTransformerData)
