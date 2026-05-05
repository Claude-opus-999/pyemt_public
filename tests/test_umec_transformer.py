from __future__ import annotations

import numpy as np

from umec_transformer import UMECTransformer, create_umec_transformer_3ph_bank


def test_umec_transformer_norton_shapes_and_reset():
    data = create_umec_transformer_3ph_bank(
        name="T1",
        S_mva=1.0,
        V1_kV=0.69,
        V2_kV=35.0,
        nodes=[
            [(1, 0), (4, 0)],
            [(2, 0), (5, 0)],
            [(3, 0), (6, 0)],
        ],
    )
    xfmr = UMECTransformer(data, dt=50e-6, verbose=False)
    G, Ih = xfmr.get_norton_equivalent()

    assert G.shape == (6, 6)
    assert Ih.shape == (6,)
    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(Ih))

    xfmr.update_history(np.ones(6))
    assert np.linalg.norm(xfmr.I_hist) > 0
    xfmr.reset_state()
    np.testing.assert_allclose(xfmr.I_hist, 0.0)
