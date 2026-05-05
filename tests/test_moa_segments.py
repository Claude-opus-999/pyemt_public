from __future__ import annotations

import numpy as np

from nonlinear_models_pscad import SegmentedMOAResistor


def test_segmented_moa_current_sign_and_segment_switch():
    moa = SegmentedMOAResistor.from_breakpoints(
        "MOA",
        [(0.0, 0.0), (100.0, 1e-3), (200.0, 0.1), (300.0, 10.0)],
        add_zero_point=False,
    )

    assert moa.current_segment_index == 0
    np.testing.assert_allclose(moa.get_current_exact(50.0), 0.5e-3)
    np.testing.assert_allclose(moa.get_current_exact(-50.0), -0.5e-3)

    changed, new_idx = moa.check_segment(250.0)
    assert changed is True
    assert new_idx >= 2
    changed_apply = moa.update_segment(250.0)
    assert changed_apply is True
    assert moa.current_segment_index == new_idx

    g, i_eq = moa.get_norton_equivalent(250.0)
    assert g > 0.0
    assert np.isfinite(i_eq)
