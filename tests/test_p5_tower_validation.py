"""P5.6: Tower case formal validation — golden metrics and regression."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import test_tower_model_latest as tower
    TOWER_AVAILABLE = True
except ImportError:
    TOWER_AVAILABLE = False


@pytest.mark.validation
@pytest.mark.skipif(not TOWER_AVAILABLE, reason="test_tower_model_latest not importable")
def test_tower_topology_and_delay_parameters():
    """Structural checks that do not run full transient."""
    tower.test_11_tower_topology_and_parameters()
    tower.test_12_tower_delay_buffers()


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.skipif(not TOWER_AVAILABLE, reason="test_tower_model_latest not importable")
def test_tower_no_flashover_metrics():
    cfg = tower.TOWER_CONFIG.copy()
    cfg["finish_time"] = 5e-6
    sim = tower.build_tower_model(cfg=cfg, verbose=False, use_lpm=False)
    r = sim["results"]

    v_top = float(np.max(np.abs(r["V_node2_kV"])))
    v_mid = float(np.max(np.abs(r["V_node4_kV"])))
    v_g01 = float(np.max(np.abs(r["V_node10_kV"])))

    assert v_top > 0, "Tower top voltage must be positive"
    assert v_mid > 0, "Tower mid voltage must be positive"
    assert v_top > v_mid, "Top voltage should exceed mid voltage"
    assert v_mid > v_g01, "Mid voltage should exceed ground voltage"


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.skipif(not TOWER_AVAILABLE, reason="test_tower_model_latest not importable")
def test_tower_lpm_voltage_consistency():
    """P0 regression: LPM internal voltage = actual gap voltage."""
    cfg = tower.TOWER_CONFIG.copy()
    cfg["finish_time"] = 1.5e-6
    sim = tower.build_tower_model(cfg=cfg, verbose=False, use_lpm=True)
    r = sim["results"]

    for name, upper, lower in [("Vbrk11", "V_node5_kV", "V_node8_kV"),
                                ("Vbrk12", "V_node6_kV", "V_node9_kV")]:
        lpm = sim["lpm_models"][name]
        actual_gap = np.abs(r[upper] - r[lower])
        internal = np.abs(np.array(lpm.voltage_history))
        np.testing.assert_allclose(
            np.max(internal), np.max(actual_gap),
            rtol=1e-10, atol=1e-7,
            err_msg=f"LPM {name}: internal peak != actual gap peak"
        )
