from __future__ import annotations

import numpy as np
import pytest

import test_tower_model_latest as tower


def test_tower_topology_and_delay_parameters_smoke():
    # These are the user's tower-case structural checks; they do not run the full transient.
    tower.test_11_tower_topology_and_parameters()
    tower.test_12_tower_delay_buffers()


@pytest.mark.slow
def test_tower_lpm_short_run_records_real_gap_voltage():
    cfg = tower.TOWER_CONFIG.copy()
    cfg["finish_time"] = 1.25e-6  # short but includes lightning start at 1 us
    sim = tower.build_tower_model(cfg=cfg, verbose=False, use_lpm=True)
    r = sim["results"]

    for name, upper, lower in [("Vbrk11", "V_node5_kV", "V_node8_kV"), ("Vbrk12", "V_node6_kV", "V_node9_kV")]:
        lpm = sim["lpm_models"][name]
        actual_gap = np.abs(r[upper] - r[lower])
        internal = np.abs(np.array(lpm.voltage_history))  # LPM stores gap voltage history in kV
        assert len(internal) == len(actual_gap)
        np.testing.assert_allclose(np.max(internal), np.max(actual_gap), rtol=1e-10, atol=1e-7)
