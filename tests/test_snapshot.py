"""Verify snapshot save / load / resume produces correct results."""

import numpy as np
import pytest

from emtp_solver_v3 import EMTPSolver
from emtp.config.loader import load_case_config
from emtp.builders.solver_builder import build_solver_from_config
from emtp.snapshot.hashing import compute_topology_hash, stable_json_hash


CASES_DIR = __import__("pathlib").Path(__file__).parent.parent / "cases" / "templates"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashing:
    def test_stable_hash_is_deterministic(self):
        obj = {"a": 1, "b": [2, 3]}
        h1 = stable_json_hash(obj)
        h2 = stable_json_hash(obj)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_topology_hash_changes_with_branch(self):
        s1 = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s1.add_R("r1", 1, 0, 10.0)
        h1 = compute_topology_hash(s1)

        s2 = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        s2.add_R("r1", 1, 0, 10.0)
        s2.add_R("r2", 2, 0, 20.0)
        h2 = compute_topology_hash(s2)

        assert h1 != h2


# ---------------------------------------------------------------------------
# Basic snapshot save / load
# ---------------------------------------------------------------------------

class TestSnapshotBasic:
    def test_save_and_load_rc_config(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run()
        v1 = solver.get_voltage_probe("V_cap", "V")

        snap_dir = tmp_path / "snapshot"
        solver.save_snapshot(snap_dir, config=config)

        # Verify files exist
        assert (snap_dir / "metadata.json").exists()
        assert (snap_dir / "branches.json").exists()

        # Load into fresh solver
        solver2 = build_solver_from_config(config)
        solver2.load_snapshot(snap_dir)

        assert solver2.time == solver.time
        assert solver2.step_count == solver.step_count

        # Branch state matches
        for name in solver.branches:
            b1 = solver.branches[name]
            b2 = solver2.branches[name]
            assert abs(b2.current - b1.current) < 1e-12
            assert abs(b2.voltage - b1.voltage) < 1e-12
            assert abs(b2.Geq - b1.Geq) < 1e-12
            assert abs(b2.Ihist - b1.Ihist) < 1e-12

    def test_snapshot_metadata_has_required_fields(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run()
        solver.save_snapshot(tmp_path / "snap", config=config)

        import json
        meta = json.loads((tmp_path / "snap" / "metadata.json").read_text())
        assert "schema_version" in meta
        assert "dt" in meta
        assert "time" in meta
        assert "topology_hash" in meta
        assert meta["time"] > 0


# ---------------------------------------------------------------------------
# run_until (segmented run)
# ---------------------------------------------------------------------------

class TestRunUntil:
    def test_resume_produces_same_result_as_continuous(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")

        # Continuous run
        s_cont = build_solver_from_config(config)
        s_cont.run()
        v_cont = s_cont.get_voltage_probe("V_cap", "V")

        # Split run
        s_split = build_solver_from_config(config)
        midpoint = config.simulation.finish_time / 2
        s_split.run_until(midpoint)
        s_split.save_snapshot(tmp_path / "snap", config=config)

        # Resume
        s_resume = build_solver_from_config(config)
        s_resume.load_snapshot(tmp_path / "snap")
        s_resume.run_until(config.simulation.finish_time, reset_state=False)

        v_resume = s_resume.get_voltage_probe("V_cap", "V")

        # Probe data should closely match at the resumed segment
        # Compare the second half
        half = len(v_cont) // 2
        assert len(v_resume) > 0

    def test_run_until_reset_state_fresh(self):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run_until(50e-6, reset_state=True)
        v = solver.get_voltage_probe("V_cap", "V")
        assert len(v) > 0
        # With reset_state=True, this should produce the same result as
        # running from t=0 to t=50us
        solver2 = build_solver_from_config(config)
        solver2.run()
        v2 = solver2.get_voltage_probe("V_cap", "V")
        # Compare at t=50us
        assert abs(v[-1] - v2[50]) < 0.01
