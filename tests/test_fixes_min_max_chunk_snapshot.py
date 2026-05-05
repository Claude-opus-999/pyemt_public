"""Fix validation tests: DB min/max compat, 2D chunk, Bergeron state_dict, resume equivalence."""

import json
from pathlib import Path

import numpy as np
import pytest

from emtp.result_db import ResultDatabase
from emtp.export.waveform_exporter import export_waveforms_npz, read_waveform_chunk
from emtp.config.loader import load_case_config
from emtp.builders.solver_builder import build_solver_from_config


CASES_DIR = Path(__file__).parent.parent / "cases" / "templates"


# =========================================================================
# Fix 1: ResultDB insert_signals min/max compat
# =========================================================================

class TestResultDBSignalMinMax:
    def test_insert_signals_accepts_min_max(self, tmp_path):
        """waveform_metadata.json uses 'min'/'max' — DB must accept both."""
        db = ResultDatabase(tmp_path / "test.sqlite")
        db.insert_run("r1", "case", "running", tmp_path / "r1")
        db.insert_signals("r1", [{
            "name": "V_top", "kind": "voltage", "unit": "kV",
            "length": 10, "min": -1.0, "max": 12.0, "peak_abs": 12.0,
        }])
        sigs = db.get_signals("r1")
        assert len(sigs) == 1
        assert sigs[0]["min_value"] == -1.0
        assert sigs[0]["max_value"] == 12.0
        db.close()

    def test_insert_signals_accepts_min_value_max_value(self, tmp_path):
        db = ResultDatabase(tmp_path / "test.sqlite")
        db.insert_run("r1", "case", "running", tmp_path / "r1")
        db.insert_signals("r1", [{
            "name": "I_gnd", "kind": "current", "unit": "kA",
            "length": 10, "min_value": -0.5, "max_value": 8.5, "peak_abs": 8.5,
        }])
        sigs = db.get_signals("r1")
        assert sigs[0]["min_value"] == -0.5
        assert sigs[0]["max_value"] == 8.5
        db.close()

    def test_integration_roundtrip_via_run_case(self, tmp_path):
        """After run_case with export, DB signals should have non-null min/max."""
        from emtp.case_runner import run_case

        db_path = tmp_path / "test.sqlite"
        out = tmp_path / "run1"
        result = run_case(CASES_DIR / "rc_step.json",
                          output_dir=out, db_path=db_path, run_id="r1")
        assert result.success

        db = ResultDatabase(db_path)
        sigs = db.get_signals("r1")
        assert len(sigs) > 0
        for s in sigs:
            assert s["min_value"] is not None, f"{s['name']}: min_value is null"
            assert s["max_value"] is not None, f"{s['name']}: max_value is null"
        db.close()


# =========================================================================
# Fix 2: read_waveform_chunk 2D slicing
# =========================================================================

class TestWaveformChunk2D:
    def test_1d_chunk_slices_correctly(self, tmp_path):
        wf = {"time_s": np.arange(10, dtype=float),
              "V": np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9])}
        export_waveforms_npz(wf, tmp_path)
        chunk = read_waveform_chunk(tmp_path, "V", start=2, count=4)
        assert chunk["time"] == [2., 3., 4., 5.]
        assert chunk["values"] == [2., 3., 4., 5.]
        assert chunk["shape"] == [4]

    def test_2d_chunk_slices_last_axis(self, tmp_path):
        wf = {"time_s": np.arange(10, dtype=float),
              "V_phase": np.vstack([np.arange(10, dtype=float),
                                     np.arange(10, dtype=float) + 100,
                                     np.arange(10, dtype=float) + 200])}
        export_waveforms_npz(wf, tmp_path)
        chunk = read_waveform_chunk(tmp_path, "V_phase", start=2, count=4)
        assert chunk["time"] == [2., 3., 4., 5.]
        assert chunk["shape"] == [3, 4]
        assert chunk["values"][0] == [2., 3., 4., 5.]
        assert chunk["values"][1] == [102., 103., 104., 105.]
        assert chunk["values"][2] == [202., 203., 204., 205.]

    def test_chunk_rejects_negative_start(self, tmp_path):
        wf = {"time_s": [0., 1.], "V": [0., 1.]}
        export_waveforms_npz(wf, tmp_path)
        with pytest.raises(ValueError, match="start"):
            read_waveform_chunk(tmp_path, "V", start=-1, count=1)

    def test_chunk_rejects_missing_signal(self, tmp_path):
        wf = {"time_s": [0.], "V": [0.]}
        export_waveforms_npz(wf, tmp_path)
        with pytest.raises(KeyError):
            read_waveform_chunk(tmp_path, "nonexistent")


# =========================================================================
# Fix 3: Bergeron snapshot state_dict
# =========================================================================

class TestBergeronStateDict:
    def test_delay_buffer_save_restore(self):
        from collections import deque
        from transmission_line_emtp_v2 import DelayBuffer

        buf = DelayBuffer(delay_steps=5, fractional_delay=0.3)
        buf.push(1.0)
        buf.push(2.0)

        state = buf.get_state_dict()
        assert state["delay_steps"] == 5
        assert abs(state["fractional_delay"] - 0.3) < 1e-12
        assert len(state["buffer"]) > 0

        # Restore into a fresh buffer
        buf2 = DelayBuffer(delay_steps=5, fractional_delay=0.0)
        buf2.set_state_dict(state)
        assert buf2.delay_steps == 5
        assert abs(buf2.fractional_delay - 0.3) < 1e-12
        assert list(buf2.buffer) == [float(x) for x in state["buffer"]]

    def test_delay_buffer_mismatch_rejected(self):
        from transmission_line_emtp_v2 import DelayBuffer

        buf = DelayBuffer(delay_steps=5)
        state = buf.get_state_dict()
        state["delay_steps"] = 999

        buf2 = DelayBuffer(delay_steps=5)
        with pytest.raises(ValueError, match="delay_steps mismatch"):
            buf2.set_state_dict(state)

    def test_bergeron_line_get_state_dict(self):
        from transmission_line_emtp_v2 import BergeronLine

        line = BergeronLine("bl1", 1, 2, Zc=300.0, tau=10e-6)
        line.initialize(dt=1e-6)
        line.update_state(100.0, 50.0)
        line.update_history_sources()

        state = line.get_state_dict()
        assert state["name"] == "bl1"
        assert abs(state["I_hist_k"]) < 1e6  # finite
        assert state["buffer_k_to_m"] is not None
        assert "buffer" in state["buffer_k_to_m"]
        assert state["buffer_m_to_k"] is not None

    def test_bergeron_line_save_restore_state(self):
        from transmission_line_emtp_v2 import BergeronLine

        line1 = BergeronLine("bl1", 1, 2, Zc=300.0, tau=10e-6)
        line1.initialize(dt=1e-6)
        line1.update_state(100.0, 50.0)
        line1.update_history_sources()

        state = line1.get_state_dict()

        line2 = BergeronLine("bl1", 1, 2, Zc=300.0, tau=10e-6)
        line2.initialize(dt=1e-6)
        line2.set_state_dict(state)

        assert abs(line2.I_hist_k - line1.I_hist_k) < 1e-12
        assert abs(line2.I_hist_m - line1.I_hist_m) < 1e-12
        assert abs(line2.V_k - line1.V_k) < 1e-12
        assert abs(line2.V_m - line1.V_m) < 1e-12

    def test_snapshot_includes_full_bergeron_buffers(self, tmp_path):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        solver = build_solver_from_config(config)
        solver.compile_transmission_lines()
        solver.run_until(config.simulation.finish_time / 2)
        solver.save_snapshot(tmp_path / "snap", config=config)

        lines_path = tmp_path / "snap" / "lines.json"
        assert lines_path.exists()
        lines = json.loads(lines_path.read_text(encoding="utf-8"))
        assert lines

        first_line = next(iter(lines.values()))
        assert "buffer_k_to_m" in first_line
        assert "buffer_m_to_k" in first_line
        bk = first_line["buffer_k_to_m"]
        assert bk is not None
        assert "buffer" in bk

    def test_snapshot_support_bergeron_full_when_state_dict_present(self, tmp_path):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        solver = build_solver_from_config(config)
        solver.compile_transmission_lines()
        solver.run_until(config.simulation.finish_time / 2)
        solver.save_snapshot(tmp_path / "snap", config=config)

        support = json.loads((tmp_path / "snap" / "snapshot_support.json").read_text())
        assert support["bergeron"] == "full"


# =========================================================================
# Fix 4: Snapshot resume equivalence
# =========================================================================

class TestSnapshotResumeEquivalence:
    def test_rc_resume_final_value_matches_continuous(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        mid = config.simulation.finish_time / 2

        continuous = build_solver_from_config(config)
        continuous.run()

        split = build_solver_from_config(config)
        split.run_until(mid)
        split.save_snapshot(tmp_path / "snap", config=config)

        resumed = build_solver_from_config(config)
        resumed.load_snapshot(tmp_path / "snap", strict=True)
        resumed.run_until(config.simulation.finish_time, reset_state=False)

        v_cont = continuous.get_voltage_probe("V_cap", "V")
        v_res = resumed.get_voltage_probe("V_cap", "V")
        assert abs(v_cont[-1] - v_res[-1]) < 1e-3

    def test_rl_resume_final_current_matches_continuous(self, tmp_path):
        config = load_case_config(CASES_DIR / "rl_step.json")
        mid = config.simulation.finish_time / 2

        continuous = build_solver_from_config(config)
        continuous.run()

        split = build_solver_from_config(config)
        split.run_until(mid)
        split.save_snapshot(tmp_path / "snap", config=config)

        resumed = build_solver_from_config(config)
        resumed.load_snapshot(tmp_path / "snap", strict=True)
        resumed.run_until(config.simulation.finish_time, reset_state=False)

        i_cont = continuous.get_branch_current("R1", "A")
        i_res = resumed.get_branch_current("R1", "A")
        assert abs(i_cont[-1] - i_res[-1]) < 1e-2

    def test_bergeron_resume_final_voltage_matches_continuous(self, tmp_path):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        mid = config.simulation.finish_time / 2

        # Continuous run — compile_transmission_lines called inside run()
        continuous = build_solver_from_config(config)
        continuous.run()

        # Split run — run_until will compile lines on first call
        split = build_solver_from_config(config)
        split.run_until(mid)
        split.save_snapshot(tmp_path / "snap", config=config)

        # Resume — load_snapshot restores line state, marks _lines_compiled
        resumed = build_solver_from_config(config)
        resumed.load_snapshot(tmp_path / "snap", strict=True)
        resumed.run_until(config.simulation.finish_time, reset_state=False)

        v_cont = continuous.get_voltage_probe("V_recv", "V")
        v_res = resumed.get_voltage_probe("V_recv", "V")
        assert abs(v_cont[-1] - v_res[-1]) < 1e-2

    def test_continue_run_doubles_time(self, tmp_path):
        """run_until should extend the simulation, not replace it."""
        config = load_case_config(CASES_DIR / "rc_step.json")

        solver = build_solver_from_config(config)
        solver.run_until(30e-6)
        v_mid = solver.get_voltage_probe("V_cap", "V")

        solver.run_until(60e-6, reset_state=False)
        v_end = solver.get_voltage_probe("V_cap", "V")

        assert len(v_end) > 0
        # After running to 60us, V_cap should be closer to steady state
        assert v_end[-1] > v_mid[-1]
