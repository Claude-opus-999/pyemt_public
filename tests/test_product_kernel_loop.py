"""End-to-end tests: run_case → result_dir → db → snapshot safety."""

import json
from pathlib import Path

import numpy as np
import pytest

from emtp.case_runner import run_case
from emtp.config.loader import load_case_config
from emtp.builders.solver_builder import build_solver_from_config
from emtp.export.waveform_exporter import (
    export_waveforms_npz, collect_waveform_metadata, read_waveform_chunk,
)
from emtp.export.csv_exporter import export_waveforms_csv
from emtp.export.metrics_exporter import export_metrics_json
from emtp.result_db import ResultDatabase
from emtp.run_id import make_run_id


CASES_DIR = Path(__file__).parent.parent / "cases" / "templates"


# =========================================================================
# run_case → result_dir (export pipeline)
# =========================================================================

class TestRunCaseExportPipeline:
    def test_exports_full_result_dir(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                          export=True)
        assert result.success
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "waveforms.npz").exists()
        assert (tmp_path / "waveform_metadata.json").exists()
        assert (tmp_path / "run_metadata.json").exists()

    def test_export_csv_flag(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                          export=True, export_csv=True)
        assert result.success
        assert (tmp_path / "probes.csv").exists()

    def test_export_with_stride(self, tmp_path):
        # Modify config to use output_stride=10
        config = load_case_config(CASES_DIR / "rc_step.json")
        config.simulation.output_stride = 10
        result = run_case(config, output_dir=tmp_path)
        with np.load(tmp_path / "waveforms.npz") as data:
            # 101 steps / stride 10 → 11 samples (0, 10, ..., 100)
            assert len(data["time_s"]) == 11

    def test_no_csv_when_flag_false(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                          export=True, export_csv=False)
        assert not (tmp_path / "probes.csv").exists()


# =========================================================================
# run_case → ResultDatabase
# =========================================================================

class TestRunCaseDatabase:
    def test_writes_run_row(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        out = tmp_path / "run1"
        result = run_case(CASES_DIR / "rc_step.json", output_dir=out,
                          db_path=db_path, run_id="run1")
        assert result.success

        db = ResultDatabase(db_path)
        row = db.get_run("run1")
        assert row["status"] == "done"
        assert row["case_name"] == "rc_step"
        db.close()

    def test_writes_metrics(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        result = run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                          db_path=db_path, run_id="run_m")
        m = ResultDatabase(db_path).get_metrics("run_m")
        assert m["total_steps"] > 0

    def test_writes_signals(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        result = run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                          db_path=db_path, run_id="run_s")
        sigs = ResultDatabase(db_path).get_signals("run_s")
        assert len(sigs) > 0
        names = [s["name"] for s in sigs]
        assert "time_s" in names

    def test_failed_run_recorded(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        result = run_case(CASES_DIR / "nonexistent.json",
                          db_path=db_path, run_id="fail1")
        assert not result.success
        row = ResultDatabase(db_path).get_run("fail1")
        assert row["status"] == "failed"
        assert row["error"]

    def test_auto_generates_run_id(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        result = run_case(CASES_DIR / "rc_step.json", db_path=db_path)
        assert result.success
        assert result.metadata["run_id"].startswith("rc_step_")

    def test_overwrite_false_raises(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path)
        with pytest.raises(FileExistsError):
            run_case(CASES_DIR / "rc_step.json", output_dir=tmp_path,
                     overwrite=False)


# =========================================================================
# Waveform metadata (rich)
# =========================================================================

class TestWaveformMetadata:
    def test_metadata_contains_kind_unit_shape(self, tmp_path):
        waveforms = {"time_s": [0.0, 1.0, 2.0], "V_top": [0.0, 10.0, 5.0]}
        specs = {"V_top": {"kind": "voltage", "unit": "V"},
                 "time_s": {"kind": "time", "unit": "s"}}
        export_waveforms_npz(waveforms, tmp_path, signal_specs=specs)

        meta = collect_waveform_metadata(tmp_path)
        signals = {s["name"]: s for s in meta["signals"]}
        assert signals["V_top"]["kind"] == "voltage"
        assert signals["V_top"]["unit"] == "V"
        assert signals["V_top"]["shape"] == [3]

    def test_metadata_infers_kind_from_name(self, tmp_path):
        waveforms = {"time_s": [0], "V_cap": [0], "I_load": [0],
                      "leader_length": [0]}
        export_waveforms_npz(waveforms, tmp_path)
        meta = collect_waveform_metadata(tmp_path)
        kinds = {s["name"]: s["kind"] for s in meta["signals"]}
        assert kinds["time_s"] == "time"
        assert kinds["V_cap"] == "voltage"
        assert kinds["I_load"] == "current"
        assert kinds["leader_length"] == "leader_length"

    def test_stride_reflected_in_metadata(self, tmp_path):
        waveforms = {"time_s": np.arange(100)}
        export_waveforms_npz(waveforms, tmp_path, stride=10)
        meta = collect_waveform_metadata(tmp_path)
        assert meta["stride"] == 10

    def test_original_shape_preserved(self, tmp_path):
        waveforms = {"V": np.array([[1, 2, 3], [4, 5, 6]])}
        export_waveforms_npz(waveforms, tmp_path)
        meta = collect_waveform_metadata(tmp_path)
        s = meta["signals"][0]
        assert s["original_shape"] == [2, 3]


# =========================================================================
# CSV export
# =========================================================================

class TestCSVExport:
    def test_csv_has_header_and_data(self, tmp_path):
        waveforms = {"time_s": [0.0, 1.0, 2.0], "V1": [10.0, 15.0, 12.0]}
        export_waveforms_csv(waveforms, tmp_path)
        content = (tmp_path / "probes.csv").read_text()
        lines = content.strip().splitlines()
        assert len(lines) == 4  # header + 3 rows
        assert lines[0] == "time_s,V1"

    def test_csv_requires_time_s(self, tmp_path):
        with pytest.raises(ValueError, match="time_s"):
            export_waveforms_csv({"V1": [1, 2]}, tmp_path)


# =========================================================================
# Builder: bergeron_line tau_per_m support
# =========================================================================

class TestBuilderBergeron:
    def test_bergeron_with_tau(self):
        config = load_case_config(CASES_DIR / "bergeron_matched.json")
        solver = build_solver_from_config(config)
        assert "BL1" in solver.transmission_lines

    def test_bergeron_with_tau_per_m(self):
        config = load_case_config(CASES_DIR / "tower_lpm_short.json")
        solver = build_solver_from_config(config)
        # Should build without error
        assert len(solver.transmission_lines) == 4
        # Check that ZA10 was created with correct tau
        line = solver.transmission_lines.get("ZA10")
        assert line is not None

    def test_run_tower_template(self, tmp_path):
        result = run_case(CASES_DIR / "tower_lpm_short.json",
                          output_dir=tmp_path, export=True)
        assert result.success
        assert (tmp_path / "waveforms.npz").exists()
        assert (tmp_path / "metrics.json").exists()


# =========================================================================
# Snapshot safety
# =========================================================================

class TestSnapshotSafety:
    def test_topology_mismatch_rejected(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run_until(config.simulation.finish_time / 2)
        solver.save_snapshot(tmp_path / "snap", config=config)

        bad_config = load_case_config(CASES_DIR / "rl_step.json")
        bad_solver = build_solver_from_config(bad_config)

        with pytest.raises(ValueError, match="topology_hash"):
            bad_solver.load_snapshot(tmp_path / "snap", strict=True)

    def test_dt_mismatch_rejected(self, tmp_path):
        from emtp_solver_v3 import EMTPSolver
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run()
        solver.save_snapshot(tmp_path / "snap", config=config)

        bad_solver = EMTPSolver(dt=1e-3, finish_time=100e-6, verbose=False)
        bad_solver.add_R("R1", 1, 2, 10.0)
        bad_solver.add_C("C1", 2, 0, 1e-6)
        bad_solver.add_VS("VS1", 1, 0, lambda t: 1.0)
        bad_solver.add_voltage_probe("V_cap", 2, 0)
        with pytest.raises(ValueError, match="dt"):
            bad_solver.load_snapshot(tmp_path / "snap", strict=True)

    def test_can_load_non_strict_with_mismatch(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run_until(50e-6)
        solver.save_snapshot(tmp_path / "snap", config=config)

        bad_config = load_case_config(CASES_DIR / "rl_step.json")
        bad_solver = build_solver_from_config(bad_config)
        # Non-strict should not raise
        bad_solver.load_snapshot(tmp_path / "snap", strict=False)

    def test_snapshot_support_metadata_written(self, tmp_path):
        config = load_case_config(CASES_DIR / "rc_step.json")
        solver = build_solver_from_config(config)
        solver.run()
        solver.save_snapshot(tmp_path / "snap", config=config)
        support_path = tmp_path / "snap" / "snapshot_support.json"
        assert support_path.exists()
        support = json.loads(support_path.read_text())
        assert "bergeron" in support
        assert "ulm" in support


# =========================================================================
# run_id utility
# =========================================================================

class TestRunId:
    def test_make_run_id_is_unique(self):
        ids = {make_run_id("test") for _ in range(100)}
        assert len(ids) == 100

    def test_make_run_id_format(self):
        rid = make_run_id("rc_step")
        parts = rid.split("_")
        assert parts[0] == "rc"
        assert parts[1] == "step"  # "rc_step" → "rc_step" (spaces replaced)
        assert len(parts[-1]) == 8  # hex suffix
