"""Test waveform export, downsampling, chunk read, metrics, and ResultDatabase."""

import json
from pathlib import Path

import numpy as np
import pytest

from emtp.export.waveform_exporter import (
    export_waveforms_npz,
    collect_waveform_metadata,
    read_waveform_chunk,
)
from emtp.export.metrics_exporter import export_metrics_json
from emtp.result_db import ResultDatabase
from emtp.config.loader import load_case_config
from emtp.case_runner import run_case


CASES_DIR = Path(__file__).parent.parent / "cases" / "templates"


# ---------------------------------------------------------------------------
# Waveform export
# ---------------------------------------------------------------------------

class TestWaveformExport:
    @pytest.fixture
    def sample_waveforms(self):
        t = np.linspace(0, 1e-3, 1000)
        return {"time_s": t, "V_cap": np.sin(2 * np.pi * 1000 * t)}

    def test_export_creates_npz_and_metadata(self, tmp_path, sample_waveforms):
        export_waveforms_npz(sample_waveforms, tmp_path)
        assert (tmp_path / "waveforms.npz").exists()
        assert (tmp_path / "waveform_metadata.json").exists()

    def test_export_roundtrip(self, tmp_path, sample_waveforms):
        export_waveforms_npz(sample_waveforms, tmp_path)
        with np.load(tmp_path / "waveforms.npz") as data:
            assert "time_s" in data
            assert len(data["time_s"]) == 1000

    def test_export_with_stride(self, tmp_path, sample_waveforms):
        export_waveforms_npz(sample_waveforms, tmp_path, stride=10)
        with np.load(tmp_path / "waveforms.npz") as data:
            assert len(data["time_s"]) == 100

    def test_metadata_has_signal_info(self, tmp_path, sample_waveforms):
        export_waveforms_npz(sample_waveforms, tmp_path)
        meta = collect_waveform_metadata(tmp_path)
        names = [s["name"] for s in meta["signals"]]
        assert "time_s" in names
        assert "V_cap" in names
        assert meta["stride"] == 1


class TestWaveformChunk:
    def test_chunk_read(self, tmp_path):
        t = np.arange(500, dtype=np.float64) * 1e-6
        v = np.sin(2 * np.pi * 1000 * t)
        export_waveforms_npz({"time_s": t, "V": v}, tmp_path)

        chunk = read_waveform_chunk(tmp_path, "V", start=100, count=50)
        assert chunk["signal"] == "V"
        assert chunk["count"] == 50
        assert len(chunk["values"]) == 50
        assert len(chunk["time"]) == 50

    def test_chunk_clips_to_end(self, tmp_path):
        t = np.arange(100, dtype=np.float64)
        export_waveforms_npz({"time_s": t, "V": t}, tmp_path)

        chunk = read_waveform_chunk(tmp_path, "V", start=80, count=50)
        assert chunk["count"] == 20  # clipped to available data


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

class TestMetricsExport:
    def test_export_metrics_json(self, tmp_path):
        metrics = {"peak_V": 1.05, "flashover": True, "notes": "test"}
        path = export_metrics_json(metrics, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["peak_V"] == 1.05


# ---------------------------------------------------------------------------
# ResultDatabase
# ---------------------------------------------------------------------------

class TestResultDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        db = ResultDatabase(tmp_path / "test.sqlite")
        yield db
        db.close()

    def test_insert_and_get_run(self, db):
        db.insert_run("r1", "rc_step", "running", "/tmp/runs/r1")
        run = db.get_run("r1")
        assert run["case_name"] == "rc_step"
        assert run["status"] == "running"

    def test_update_run_done(self, db):
        db.insert_run("r2", "rc_step", "running")
        db.update_run_done("r2", elapsed_s=0.5)
        run = db.get_run("r2")
        assert run["status"] == "done"
        assert run["elapsed_s"] == 0.5

    def test_update_run_failed(self, db):
        db.insert_run("r3", "bad_case", "running")
        db.update_run_failed("r3", "config error")
        run = db.get_run("r3")
        assert run["status"] == "failed"
        assert "config error" in run["error"]

    def test_insert_and_get_metrics(self, db):
        db.insert_run("r4", "test", "done")
        db.insert_metrics("r4", {"V_peak": 1.05, "I_peak_kA": 2.3})
        metrics = db.get_metrics("r4")
        assert metrics["V_peak"] == 1.05
        assert metrics["I_peak_kA"] == 2.3

    def test_list_recent_runs(self, db):
        db.insert_run("r5", "case_a", "done")
        db.insert_run("r6", "case_b", "done")
        runs = db.list_recent_runs(10)
        assert len(runs) == 2

    def test_insert_signals(self, db):
        db.insert_run("r7", "test", "done")
        signals = [
            {"name": "V_cap", "kind": "voltage", "unit": "V",
             "length": 1000, "min_value": 0.0, "max_value": 1.0, "peak_abs": 1.0},
        ]
        db.insert_signals("r7", signals)
        sigs = db.get_signals("r7")
        assert len(sigs) == 1
        assert sigs[0]["name"] == "V_cap"

    def test_empty_metrics(self, db):
        db.insert_run("r8", "test", "done")
        assert db.get_metrics("r8") == {}

    def test_text_metrics(self, db):
        db.insert_run("r9", "test", "done")
        db.insert_metrics("r9", {"status": "ok"})
        metrics = db.get_metrics("r9")
        assert metrics["status"] == "ok"


# ---------------------------------------------------------------------------
# Integrated: run_case with export
# ---------------------------------------------------------------------------

class TestRunCaseWithExport:
    def test_export_waveforms_from_run_case(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json")
        assert result.success

        export_waveforms_npz(result.waveforms, tmp_path)
        assert (tmp_path / "waveforms.npz").exists()

    def test_export_metrics_from_run_case(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json")
        export_metrics_json(result.metrics, tmp_path)
        assert (tmp_path / "metrics.json").exists()

    def test_run_case_to_database(self, tmp_path):
        result = run_case(CASES_DIR / "rc_step.json")
        assert result.success

        db = ResultDatabase(tmp_path / "runs.sqlite")
        run_id = "rc_test_001"
        db.insert_run(run_id, result.case_name, "running", str(tmp_path))
        db.update_run_done(run_id, result.metadata["elapsed_s"])
        db.insert_metrics(run_id, result.metrics)

        run = db.get_run(run_id)
        assert run["status"] == "done"
        db.close()
