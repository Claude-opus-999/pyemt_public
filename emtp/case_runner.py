"""High-level convenience: load → build → run → bundle → export → db.

Full pipeline::

    result = run_case("cases/templates/rc_step.json",
                       output_dir="runs/rc_001",
                       db_path="runs/history.sqlite")
    print(result.metrics)
"""

from __future__ import annotations

import json
import shutil
import time as _perf_time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

from emtp.config.loader import load_case_config
from emtp.builders.solver_builder import build_solver_from_config
from emtp.result_bundle import ResultBundle
from emtp.run_id import make_run_id
from emtp.export.metrics_exporter import export_metrics_json
from emtp.export.waveform_exporter import export_waveforms_npz
from emtp.export.csv_exporter import export_waveforms_csv
from emtp.result_db import ResultDatabase


def run_case(
    config_or_path: Union[str, Path, object],
    output_dir: Union[str, Path, None] = None,
    *,
    db_path: Union[str, Path, None] = None,
    run_id: Optional[str] = None,
    export: bool = True,
    export_csv: bool = False,
    overwrite: bool = True,
) -> ResultBundle:
    """Load, build, simulate, export and return a :class:`ResultBundle`.

    Parameters
    ----------
    config_or_path:
        Path to a JSON config file, or an already-loaded
        :class:`~emtp.config.CaseConfig`.
    output_dir:
        Directory for results.  When given (and *export* is ``True``),
        writes ``config.json``, ``metrics.json``, ``waveforms.npz``,
        ``waveform_metadata.json``, ``run_metadata.json``, and
        optionally ``probes.csv``.
    db_path:
        Path to a SQLite database (created if absent).  When given,
        the run row, metrics and waveform signal records are inserted
        automatically.
    run_id:
        Unique run identifier.  Auto-generated from *case_name* when
        omitted.
    export:
        When ``True`` (default), materialise result files in *output_dir*.
    export_csv:
        When ``True``, write ``probes.csv`` alongside the NPZ.
    overwrite:
        When ``True`` (default), allow overwriting an existing *output_dir*.
    """
    # -- resolve config -------------------------------------------------------
    config_path: Optional[Path] = None
    if isinstance(config_or_path, (str, Path)):
        config_path = Path(config_or_path)

    run_id = run_id or make_run_id(
        config_or_path.stem if isinstance(config_or_path, Path)
        else getattr(config_or_path, "case_name", "case")
    )
    result_dir = Path(output_dir) if output_dir else None

    db: Optional[ResultDatabase] = None
    if db_path:
        db = ResultDatabase(db_path)

    if result_dir:
        if not overwrite and result_dir.exists():
            raise FileExistsError(
                f"Result directory {result_dir} already exists. "
                "Use overwrite=True to replace it."
            )
        result_dir.mkdir(parents=True, exist_ok=True)

    # -- simulate (config load inside try for error reporting) ---------------
    # Pre-insert the DB run row so failures during config-load / solver-run
    # are still recorded.
    case_name_hint = (
        config_path.stem if config_path
        else getattr(config_or_path, "case_name", "run") if not isinstance(config_or_path, (str, Path))
        else "run"
    )
    if db:
        db.insert_run(
            run_id=run_id,
            case_name=case_name_hint,
            status="running",
            result_dir=result_dir,
            config_path=config_path,
        )

    try:
        if isinstance(config_or_path, (str, Path)):
            config = load_case_config(config_path)  # type: ignore[arg-type]
        else:
            config = config_or_path

        # Update DB row with the true case_name from the loaded config
        if db and config.case_name != case_name_hint:
            db.insert_run(
                run_id=run_id,
                case_name=config.case_name,
                status="running",
                result_dir=result_dir,
                config_path=config_path,
            )

        solver = build_solver_from_config(config)

        t0 = _perf_time.perf_counter()
        solver.run()
        elapsed = _perf_time.perf_counter() - t0

        metrics = _collect_metrics(solver, config)
        waveforms = _collect_waveforms(solver, config)

        metadata = {
            "run_id": run_id,
            "elapsed_s": round(elapsed, 6),
            "dt": config.simulation.dt,
            "finish_time": config.simulation.finish_time,
            "n_steps": solver.step_count,
            "case_name": config.case_name,
        }

        bundle = ResultBundle(
            case_name=config.case_name,
            success=True,
            metrics=metrics,
            waveforms=waveforms,
            metadata=metadata,
            result_dir=result_dir,
        )

        # -- export -----------------------------------------------------------
        if export and result_dir:
            _export_case_outputs(
                bundle=bundle,
                config=config,
                result_dir=result_dir,
                config_path=config_path,
                export_csv=export_csv,
            )

        # -- database: finalise -----------------------------------------------
        if db:
            db.insert_metrics(run_id, metrics)
            signals = _load_exported_signals(result_dir) if result_dir else []
            if signals:
                db.insert_signals(run_id, signals)
            db.update_run_done(run_id, elapsed_s=elapsed)

        return bundle

    except Exception as exc:
        if db:
            db.update_run_failed(run_id, str(exc))

        case_name = "unknown"
        try:
            if "config" in dir() and config is not None:
                case_name = getattr(config, "case_name", "unknown")
        except Exception:
            pass

        return ResultBundle(
            case_name=case_name,
            success=False,
            metrics={},
            waveforms={},
            result_dir=result_dir,
            error=str(exc),
            metadata={"run_id": run_id},
        )

    finally:
        if db:
            db.close()


# =========================================================================
# Internal: export pipeline
# =========================================================================

def _export_case_outputs(
    *,
    bundle: ResultBundle,
    config,
    result_dir: Path,
    config_path: Optional[Path],
    export_csv: bool,
) -> None:
    """Write config, metrics, waveforms, metadata inside *result_dir*."""
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json — copy original or serialise dataclass
    if config_path is not None:
        shutil.copyfile(config_path, result_dir / "config.json")
    else:
        with (result_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    # 2. metrics.json
    export_metrics_json(bundle.metrics, result_dir)

    # 3. waveforms.npz + waveform_metadata.json
    stride = max(1, int(getattr(config.simulation, "output_stride", 1)))
    export_waveforms_npz(
        bundle.waveforms,
        result_dir,
        stride=stride,
        signal_specs=_build_signal_specs(config),
    )

    # 4. run_metadata.json
    with (result_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2, ensure_ascii=False)

    # 5. optional CSV
    if export_csv:
        export_waveforms_csv(
            bundle.waveforms,
            result_dir,
            stride=stride,
        )


def _build_signal_specs(config) -> dict:
    """Derive signal kind/unit hints from config probes."""
    specs: dict = {
        "time_s": {"kind": "time", "unit": "s"},
    }
    for probe in config.probes:
        name = probe.get("name", "")
        kind = probe.get("kind", "")
        unit = probe.get("unit", "")
        if kind == "voltage" and not unit:
            unit = "V"
        elif kind == "branch_current" and not unit:
            unit = "A"
        specs[name] = {"kind": kind, "unit": unit}
    return specs


def _load_exported_signals(result_dir: Path) -> List[dict]:
    meta_path = result_dir / "waveform_metadata.json"
    if not meta_path.exists():
        return []
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("signals", [])


# =========================================================================
# Internal: metrics & waveform collectors
# =========================================================================

def _collect_metrics(solver, config) -> dict:
    stats = solver._stats.copy()
    metrics = {
        "total_steps": stats.get("total_steps", 0),
        "G_rebuilds": stats.get("G_rebuilds", 0),
        "G_cache_hits": stats.get("G_cache_hits", 0),
        "segment_switches": stats.get("segment_switches", 0),
        "segment_resolves": stats.get("segment_resolves", 0),
        "lpm_flashovers": stats.get("lpm_flashovers", 0),
        "lpm_extinctions": stats.get("lpm_extinctions", 0),
        "transformer_saturation_switches": stats.get(
            "transformer_saturation_switches", 0,
        ),
    }
    try:
        for name in solver._voltage_probe_names:
            data = solver.get_voltage_probe(name, "V")
            if len(data):
                metrics[f"probe_{name}_peak_V"] = float(abs(data).max())
    except Exception:
        pass
    return metrics


def _collect_waveforms(solver, config) -> dict:
    waveforms: dict = {}
    try:
        waveforms["time_s"] = solver.get_time("s")
    except Exception:
        pass
    try:
        for name in solver._voltage_probe_names:
            waveforms[name] = solver.get_voltage_probe(name, "V")
    except Exception:
        pass
    try:
        for name in solver._branch_current_probe_names:
            waveforms[name] = solver.get_branch_current_probe(name, "A")
    except Exception:
        pass
    return waveforms
