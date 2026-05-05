"""High-level convenience: load → build → run → bundle.

Usage::

    from emtp.case_runner import run_case

    result = run_case("cases/templates/rc_step.json")
    print(result.metrics)
"""

from __future__ import annotations

import time as _perf_time
from pathlib import Path
from typing import Union

from emtp.config.loader import load_case_config
from emtp.builders.solver_builder import build_solver_from_config
from emtp.result_bundle import ResultBundle


def run_case(
    config_or_path: Union[str, Path, object],
    output_dir: Union[str, Path, None] = None,
) -> ResultBundle:
    """Load, build, simulate and return a :class:`ResultBundle`.

    Parameters
    ----------
    config_or_path:
        Path to a JSON config file, or an already-loaded
        :class:`~emtp.config.CaseConfig`.
    output_dir:
        If given, export results to this directory (reserved for
        waveform/metrics exporter — PR Phase 3).
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_case_config(config_or_path)
    else:
        config = config_or_path

    result_dir = Path(output_dir) if output_dir else None

    try:
        solver = build_solver_from_config(config)

        t0 = _perf_time.perf_counter()
        solver.run()
        elapsed = _perf_time.perf_counter() - t0

        metrics = _collect_metrics(solver, config)
        waveforms = _collect_waveforms(solver, config)

        return ResultBundle(
            case_name=config.case_name,
            success=True,
            metrics=metrics,
            waveforms=waveforms,
            metadata={
                "elapsed_s": elapsed,
                "dt": config.simulation.dt,
                "finish_time": config.simulation.finish_time,
                "n_steps": solver.step_count,
            },
            result_dir=result_dir,
        )

    except Exception as exc:
        return ResultBundle(
            case_name=config.case_name
            if hasattr(config, "case_name")
            else "unknown",
            success=False,
            metrics={},
            waveforms={},
            result_dir=result_dir,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal collectors (expanded in Phase 3)
# ---------------------------------------------------------------------------

def _collect_metrics(solver, config) -> dict:
    """Collect scalar metrics from solver stats and probe peaks."""
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
    # Add probe peak values
    try:
        for name in solver._voltage_probe_names:
            data = solver.get_voltage_probe(name, "V")
            metrics[f"probe_{name}_peak_V"] = float(
                abs(data).max() if len(data) else 0.0
            )
    except Exception:
        pass

    return metrics


def _collect_waveforms(solver, config) -> dict:
    """Collect time-series waveforms from solver result APIs."""
    waveforms = {}

    try:
        waveforms["time_s"] = solver.get_time("s")
    except Exception:
        pass

    # Voltage probes
    try:
        for name in solver._voltage_probe_names:
            waveforms[name] = solver.get_voltage_probe(name, "V")
    except Exception:
        pass

    # Branch current probes
    try:
        for name in solver._branch_current_probe_names:
            waveforms[name] = solver.get_branch_current_probe(name, "A")
    except Exception:
        pass

    return waveforms
