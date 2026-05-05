"""Save solver state to a snapshot directory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import SnapshotMetadata
from .hashing import compute_config_hash, compute_topology_hash


def save_snapshot(
    solver, path, *, config=None, notes: str = "", solver_version: str = "0.2.0",
) -> None:
    """Save the current solver state to *path*.

    Creates the directory if it does not exist and writes::

        metadata.json   — SnapshotMetadata
        branches.json   — per-branch dynamic state
        arrays.npz      — optional large arrays (last solution, etc.)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # -- metadata ------------------------------------------------------------
    config_hash = ""
    if config is not None:
        config_hash = compute_config_hash(config)

    topology_hash = compute_topology_hash(solver)

    meta = SnapshotMetadata(
        schema_version="0.1.0",
        solver_version=solver_version,
        case_name=getattr(config, "case_name", "") if config else "",
        time=solver.time,
        step_index=solver.step_count,
        dt=solver.dt,
        finish_time=solver.finish_time,
        config_hash=config_hash,
        topology_hash=topology_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )

    with (path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_dataclass_to_dict(meta), f, indent=2, ensure_ascii=False)

    # -- branches ------------------------------------------------------------
    branches_data = []
    for br in solver.branches.values():
        branches_data.append({
            "name": br.name,
            "current": float(br.current),
            "voltage": float(br.voltage),
            "current_prev": float(br.current_prev),
            "voltage_prev": float(br.voltage_prev),
            "Geq": float(br.Geq),
            "Ihist": float(br.Ihist),
            "Rp": float(getattr(br, "Rp", 0.0)),
            "Geq_damping": float(getattr(br, "Geq_damping", 0.0)),
            "is_closed": bool(getattr(br, "is_closed", False)),
            "value": float(br.value),
        })

    with (path / "branches.json").open("w", encoding="utf-8") as f:
        json.dump(branches_data, f, indent=2, ensure_ascii=False)

    # -- arrays --------------------------------------------------------------
    arrays = {}
    np.savez_compressed(path / "arrays.npz", **arrays)

    # -- line states ---------------------------------------------------------
    _save_line_states(solver, path)

    # -- LPM states ----------------------------------------------------------
    _save_lpm_states(solver, path)


def _save_line_states(solver, path) -> None:
    """Save per-line dynamic state.

    Lines with a ``get_state_dict`` method (e.g. BergeronLine) export
    full dynamic state including delay buffers.  Other lines fall back
    to saving I_hist only, with ``snapshot_support: partial``.
    """
    line_states = {}
    support_info: dict = {}

    for name, line in solver.transmission_lines.items():
        if hasattr(line, "get_state_dict"):
            line_states[name] = line.get_state_dict()
            support_info[name] = "full"
        else:
            state: dict = {"snapshot_support": "partial"}
            if hasattr(line, "I_hist_k"):
                state["I_hist_k"] = _to_jsonable(line.I_hist_k)
            if hasattr(line, "I_hist_m"):
                state["I_hist_m"] = _to_jsonable(line.I_hist_m)
            line_states[name] = state
            support_info[name] = "partial"

    if line_states:
        with (path / "lines.json").open("w", encoding="utf-8") as f:
            json.dump(line_states, f, indent=2, ensure_ascii=False)

    # snapshot support metadata — detect dynamically
    support_meta = _detect_snapshot_support(solver)
    with (path / "snapshot_support.json").open("w", encoding="utf-8") as f:
        json.dump(support_meta, f, indent=2, ensure_ascii=False)


def _detect_snapshot_support(solver) -> dict:
    """Determine snapshot support level for each model category."""
    support = {
        "branch": "partial",
        "lpm": "partial",
        "umec": "partial",
        "ulm": "unsupported",
        "bergeron": "unsupported",
    }

    lines = getattr(solver, "transmission_lines", {})
    if lines:
        all_full = all(
            hasattr(line, "get_state_dict") and hasattr(line, "set_state_dict")
            for line in lines.values()
        )
        support["bergeron"] = "full" if all_full else "partial"

    return support


def _to_jsonable(value):
    """Convert a numeric or array value to a JSON-serializable form."""
    if isinstance(value, (int, float)):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _save_lpm_states(solver, path) -> None:
    """Save LPM insulator states."""
    lpm_data = {}
    for name, lpm in getattr(solver, "_lpm_elements", {}).items():
        lpm_data[name] = {
            "is_flashed_over": bool(getattr(lpm, "is_flashed_over", False)),
            "R_current": float(getattr(lpm, "R_current", 1e9)),
            "G_current": float(getattr(lpm, "G_current", 1e-9)),
        }
        if hasattr(lpm, "leader_length"):
            lpm_data[name]["leader_length"] = float(lpm.leader_length)

    if lpm_data:
        with (path / "lpm.json").open("w", encoding="utf-8") as f:
            json.dump(lpm_data, f, indent=2, ensure_ascii=False)


def _dataclass_to_dict(obj) -> dict:
    """Convert a dataclass instance to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dataclass_to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
    return obj
