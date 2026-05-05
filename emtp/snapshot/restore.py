"""Restore solver state from a snapshot directory."""

from __future__ import annotations

import json
from pathlib import Path


def load_snapshot_into_solver(solver, path, *, strict: bool = True) -> None:
    """Restore dynamic state from *path* into an already-configured *solver*.

    The solver must already have the correct topology (branches, lines,
    transformers).  This function only restores dynamic state: branch
    currents/voltages, history sources, LPM state, line state, etc.

    Parameters
    ----------
    solver:
        Pre-configured :class:`EMTPSolver` with matching topology.
    path:
        Snapshot directory (contains metadata.json, branches.json, ...).
    strict:
        If ``True``, raise when a snapshot entry references an unknown
        branch/line/transformer.
    """
    path = Path(path)

    # -- metadata -----------------------------------------------------------
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    if strict:
        if abs(meta.get("dt", 0) - solver.dt) > 1e-30:
            raise ValueError("Snapshot dt does not match solver dt")

    # -- branches -----------------------------------------------------------
    if (path / "branches.json").exists():
        branch_states = json.loads((path / "branches.json").read_text(encoding="utf-8"))
        by_name = {b.name: b for b in solver.branches.values()}

        for state in branch_states:
            name = state["name"]
            if name not in by_name:
                if strict:
                    raise ValueError(f"Snapshot branch {name!r} not found in solver")
                continue
            br = by_name[name]
            br.current = state.get("current", 0.0)
            br.voltage = state.get("voltage", 0.0)
            br.current_prev = state.get("current_prev", 0.0)
            br.voltage_prev = state.get("voltage_prev", 0.0)
            br.Geq = state.get("Geq", 0.0)
            br.Ihist = state.get("Ihist", 0.0)
            br.Rp = state.get("Rp", 0.0)
            br.Geq_damping = state.get("Geq_damping", 0.0)
            br.is_closed = state.get("is_closed", False)
            br.value = state.get("value", br.value)

    # -- line states --------------------------------------------------------
    if (path / "lines.json").exists():
        line_states = json.loads((path / "lines.json").read_text(encoding="utf-8"))
        for name, state in line_states.items():
            if name not in solver.transmission_lines:
                if strict:
                    raise ValueError(f"Snapshot line {name!r} not found in solver")
                continue
            line = solver.transmission_lines[name]
            if "I_hist_k" in state and hasattr(line, "I_hist_k"):
                line.I_hist_k = state["I_hist_k"]
            if "I_hist_m" in state and hasattr(line, "I_hist_m"):
                line.I_hist_m = state["I_hist_m"]

    # -- LPM states ---------------------------------------------------------
    if (path / "lpm.json").exists():
        lpm_data = json.loads((path / "lpm.json").read_text(encoding="utf-8"))
        for name, state in lpm_data.items():
            if name not in solver._lpm_elements:
                if strict:
                    raise ValueError(f"Snapshot LPM {name!r} not found in solver")
                continue
            lpm = solver._lpm_elements[name]
            if hasattr(lpm, "is_flashed_over"):
                lpm.is_flashed_over = state.get("is_flashed_over", False)
            if hasattr(lpm, "R_current"):
                lpm.R_current = state.get("R_current", 1e9)
            if hasattr(lpm, "G_current"):
                lpm.G_current = state.get("G_current", 1e-9)
            if hasattr(lpm, "leader_length") and "leader_length" in state:
                lpm.leader_length = state.get("leader_length", 0.0)

    # -- sync solver metadata -----------------------------------------------
    solver.time = meta.get("time", 0.0)
    solver.step_count = meta.get("step_index", 0)

    # Force MNA rebuild on next solve
    solver._reset_caches()
    solver.mark_topology_changed("snapshot restore")
