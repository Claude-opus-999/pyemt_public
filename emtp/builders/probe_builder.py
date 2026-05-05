"""Dispatch each probe dict to the correct solver add_* method."""


def add_probe_to_solver(solver, probe: dict) -> None:
    """Add a single probe to the solver based on its ``kind`` key."""
    kind = probe["kind"]

    if kind == "voltage":
        solver.add_voltage_probe(
            probe["name"],
            probe["node_pos"],
            probe.get("node_neg", 0),
        )
        return

    if kind == "branch_current":
        solver.add_branch_current_probe(
            probe["name"],
            probe["branch"],
        )
        return

    raise ValueError(f"Unsupported probe kind: {kind!r}")
