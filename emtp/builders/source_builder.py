"""Dispatch each source dict to the correct solver add_* method."""


def add_source_to_solver(solver, source: dict) -> None:
    """Add a single source to the solver based on its ``kind`` key."""
    kind = source["kind"]

    if kind == "current":
        current = source["current"]
        if isinstance(current, (int, float)):
            current = float(current)
        solver.add_IS(
            source["name"],
            source["node_from"],
            source["node_to"],
            current,
        )
        return

    if kind == "voltage":
        voltage = source["voltage"]
        if isinstance(voltage, (int, float)):
            const = float(voltage)
            voltage = lambda t, v=const: v  # noqa: E731
        solver.add_VS(
            source["name"],
            source["node_pos"],
            source["node_neg"],
            voltage,
        )
        return

    if kind == "standard_double_exponential_current":
        solver.add_standard_double_exponential_current_source(
            name=source["name"],
            node_from=source["node_from"],
            node_to=source["node_to"],
            waveform_type=source["waveform_type"],
            peak=source["peak"],
            PERC=source.get("PERC", 30),
            Tstart=source.get("Tstart", 0.0),
            Tstop=source.get("Tstop"),
        )
        return

    if kind == "lightning_current":
        solver.add_lightning_IS(
            source["name"],
            source["node_from"],
            source["node_to"],
            model=source.get("model", "heidlerf"),
            peak=source["peak"],
            T1=source["T1"],
            T2=source["T2"],
            n=source.get("n", 10.0),
            PERC=source.get("PERC", 30),
            Tstart=source.get("Tstart", 0.0),
            Tstop=source.get("Tstop"),
        )
        return

    raise ValueError(f"Unsupported source kind: {kind!r}")
