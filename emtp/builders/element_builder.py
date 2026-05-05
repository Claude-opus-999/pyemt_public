"""Dispatch each element dict to the correct solver add_* method."""


def add_element_to_solver(solver, element: dict) -> None:
    """Add a single element to the solver based on its ``kind`` key."""
    kind = element["kind"]

    if kind == "resistor":
        solver.add_R(
            element["name"],
            element["node_from"],
            element["node_to"],
            element["R"],
        )
        return

    if kind == "inductor":
        solver.add_L(
            element["name"],
            element["node_from"],
            element["node_to"],
            element["L"],
            Rp=element.get("Rp", 0.0),
        )
        return

    if kind == "capacitor":
        solver.add_C(
            element["name"],
            element["node_from"],
            element["node_to"],
            element["C"],
            Rp=element.get("Rp", 0.0),
        )
        return

    if kind == "series_rl":
        solver.add_series_RL(
            element["name"],
            element["node_from"],
            element["node_to"],
            R=element["R"],
            L=element["L"],
        )
        return

    if kind == "switch":
        solver.add_SW(
            element["name"],
            element["node_from"],
            element["node_to"],
            t_close=element.get("t_close", -1.0),
            t_open=element.get("t_open", -1.0),
            R_closed=element.get("R_closed", 1e-6),
            R_open=element.get("R_open", 1e9),
            initially_closed=element.get("initially_closed", False),
        )
        return

    if kind == "bergeron_line":
        solver.add_bergeron_line(
            element["name"],
            element["node_k"],
            element["node_m"],
            Zc=element["Zc"],
            tau=element["tau"],
        )
        return

    if kind == "lpm_insulator":
        solver.add_insulator_LPM(
            element["name"],
            element["node_from"],
            element["node_to"],
            gap_length=element["gap_length"],
            k=element.get("k", 1.0e-6),
            E0=element.get("E0", 600.0),
            R_arc=element.get("R_arc", 1.0),
            R_open=element.get("R_open", 1e9),
            altitude_m=element.get("altitude_m", 0.0),
        )
        return

    if kind == "umec_transformer":
        from umec_transformer import UMECTransformer
        data = _build_umec_data(element)
        solver.add_UMEC_transformer(element["name"], data)
        return

    raise ValueError(f"Unsupported element kind: {kind!r}")


def _build_umec_data(element: dict):
    """Minimal UMEC data builder — extended later with full config support."""
    from umec_transformer import UMECTransformerData
    return UMECTransformerData(**element.get("data", {}))
