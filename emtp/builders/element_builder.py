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
        tau = _resolve_bergeron_tau(element)
        solver.add_bergeron_line(
            element["name"],
            element["node_k"],
            element["node_m"],
            Zc=element["Zc"],
            tau=tau,
        )
        return

    if kind == "ulm_line":
        solver.add_ULM_line(
            name=element["name"],
            nodes_send=element["nodes_send"],
            nodes_recv=element["nodes_recv"],
            length=float(
                element["length"] if "length" in element
                else _resolve_lcp_length(element)
            ),
            generate_fitulm=bool(element.get("generate_fitulm", False)),
            fitulm_path=element.get("fitulm_path", None),
            lcp_spec=element.get("lcp_spec", None),
            cache_dir=element.get("cache_dir", ".lcp_cache"),
            force_recompute=bool(element.get("force_recompute", False)),
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


def _resolve_lcp_length(element: dict) -> float:
    """When lcp_spec is provided, length defaults to lcp_spec.length."""
    lcp_spec = element.get("lcp_spec")
    if lcp_spec is not None:
        return float(lcp_spec.length)
    raise ValueError(
        f"ulm_line {element.get('name')!r} requires 'length' "
        "or 'lcp_spec' with a length attribute"
    )


def _resolve_bergeron_tau(element: dict) -> float:
    """Return propagation delay from either ``tau`` or ``tau_per_m * length_m``."""
    if "tau" in element:
        return float(element["tau"])
    if "tau_per_m" in element and "length_m" in element:
        return float(element["tau_per_m"]) * float(element["length_m"])
    raise ValueError(
        f"bergeron_line {element.get('name')!r} must provide either "
        "'tau' or both 'tau_per_m' and 'length_m'"
    )


def _build_umec_data(element: dict):
    """UMEC transformer data builder."""
    from umec_transformer import UMECTransformerData
    from umec_transformer import create_umec_transformer_3ph_bank

    if "data" in element:
        return UMECTransformerData(**element["data"])

    if "factory" in element:
        factory = element["factory"]
        if factory.get("kind") == "3ph_bank":
            params = {k: v for k, v in factory.items() if k != "kind"}
            return create_umec_transformer_3ph_bank(**params)

    raise ValueError(
        f"umec_transformer {element.get('name')!r} requires either "
        "'data' or 'factory'"
    )
