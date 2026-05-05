"""Load a case config from a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .schema import CaseConfig, SimulationOptions
from .defaults import DEFAULT_SIMULATION
from .validator import validate_case_config


def load_case_config(path: Union[str, Path]) -> CaseConfig:
    """Load and validate a case configuration from *path*.

    Returns a fully populated :class:`CaseConfig`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # -- simulation block (merge defaults) -----------------------------------
    sim_raw = raw.get("simulation", {})
    for key, default in DEFAULT_SIMULATION.items():
        sim_raw.setdefault(key, default)

    sim = SimulationOptions(
        dt=sim_raw["dt"],
        finish_time=sim_raw["finish_time"],
        verbose=sim_raw.get("verbose", DEFAULT_SIMULATION["verbose"]),
        record_all_node_voltages=sim_raw.get(
            "record_all_node_voltages",
            DEFAULT_SIMULATION["record_all_node_voltages"],
        ),
        record_line_history=sim_raw.get(
            "record_line_history",
            DEFAULT_SIMULATION["record_line_history"],
        ),
        record_branch_history=sim_raw.get(
            "record_branch_history",
            DEFAULT_SIMULATION["record_branch_history"],
        ),
        record_source_history=sim_raw.get(
            "record_source_history",
            DEFAULT_SIMULATION["record_source_history"],
        ),
        pre_sample_sources=sim_raw.get(
            "pre_sample_sources",
            DEFAULT_SIMULATION["pre_sample_sources"],
        ),
        use_rhs_plan=sim_raw.get(
            "use_rhs_plan",
            DEFAULT_SIMULATION["use_rhs_plan"],
        ),
        output_stride=sim_raw.get(
            "output_stride",
            DEFAULT_SIMULATION["output_stride"],
        ),
        probe_stride=sim_raw.get(
            "probe_stride",
            DEFAULT_SIMULATION["probe_stride"],
        ),
    )

    config = CaseConfig(
        schema_version=raw.get("schema_version", "0.1.0"),
        case_name=raw["case_name"],
        description=raw.get("description", ""),
        simulation=sim,
        nodes=raw.get("nodes", []),
        elements=raw.get("elements", []),
        sources=raw.get("sources", []),
        probes=raw.get("probes", []),
        outputs=raw.get("outputs", {}),
        metadata=raw.get("metadata", {}),
    )

    validate_case_config(config)
    return config
