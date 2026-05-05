"""Configuration data types for EMTP case definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SimulationOptions:
    """Time-stepping and output options for a single simulation run."""

    dt: float
    finish_time: float
    verbose: bool = False

    record_all_node_voltages: bool = False
    record_line_history: bool = True
    record_branch_history: bool = True
    record_source_history: bool = True

    pre_sample_sources: bool = True
    use_rhs_plan: bool = True

    output_stride: int = 1
    probe_stride: int = 1


@dataclass
class CaseConfig:
    """Complete EMTP simulation case definition.

    Mirrors the JSON config format.  Nodes, elements, sources, probes
    are lists of raw dicts that downstream builders interpret.
    """

    schema_version: str
    case_name: str
    simulation: SimulationOptions

    description: str = ""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    elements: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
