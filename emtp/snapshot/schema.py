"""Snapshot metadata and container types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SnapshotMetadata:
    """Immutable metadata written alongside a solver snapshot."""

    schema_version: str
    solver_version: str = "0.2.0"
    case_name: str = ""
    time: float = 0.0
    step_index: int = 0
    dt: float = 0.0
    finish_time: float = 0.0
    config_hash: str = ""
    topology_hash: str = ""
    created_at: str = ""
    notes: str = ""
