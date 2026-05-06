"""ProbeManager — unified probe registration and metadata.

PR3: Extracts probe registration from EMTPSolver into its own module.
Solver facade methods (add_voltage_probe, get_voltage_probe, etc.)
forward to ProbeManager while keeping backward-compatible behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ProbeSpec:
    """Immutable probe specification."""

    __slots__ = ("name", "kind", "target", "scale", "_hash")

    def __init__(
        self,
        name: str,
        kind: str,  # "voltage_diff" | "branch_current"
        target: tuple,
        scale: float = 1.0,
    ):
        self.name = name
        self.kind = kind
        self.target = target
        self.scale = scale
        self._hash = hash((name, kind, target))

    def __repr__(self):
        return f"ProbeSpec({self.name!r}, kind={self.kind!r}, target={self.target})"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, ProbeSpec):
            return NotImplemented
        return (self.name, self.kind, self.target) == (other.name, other.kind, other.target)


class ProbeManager:
    """Manages probe registration, storage indexing, and result retrieval.

    Delegates array allocation to ResultStore (via the solver).
    This class owns the *which probes exist* and *what they measure*;
    ResultStore owns *where the values are stored*.
    """

    def __init__(self):
        self._probes: Dict[str, ProbeSpec] = {}

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def add_voltage_probe(
        self,
        name: str,
        node_pos: int,
        node_neg: int = 0,
        scale: float = 1.0,
    ) -> ProbeSpec:
        self._ensure_unique(name)
        spec = ProbeSpec(
            name=name,
            kind="voltage_diff",
            target=(int(node_pos), int(node_neg)),
            scale=scale,
        )
        self._probes[str(name)] = spec
        return spec

    def add_branch_current_probe(
        self,
        name: str,
        branch_name: str,
        scale: float = 1.0,
    ) -> ProbeSpec:
        self._ensure_unique(name)
        spec = ProbeSpec(
            name=name,
            kind="branch_current",
            target=(str(branch_name),),
            scale=scale,
        )
        self._probes[str(name)] = spec
        return spec

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    @property
    def voltage_probe_names(self) -> List[str]:
        return [s.name for s in self._probes.values() if s.kind == "voltage_diff"]

    @property
    def branch_current_probe_names(self) -> List[str]:
        return [s.name for s in self._probes.values() if s.kind == "branch_current"]

    @property
    def names(self) -> List[str]:
        return list(self._probes)

    def get_spec(self, name: str) -> ProbeSpec:
        if name not in self._probes:
            raise KeyError(f"Probe not found: {name!r}")
        return self._probes[name]

    def has(self, name: str) -> bool:
        return name in self._probes

    def list_by_kind(self) -> Dict[str, List[str]]:
        return {
            "voltage": self.voltage_probe_names,
            "branch_current": self.branch_current_probe_names,
        }

    def probe_index(self) -> Dict[str, int]:
        """Build {name: column_index} maps for result arrays."""
        return {
            "voltage": {n: i for i, n in enumerate(self.voltage_probe_names)},
            "branch_current": {n: i for i, n in enumerate(self.branch_current_probe_names)},
        }

    # -----------------------------------------------------------------
    # Sampling (called per step)
    # -----------------------------------------------------------------

    def sample_voltage(
        self,
        solution: np.ndarray,
        indexer_lookup,
        branch_state: dict | None = None,
    ) -> Dict[str, float]:
        """Extract voltage probe values from a solution vector."""
        result = {}
        for spec in self._probes.values():
            if spec.kind != "voltage_diff":
                continue
            pos_idx = indexer_lookup(spec.target[0])
            neg_idx = indexer_lookup(spec.target[1]) if spec.target[1] != 0 else None
            v_pos = solution[pos_idx] if pos_idx is not None else 0.0
            v_neg = solution[neg_idx] if neg_idx is not None else 0.0
            result[spec.name] = float((v_pos - v_neg) * spec.scale)
        return result

    def sample_branch_current(
        self,
        solution: np.ndarray,
        branch_state: dict,
        indexer_lookup,
    ) -> Dict[str, float]:
        """Extract branch current values from branch state dict."""
        result = {}
        for spec in self._probes.values():
            if spec.kind != "branch_current":
                continue
            br_name = spec.target[0]
            if br_name in branch_state:
                value = float(branch_state[br_name])
                result[spec.name] = value * spec.scale
            else:
                result[spec.name] = 0.0
        return result

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _ensure_unique(self, name: str) -> None:
        if name in self._probes:
            raise ValueError(f"Duplicate probe name: {name!r}")
