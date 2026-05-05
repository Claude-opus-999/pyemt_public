"""Stable hashing helpers for snapshot integrity checks."""

from __future__ import annotations

import hashlib
import json


def stable_json_hash(obj) -> str:
    """Return a deterministic SHA-256 hex digest for a JSON-serializable *obj*."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_config_hash(config) -> str:
    """Compute config hash from a :class:`CaseConfig` or compatible dict."""
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        return stable_json_hash(asdict(config))
    return stable_json_hash(config)


def compute_topology_hash(solver) -> str:
    """Return a hash that changes only when circuit topology changes.

    Covers branch names/types/nodes, line names, and VS names.
    """
    topology = {
        "branches": sorted(
            (
                b.name,
                str(b.element_type),
                b.node_from,
                b.node_to,
            )
            for b in solver.branches.values()
            if hasattr(b, "name")
        ),
        "lines": sorted(
            name for name in getattr(solver, "transmission_lines", {})
        ),
        "transformers": sorted(
            name for name in getattr(solver, "transformers", {})
        ),
        "voltage_sources": sorted(
            vs.name for vs in getattr(solver, "voltage_sources", {}).values()
        ),
    }
    return stable_json_hash(topology)
