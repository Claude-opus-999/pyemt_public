"""Typed records for objects managed by SimulationRegistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ElementRecord:
    """A two-terminal circuit element registered in the simulation."""

    name: str
    kind: str  # "resistor", "inductor", "capacitor", "switch", "series_rl"
    nodes: tuple[int, ...]
    device: Any | None = None
    metadata: dict | None = field(default_factory=dict)


@dataclass
class SourceRecord:
    """An independent current or voltage source."""

    name: str
    kind: Literal["voltage", "current"]
    nodes: tuple[int, int]  # (pos, neg) for voltage; (from, to) for current
    source: Any
    metadata: dict | None = field(default_factory=dict)


@dataclass
class MultiPortRecord:
    """A multi-port device (Bergeron line, ULM line, UMEC transformer)."""

    name: str
    kind: str  # "bergeron", "ulm", "umec"
    terminals: tuple[int, ...]  # flat tuple of all terminal nodes
    device: Any
    metadata: dict | None = field(default_factory=dict)
