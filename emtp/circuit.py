"""CircuitModel — container for all circuit elements owned by EMTPSolver.

Extracted from ``EMTPSolver`` so the data containers (branches,
devices, sources, lines, transformers, nodes) live in a standalone
object that can be passed to ``MNAAssembler``, ``TimeStepper``, and
``ResolveManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from emtp.nodes import NodeBook, NodeIndexer
from emtp.types import Branch, CurrentSource, LineData, VoltageSource


@dataclass
class CircuitModel:
    """All circuit elements managed by the solver.

    The containers are mutable; the dataclass only provides defaults.
    """

    # -- nodes ----------------------------------------------------------------
    indexer: NodeIndexer = field(default_factory=NodeIndexer)
    nodes: NodeBook = field(default_factory=NodeBook)

    # -- two-terminal devices --------------------------------------------------
    devices: List[Any] = field(default_factory=list)
    branches: Dict[str, Branch] = field(default_factory=dict)

    # -- multi-port devices ----------------------------------------------------
    multiport_devices: List[Any] = field(default_factory=list)

    # -- sources ---------------------------------------------------------------
    current_sources: Dict[str, CurrentSource] = field(default_factory=dict)
    voltage_sources: Dict[str, VoltageSource] = field(default_factory=dict)

    # -- transmission lines ----------------------------------------------------
    transmission_lines: Dict[str, Any] = field(default_factory=dict)
    lines: Dict[str, LineData] = field(default_factory=dict)

    # -- transformers ----------------------------------------------------------
    transformers: Dict[str, Any] = field(default_factory=dict)

    # -- node tracking ---------------------------------------------------------
    _node_set: set = field(default_factory=set)
    _vs_node_set: set = field(default_factory=set)

    @property
    def num_nodes(self) -> int:
        return max(self._node_set) if self._node_set else 0

    def update_nodes(self, *node_ids) -> None:
        for n in node_ids:
            if isinstance(n, (list, tuple)):
                for nn in n:
                    if nn > 0:
                        self._node_set.add(int(nn))
                        self.indexer.register(int(nn))
            elif isinstance(n, (int, float)) and n > 0:
                self._node_set.add(int(n))
                self.indexer.register(int(n))

    def is_empty(self) -> bool:
        return (
            not self.branches
            and not self.current_sources
            and not self.voltage_sources
            and not self.transmission_lines
            and not self.transformers
        )
