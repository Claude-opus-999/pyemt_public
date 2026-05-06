"""SimulationRegistry — unified object registry for all simulation state.

PR2: Introduces the registry as a shadow state (dual-write alongside
existing solver containers).  Future PRs will make it the single
source of truth.
"""

from .simulation_registry import SimulationRegistry
from .records import ElementRecord, SourceRecord, MultiPortRecord

__all__ = [
    "SimulationRegistry",
    "ElementRecord",
    "SourceRecord",
    "MultiPortRecord",
]
