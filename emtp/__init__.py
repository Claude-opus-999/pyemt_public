"""EMTP electromagnetic transient simulation solver package."""

from .nodes import NodeBook, NodeIndexer

__all__ = [
    "NodeBook",
    "NodeIndexer",
]


def __getattr__(name):
    """Defer solver import until actually accessed."""
    if name == "EMTPSolver":
        from .solver import EMTPSolver
        return EMTPSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
