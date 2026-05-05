"""Shared data types: enums, dataclasses, and validation structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ElementType(Enum):
    """Circuit element type."""
    RESISTOR           = "R"
    INDUCTOR           = "L"
    CAPACITOR          = "C"
    CURRENT_SOURCE     = "IS"
    VOLTAGE_SOURCE     = "VS"
    SWITCH             = "SW"
    NONLINEAR_RESISTOR = "NR"
    TRANSMISSION_LINE  = "TL"
    BERGERON_LINE      = "BL"
    JMARTI_LINE        = "JL"
    SERIES_RL          = "SERIES_RL"


@dataclass
class Branch:
    """Circuit branch: basic parameters, state, trapezoidal Norton equivalent."""

    name: str
    element_type: ElementType
    node_from: int
    node_to: int
    value: float  # R[ohm], L[H], C[F]

    # instantaneous state
    current: float = 0.0
    voltage: float = 0.0
    current_prev: float = 0.0
    voltage_prev: float = 0.0

    # implicit trapezoidal Norton equivalent
    Geq: float = 0.0
    Ihist: float = 0.0

    # parallel damping (for L/C numerical damping)
    Rp: float = 0.0
    Geq_damping: float = 0.0

    # switch
    is_closed: bool = False
    R_closed: float = 1e-6
    R_open: float = 1e9
    t_close: float = -1.0   # <0 means no action
    t_open: float = -1.0

    # nonlinear model (NonlinearResistorModel instance)
    nonlinear_model: Any = None

    # composite element extended params and state (SERIES_RL etc.)
    params: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    # output history
    current_history: List[float] = field(default_factory=list)
    voltage_history: List[float] = field(default_factory=list)


@dataclass
class CurrentSource:
    """Independent current source."""

    name: str
    node_from: int
    node_to: int
    current_func: Callable[[float], float]
    current_history: List[float] = field(default_factory=list)

    def current_at(self, t: float) -> float:
        return self.current_func(t)


@dataclass
class LineData:
    """Lightweight solver reference to a transmission line."""

    name: str
    node_k: int
    node_m: int
    interface: Any  # TransmissionLineInterface

    I_k_history: List[float] = field(default_factory=list)
    I_m_history: List[float] = field(default_factory=list)
    V_k_history: List[float] = field(default_factory=list)
    V_m_history: List[float] = field(default_factory=list)


@dataclass
class VoltageSource:
    """Ideal voltage source (MNA modified nodal analysis).

    In MNA augmented equations the voltage source introduces extra
    constraint rows/columns.  After solving, ``current`` is read
    directly from the augmented component of the MNA solution vector.
    Positive direction: from node_pos through external circuit to node_neg.
    """

    name: str
    node_pos: int
    node_neg: int
    voltage_func: Callable[[float], float]
    current: float = 0.0
    current_history: list = field(default_factory=list)

    def voltage_at(self, t: float) -> float:
        return self.voltage_func(t)


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "error", "warning", "info"
    code: str
    message: str
    related_nodes: list = field(default_factory=list)
    related_branches: list = field(default_factory=list)


@dataclass
class ValidationReport:
    """Collected results from validate_circuit()."""

    issues: list = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]

    def __bool__(self) -> bool:
        return not self.has_errors


@dataclass
class RHSPlan:
    """Pre-compiled index arrays for fast RHS assembly.

    Built once before the main loop; reused every step without
    re-iterating Python device objects.
    """

    # Reactive branch history sources (L, C, SERIES_RL, NONLINEAR_RESISTOR)
    dyn_branch_names: List[str] = field(default_factory=list)
    dyn_branch_nf_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    dyn_branch_nt_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    dyn_branch_type: List[str] = field(default_factory=list)

    # Independent current sources
    isource_names: List[str] = field(default_factory=list)
    isource_nf_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    isource_nt_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # UMEC transformer port indices
    xfmr_names: List[str] = field(default_factory=list)
    xfmr_port_nf_idx: List[np.ndarray] = field(default_factory=list)
    xfmr_port_nt_idx: List[np.ndarray] = field(default_factory=list)


__all__ = [
    "ElementType",
    "Branch",
    "CurrentSource",
    "LineData",
    "VoltageSource",
    "ValidationIssue",
    "ValidationReport",
    "RHSPlan",
]
