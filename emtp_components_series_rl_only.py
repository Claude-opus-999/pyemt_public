"""EMTP 求解器的基础数据结构与枚举。

雷电波形从 :mod:`lightning_waveform` 导入;非线性模型从
:mod:`nonlinear_models_pscad` 导入。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List

# Prefer the original lightning_waveform module when it exists; otherwise use
# the ATP-compatible lightning current generator uploaded with this solver.
try:
    from lightning_waveform import (
        STANDARD_DOUBLE_EXPONENTIAL_PARAMS,
        LightningWaveform,
        create_custom_waveform,
        create_lightning_waveform,
    )
except ImportError:
    from atp_lightning_current_generator_simplified import (
        STANDARD_DOUBLE_EXPONENTIAL_PARAMS,
        LightningWaveform,
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
    )

    def create_lightning_waveform(*args, **kwargs):
        """Compatibility wrapper around create_lightning_current_source()."""
        return create_lightning_current_source(*args, **kwargs)

    def create_custom_waveform(*args, **kwargs):
        """Compatibility wrapper around create_lightning_current_source()."""
        return create_lightning_current_source(*args, **kwargs)



class ElementType(Enum):
    """电路元件类型。"""
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
    """电路支路:基本参数、状态量、隐式梯形等效参数、开关与非线性拓展。"""

    name: str
    element_type: ElementType
    node_from: int
    node_to: int
    value: float  # R[Ω], L[H], C[F]

    # 瞬时状态
    current: float = 0.0
    voltage: float = 0.0
    current_prev: float = 0.0
    voltage_prev: float = 0.0

    # 隐式梯形法诺顿等效
    Geq: float = 0.0
    Ihist: float = 0.0

    # 并联阻尼(用于 L/C 的数值阻尼)
    Rp: float = 0.0
    Geq_damping: float = 0.0

    # 开关
    is_closed: bool = False
    R_closed: float = 1e-6
    R_open: float = 1e9
    t_close: float = -1.0   # <0 表示不动作
    t_open: float = -1.0

    # 非线性模型(NonlinearResistorModel 实例)
    nonlinear_model: Any = None

    # 复合元件扩展参数与状态,用于 SERIES_RL 等
    params: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    # 输出用历史
    current_history: List[float] = field(default_factory=list)
    voltage_history: List[float] = field(default_factory=list)


@dataclass
class CurrentSource:
    """独立电流源。"""
    name: str
    node_from: int
    node_to: int
    current_func: Callable[[float], float]
    current_history: List[float] = field(default_factory=list)

    def current_at(self, t: float) -> float:
        return self.current_func(t)


@dataclass
class LineData:
    """求解器对传输线的轻量引用结构。"""
    name: str
    node_k: int
    node_m: int
    interface: Any  # TransmissionLineInterface

    I_k_history: List[float] = field(default_factory=list)
    I_m_history: List[float] = field(default_factory=list)
    V_k_history: List[float] = field(default_factory=list)
    V_m_history: List[float] = field(default_factory=list)


__all__ = [
    'ElementType', 'Branch', 'CurrentSource', 'LineData',
    # 从 lightning_waveform 重导出
    'LightningWaveform', 'STANDARD_DOUBLE_EXPONENTIAL_PARAMS',
    'create_lightning_waveform', 'create_custom_waveform',

]
