r"""无损 Bergeron 常参数行波传输线模型。

去除了原有的损耗特性与外部数据结构解析依赖，目前仅保留无损 Bergeron 模型。

核心关系
--------
    Zc = sqrt(L/C)
    v  = 1/sqrt(L' C')
    tau = l / v

无损 Bergeron 诺顿等效(两端 G_eq 相同):
    I_k = G_eq · V_k + I_hist_k
    I_hist_k(t) = -1 / Z_c · e_m(t - tau)

行波变量 e = V + Z_c · I
延时缓冲区对非整数倍 tau/dt 采用线性插值。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 线路计算工具
# ---------------------------------------------------------------------------

class LineCalculator:
    """无损输电线路参数计算工具。"""

    @staticmethod
    def calculate_characteristic_impedance(L: float, C: float) -> float:
        r"""\( Z_c = \sqrt{L/C} \)。"""
        if L <= 0 or C <= 0:
            logger.warning("L 或 C 非正,返回默认 Zc=1.0 Ω")
            return 1.0
        return float(np.sqrt(L / C))

    @staticmethod
    def calculate_wave_velocity(L_per_m: float, C_per_m: float) -> float:
        r"""\( v = 1/\sqrt{L' C'} \)。"""
        if L_per_m <= 0 or C_per_m <= 0:
            # 当参数异常时，采用接近光速的默认波速
            v = 299792458.0 * 0.98
            logger.warning("L' 或 C' 非正,返回默认波速 %.0f km/s", v / 1e3)
            return v
        return 1.0 / np.sqrt(L_per_m * C_per_m)

    @staticmethod
    def calculate_propagation_delay(length_km: float, velocity: float) -> float:
        r"""\( \tau = l/v \),length 单位 km,velocity 单位 m/s。"""
        if velocity <= 0:
            raise ValueError("波速必须为正值")
        return length_km * 1000.0 / velocity


# ---------------------------------------------------------------------------
# 延时缓冲区
# ---------------------------------------------------------------------------

class DelayBuffer:
    """环形延时缓冲区,支持分数延时的线性插值。"""

    def __init__(self, delay_steps: int, fractional_delay: float = 0.0):
        """
        Parameters
        ----------
        delay_steps : int
            整数延时步数 floor(tau/dt)。
        fractional_delay : float
            小数部分,取值 [0, 1)。
        """
        self.delay_steps = max(delay_steps, 1)
        self.fractional_delay = max(0.0, min(fractional_delay, 0.9999))

        size = self.delay_steps + 2
        self.buffer = deque([0.0] * size, maxlen=size)

    def push(self, value: float) -> None:
        self.buffer.append(value)

    def get_delayed(self) -> float:
        """按 frac 线性插值: (1-f) · v(later) + f · v(earlier)。"""
        if self.fractional_delay < 1e-10:
            return self.buffer[2]
        v_earlier = self.buffer[1]
        v_later = self.buffer[2]
        return ((1.0 - self.fractional_delay) * v_later
                + self.fractional_delay * v_earlier)

    def reset(self, initial_value: float = 0.0) -> None:
        size = self.delay_steps + 2
        self.buffer = deque([initial_value] * size, maxlen=size)

    def get_state_dict(self) -> dict:
        return {
            "delay_steps": int(self.delay_steps),
            "fractional_delay": float(self.fractional_delay),
            "buffer": [float(x) for x in list(self.buffer)],
        }

    def set_state_dict(self, state: dict) -> None:
        expected_steps = int(state["delay_steps"])
        if expected_steps != int(self.delay_steps):
            raise ValueError(
                f"DelayBuffer delay_steps mismatch: "
                f"snapshot={expected_steps}, current={self.delay_steps}"
            )
        self.fractional_delay = float(state["fractional_delay"])
        values = [float(x) for x in state["buffer"]]
        self.buffer = deque(values, maxlen=self.buffer.maxlen)

    @classmethod
    def create_for_delay(cls, tau: float, dt: float) -> 'DelayBuffer':
        if dt <= 0:
            raise ValueError("时间步长必须为正")
        if tau < 0:
            raise ValueError("延时不能为负")
        if tau < dt:
            logger.warning(
                "Propagation delay tau=%g is smaller than dt=%g. "
                "Delay is rounded to one time step.",
                tau,
                dt,
            )
        exact_steps = tau / dt
        delay_steps = int(exact_steps)
        return cls(delay_steps, exact_steps - delay_steps)


def _ensure_delay_buffer(existing, state: dict) -> 'DelayBuffer':
    """Return *existing* if not None, otherwise create a DelayBuffer from state."""
    if existing is not None:
        return existing
    return DelayBuffer(
        delay_steps=int(state["delay_steps"]),
        fractional_delay=float(state["fractional_delay"]),
    )


# ---------------------------------------------------------------------------
# 传输线接口基类
# ---------------------------------------------------------------------------

class TransmissionLineInterface(ABC):
    """输电线路时域接口基类。"""

    def __init__(self, name: str, node_k: int, node_m: int):
        self.name = name
        self.node_k = node_k
        self.node_m = node_m

        self.G_eq: float = 0.0
        self.G_eq_k: float = 0.0
        self.G_eq_m: float = 0.0

        self.I_hist_k: float = 0.0
        self.I_hist_m: float = 0.0

        self.V_k: float = 0.0
        self.V_m: float = 0.0
        self.I_k: float = 0.0
        self.I_m: float = 0.0

        self.V_k_history: List[float] = []
        self.V_m_history: List[float] = []
        self.I_k_history: List[float] = []
        self.I_m_history: List[float] = []

    @abstractmethod
    def initialize(self, dt: float) -> None: ...

    @abstractmethod
    def update_history_sources(self) -> None: ...

    @abstractmethod
    def update_state(
        self,
        V_k: float,
        V_m: float,
        record_history: bool = True,
    ) -> None: ...

    def get_norton_equivalent_k(self) -> Tuple[float, float]:
        return self.G_eq_k, self.I_hist_k

    def get_norton_equivalent_m(self) -> Tuple[float, float]:
        return self.G_eq_m, self.I_hist_m

    def record_history(self) -> None:
        self.V_k_history.append(self.V_k)
        self.V_m_history.append(self.V_m)
        self.I_k_history.append(self.I_k)
        self.I_m_history.append(self.I_m)


# ---------------------------------------------------------------------------
# Bergeron 线路实现
# ---------------------------------------------------------------------------

class BergeronLine(TransmissionLineInterface):
    r"""无损 Bergeron 常参数行波模型。

    诺顿等效:
        I_k = G_eq · V_k + I_hist_k,
        I_hist_k(t) = -e_m(t - tau) / Z_c,
    行波变量 e = V + Z_c · I
    """

    def __init__(self, name: str, node_k: int, node_m: int,
                 Zc: float, tau: float):
        super().__init__(name, node_k, node_m)

        self.Zc = Zc
        self.tau = tau

        self.buffer_k_to_m: Optional[DelayBuffer] = None
        self.buffer_m_to_k: Optional[DelayBuffer] = None
        self.dt: float = 0.0
        self.delay_steps: int = 1

        logger.info("创建无损 Bergeron 线路 %s: Zc=%.2fΩ, τ=%.2fμs",
                    name, self.Zc, self.tau * 1e6)

    def initialize(self, dt: float) -> None:
        self.dt = dt
        self.buffer_k_to_m = DelayBuffer.create_for_delay(self.tau, dt)
        self.buffer_m_to_k = DelayBuffer.create_for_delay(self.tau, dt)
        self.delay_steps = self.buffer_k_to_m.delay_steps
        self.I_hist_k = 0.0
        self.I_hist_m = 0.0
        self.V_k = 0.0
        self.V_m = 0.0
        self.I_k = 0.0
        self.I_m = 0.0
        self.V_k_history.clear()
        self.V_m_history.clear()
        self.I_k_history.clear()
        self.I_m_history.clear()

        # 无损下两侧等效导纳相同，均等于 1/Zc
        self.G_eq = 1.0 / max(self.Zc, 1e-9)
        self.G_eq_k = self.G_eq
        self.G_eq_m = self.G_eq

        logger.debug(
            "无损 Bergeron %s 初始化: dt=%.3fμs, delay=%d+%.4f, G_eq=%.6e",
            self.name, dt * 1e6, self.delay_steps,
            self.buffer_k_to_m.fractional_delay, self.G_eq,
        )

    def update_history_sources(self) -> None:
        if self.buffer_k_to_m is None or self.buffer_m_to_k is None:
            return
        delayed_e_k = self.buffer_k_to_m.get_delayed()  # k 端发出的行波到达 m
        delayed_e_m = self.buffer_m_to_k.get_delayed()  # m 端发出的行波到达 k

        # 无损下 alpha = 1.0
        self.I_hist_k = -delayed_e_m / self.Zc
        self.I_hist_m = -delayed_e_k / self.Zc

    def update_state(
        self,
        V_k: float,
        V_m: float,
        record_history: bool = True,
    ) -> None:
        self.V_k = V_k
        self.V_m = V_m

        self.I_k = self.G_eq_k * V_k + self.I_hist_k
        self.I_m = self.G_eq_m * V_m + self.I_hist_m

        # 行波变量 e = V + Zc · I,存入缓冲区供下一延时使用
        if self.buffer_k_to_m is not None:
            self.buffer_k_to_m.push(V_k + self.Zc * self.I_k)
        if self.buffer_m_to_k is not None:
            self.buffer_m_to_k.push(V_m + self.Zc * self.I_m)

        if record_history:
            self.record_history()

    def get_state_dict(self) -> dict:
        """Return full dynamic state for snapshot save."""
        return {
            "name": self.name,
            "I_hist_k": float(self.I_hist_k),
            "I_hist_m": float(self.I_hist_m),
            "V_k": float(getattr(self, "V_k", 0.0)),
            "V_m": float(getattr(self, "V_m", 0.0)),
            "I_k": float(getattr(self, "I_k", 0.0)),
            "I_m": float(getattr(self, "I_m", 0.0)),
            "buffer_k_to_m": (
                self.buffer_k_to_m.get_state_dict()
                if self.buffer_k_to_m is not None
                else None
            ),
            "buffer_m_to_k": (
                self.buffer_m_to_k.get_state_dict()
                if self.buffer_m_to_k is not None
                else None
            ),
        }

    def set_state_dict(self, state: dict) -> None:
        """Restore full dynamic state from a snapshot."""
        self.I_hist_k = float(state.get("I_hist_k", self.I_hist_k))
        self.I_hist_m = float(state.get("I_hist_m", self.I_hist_m))

        if hasattr(self, "V_k"):
            self.V_k = float(state.get("V_k", self.V_k))
        if hasattr(self, "V_m"):
            self.V_m = float(state.get("V_m", self.V_m))
        if hasattr(self, "I_k"):
            self.I_k = float(state.get("I_k", self.I_k))
        if hasattr(self, "I_m"):
            self.I_m = float(state.get("I_m", self.I_m))

        if state.get("buffer_k_to_m") is not None:
            self.buffer_k_to_m = _ensure_delay_buffer(
                self.buffer_k_to_m, state["buffer_k_to_m"],
            )
            self.buffer_k_to_m.set_state_dict(state["buffer_k_to_m"])

        if state.get("buffer_m_to_k") is not None:
            self.buffer_m_to_k = _ensure_delay_buffer(
                self.buffer_m_to_k, state["buffer_m_to_k"],
            )
            self.buffer_m_to_k.set_state_dict(state["buffer_m_to_k"])

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model_type': 'Bergeron (Lossless)',
            'node_k': self.node_k,
            'node_m': self.node_m,
            'Zc': self.Zc,
            'tau': self.tau,
            'G_eq': self.G_eq,
            'delay_steps': self.delay_steps,
        }


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

class TransmissionLineFactory:
    """传输线工厂。"""

    @staticmethod
    def create_from_zc_tau(
            name: str, node_k: int, node_m: int,
            Zc: float, tau_per_m: float, length_m: float,
    ) -> BergeronLine:
        """PSCAD "Surge Impedance + Travel Time per unit length" 格式。"""
        if Zc <= 0:
            raise ValueError("波阻抗 Zc 必须为正值")
        if tau_per_m <= 0:
            raise ValueError("tau_per_m 必须为正值")
        if length_m <= 0:
            raise ValueError("length_m 必须为正值")

        tau = tau_per_m * length_m
        return BergeronLine(name, node_k, node_m, Zc, tau)


__all__ = [
    'LineCalculator',
    'TransmissionLineInterface', 'BergeronLine',
    'TransmissionLineFactory',
    'DelayBuffer',
]
