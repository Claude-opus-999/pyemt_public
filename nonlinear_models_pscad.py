"""非线性电阻模型:PSCAD 分段线性法 + CIGRE 先导发展法。

包含三类元件:

- :class:`NonlinearResistorModel` : 非线性电阻抽象基类
- :class:`SegmentedMOAResistor`    : PSCAD 风格分段线性避雷器
- :class:`MOAResistor`             : 连续解析 MOA 模型(I = I0·(V/Vref)^α)
- :class:`InsulatorFlashoverLPM`   : CIGRE 先导发展法绝缘子闪络开关

:class:`SegmentedSolverHelper` 提供多元件注册与段切换检测,供求解器集成。

参考
----
- CIGRE WG 33.01, TB 63 (1991).
- A. Pigini et al., IEEE Trans. Power Del., 4(2), 1989.
- CIGRE WG C4.23, TB 839 (2021).
- 郝艳捧等, 中国电机工程学报, 32(34):158-164, 2012.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 非线性电阻基类
# ---------------------------------------------------------------------------

class NonlinearResistorModel(ABC):
    """非线性电阻抽象基类。"""

    MIN_CONDUCTANCE: float = 1e-12
    MAX_CONDUCTANCE: float = 1e6
    VOLTAGE_EPSILON: float = 1e-12

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def get_current(self, voltage: float) -> float:
        """给定电压返回电流。"""

    @abstractmethod
    def get_conductance(self, voltage: float) -> float:
        """给定电压返回微分电导 dI/dV。"""

    def linearize_at(
        self, voltage: float, current: Optional[float] = None,
    ) -> Tuple[float, float]:
        """工作点线性化,返回 (g_eq, i_eq)。"""
        if current is None:
            current = self.get_current(voltage)
        g_eq = self.get_conductance(voltage)
        g_eq = max(self.MIN_CONDUCTANCE, min(self.MAX_CONDUCTANCE, g_eq))
        i_eq = current - g_eq * voltage
        return g_eq, i_eq

    def get_info(self) -> Dict[str, Any]:
        return {'name': self.name, 'type': self.__class__.__name__}


# ---------------------------------------------------------------------------
# 分段线性段数据结构
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """单个线性段:I = g·|V| + I_offset 在 V_min ≤ |V| < V_max 内有效。"""

    index: int
    V_min: float
    V_max: float
    g: float
    I_offset: float

    def contains(self, voltage: float) -> bool:
        v = abs(voltage)
        return self.V_min <= v < self.V_max

    def get_current(self, voltage: float) -> float:
        if voltage == 0.0:
            return 0.0
        return np.sign(voltage) * (self.g * abs(voltage) + self.I_offset)

    def get_norton_equivalent(
        self, voltage: Optional[float] = None,
    ) -> Tuple[float, float]:
        """返回 (g, i_eq),按电压符号处理对称性。"""
        if voltage is None or voltage >= 0:
            return self.g, self.I_offset
        return self.g, -self.I_offset

    def __repr__(self) -> str:
        return (f"Segment[{self.index}]: V∈[{self.V_min:.2e}, {self.V_max:.2e}), "
                f"g={self.g:.4e} S, I0={self.I_offset:.4e} A")


# ---------------------------------------------------------------------------
# PSCAD 分段线性避雷器
# ---------------------------------------------------------------------------

class SegmentedMOAResistor(NonlinearResistorModel):
    """PSCAD 风格分段线性避雷器。

    预先将 V-I 曲线离散为多个线性段,追踪当前工作段号;段切换时仅此元件的
    Norton 等效改变,求解器可做增量更新。
    """

    def __init__(self, name: str, segments: List[Segment]):
        super().__init__(name)
        self.segments = sorted(segments, key=lambda s: s.V_min)
        self.num_segments = len(self.segments)

        self._current_segment_index: int = 0
        self._segment_changed: bool = False
        self._prev_voltage: float = 0.0

        self._validate_segments()
        self.Vref = self.segments[-1].V_min if self.segments else 0.0

    def _validate_segments(self) -> None:
        if not self.segments:
            raise ValueError("segments 列表不能为空")
        for i in range(len(self.segments) - 1):
            s1, s2 = self.segments[i], self.segments[i + 1]
            tol = 1e-6 * max(s1.V_max, 1.0)
            if abs(s1.V_max - s2.V_min) > tol:
                raise ValueError(
                    f"段 {i} 与段 {i + 1} 不连续: "
                    f"V_max={s1.V_max:.6e} vs V_min={s2.V_min:.6e}"
                )

    # ---- 核心接口 -------------------------------------------------------

    def get_current(self, voltage: float) -> float:
        """使用当前段计算电流(无段切换检查)。"""
        return self.segments[self._current_segment_index].get_current(voltage)

    def get_current_exact(self, voltage: float) -> float:
        """自动查找正确段计算电流。"""
        return self.segments[self._find_segment(abs(voltage))].get_current(voltage)

    def get_conductance(self, voltage: Optional[float] = None) -> float:
        return self.segments[self._current_segment_index].g

    def get_current_segment(self) -> Segment:
        return self.segments[self._current_segment_index]

    @property
    def current_segment_index(self) -> int:
        return self._current_segment_index

    @property
    def segment_changed(self) -> bool:
        return self._segment_changed

    # ---- 分段线性法 -----------------------------------------------------

    def check_segment(self, voltage: float) -> Tuple[bool, int]:
        """判断电压是否仍在当前段内,若否返回正确的段号。"""
        current_seg = self.segments[self._current_segment_index]
        if current_seg.contains(voltage):
            return False, self._current_segment_index
        return True, self._find_segment(abs(voltage))

    def _find_segment(self, v_abs: float) -> int:
        """二分查找电压所在段。"""
        if v_abs < self.segments[0].V_min:
            return 0
        if v_abs >= self.segments[-1].V_max:
            return self.num_segments - 1

        left, right = 0, self.num_segments - 1
        while left <= right:
            mid = (left + right) // 2
            seg = self.segments[mid]
            if seg.V_min <= v_abs < seg.V_max:
                return mid
            if v_abs < seg.V_min:
                right = mid - 1
            else:
                left = mid + 1

        # 理论上不可达,保守回退线性扫描
        for i, seg in enumerate(self.segments):
            if seg.contains(v_abs):
                return i
        return self.num_segments - 1

    def update_segment(self, voltage: float) -> bool:
        """根据电压更新当前段号,返回是否发生段切换。"""
        changed, new_index = self.check_segment(voltage)
        self._segment_changed = changed
        self._prev_voltage = voltage
        if changed:
            self._current_segment_index = new_index
        return changed

    def get_norton_equivalent(
        self, voltage: Optional[float] = None,
    ) -> Tuple[float, float]:
        """返回当前段的 (g_eq, i_eq)。"""
        return self.segments[self._current_segment_index].get_norton_equivalent(
            voltage
        )

    def reset_to_segment(self, index: int = 0) -> None:
        if 0 <= index < self.num_segments:
            self._current_segment_index = index
            self._segment_changed = False
            self._prev_voltage = 0.0

    # ---- 工厂方法 -------------------------------------------------------

    @classmethod
    def from_breakpoints(
        cls, name: str,
        breakpoints: List[Tuple[float, float]],
        add_zero_point: bool = True,
    ) -> 'SegmentedMOAResistor':
        """由 V-I 断点列表 [(V, I), ...] 创建。"""
        bp = sorted(set(breakpoints), key=lambda x: x[0])
        if add_zero_point and (not bp or bp[0][0] > 1e-10):
            bp.insert(0, (0.0, 0.0))
        if len(bp) < 2:
            raise ValueError("至少需要 2 个断点")

        segments: List[Segment] = []
        for i in range(len(bp) - 1):
            V1, I1 = bp[i]
            V2, I2 = bp[i + 1]
            dV = V2 - V1
            g = 1e6 if abs(dV) < 1e-12 else (I2 - I1) / dV
            I_offset = I1 - g * V1
            segments.append(Segment(
                index=i, V_min=V1, V_max=V2,
                g=g, I_offset=I_offset,
            ))

        # 最后一段延伸至无穷,保持斜率与电流连续
        last = segments[-1]
        V_b = last.V_max
        I_b = last.g * V_b + last.I_offset
        segments.append(Segment(
            index=len(segments), V_min=V_b, V_max=float('inf'),
            g=last.g, I_offset=I_b - last.g * V_b,
        ))

        return cls(name, segments)

    @classmethod
    def from_moa_params(
        cls, name: str, Vref: float,
        alpha: float = 30.0, I0: float = 1e-3, n_segments: int = 20,
        V_range: Optional[Tuple[float, float]] = None,
        I_range: Optional[Tuple[float, float]] = None,
    ) -> 'SegmentedMOAResistor':
        """由 MOA 参数 I = I0·(V/Vref)^α 采样生成分段。

        电流对数空间均匀采样 + 关键电压点加密。
        """
        if V_range is None:
            V_range = (0.0, 2.0 * Vref)

        if I_range is None:
            I_min, I_max = I0 * 1e-12, I0 * 1e9
        else:
            I_min, I_max = I_range

        breakpoints: List[Tuple[float, float]] = [(0.0, 0.0)]

        for I in np.logspace(np.log10(I_min), np.log10(I_max), n_segments + 1):
            if I > 0:
                V = Vref * (I / I0) ** (1.0 / alpha)
                breakpoints.append((V, I))

        # 关键电压点加密
        for ratio in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95,
                      1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5):
            V = ratio * Vref
            if not any(abs(bp[0] - V) < Vref * 0.005 for bp in breakpoints):
                breakpoints.append((V, I0 * ratio ** alpha))

        breakpoints = sorted(set(breakpoints), key=lambda x: x[0])
        return cls.from_breakpoints(name, breakpoints, add_zero_point=False)

    @classmethod
    def from_file(
        cls, name: str, file_path: str,
        rated_voltage: float = 1.0, voltage_is_pu: bool = True,
    ) -> 'SegmentedMOAResistor':
        """从 PSCAD 风格文本文件读取 V-I 数据。

        文件格式:每行 ``current voltage``,以 ``#`` / ``//`` 开头为注释,
        遇 ``ENDFILE`` 结束。
        """
        breakpoints: List[Tuple[float, float]] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                if 'ENDFILE' in line.upper():
                    break
                parts = line.replace('/', ' ').replace(',', ' ').split()
                if len(parts) < 2:
                    continue
                try:
                    current = float(parts[0])
                    voltage = float(parts[1])
                except ValueError:
                    continue
                if voltage_is_pu:
                    voltage *= rated_voltage
                breakpoints.append((voltage, current))

        if len(breakpoints) < 2:
            raise ValueError(f"文件 {file_path} 中数据点不足")
        return cls.from_breakpoints(name, breakpoints)

    # ---- 辅助 -----------------------------------------------------------

    def get_all_breakpoints(self) -> List[Tuple[float, float]]:
        """返回全部段的端点 (V, I)。"""
        breakpoints = []
        for seg in self.segments:
            V = seg.V_min
            breakpoints.append((V, seg.g * V + seg.I_offset))
        if self.segments:
            last = self.segments[-1]
            if last.V_max < float('inf'):
                V = last.V_max
                breakpoints.append((V, last.g * V + last.I_offset))
        return breakpoints

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            'num_segments': self.num_segments,
            'Vref': self.Vref,
            'current_segment': self._current_segment_index,
            'segments': [str(s) for s in self.segments[:5]],
        })
        return info

    def print_segments(self) -> None:
        print(f"{self.name} 分段信息 ({self.num_segments} 段):")
        print("-" * 70)
        for seg in self.segments:
            print(f"  {seg}")
        print("-" * 70)

    def __repr__(self) -> str:
        return f"<SegmentedMOAResistor: {self.name}, {self.num_segments} segments>"





# ---------------------------------------------------------------------------
# 分段线性求解辅助器
# ---------------------------------------------------------------------------

class SegmentedSolverHelper:
    """管理多个分段线性元件,供求解器批量检测段切换。"""

    def __init__(self):
        self.elements: Dict[str, SegmentedMOAResistor] = {}
        self._resolve_count: int = 0
        self._total_steps: int = 0

    def register(self, name: str, model: SegmentedMOAResistor) -> None:
        if not isinstance(model, SegmentedMOAResistor):
            raise TypeError(f"{name} 必须是 SegmentedMOAResistor")
        self.elements[name] = model

    def unregister(self, name: str) -> None:
        self.elements.pop(name, None)

    def check_all_segments(
        self, voltages: Dict[str, float],
    ) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
        """检查所有元件,返回 (是否需要重解, {name: (g_new, i_eq_new)})。"""
        updates: Dict[str, Tuple[float, float]] = {}
        need_resolve = False
        for name, model in self.elements.items():
            v = voltages.get(name, 0.0)
            if model.update_segment(v):
                need_resolve = True
                updates[name] = model.get_norton_equivalent(v)
        if need_resolve:
            self._resolve_count += 1
        return need_resolve, updates

    def get_all_norton_equivalents(
        self, voltages: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        return {
            name: model.get_norton_equivalent(
                voltages.get(name, 0.0) if voltages else None
            )
            for name, model in self.elements.items()
        }

    def reset_all(self) -> None:
        for model in self.elements.values():
            model.reset_to_segment(0)
        self._resolve_count = 0
        self._total_steps = 0

    def step_completed(self) -> None:
        self._total_steps += 1

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_steps': self._total_steps,
            'resolve_count': self._resolve_count,
            'resolve_ratio': self._resolve_count / max(1, self._total_steps),
            'elements': list(self.elements),
        }

    def print_statistics(self) -> None:
        s = self.get_statistics()
        print(f"分段线性求解统计: 步数={s['total_steps']}, "
              f"重解次数={s['resolve_count']}, "
              f"重解比例={s['resolve_ratio']*100:.1f}%")



# ---------------------------------------------------------------------------
# CIGRE 先导发展法 (LPM)
# ---------------------------------------------------------------------------

class LPMInsulatorType(Enum):
    """CIGRE TB 63 / TB 839 表 2-28 推荐参数 (k [m²/(kV²·s)], E0 [kV/m])。"""

    AIR_GAP_POST_POS      = (0.8e-6, 600, "气体间隙/柱状绝缘子 (正极性)")
    LONG_ROD_COMP_NEG     = (1.0e-6, 670, "长棒复合绝缘子 (负极性)")
    CAP_PIN_PORCELAIN_POS = (1.2e-6, 520, "帽针型瓷绝缘子 (正极性)")
    GLASS_NEG             = (1.3e-6, 600, "玻璃绝缘子 (负极性)")
    PIGINI_POSITIVE       = (0.8e-6, 600, "Pigini 正极性模型")
    PIGINI_NEGATIVE       = (1.0e-6, 670, "Pigini 负极性模型")
    MOTOYAMA              = (1.2e-6, 520, "Motoyama 模型")

    @property
    def k(self) -> float:
        return self.value[0]

    @property
    def E0(self) -> float:
        return self.value[1]

    @property
    def description(self) -> str:
        return self.value[2]


@dataclass
class LPMConfig:
    """先导发展法参数配置。

    CIGRE 公式: v(t) = k · u(t) · [u(t)/(d - l) - E0],
    起始条件 u/(d-l) > E0,闪络条件 l ≥ d。
    """

    gap_length: float            # d, 间隙长度 (m)
    k: float = 1.0e-6            # 速度系数 m²/(kV²·s)
    E0: float = 600.0            # 临界场强 kV/m

    include_predischarge: bool = False
    q_leader: float = 50e-6      # 先导单位长度电荷 (C/m)
    L_channel: float = 1e-6      # 先导单位长度电感 (H/m)

    R_open: float = 1e9
    R_arc: float = 1.0

    allow_extinction: bool = False
    extinction_current: float = 0.0
    extinction_delay: float = 0.0

    altitude_m: float = 0.0      # 海拔气压修正

    def get_altitude_correction(self) -> float:
        """高海拔修正系数 δ = exp(-altitude / 8150),用于 E0_eff = E0·δ。"""
        if self.altitude_m <= 0:
            return 1.0
        return float(np.exp(-self.altitude_m / 8150.0))


class InsulatorFlashoverLPM:
    """CIGRE 先导发展法绝缘子闪络模型,作为电压控制型开关集成到 EMTP。

    物理过程:电晕起始 → 流注发展 → 先导传播 → 闪络。开关行为:闪络前
    R_open(≈开路),闪络后 R_arc(电弧通道)。
    """

    def __init__(self, name: str, config: LPMConfig):
        self.name = name
        self.config = config

        self.E0_eff = config.E0 * config.get_altitude_correction()

        # 先导状态
        self._leader_length: float = 0.0
        self._leader_velocity: float = 0.0
        self._is_flashed_over: bool = False
        self._is_leader_active: bool = False
        self._flashover_time: float = -1.0
        self._inception_time: float = -1.0

        # 预放电电流
        self._predischarge_current: float = 0.0

        # 电弧熄灭
        self._arc_extinction_time: float = -1.0
        self._below_extinction_since: float = -1.0

        # 等效开关参数
        self.R_current = config.R_open
        self.G_current = 1.0 / config.R_open

        # 历史
        self.leader_length_history: List[float] = []
        self.leader_velocity_history: List[float] = []
        self.voltage_history: List[float] = []
        self.state_history: List[int] = []
        self.predischarge_current_history: List[float] = []

        # 统计
        self._flashover_count: int = 0
        self._peak_leader_velocity: float = 0.0
        self._peak_voltage_kV: float = 0.0

    # ---- 核心更新 -------------------------------------------------------

    def update(
        self, voltage_V: float, dt: float,
        current_A: float = 0.0, time: float = 0.0,
    ) -> bool:
        """单步更新先导发展,返回开关状态是否改变(需重解矩阵时)。

        算法:
            1. u_eff = |u| - 预放电压降(可选)
            2. 若 u_eff/(d-l) > E0,则 v = k·u_eff·(E_gap - E0),l += v·dt
            3. l ≥ d → 闪络,开关闭合
            4. 已闪络时可选检测电弧熄灭
        """
        d = self.config.gap_length
        k = self.config.k
        E0 = self.E0_eff

        u_kV = abs(voltage_V) / 1000.0
        self._peak_voltage_kV = max(self._peak_voltage_kV, u_kV)

        old_state = self._is_flashed_over

        # 已闪络:仅检测熄灭
        if self._is_flashed_over:
            state_changed = (self._check_arc_extinction(current_A, time)
                             if self.config.allow_extinction else False)
            self._record_history(u_kV, time)
            return state_changed

        # 未闪络:先导发展计算
        u_eff_kV = u_kV
        if self.config.include_predischarge and self._leader_length > 0:
            # 预放电 i_pd = q·v,先导通道电感压降 ΔV = L·l·di/dt
            i_pd = self.config.q_leader * self._leader_velocity
            delta_v_kV = (
                self.config.L_channel * self._leader_length * i_pd / dt
            ) / 1000.0
            u_eff_kV = max(0.0, u_kV - delta_v_kV)
            self._predischarge_current = i_pd
        else:
            self._predischarge_current = 0.0

        remaining = max(d - self._leader_length, 1e-12)
        E_gap = u_eff_kV / remaining

        if E_gap > E0:
            if not self._is_leader_active:
                self._is_leader_active = True
                self._inception_time = time

            # CIGRE: v[m/μs] = k[m²/(kV²·μs)] · u[kV] · (E_gap - E0)[kV/m]
            # SI:   v[m/s]  = k · 1e6 · u · (E_gap - E0)
            velocity = max(0.0, k * 1e6 * u_eff_kV * (E_gap - E0))
            self._leader_velocity = velocity
            self._peak_leader_velocity = max(self._peak_leader_velocity, velocity)

            self._leader_length += velocity * dt

            if self._leader_length >= d:
                self._leader_length = d
                self._is_flashed_over = True
                self._flashover_time = time
                self._flashover_count += 1
                self.R_current = self.config.R_arc
                self.G_current = 1.0 / self.config.R_arc
        else:
            self._leader_velocity = 0.0
            # 场强不足:先导停止发展,保持当前长度

        state_changed = (self._is_flashed_over != old_state)
        self._record_history(u_kV, time)
        return state_changed

    def _check_arc_extinction(self, current_A: float, time: float) -> bool:
        """电弧熄灭判断:电流低于阈值并持续 extinction_delay 后开路。"""
        if abs(current_A) < self.config.extinction_current:
            if self._below_extinction_since < 0:
                self._below_extinction_since = time
            elif time - self._below_extinction_since >= self.config.extinction_delay:
                self._is_flashed_over = False
                self._leader_length = 0.0
                self._leader_velocity = 0.0
                self._is_leader_active = False
                self.R_current = self.config.R_open
                self.G_current = 1.0 / self.config.R_open
                self._arc_extinction_time = time
                self._below_extinction_since = -1.0
                return True
        else:
            self._below_extinction_since = -1.0
        return False

    def _record_history(self, u_kV: float, time: float) -> None:
        self.leader_length_history.append(self._leader_length)
        self.leader_velocity_history.append(self._leader_velocity)
        self.voltage_history.append(u_kV)
        self.state_history.append(1 if self._is_flashed_over else 0)
        self.predischarge_current_history.append(self._predischarge_current)

    # ---- 状态查询 -------------------------------------------------------

    @property
    def is_flashed_over(self) -> bool:
        return self._is_flashed_over

    @property
    def leader_length(self) -> float:
        return self._leader_length

    @property
    def leader_velocity(self) -> float:
        return self._leader_velocity

    @property
    def flashover_time(self) -> float:
        return self._flashover_time

    @property
    def inception_time(self) -> float:
        return self._inception_time

    @property
    def leader_progress(self) -> float:
        """先导进展比例 l/d ∈ [0, 1]。"""
        return self._leader_length / self.config.gap_length

    def reset(self) -> None:
        self._leader_length = 0.0
        self._leader_velocity = 0.0
        self._is_flashed_over = False
        self._is_leader_active = False
        self._flashover_time = -1.0
        self._inception_time = -1.0
        self._predischarge_current = 0.0
        self._below_extinction_since = -1.0
        self._arc_extinction_time = -1.0
        self.R_current = self.config.R_open
        self.G_current = 1.0 / self.config.R_open
        self.leader_length_history.clear()
        self.leader_velocity_history.clear()
        self.voltage_history.clear()
        self.state_history.clear()
        self.predischarge_current_history.clear()
        self._flashover_count = 0
        self._peak_leader_velocity = 0.0
        self._peak_voltage_kV = 0.0



    # ---- 信息 -----------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'InsulatorFlashoverLPM',
            'gap_length_m': self.config.gap_length,
            'k': self.config.k,
            'E0': self.config.E0,
            'E0_eff': self.E0_eff,
            'altitude_m': self.config.altitude_m,
            'R_open': self.config.R_open,
            'R_arc': self.config.R_arc,
            'is_flashed_over': self._is_flashed_over,
            'leader_length_m': self._leader_length,
            'leader_progress': self.leader_progress,
            'flashover_time_us': (self._flashover_time * 1e6
                                  if self._flashover_time >= 0 else None),
            'inception_time_us': (self._inception_time * 1e6
                                  if self._inception_time >= 0 else None),
            'peak_velocity_m_s': self._peak_leader_velocity,
            'peak_voltage_kV': self._peak_voltage_kV,
            'flashover_count': self._flashover_count,
        }

    def print_info(self) -> None:
        info = self.get_info()
        sep = "-" * 55
        print(sep)
        print(f"绝缘子 LPM 闪络模型: {info['name']}")
        print(sep)
        print(f"  间隙 d     = {info['gap_length_m']:.3f} m")
        print(f"  k          = {info['k']:.2e} m²/(kV²·s)")
        print(f"  E0         = {info['E0']:.1f} kV/m")
        if self.config.altitude_m > 0:
            print(f"  E0_eff     = {info['E0_eff']:.1f} kV/m "
                  f"(海拔 {self.config.altitude_m:.0f} m)")
        print(f"  R_open     = {info['R_open']:.2e} Ω, "
              f"R_arc = {info['R_arc']:.1f} Ω")
        print(f"  闪络状态    : {'已闪络' if info['is_flashed_over'] else '未闪络'}")
        print(f"  先导长度    : {info['leader_length_m']:.4f} m "
              f"({info['leader_progress']*100:.1f}%)")
        if info['flashover_time_us'] is not None:
            print(f"  闪络时刻    : {info['flashover_time_us']:.2f} μs")
        if info['inception_time_us'] is not None:
            print(f"  先导起始    : {info['inception_time_us']:.2f} μs")
        print(f"  峰值先导速度: {info['peak_velocity_m_s']:.2e} m/s")
        print(f"  峰值间隙电压: {info['peak_voltage_kV']:.1f} kV")
        print(sep)

    def __repr__(self) -> str:
        state = "FLASHED" if self._is_flashed_over else "OPEN"
        return (f"<InsulatorFlashoverLPM: {self.name}, "
                f"d={self.config.gap_length:.3f}m, "
                f"leader={self.leader_progress*100:.1f}%, {state}>")


# 工程经验:每片悬式绝缘子高度 146 mm
_INSULATOR_DISC_HEIGHT_M = 0.146
_INSULATOR_DISC_COUNTS = {
    10: 1, 35: 3, 66: 5, 110: 7, 220: 13,
    330: 19, 500: 25, 750: 33, 1000: 45,
}



__all__ = [
    'NonlinearResistorModel',
    'Segment',
    'SegmentedSolverHelper',
    'LPMInsulatorType', 'LPMConfig', 'InsulatorFlashoverLPM',
]
