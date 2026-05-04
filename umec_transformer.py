"""
UMEC 变压器模型 — 三相组 (THREE_PHASE_BANK) 专用
=================================================

三相组 = 三台独立单相变压器，各相无磁耦合。
每相等价于一个独立的多绕组耦合电感，
用梯形积分离散化为多端口诺顿等效。

基于 PSCAD/EMTDC 手册 Chapter 6 与 Enright et al. (1997)。

使用方法:
    >>> data = UMECTransformerData(...)
    >>> xfmr = UMECTransformer(data, dt=50e-6)
    >>> G_eq, I_hist = xfmr.get_norton_equivalent()

作者: EMTP Project
版本: 2.0
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 绕组接法
# =============================================================================

class WindingType:
    """绕组接法"""
    Y = 'Y'
    Y_GND = 'Y_gnd'
    DELTA = 'Delta'


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class UMECTransformerData:
    """
    三相组 UMEC 变压器输入参数

    Parameters
    ----------
    name : str
        变压器名称
    S_rated : float
        三相额定容量 (VA)
    freq : float
        额定频率 (Hz)
    V_rated_LL : list of float
        各绕组额定线电压 (V, RMS)
    winding_types : list of str
        各绕组接法: 'Y', 'Y_gnd', 'Delta'
    X_leak_pu : float
        漏抗标幺值（一次侧测量）
    Im_percent : float
        额定电压下励磁电流百分比 (%)
    NLL_pu : float
        空载损耗标幺值
    CL_pu : float
        铜损标幺值
    enable_saturation : bool
        是否启用饱和
    sat_V_pu : list of float
        饱和曲线电压数据点 (pu)
    sat_I_percent : list of float
        饱和曲线电流数据点 (%)
    nodes : list of list of tuple
        节点映射 nodes[phase][winding] = (node_from, node_to)
    """
    name: str
    S_rated: float = 1e6
    freq: float = 50.0
    V_rated_LL: List[float] = field(default_factory=lambda: [690.0, 35000.0])
    winding_types: List[str] = field(default_factory=lambda: ['Y', 'Delta'])
    X_leak_pu: float = 0.08
    Im_percent: float = 1.0
    NLL_pu: float = 0.0
    CL_pu: float = 0.0
    enable_saturation: bool = False
    sat_V_pu: List[float] = field(default_factory=list)
    sat_I_percent: List[float] = field(default_factory=list)
    nodes: Optional[List[List[Tuple[int, int]]]] = None

    n_phases: int = 3

    @property
    def num_windings_per_phase(self) -> int:
        return len(self.V_rated_LL)

    @property
    def total_windings(self) -> int:
        return self.n_phases * self.num_windings_per_phase


# =============================================================================
# 饱和模型（分段线性）
# =============================================================================

class UMECSaturationModel:
    """
    铁芯分段线性饱和模型

    UMEC 饱和: 直接修改磁导 Pw，改变等效支路电导。
    饱和曲线定义为 (V_pu, I_percent) 数据点对。

    Parameters
    ----------
    V_pu : array
        电压数据点 (pu)
    I_percent : array
        电流数据点 (%)
    V_rated_phase : float
        额定相电压 (V, peak)
    I_base : float
        基准电流 (A, peak)
    Pw_nominal : float
        额定磁导
    omega : float
        额定角频率
    """

    def __init__(self, V_pu: np.ndarray, I_percent: np.ndarray,
                 V_rated_phase: float, I_base: float,
                 Pw_nominal: float, omega: float):

        self.V_pu = np.asarray(V_pu, dtype=float)
        self.I_percent = np.asarray(I_percent, dtype=float)
        self.V_rated_phase = V_rated_phase
        self.I_base = I_base
        self.Pw_nominal = Pw_nominal
        self.omega = omega
        self.num_points = len(V_pu)

        # (V_pu, I_percent) → (flux_linkage, magnetizing_current)
        self.flux_points = self.V_pu * V_rated_phase / omega   # Wb·turns
        self.current_points = self.I_percent / 100.0 * I_base  # A

        self._compute_segments()
        self.current_segment = 0

    def _compute_segments(self):
        """计算每段增量电感"""
        n = self.num_points
        self.num_segments = max(n - 1, 1)
        self.L_segments = np.zeros(self.num_segments)

        for k in range(self.num_segments):
            d_flux = self.flux_points[k + 1] - self.flux_points[k]
            d_current = self.current_points[k + 1] - self.current_points[k]

            if abs(d_current) < 1e-30:
                self.L_segments[k] = 1e10
            elif abs(d_flux) < 1e-30:
                self.L_segments[k] = 1e-6
            else:
                self.L_segments[k] = d_flux / d_current

    def find_segment(self, flux: float) -> int:
        """根据磁通链绝对值确定当前段索引"""
        abs_flux = abs(flux)
        for k in range(self.num_segments):
            if abs_flux <= self.flux_points[k + 1]:
                return k
        return self.num_segments - 1

    def get_incremental_inductance(self, flux: float) -> float:
        """获取当前磁通对应的增量电感"""
        return self.L_segments[self.find_segment(flux)]

    def get_magnetizing_current(self, flux: float) -> float:
        """从分段线性曲线插值励磁电流"""
        abs_flux = abs(flux)
        sign = np.sign(flux) if flux != 0 else 1.0
        i_mag = np.interp(abs_flux, self.flux_points, self.current_points)
        return sign * i_mag


# =============================================================================
# UMEC 变压器（三相组专用）
# =============================================================================

class UMECTransformer:
    """
    三相组 UMEC 变压器模型

    三相组各相独立（无相间磁耦合），相磁导矩阵为对角阵。
    每相: L_mag = N² × Pw，饱和时直接改 Pw 即可。

    离散化:
      [V] = [R][i] + [L] d[i]/dt  →  梯形法  →
      i(t) = G_eq V(t) + I_hist(t)

    Parameters
    ----------
    data : UMECTransformerData
    dt : float
        时间步长 (s)
    verbose : bool
    """

    def __init__(self, data: UMECTransformerData, dt: float, verbose: bool = False):
        self.data = data
        self.dt = dt
        self.verbose = verbose

        self.n_phases = data.n_phases
        self.n_windings = data.num_windings_per_phase
        self.m = data.total_windings

        # 1. 基本参数
        self.omega = 2.0 * np.pi * data.freq
        self.V_phase = self._compute_phase_voltages()
        self.N = self._build_turns_vector()

        self.S_per_phase = data.S_rated / data.n_phases
        self.Z_base = np.zeros(self.m)
        self.I_base = np.zeros(self.m)
        for ph in range(self.n_phases):
            for w in range(self.n_windings):
                idx = ph * self.n_windings + w
                self.Z_base[idx] = self.V_phase[w] ** 2 / self.S_per_phase
                self.I_base[idx] = self.S_per_phase / self.V_phase[w]

        # 2. 磁导（各相独立，对角）
        self.Pw = self._compute_permeance()
        self.Pw_per_phase = np.full(self.n_phases, self.Pw)

        # 3. 电感矩阵
        self.L = self._build_inductance_matrix()

        # 4. 电阻矩阵
        self.R = self._build_resistance_matrix()

        # 5. 空载损耗并联电阻
        self.R_core = self._compute_core_loss_resistance()

        # 6. 梯形积分离散化
        self._discretize()

        # 7. 状态
        self.I_hist = np.zeros(self.m)
        self.V_prev = np.zeros(self.m)
        self.I_prev = np.zeros(self.m)
        self.flux = np.zeros(self.m)

        # 8. 饱和
        self.saturation_models: List[UMECSaturationModel] = []
        self._has_saturation = False
        if data.enable_saturation and len(data.sat_V_pu) > 1:
            self._init_saturation()

        if verbose:
            self._print_info()

    # =========================================================================
    # 参数计算
    # =========================================================================

    def _compute_phase_voltages(self) -> np.ndarray:
        """Y 接法: V_phase = V_LL/√3 (三相)；Delta: V_phase = V_LL"""
        V_phase = np.zeros(self.n_windings)
        for w in range(self.n_windings):
            V_LL = self.data.V_rated_LL[w]
            wtype = self.data.winding_types[w]
            if wtype in (WindingType.Y, WindingType.Y_GND):
                V_phase[w] = V_LL / np.sqrt(3.0)
            else:
                V_phase[w] = V_LL
        return V_phase

    def _build_turns_vector(self) -> np.ndarray:
        """等效匝数 N_i = V_phase_i (kV)"""
        N = np.zeros(self.m)
        for ph in range(self.n_phases):
            for w in range(self.n_windings):
                idx = ph * self.n_windings + w
                N[idx] = self.V_phase[w] / 1e3
        return N

    def _compute_permeance(self) -> float:
        """
        从开路试验推导磁导

        三相组各相独立:
          L_mag = V_phase / (ω × I_OC)
          Pw = L_mag / N²    (N = V_phase_kV)
        """
        V1 = self.V_phase[0]
        I_base1 = self.S_per_phase / V1
        I_OC = self.data.Im_percent / 100.0 * I_base1

        L_mag = V1 / (self.omega * I_OC)
        N1_kV = V1 / 1e3
        Pw = L_mag / (N1_kV ** 2)

        if self.verbose:
            print(f"  磁导: Pw = {Pw:.6e}")
        return Pw

    def _build_inductance_matrix(self) -> np.ndarray:
        """
        电感矩阵 L (m×m)

        三相组无相间耦合，分块对角。
        同相不同绕组: L[i,j] = N[i] × Pw[ph] × N[j]
        加漏感到对角线。
        """
        m = self.m
        L = np.zeros((m, m))

        for i in range(m):
            ph_i = i // self.n_windings
            for j in range(m):
                ph_j = j // self.n_windings
                if ph_i == ph_j:
                    L[i, j] = self.N[i] * self.Pw_per_phase[ph_i] * self.N[j]

        # 漏感
        L_leak = self._compute_leakage_inductances()
        for i in range(m):
            L[i, i] += L_leak[i]

        # 正定检查
        eigvals = np.linalg.eigvalsh(L)
        if np.any(eigvals <= 0):
            min_eig = np.min(eigvals)
            logger.warning(f"电感矩阵非正定 (min_eig={min_eig:.2e})，添加正则化")
            L += np.eye(m) * abs(min_eig) * 1.01

        if self.verbose:
            print(f"  电感矩阵 L ({m}×{m}):")
            print(f"    对角元素: {np.diag(L)}")
            print(f"    条件数: {np.linalg.cond(L):.2e}")
        return L

    def _compute_leakage_inductances(self) -> np.ndarray:
        """从短路试验数据计算漏感并分配到各绕组"""
        m = self.m
        L_leak = np.zeros(m)

        Z_base1 = self.V_phase[0] ** 2 / self.S_per_phase
        L_leak_total = self.data.X_leak_pu * Z_base1 / self.omega

        if self.n_windings == 2:
            L1 = L_leak_total / 2.0
            a = self.V_phase[0] / self.V_phase[1]
            L2 = L1 / (a ** 2)
            for ph in range(self.n_phases):
                L_leak[ph * self.n_windings + 0] = L1
                L_leak[ph * self.n_windings + 1] = L2
        elif self.n_windings >= 3:
            for ph in range(self.n_phases):
                for w in range(self.n_windings):
                    idx = ph * self.n_windings + w
                    L_leak[idx] = L_leak_total / (2.0 * self.n_windings)
        else:
            for ph in range(self.n_phases):
                L_leak[ph] = L_leak_total

        return L_leak

    def _build_resistance_matrix(self) -> np.ndarray:
        """绕组电阻 (对角)"""
        m = self.m
        R = np.zeros((m, m))

        if self.data.CL_pu > 0:
            P_cu_total = self.data.CL_pu * self.data.S_rated
            P_cu_per_winding = P_cu_total / m
            for i in range(m):
                R[i, i] = P_cu_per_winding / (self.I_base[i] ** 2)
        else:
            for i in range(m):
                R[i, i] = 1e-6

        return R

    def _compute_core_loss_resistance(self) -> np.ndarray:
        """空载损耗等效并联电阻"""
        m = self.m
        R_core = np.full(m, np.inf)

        if self.data.NLL_pu > 0:
            P_nll_total = self.data.NLL_pu * self.data.S_rated
            P_per_winding = P_nll_total / m
            for i in range(m):
                w = i % self.n_windings
                R_core[i] = self.V_phase[w] ** 2 / P_per_winding
        return R_core

    # =========================================================================
    # 梯形积分离散化
    # =========================================================================

    def _discretize(self):
        """
        梯形法离散化

          [Z_eq] = 2/Δt [L] + [R]
          [G_eq] = [Z_eq]⁻¹
          [H]    = [G_eq] (2/Δt [L] - [R])

          i(t) = G_eq V(t) + I_hist(t)
          I_hist(t+Δt) = G_eq V(t) + H i(t)
        """
        two_L_dt = 2.0 * self.L / self.dt
        self.Z_eq = two_L_dt + self.R
        self.G_eq = np.linalg.inv(self.Z_eq)
        self.H_hist = self.G_eq @ (two_L_dt - self.R)

        self.G_core = np.zeros(self.m)
        for i in range(self.m):
            if np.isfinite(self.R_core[i]) and self.R_core[i] > 0:
                self.G_core[i] = 1.0 / self.R_core[i]

    # =========================================================================
    # 饱和初始化
    # =========================================================================

    def _init_saturation(self):
        """初始化分段线性饱和模型（每相一个）"""
        data = self.data
        for ph in range(self.n_phases):
            V_rated = self.V_phase[0]
            I_base = self.I_base[ph * self.n_windings]
            model = UMECSaturationModel(
                V_pu=np.array(data.sat_V_pu),
                I_percent=np.array(data.sat_I_percent),
                V_rated_phase=V_rated * np.sqrt(2),
                I_base=I_base * np.sqrt(2),
                Pw_nominal=self.Pw,
                omega=self.omega,
            )
            self.saturation_models.append(model)
        self._has_saturation = True

        if self.verbose:
            print(f"  饱和模型: {self.n_phases} 相, "
                  f"{self.saturation_models[0].num_segments} 段/相")

    # =========================================================================
    # 诺顿等效接口
    # =========================================================================

    def get_norton_equivalent(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回多端口诺顿等效 (G_total, I_hist)

        G_total = G_eq + diag(G_core)
        """
        G_total = self.G_eq.copy()
        for i in range(self.m):
            G_total[i, i] += self.G_core[i]
        return G_total, self.I_hist.copy()

    def get_port_nodes(self) -> List[Tuple[int, int]]:
        """获取端口节点对列表"""
        if self.data.nodes is None:
            raise ValueError(f"变压器 {self.data.name} 的节点未设置")
        port_nodes = []
        for ph_nodes in self.data.nodes:
            for node_pair in ph_nodes:
                port_nodes.append(node_pair)
        return port_nodes

    # =========================================================================
    # 状态更新
    # =========================================================================

    def update_history(self, V_ports: np.ndarray):
        """
        梯形积分历史项更新

          I_now = G_eq V(t) + I_hist(t)
          I_hist(t+Δt) = G_eq V(t) + H I_now
          λ(t+Δt) = λ(t) + Δt/2 (V(t) + V(t-Δt))
        """
        I_now = self.G_eq @ V_ports + self.I_hist
        self.I_hist = self.G_eq @ V_ports + self.H_hist @ I_now
        self.flux += 0.5 * self.dt * (V_ports + self.V_prev)
        self.I_prev = I_now.copy()
        self.V_prev = V_ports.copy()

    def reset_state(self) -> None:
        """Reset dynamic history, flux, and saturation segment state."""
        self.I_hist.fill(0.0)
        self.V_prev.fill(0.0)
        self.I_prev.fill(0.0)
        self.flux.fill(0.0)
        if self._has_saturation:
            for model in self.saturation_models:
                model.current_segment = 0
            self.Pw_per_phase[:] = self.Pw
            self.L = self._build_inductance_matrix()
            self._discretize()

    # =========================================================================
    # 饱和处理
    # =========================================================================

    def check_saturation(self, V_ports: np.ndarray) -> Tuple[bool, dict]:
        """
        检查是否需要饱和段切换

        Returns
        -------
        need_update : bool
        updates : dict
        """
        if not self._has_saturation:
            return False, {}

        need_update = False
        updates = {}

        flux_trial = self.flux + 0.5 * self.dt * (V_ports + self.V_prev)

        for ph in range(self.n_phases):
            idx = ph * self.n_windings
            flux_ph = flux_trial[idx]
            model = self.saturation_models[ph]
            new_seg = model.find_segment(flux_ph)

            if new_seg != model.current_segment:
                need_update = True
                old_seg = model.current_segment
                model.current_segment = new_seg
                updates[ph] = {
                    'old_seg': old_seg,
                    'new_seg': new_seg,
                    'L_new': model.L_segments[new_seg],
                }

        if need_update:
            self._update_saturation_parameters()

        return need_update, updates

    def _update_saturation_parameters(self):
        """
        饱和段切换后更新磁导、电感矩阵并重新离散化

        三相组: L_mag = N² × Pw → Pw = L_mag / N²
        """
        L_leak = self._compute_leakage_inductances()

        for ph in range(self.n_phases):
            if ph < len(self.saturation_models):
                model = self.saturation_models[ph]
                seg = model.current_segment
                L_inc = model.L_segments[seg]

                idx = ph * self.n_windings
                L_mag = max(L_inc - L_leak[idx], 1e-12)
                N_sq = self.N[idx] ** 2
                self.Pw_per_phase[ph] = L_mag / N_sq

        self.L = self._build_inductance_matrix()
        self._discretize()

    # =========================================================================
    # 信息输出
    # =========================================================================

    def _print_info(self):
        """打印变压器参数"""
        data = self.data
        print(f"\n{'=' * 50}")
        print(f"UMEC 变压器 (三相组): {data.name}")
        print(f"{'=' * 50}")
        print(f"  额定容量: {data.S_rated / 1e6:.2f} MVA")
        print(f"  额定频率: {data.freq:.1f} Hz")
        print(f"  绕组数/相: {self.n_windings}")
        print(f"  总端口数: {self.m}")

        for w in range(self.n_windings):
            print(f"  绕组 #{w + 1}: {data.V_rated_LL[w] / 1e3:.3f} kV (LL), "
                  f"{self.V_phase[w] / 1e3:.3f} kV (相), "
                  f"{data.winding_types[w]}, N={self.N[w]:.4f}")

        print(f"  漏抗: {data.X_leak_pu:.4f} pu")
        print(f"  励磁电流: {data.Im_percent:.2f} %")
        print(f"  磁导: Pw={self.Pw:.6e}")

        if self.n_windings >= 2:
            a = self.V_phase[0] / self.V_phase[1]
            print(f"  匝比 (1:2): {a:.6f}")

        print(f"  饱和: {'启用' if self._has_saturation else '禁用'}")
        print(f"{'=' * 50}")

    def get_info(self) -> Dict[str, Any]:
        """获取变压器信息字典"""
        return {
            'name': self.data.name,
            'S_rated': self.data.S_rated,
            'freq': self.data.freq,
            'n_phases': self.n_phases,
            'n_windings': self.n_windings,
            'V_phase': self.V_phase.tolist(),
            'N': self.N.tolist(),
            'Pw': self.Pw,
            'Pw_per_phase': self.Pw_per_phase.tolist(),
            'L_diag': np.diag(self.L).tolist(),
            'has_saturation': self._has_saturation,
        }


# =============================================================================
# 便捷创建函数
# =============================================================================

def create_umec_transformer_3ph_bank(
    name: str,
    S_mva: float,
    V1_kV: float,
    V2_kV: float,
    wtype1: str = 'Y',
    wtype2: str = 'Delta',
    X_leak_pu: float = 0.08,
    Im_percent: float = 1.0,
    freq: float = 50.0,
    NLL_pu: float = 0.0,
    CL_pu: float = 0.0,
    nodes: Optional[List[List[Tuple[int, int]]]] = None,
    **kwargs
) -> UMECTransformerData:
    """
    便捷函数: 创建三相组两绕组变压器数据

    Parameters
    ----------
    S_mva : float
        额定容量 (MVA)
    V1_kV, V2_kV : float
        一次/二次侧额定线电压 (kV)
    """
    return UMECTransformerData(
        name=name,
        S_rated=S_mva * 1e6,
        freq=freq,
        V_rated_LL=[V1_kV * 1e3, V2_kV * 1e3],
        winding_types=[wtype1, wtype2],
        X_leak_pu=X_leak_pu,
        Im_percent=Im_percent,
        NLL_pu=NLL_pu,
        CL_pu=CL_pu,
        nodes=nodes,
        **kwargs
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'WindingType',
    'UMECTransformerData',
    'UMECSaturationModel',
    'UMECTransformer',
    'create_umec_transformer_3ph_bank',
]
