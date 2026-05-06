"""Main EMTPSolver implementation.

Supports:
- Multi-phase transmission lines (Bergeron / ULM)
- PSCAD segmented nonlinear elements (MOA arresters)
- MNA (Modified Nodal Analysis) with ideal voltage sources
- UMEC multi-port transformer model
- CIGRE leader-progression-model insulator flashover (LPM)

Performance:
- MNA sparse matrix (scipy.sparse CSC): rebuilt only on topology changes
- SuperLU sparse LU decomposition: cached when topology is unchanged
- Pre-allocated output arrays eliminate dynamic list-appends
"""


from __future__ import annotations

import logging
import os
import time as _perf_time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# 稀疏求解器: SuperLU
#
# 仅使用 scipy.sparse.linalg.splu 进行 CSC 稀疏 LU 分解。
# 已移除 KLU / UMFPACK 可选加速支持，避免额外库依赖和环境差异。
# ---------------------------------------------------------------------------
_SPARSE_SOLVER_NAME: str = "SuperLU"


def _sparse_factorize(A: 'sp.csc_matrix') -> Any:
    """对 CSC 矩阵做稀疏 LU 分解。

    返回对象满足 .solve(rhs) → ndarray 接口。
    使用 scipy.sparse.linalg.splu，在 SuperLU 后端上运行。
    """
    return splu(A, permc_spec='MMD_AT_PLUS_A')

try:
    from atp_lightning_current_generator_simplified import (
        BaseLightningCurrentSource,
        TWOEXPFCurrentSource,
        HEIDLERFCurrentSource,
        LightningWaveform,
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
    )
except ImportError:
    BaseLightningCurrentSource = ()
    TWOEXPFCurrentSource = None
    HEIDLERFCurrentSource = None
    LightningWaveform = None
    create_lightning_current_source = None
    create_standard_twoexpf_current_source = None

try:
    from nonlinear_models_pscad import (
        InsulatorFlashoverLPM,
        LPMConfig,
        LPMInsulatorType,
        SegmentedSolverHelper,
        SegmentedMOAResistor,
    )
    NONLINEAR_AVAILABLE = True
except ImportError:
    NONLINEAR_AVAILABLE = False

    class _UnavailableNonlinear:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "nonlinear_models_pscad.py is required for MOA/LPM nonlinear components"
            )

    class SegmentedSolverHelper:
        def register(self, *args, **kwargs):
            raise ImportError(
                "nonlinear_models_pscad.py is required for segmented nonlinear components"
            )
        def reset_all(self):
            return None
        def check_all_segments(self, voltages):
            return False, {}

    InsulatorFlashoverLPM = _UnavailableNonlinear
    LPMConfig = _UnavailableNonlinear
    LPMInsulatorType = _UnavailableNonlinear
    SegmentedMOAResistor = _UnavailableNonlinear

try:
    from transmission_line_emtp_v2 import (
        BergeronLine,
        TransmissionLineInterface,
    )
    TRANSMISSION_LINE_AVAILABLE = True
except ImportError:
    TRANSMISSION_LINE_AVAILABLE = False

    class TransmissionLineInterface:
        """Placeholder used when transmission_line_emtp_v2.py is not installed."""
        pass

    class BergeronLine(TransmissionLineInterface):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "transmission_line_emtp_v2.py is required for BergeronLine components"
            )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export from emtp package modules (P3 modularisation)
# These were originally defined inline; now sourced from the new package.
# ---------------------------------------------------------------------------
from emtp.nodes import NodeBook, NodeIndexer           # noqa: E402, F811
from emtp.types import (                                # noqa: E402, F811
    VoltageSource, ValidationIssue, ValidationReport,
    RHSPlan, ElementType, Branch, CurrentSource, LineData,
)
from emtp.sparse_solver import (                        # noqa: E402
    _SPARSE_SOLVER_NAME as _SPARSE_SOLVER_NAME,
    _sparse_factorize as _sparse_factorize,
    SparseLinearSolver,
)
from emtp.stamping import COOStamper, StampingEngine    # noqa: E402, F811
from emtp.devices import (                               # noqa: E402, F811
    Device,
    ResistorDevice,
    InductorDevice,
    CapacitorDevice,
    SwitchDevice,
    SeriesRLDevice,
    NonlinearResistorDevice,
    LPMFlashoverDevice,
    _update_series_rl_history_static,
)
from emtp.runtime import DynamicDeviceRuntime           # noqa: E402, F811
from emtp.runtime.resolve import ResolveManager
from emtp.runtime.stepper import TimeStepper
from emtp.results.store import ResultStore
from emtp.lines.bergeron import BergeronLineDevice
from emtp.lines.ulm import ULMLineDevice
from emtp.lines.fitulm_resolver import FitULMSpec, FitULMResolver
from emtp.transformers.umec import UMECTransformerDevice
from emtp.results import (                                # noqa: E402
    scale_probe_values,
    scale_values,
    node_voltage_from_solution,
    branch_voltage_from_solution,
    branch_current_from_solution,
)

# ---------------------------------------------------------------------------
# 可选模块:ULM / UMEC
# ---------------------------------------------------------------------------

try:
    from ulm_transmission_line_PARA import (
        FitULMData,
        FitULMReader,
        ULMLine,
        ULMModel,
        ULMBatchPack,
    )
    ULM_AVAILABLE = True
except ImportError:
    ULM_AVAILABLE = False
    ULMLine = None
    ULMModel = None
    FitULMReader = None
    FitULMData = None
    ULMBatchPack = None

try:
    from umec_transformer import (
        UMECTransformer,
        UMECTransformerData,
        WindingType,
        create_umec_transformer_3ph_bank,
    )
    UMEC_AVAILABLE = True
except ImportError:
    UMEC_AVAILABLE = False
    UMECTransformer = None
    UMECTransformerData = None
    WindingType = None
    create_umec_transformer_3ph_bank = None



# ---------------------------------------------------------------------------

class EMTPSolver:
    """EMTP 电磁暂态仿真求解器。

    支持元件
    --------
    - 基本: R, L, C, 开关
    - 电源: 电流源(含雷电波形), 理想电压源
    - 非线性: 分段线性 MOA 避雷器
    - 传输线: Bergeron, ULM (单相/多相)
    - UMEC 变压器: 三相组 / 三相三柱 / 三相五柱
    - LPM 绝缘子闪络开关

    MNA 修正节点分析
    ----------------
    构建 (n+m)×(n+m) 增广系统:

        ┌       ┐ ┌     ┐   ┌   ┐
        │ G   B │ │  v  │   │ I │
        │       │ │     │ = │   │
        │ C   D │ │ i_s │   │ E │
        └       ┘ └     ┘   └   ┘

    其中 n 为节点数, m 为理想电压源数。
    G: n×n 节点导纳矩阵
    B: n×m 电压源关联矩阵
    C: m×n = Bᵀ (理想电压源)
    D: m×m 零矩阵 (理想电压源)
    解向量直接包含所有节点电压 v 与电压源电流 i_s。

    稀疏求解器
    ----------
    矩阵以 scipy.sparse CSC 格式存储,使用 scipy.sparse.linalg.splu
    进行 SuperLU 稀疏 LU 分解。

    非线性求解(PSCAD 分段线性法)
    ---------------------------
    预先将 V-I 曲线离散为若干线性段;每步用当前段的诺顿等效求解,
    解出后检查段边界,仅在段切换时更新矩阵并重解。
    """

    _MAX_SEG_ITER: int = 5  # 最大段切换迭代次数
    _LU_SINGULAR_REG: float = 1e-12  # 奇异矩阵正则化系数

    def __init__(
        self, dt: float = 1e-6, finish_time: float = 100e-6,
        verbose: bool = True,
        line_compile_workers: Optional[int] = None,
        compile_lines_on_add: bool = False,
        ulm_batch_mode: str = "auto",
        ulm_batch_parallel_threshold_factor: int = 2,
        record_line_history: bool = False,
        record_branch_history: bool = False,
        record_source_history: bool = False,
        sync_line_state_each_step: bool = False,
        allow_singular_regularization: bool = False,
        record_all_node_voltages: bool = True,
        max_result_memory_mb: Optional[float] = None,
        pre_sample_sources: bool = False,
        use_rhs_plan: bool = False,
        use_multiport_lines: bool = False,
        use_multiport_transformers: bool = False,
    ):
        """
        Parameters
        ----------
        dt : float
            时间步长 (s),默认 1 μs。
        finish_time : float
            仿真结束时间 (s)。
        verbose : bool
            是否输出详细日志。
        record_all_node_voltages : bool
            是否记录所有节点电压历史。开启时会分配
            ``num_nodes × n_steps`` 的完整电压矩阵，
            大型网络建议设为 False 并改用探针获取关心量。
        max_result_memory_mb : float or None
            结果缓存内存上限(MB)。若估计值超过此阈值,
            会在运行前发出 warning。None 表示不限制。
        pre_sample_sources : bool
            在仿真前对独立电流源和电压源进行预采样。
            可减少每步 Python 函数调用开销,但会改变
            source function 的调用时机。默认 False。
        use_rhs_plan : bool
            预编译 RHS 装配的拓扑索引,用 flat arrays 替代
            每步 Python 对象遍历。与 pre_sample_sources
            搭配效果最佳。默认 False。
        """
        self.dt = dt
        self.finish_time = finish_time
        self.verbose = verbose
        self.line_compile_workers = line_compile_workers
        self.compile_lines_on_add = compile_lines_on_add
        self.ulm_batch_mode = str(ulm_batch_mode).lower()
        self.ulm_batch_parallel_threshold_factor = int(ulm_batch_parallel_threshold_factor)
        self.record_line_history = bool(record_line_history)
        self.record_branch_history = bool(record_branch_history)
        self.record_source_history = bool(record_source_history)
        self.sync_line_state_each_step = bool(sync_line_state_each_step)
        self.allow_singular_regularization = bool(allow_singular_regularization)
        self.record_all_node_voltages = bool(record_all_node_voltages)
        self.max_result_memory_mb = (
            float(max_result_memory_mb) if max_result_memory_mb is not None else None
        )
        self.pre_sample_sources = bool(pre_sample_sources)
        self.use_rhs_plan = bool(use_rhs_plan)
        self.use_multiport_lines = bool(use_multiport_lines)
        self.use_multiport_transformers = bool(use_multiport_transformers)
        self._rhs_plan: Optional[RHSPlan] = None
        self._rhs_plan_dirty: bool = True
        self._active_mna_solver_name = _SPARSE_SOLVER_NAME
        if self.ulm_batch_mode not in {'auto', 'parallel', 'serial', 'off'}:
            raise ValueError(
                "ulm_batch_mode 必须是 'auto'、'parallel'、'serial' 或 'off'，"
                f"当前为 {ulm_batch_mode!r}"
            )
        self._lines_compiled: bool = False

        # ---- 元件存储 ----
        self.branches: Dict[str, Branch] = {}
        self._devices: List[Device] = []
        self._multiport_devices: List[Any] = []
        self.current_sources: Dict[str, CurrentSource] = {}
        self.voltage_sources: Dict[str, VoltageSource] = {}
        self.transmission_lines: Dict[str, TransmissionLineInterface] = {}
        self.transformers: Dict[str, 'UMECTransformer'] = {}
        self.lines: Dict[str, LineData] = {}  # 兼容旧版

        # ---- 节点管理 ----
        self.num_nodes: int = 0
        self._node_set: set = set()
        self._vs_node_set: set = set()  # 电压源正端节点集合
        self._indexer = NodeIndexer()   # external id ↔ compact index
        self._runtime = DynamicDeviceRuntime(self.dt)
        self._resolve_mgr = ResolveManager(max_iter=self._MAX_SEG_ITER)
        self._stepper = TimeStepper()
        self._result_store: Optional[ResultStore] = None
        self._stamping = StampingEngine(
            self._indexer,
            allow_singular_regularization=allow_singular_regularization,
        )

        # ---- 命名节点管理 ----
        # 允许使用字符串节点名(如 "T1.tower_top"),
        # 内部自动转换为整数节点供 MNA 装配使用。
        self.nodes = NodeBook(start=1)

        # ---- 轻量探针记录 ----
        # 只记录用户指定的节点/支路波形，避免开启全量 history。
        self.voltage_probes: Dict[str, Dict[str, int]] = {}
        self.branch_current_probes: Dict[str, Dict[str, str]] = {}

        self._voltage_probe_names: List[str] = []
        self._branch_current_probe_names: List[str] = []

        self._voltage_probe_index: Dict[str, int] = {}
        self._branch_current_probe_index: Dict[str, int] = {}

        self._voltage_probe_data: Optional[np.ndarray] = None
        self._branch_current_probe_data: Optional[np.ndarray] = None

        # ---- 时间与结果 ----
        self.time: float = 0.0
        self.step_count: int = 0
        self.time_array: list = []
        self.voltage_results: Dict[int, list] = {}
        self._actual_steps: int = 0
        self._results_valid: bool = False
        self._is_running: bool = False

        # ---- 分段线性法 ----
        self._seg_node_map: Dict[str, Tuple[int, int]] = {}
        self.seg_helper = SegmentedSolverHelper()
        self._has_nonlinear: bool = False

        # ---- LPM 绝缘子闪络 ----
        self._lpm_elements: Dict[str, InsulatorFlashoverLPM] = {}
        self._lpm_node_map: Dict[str, Tuple[int, int]] = {}
        self._lpm_flashover_log: list = []

        # ---- MNA 稀疏矩阵缓存 ----
        self._G_dirty: bool = True
        self._cached_MNA: Optional[sp.csc_matrix] = None  # MNA 系统矩阵
        self._cached_splu: Optional[Any] = None            # SuperLU 分解
        self._mna_size: int = 0                             # n + m
        self._vs_list: Optional[List[VoltageSource]] = None # 有序电压源列表
        self._vs_index_map: Optional[Dict[str, int]] = None # name → 增广索引

        # ---- 统计 ----
        self._stats: Dict[str, Any] = self._fresh_stats()

        # ---- 计时 ----
        self._timing: Dict[str, float] = defaultdict(float)

        # ---- ULM batch 运行时缓存 ----
        self._ulm_batch: Optional[Any] = None
        self._ulm_batch_meta: list = []
        self._ulm_batch_line_index: Dict[int, int] = {}
        self._line_inject_maps: list = []
        self._line_inject_maps_nonbatch: list = []
        self._line_vk_bufs: Dict[str, np.ndarray] = {}
        self._line_vm_bufs: Dict[str, np.ndarray] = {}
        self._ulm_batch_k_rows: Optional[np.ndarray] = None
        self._ulm_batch_k_slots: Optional[np.ndarray] = None
        self._ulm_batch_k_nodes: Optional[np.ndarray] = None
        self._ulm_batch_k_valid: Optional[np.ndarray] = None
        self._ulm_batch_m_rows: Optional[np.ndarray] = None
        self._ulm_batch_m_slots: Optional[np.ndarray] = None
        self._ulm_batch_m_nodes: Optional[np.ndarray] = None
        self._ulm_batch_m_valid: Optional[np.ndarray] = None
        self._ulm_batch_k_rows_v: Optional[np.ndarray] = None
        self._ulm_batch_k_slots_v: Optional[np.ndarray] = None
        self._ulm_batch_k_nodes_v: Optional[np.ndarray] = None
        self._ulm_batch_m_rows_v: Optional[np.ndarray] = None
        self._ulm_batch_m_slots_v: Optional[np.ndarray] = None
        self._ulm_batch_m_nodes_v: Optional[np.ndarray] = None
        self._rhs_buf: Optional[np.ndarray] = None

        # ---- 源预采样缓存 ----
        self._current_source_samples: Dict[str, np.ndarray] = {}
        self._voltage_source_samples: Dict[str, np.ndarray] = {}

    @staticmethod
    def _fresh_stats() -> Dict[str, Any]:
        return {
            'total_steps': 0,
            'segment_switches': 0,
            'segment_resolves': 0,
            'max_seg_iter': 0,
            'lpm_resolves': 0,
            'lpm_flashovers': 0,
            'transformer_saturation_resolves': 0,
            'transformer_saturation_switches': 0,
        }

    def mark_topology_changed(self, reason: str = "") -> None:
        """Invalidate cached MNA matrix/factorization after topology changes."""
        self._stamping.mark_dirty()
        self._vs_list = None
        self._vs_index_map = None
        self._rhs_plan_dirty = True
        if not getattr(self, "_is_running", False):
            self._invalidate_results()
        if self.verbose and reason:
            logger.debug("MNA matrix marked dirty: %s", reason)

    def _invalidate_results(self) -> None:
        """Mark stored outputs stale after topology/probe/config changes."""
        self._results_valid = False
        self._actual_steps = 0

    def _ensure_unique_device_name(self, name: str, kind: str = "device") -> None:
        """Reject duplicate public device names before mutating topology."""
        name = str(name)
        containers = (
            ("branch", self.branches),
            ("current source", self.current_sources),
            ("voltage source", self.voltage_sources),
            ("transmission line", self.transmission_lines),
            ("transformer", self.transformers),
        )
        for existing_kind, container in containers:
            if name in container:
                raise ValueError(
                    f"{kind} name {name!r} conflicts with existing {existing_kind}"
                )

    def _ensure_unique_probe_name(self, name: str) -> None:
        """Reject duplicate probe names across all probe namespaces."""
        name = str(name)
        if name in self.voltage_probes or name in self.branch_current_probes:
            raise ValueError(f"Probe name {name!r} already exists")

    def _require_run_completed(self) -> None:
        """Raise a clear error when result APIs are used before run()."""
        if (
            not bool(getattr(self, "_results_valid", False))
            or int(getattr(self, "_actual_steps", 0)) <= 0
        ):
            raise RuntimeError(
                "Simulation results are not available or are stale; call run() first."
            )

    # =========================================================================
    # 节点管理
    # =========================================================================

    def _update_node_count(self, *nodes) -> None:
        """更新节点集合。参数可为整数或整数序列。"""
        for n in nodes:
            if isinstance(n, (list, tuple, np.ndarray)):
                for node in n:
                    if node > 0:
                        self._node_set.add(int(node))
                        self._indexer.register(int(node))
            elif isinstance(n, (int, np.integer)) and n > 0:
                self._node_set.add(int(n))
                self._indexer.register(int(n))
        self.num_nodes = max(self._node_set) if self._node_set else 0

    # ---- 命名节点解析 (NodeBook 桥接) ----

    def node(self, node: Union[str, int, np.integer]) -> int:
        """把节点名或节点号解析成整数节点号(对外公开)。"""
        return self.nodes.get(node)

    def node_name(self, node_id: int) -> Optional[str]:
        """由整数节点号反查节点名。"""
        return self.nodes.name_of(node_id)

    def bind_node(self, name: str, node_id: Optional[int] = None) -> int:
        """手动绑定节点名,兼容已有整数节点模型。"""
        return self.nodes.reserve(name, node_id)

    def alias_node(self, alias_name: str, existing: Union[str, int]) -> int:
        """给已有节点增加别名。"""
        return self.nodes.alias(alias_name, existing)

    def _resolve_node(self, node: Union[str, int, np.integer]) -> int:
        """内部使用:解析单个节点(字符串或整数)为整数节点号。"""
        return self.nodes.get(node)

    def _resolve_existing_node(self, node: Union[str, int, np.integer]) -> int:
        """Resolve a node reference without creating a new named node."""
        if isinstance(node, (int, np.integer)):
            node_id = int(node)
            if node_id < 0:
                raise ValueError(f"节点编号必须 >= 0,当前为 {node}")
            return node_id

        name = str(node)
        if name in self.nodes.GROUND_NAMES:
            return 0
        if name not in self.nodes:
            raise ValueError(
                f"节点 {name!r} 尚未在电路中定义；探针不会自动创建节点"
            )
        return self.nodes.get(name)

    def _resolve_nodes(self, nodes):
        """内部使用:递归解析节点或节点列表。"""
        if isinstance(nodes, (list, tuple, np.ndarray)):
            return [self._resolve_node(n) for n in nodes]
        return self._resolve_node(nodes)

    def _node_label(self, node: int) -> str:
        """用于打印:返回带名称的节点标签,如 'T1.tower_top(5)'。"""
        if node == 0:
            return "GND(0)"
        name = self.node_name(node)
        if name:
            return f"{name}({node})"
        return str(node)

    # =========================================================================
    # 轻量探针 API
    # =========================================================================

    def add_voltage_probe(
        self,
        name: str,
        node_pos: Union[str, int],
        node_neg: Union[str, int] = 0,
    ) -> None:
        """注册电压探针，记录 V(node_pos) - V(node_neg)。

        探针只引用既有节点，不改变电路拓扑；字符串节点名必须已经由元件
        或 reserve_node()/alias_node() 注册。
        """
        self._ensure_unique_probe_name(name)
        node_pos_id = self._resolve_existing_node(node_pos)
        node_neg_id = self._resolve_existing_node(node_neg)
        self.voltage_probes[str(name)] = {
            "node_pos": int(node_pos_id),
            "node_neg": int(node_neg_id),
        }
        self._invalidate_results()

    def add_branch_current_probe(self, name: str, branch_name: str) -> None:
        """注册普通支路电流探针。"""
        self._ensure_unique_probe_name(name)
        self.branch_current_probes[str(name)] = {
            "branch_name": str(branch_name),
        }
        self._invalidate_results()

    def _init_probe_storage(self, n_steps: int) -> None:
        """仿真开始前预分配探针结果数组。"""
        self._voltage_probe_names = list(self.voltage_probes.keys())
        self._branch_current_probe_names = list(self.branch_current_probes.keys())

        self._voltage_probe_index = {
            name: i for i, name in enumerate(self._voltage_probe_names)
        }
        self._branch_current_probe_index = {
            name: i for i, name in enumerate(self._branch_current_probe_names)
        }

        self._voltage_probe_data = None
        self._branch_current_probe_data = None

        if self._voltage_probe_names:
            self._voltage_probe_data = np.empty(
                (n_steps, len(self._voltage_probe_names)),
                dtype=np.float64,
            )

        if self._branch_current_probe_names:
            self._branch_current_probe_data = np.empty(
                (n_steps, len(self._branch_current_probe_names)),
                dtype=np.float64,
            )

    @staticmethod
    def _scale_probe_values(values: np.ndarray, unit: Optional[str]) -> np.ndarray:
        return scale_probe_values(values, unit)

    @staticmethod
    def _scale_values(
        values: np.ndarray,
        unit: Optional[str],
        scale_map: Dict[str, float],
        quantity: str,
    ) -> np.ndarray:
        return scale_values(values, unit, scale_map, quantity)

    def _node_voltage_from_solution(self, V: np.ndarray, node: int) -> float:
        return node_voltage_from_solution(V, node, self._indexer.to_compact)

    def _branch_voltage_from_solution(self, V: np.ndarray, branch: Branch) -> float:
        return branch_voltage_from_solution(V, branch, self._indexer.to_compact)

    def _branch_current_from_solution(self, V: np.ndarray, branch: Branch) -> float:
        return branch_current_from_solution(V, branch, self._indexer.to_compact)

    def _record_voltage_probes(self, step: int, V: np.ndarray) -> None:
        if self._voltage_probe_data is None:
            return

        for j, name in enumerate(self._voltage_probe_names):
            p = self.voltage_probes[name]
            vp = self._node_voltage_from_solution(V, p["node_pos"])
            vn = self._node_voltage_from_solution(V, p["node_neg"])
            self._voltage_probe_data[step, j] = vp - vn

    def _record_branch_current_probes(self, step: int, V: np.ndarray) -> None:
        if self._branch_current_probe_data is None:
            return

        for j, name in enumerate(self._branch_current_probe_names):
            p = self.branch_current_probes[name]
            branch_name = p["branch_name"]
            if branch_name not in self.branches:
                raise ValueError(f"支路电流探针 {name} 引用的支路不存在: {branch_name}")

            br = self.branches[branch_name]
            self._branch_current_probe_data[step, j] = (
                self._branch_current_from_solution(V, br)
            )

    def _record_probes(self, step: int, V: np.ndarray) -> None:
        """每步求解并完成状态更新后记录全部探针。"""
        self._record_voltage_probes(step, V)
        self._record_branch_current_probes(step, V)

    def get_voltage_probe(self, name: str, unit: Optional[str] = "V") -> np.ndarray:
        self._require_run_completed()
        if name not in self._voltage_probe_index:
            raise KeyError(f"电压探针不存在: {name}")
        if self._voltage_probe_data is None:
            raise RuntimeError(f"电压探针 {name!r} 未记录数据")
        idx = self._voltage_probe_index[name]
        actual = getattr(self, "_actual_steps", self._voltage_probe_data.shape[0])
        data = self._voltage_probe_data[:actual, idx]
        return self._scale_probe_values(data, unit)

    def get_branch_current_probe(
        self,
        name: str,
        unit: Optional[str] = "A",
    ) -> np.ndarray:
        self._require_run_completed()
        if name not in self._branch_current_probe_index:
            raise KeyError(f"支路电流探针不存在: {name}")
        if self._branch_current_probe_data is None:
            raise RuntimeError(f"支路电流探针 {name!r} 未记录数据")
        idx = self._branch_current_probe_index[name]
        actual = getattr(self, "_actual_steps", self._branch_current_probe_data.shape[0])
        data = self._branch_current_probe_data[:actual, idx]
        return self._scale_probe_values(data, unit)

    def get_probe(self, name: str, unit: Optional[str] = None) -> np.ndarray:
        """统一读取探针波形。unit 可用 V/kV/mV 或 A/kA/mA。"""
        if name in self._voltage_probe_index:
            return self.get_voltage_probe(name, unit or "V")
        if name in self._branch_current_probe_index:
            return self.get_branch_current_probe(name, unit or "A")
        raise KeyError(f"探针不存在: {name}")

    def list_probes(self) -> Dict[str, List[str]]:
        """列出已注册探针。"""
        return {
            "voltage": list(self._voltage_probe_names or self.voltage_probes.keys()),
            "branch_current": list(
                self._branch_current_probe_names or self.branch_current_probes.keys()
            ),
        }


    # =========================================================================
    # 基本元件
    # =========================================================================

    def add_R(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        R: float,
    ) -> None:
        """添加电阻。支持整数节点号或字符串节点名。"""
        self._ensure_unique_device_name(name, "resistor")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        if R <= 0:
            raise ValueError(f"电阻值必须为正: R={R}")
        dev = ResistorDevice(name, node_from, node_to, R)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add resistor: {name}")

    def add_L(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        L: float, Rp: Optional[float] = None,
    ) -> None:
        """添加电感(隐式梯形:G_eq = Δt/(2L))。支持字符串节点名。"""
        self._ensure_unique_device_name(name, "inductor")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        if L <= 0:
            raise ValueError(f"电感值必须为正: L={L}")
        dev = InductorDevice(name, node_from, node_to, L, self.dt,
                             Rp=Rp if Rp else 0.0)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add inductor: {name}")

    def add_C(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        C: float, Rp: Optional[float] = None,
    ) -> None:
        """添加电容(隐式梯形:G_eq = 2C/Δt)。支持字符串节点名。"""
        self._ensure_unique_device_name(name, "capacitor")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        if C <= 0:
            raise ValueError(f"电容值必须为正: C={C}")
        dev = CapacitorDevice(name, node_from, node_to, C, self.dt,
                              Rp=Rp if Rp else 0.0)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add capacitor: {name}")

    def add_resistor(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        R: float,
    ) -> None:
        """Alias for add_R()."""
        self.add_R(name, node_from, node_to, R)

    def add_inductor(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        L: float,
        Rp: Optional[float] = None,
    ) -> None:
        """Alias for add_L()."""
        self.add_L(name, node_from, node_to, L, Rp=Rp)

    def add_capacitor(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        C: float,
        Rp: Optional[float] = None,
    ) -> None:
        """Alias for add_C()."""
        self.add_C(name, node_from, node_to, C, Rp=Rp)


    def add_series_RL(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        R: float, L: float,
    ) -> None:
        """添加无中间节点的串联 RL 二端支路。

        该支路不会拆成 R 和 L 两个元件，也不会创建内部节点。
        它直接在 node_from 与 node_to 之间形成一个二端 Norton 等效，
        用于缩减 MNA 矩阵节点数。

        梯形离散：
            i = G_L * v_L + I_L_hist
            v_L = v_branch - R * i

        合并得到：
            i = G_eq * v_branch + I_eq
            G_eq = G_L / (1 + R * G_L)
            I_eq = I_L_hist / (1 + R * G_L)
        """
        self._ensure_unique_device_name(name, "series RL branch")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        if R < 0:
            raise ValueError(f"串联 RL 的 R 不能为负: R={R}")
        if L <= 0:
            raise ValueError(f"电感值必须为正: L={L}")

        dev = SeriesRLDevice(name, node_from, node_to, R, L, self.dt)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add series RL: {name}")

    def add_SW(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        t_close: float = -1.0, t_open: float = -1.0,
        R_closed: float = 1e-6, R_open: float = 1e9,
        initially_closed: bool = False,
    ) -> None:
        """添加定时开关。t_close / t_open < 0 表示不动作。支持字符串节点名。"""
        self._ensure_unique_device_name(name, "switch")
        if R_closed <= 0 or R_open <= 0:
            raise ValueError("开关 R_closed/R_open 必须为正")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        dev = SwitchDevice(name, node_from, node_to,
                           t_close, t_open, R_closed, R_open,
                           bool(initially_closed))
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add switch: {name}")

    def add_switch(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        t_close: float = -1.0,
        t_open: float = -1.0,
        R_closed: float = 1e-6,
        R_open: float = 1e9,
        initially_closed: bool = False,
    ) -> None:
        """Alias for add_SW()."""
        self.add_SW(
            name, node_from, node_to, t_close=t_close, t_open=t_open,
            R_closed=R_closed, R_open=R_open,
            initially_closed=initially_closed,
        )

    def add_IS(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        current_func: Union[Callable[[float], float], float, LightningWaveform],
    ) -> None:
        """添加电流源,支持 LightningWaveform、常数或函数。支持字符串节点名。"""
        self._ensure_unique_device_name(name, "current source")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        current_func = self._coerce_current_source_function(current_func)

        self.current_sources[name] = CurrentSource(
            name=name, node_from=node_from, node_to=node_to,
            current_func=current_func,
        )
        self._update_node_count(node_from, node_to)
        self.mark_topology_changed(f"add current source: {name}")

    def add_current_source(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        current_func: Union[Callable[[float], float], float, LightningWaveform],
    ) -> None:
        """Alias for add_IS()."""
        self.add_IS(name, node_from, node_to, current_func)


    @staticmethod
    def _lightning_source_to_callable(source: Any) -> Callable[[float], float]:
        """Adapt ATP lightning-current source objects to solver current_func(t).

        The ATP generator stores waveform parameters but intentionally removed
        sampling/evaluation helpers.  The EMTP solver only needs a scalar
        function of time, so we reconstruct it from the source's raw waveform,
        peak scale, Tstart and optional Tstop.

        For the fallback path (no ``current_at`` method), a reusable scalar
        buffer avoids allocating ``np.array([t_rel])`` on every time-step.
        """
        # Prefer scalar fast path (avoids per-step np.array allocation).
        if hasattr(source, "current_at_scalar"):
            return lambda t, src=source: float(src.current_at_scalar(t))
        if hasattr(source, "current_at"):
            return lambda t, src=source: float(src.current_at(t))

        _buf = np.empty(1, dtype=float)

        def current_at(t: float, src=source, buf=_buf) -> float:
            t_rel = float(t) - float(getattr(src, "Tstart", 0.0))
            if t_rel < 0.0:
                return 0.0
            tstop = getattr(src, "Tstop", None)
            if tstop is not None and float(t) > float(tstop):
                return 0.0
            buf[0] = t_rel
            raw = float(src._raw(buf)[0])
            return float(src.peak) * float(src.k_factor) * raw
        return current_at

    @classmethod
    def _coerce_current_source_function(
        cls,
        current_func: Union[Callable[[float], float], float, Any],
    ) -> Callable[[float], float]:
        """Normalize constants, legacy LightningWaveform and ATP sources."""
        if isinstance(current_func, (int, float)):
            const_val = float(current_func)
            return lambda t, v=const_val: v

        if hasattr(current_func, "get_waveform_function"):
            return current_func.get_waveform_function()

        if BaseLightningCurrentSource and isinstance(current_func, BaseLightningCurrentSource):
            return cls._lightning_source_to_callable(current_func)

        if callable(current_func):
            return current_func

        raise TypeError(
            "current_func must be a callable, a number, a legacy LightningWaveform, "
            "or an ATP lightning current source object"
        )

    def add_lightning_IS(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        *,
        model: str = "heidlerf",
        peak: float,
        T1: float,
        T2: float,
        n: float = 10.0,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
        **kwargs: Any,
    ) -> Any:
        """Create an ATP-compatible lightning current source and add it as IS.

        Returns the created ATP source object so callers can inspect parameters
        with get_info()/print_info().
        """
        self._ensure_unique_device_name(name, "lightning current source")
        if create_lightning_current_source is None:
            raise ImportError(
                "atp_lightning_current_generator_simplified.py is required for add_lightning_IS"
            )
        source = create_lightning_current_source(
            model=model, peak=peak, T1=T1, T2=T2, n=n, PERC=PERC,
            Tstart=Tstart, Tstop=Tstop, atp_compatible=atp_compatible,
            description=description, **kwargs,
        )
        self.add_IS(name, node_from, node_to, source)
        return source

    def add_lightning_current_source(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        *,
        model: str = "heidlerf",
        peak: float,
        T1: float,
        T2: float,
        n: float = 10.0,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
        **kwargs: Any,
    ) -> Any:
        """Recommended alias for :meth:`add_lightning_IS`."""
        return self.add_lightning_IS(
            name, node_from, node_to,
            model=model, peak=peak, T1=T1, T2=T2, n=n, PERC=PERC,
            Tstart=Tstart, Tstop=Tstop, atp_compatible=atp_compatible,
            description=description, **kwargs,
        )

    def add_standard_twoexpf_IS(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        *,
        waveform_type: str,
        peak: float,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
    ) -> Any:
        """Add a standard-library TWOEXPF lightning current source as IS."""
        self._ensure_unique_device_name(name, "lightning current source")
        if create_standard_twoexpf_current_source is None:
            raise ImportError(
                "atp_lightning_current_generator_simplified.py is required for add_standard_twoexpf_IS"
            )
        source = create_standard_twoexpf_current_source(
            waveform_type=waveform_type, peak=peak, PERC=PERC,
            Tstart=Tstart, Tstop=Tstop, atp_compatible=atp_compatible,
            description=description,
        )
        self.add_IS(name, node_from, node_to, source)
        return source

    def add_standard_double_exponential_current_source(
        self,
        name: str,
        node_from: Union[str, int],
        node_to: Union[str, int],
        *,
        waveform_type: str,
        peak: float,
        PERC: int = 30,
        Tstart: float = 0.0,
        Tstop: Optional[float] = None,
        atp_compatible: bool = True,
        description: str = "",
    ) -> Any:
        """Recommended alias for :meth:`add_standard_twoexpf_IS`."""
        return self.add_standard_twoexpf_IS(
            name, node_from, node_to,
            waveform_type=waveform_type, peak=peak, PERC=PERC,
            Tstart=Tstart, Tstop=Tstop, atp_compatible=atp_compatible,
            description=description,
        )

    # =========================================================================
    # 理想电压源 (MNA 增广方程)
    # =========================================================================

    def add_VS(
        self, name: str,
        node_pos: Union[str, int], node_neg: Union[str, int],
        voltage_func: Union[Callable[[float], float], float],
    ) -> None:
        """添加理想电压源。支持字符串节点名。

        MNA 增广方程中,每个电压源引入一个额外方程:
        e(node_pos) - e(node_neg) = voltage_func(t)。
        node_neg=0 (或 "GND") 表示接地。

        Raises
        ------
        ValueError
            node_pos <= 0 或节点已被其它电压源指定。
        """
        self._ensure_unique_device_name(name, "voltage source")
        node_pos = self._resolve_node(node_pos)
        node_neg = self._resolve_node(node_neg)
        if node_pos <= 0:
            raise ValueError(f"电压源 {name} 正端必须 > 0,当前 {node_pos}")
        if node_pos == node_neg:
            raise ValueError(f"电压源 {name} 的正负端不能是同一节点")
        if node_pos in self._vs_node_set:
            raise ValueError(f"节点 {node_pos} 已被另一个电压源指定,不能重复")
        if node_neg > 0 and node_neg in self._vs_node_set:
            raise ValueError(
                f"节点 {node_neg} 已被电压源指定,不能同时作为另一电压源的负端"
            )

        if isinstance(voltage_func, (int, float)):
            const_val = float(voltage_func)
            voltage_func = lambda t, v=const_val: v

        source = VoltageSource(
            name=name, node_pos=node_pos, node_neg=node_neg,
            voltage_func=voltage_func,
        )
        self.voltage_sources[name] = source
        self._vs_node_set.add(node_pos)
        self._update_node_count(node_pos, node_neg)
        self.mark_topology_changed(f"add voltage source: {name}")

        logger.debug("添加电压源 %s: (%d-%d), V(0)=%.2f",
                     name, node_pos, node_neg, source.voltage_at(0.0))

    def add_voltage_source(
        self,
        name: str,
        node_pos: Union[str, int],
        node_neg: Union[str, int],
        voltage_func: Union[Callable[[float], float], float],
    ) -> None:
        """Alias for add_VS()."""
        self.add_VS(name, node_pos, node_neg, voltage_func)

    # =========================================================================
    # 分段线性避雷器 (MOA)
    # =========================================================================


    def add_MOA_from_file(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        file_path: str, rated_voltage: float,
        voltage_is_pu: bool = True,
    ) -> None:
        """从 V-I 数据文件添加 MOA。支持字符串节点名。

        文件格式::

            # 注释
            current voltage
            ...
            ENDFILE
        """
        self._ensure_unique_device_name(name, "segmented MOA")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        model = SegmentedMOAResistor.from_file(
            name=name, file_path=file_path,
            rated_voltage=rated_voltage, voltage_is_pu=voltage_is_pu,
        )
        self._register_segmented_moa(
            name, node_from, node_to, model, Rp=rated_voltage * 1e3,
        )
        logger.debug("从文件添加 MOA %s: %s, Vn=%.1fkV, 分段=%d",
                     name, file_path, rated_voltage / 1e3, model.num_segments)


    def _register_segmented_moa(
        self, name: str, node_from: int, node_to: int,
        model: SegmentedMOAResistor, Rp: float,
    ) -> None:
        """通用注册:构造 Branch + 写入 seg_helper + 更新节点。"""
        self._ensure_unique_device_name(name, "segmented MOA")
        g_init, i_init = model.get_norton_equivalent(0)
        dev = NonlinearResistorDevice(name, node_from, node_to,
                                      g_init, i_init, model, Rp)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self.seg_helper.register(name, model)
        self._seg_node_map[name] = (node_from, node_to)
        self._has_nonlinear = True
        self._devices.append(dev)
        self.mark_topology_changed(f"add segmented nonlinear: {name}")

    # =========================================================================
    # LPM 绝缘子闪络开关
    # =========================================================================

    def add_insulator_LPM(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        gap_length: float,
        k: float = 1.0e-6, E0: float = 600.0,
        R_arc: float = 1.0, R_open: float = 1e9,
        altitude_m: float = 0.0,
        include_predischarge: bool = False,
        allow_extinction: bool = False,
        extinction_current: float = 0.0,
        **kwargs,
    ) -> InsulatorFlashoverLPM:
        """添加 CIGRE 先导发展法闪络开关。支持字符串节点名。

        并联在绝缘子两端,间隙电压满足先导发展条件且先导桥接时闭合。
        CIGRE 速度公式: v(t) = k · u(t) · [ u(t)/(d-l) - E₀ ]
        """
        self._ensure_unique_device_name(name, "LPM switch")
        if R_arc <= 0 or R_open <= 0:
            raise ValueError("LPM R_arc/R_open 必须为正")
        node_from = self._resolve_node(node_from)
        node_to = self._resolve_node(node_to)
        config = LPMConfig(
            gap_length=gap_length, k=k, E0=E0,
            R_open=R_open, R_arc=R_arc,
            altitude_m=altitude_m,
            include_predischarge=include_predischarge,
            allow_extinction=allow_extinction,
            extinction_current=extinction_current,
            **kwargs,
        )
        lpm_model = InsulatorFlashoverLPM(name, config)

        dev = LPMFlashoverDevice(name, node_from, node_to, R_open, R_arc)
        self.branches[name] = dev._branch
        self._update_node_count(node_from, node_to)
        self._devices.append(dev)
        self.mark_topology_changed(f"add LPM switch: {name}")

        self._lpm_elements[name] = lpm_model
        self._lpm_node_map[name] = (node_from, node_to)

        logger.debug(
            "添加 LPM %s: (%d-%d), d=%.3fm, k=%.2e, E0=%.0fkV/m, "
            "R_arc=%.1fΩ, R_open=%.2eΩ",
            name, node_from, node_to,
            gap_length, k, E0, R_arc, R_open,
        )
        return lpm_model

    def add_lpm_flashover_insulator(
        self, name: str,
        node_from: Union[str, int], node_to: Union[str, int],
        gap_length: float,
        k: float = 1.0e-6, E0: float = 600.0,
        R_arc: float = 1.0, R_open: float = 1e9,
        altitude_m: float = 0.0,
        include_predischarge: bool = False,
        allow_extinction: bool = False,
        extinction_current: float = 0.0,
        **kwargs,
    ) -> Any:
        """Recommended alias for :meth:`add_insulator_LPM`.

        Adds a CIGRE leader-progression-model insulator flashover switch
        in parallel with the insulator.  The gap flashes over when the
        leader bridges the gap under the applied voltage.

        Parameters
        ----------
        gap_length : float
            Insulator gap length in **metres**.
        k : float
            CIGRE velocity constant (m²/(kV²·s)).
        E0 : float
            Critical electric field in **kV/m**.
        R_arc : float
            Arc resistance in **ohms** after flashover.
        R_open : float
            Open-circuit resistance in **ohms** before flashover.
        altitude_m : float
            Altitude above sea level in **metres** (affects E0).
        """
        return self.add_insulator_LPM(
            name, node_from, node_to,
            gap_length=gap_length, k=k, E0=E0,
            R_arc=R_arc, R_open=R_open,
            altitude_m=altitude_m,
            include_predischarge=include_predischarge,
            allow_extinction=allow_extinction,
            extinction_current=extinction_current,
            **kwargs,
        )


    # ---- LPM 结果获取 ----

    def get_insulator_leader_length(
        self, name: str, unit: str = 'm',
    ) -> np.ndarray:
        """先导长度历史。unit ∈ {'m','mm','cm'}。"""
        self._require_lpm(name)
        L = np.array(self._lpm_elements[name].leader_length_history)
        if unit == 'mm':
            return L * 1e3
        if unit == 'cm':
            return L * 1e2
        return L

    def get_insulator_leader_velocity(self, name: str) -> np.ndarray:
        """先导速度历史 (m/s)。"""
        self._require_lpm(name)
        return np.array(self._lpm_elements[name].leader_velocity_history)

    def get_insulator_voltage(self, name: str, unit: str = 'kV') -> np.ndarray:
        """间隙电压历史。默认 kV。"""
        self._require_lpm(name)
        V = np.array(self._lpm_elements[name].voltage_history)
        return V * 1e3 if unit == 'V' else V

    def get_insulator_state(self, name: str) -> np.ndarray:
        """开关状态历史 (0=开路, 1=闪络)。"""
        self._require_lpm(name)
        return np.array(self._lpm_elements[name].state_history)

    def get_insulator_info(self, name: str) -> Dict[str, Any]:
        self._require_lpm(name)
        return self._lpm_elements[name].get_info()

    def get_flashover_log(self) -> list:
        return list(self._lpm_flashover_log)

    def _require_lpm(self, name: str) -> None:
        if name not in self._lpm_elements:
            raise ValueError(f"LPM 绝缘子 {name} 不存在")

    # =========================================================================
    # =========================================================================
    # PSCAD-style transmission-line compile stage
    # =========================================================================

    def compile_transmission_lines(
        self,
        max_workers: Optional[int] = None,
        reserve_cores: int = 1,
        force: bool = False,
    ) -> None:
        """PSCAD 风格的线路编译阶段：把每条传输线分配到不同 CPU 核预初始化。"""
        lines = list(self.transmission_lines.values())
        if not lines:
            self._lines_compiled = True
            return
        if self._lines_compiled and not force:
            return

        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            max_workers = max(1, cpu_count - max(0, reserve_cores))
        max_workers = max(1, min(int(max_workers), len(lines)))

        def _compile_one(line: TransmissionLineInterface) -> str:
            line.initialize(self.dt)
            line.update_history_sources()
            return line.name

        t0 = _perf_time.perf_counter()
        if max_workers == 1 or len(lines) == 1:
            for line in lines:
                _compile_one(line)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_compile_one, line) for line in lines]
                for fut in as_completed(futures):
                    fut.result()

        self._lines_compiled = True
        self._timing['line_compile'] += _perf_time.perf_counter() - t0
        if self.verbose:
            logger.info(
                "PSCAD式线路编译完成: %d 条线路, workers=%d, 预留核心=%d",
                len(lines), max_workers, reserve_cores,
            )

    def _choose_ulm_batch_parallel(self, n_lines: int) -> bool:
        """选择 ULM batch kernel 类型。"""
        mode = getattr(self, 'ulm_batch_mode', 'auto')

        if mode == 'parallel':
            return True
        if mode == 'serial':
            return False
        if mode == 'off':
            return False
        if mode != 'auto':
            raise ValueError(f"未知 ulm_batch_mode: {mode!r}")

        try:
            from numba import get_num_threads
            n_threads = int(get_num_threads())
        except Exception:
            n_threads = os.cpu_count() or 1

        threshold_factor = max(
            1,
            int(getattr(self, 'ulm_batch_parallel_threshold_factor', 4)),
        )
        return int(n_lines) >= threshold_factor * max(1, n_threads)

    def _build_ulm_batch_runtime(self) -> None:
        """在线路已编译后创建/刷新 ULM batch 运行时。

        ``ulm_batch_mode`` 控制运行方式：
        - off      : 关闭 ULMBatchPack，走逐线 ``line.full_step`` fallback。
        - serial   : ULMBatchPack + serial njit kernel。
        - parallel : ULMBatchPack + Numba parallel/prange kernel。
        - auto     : 根据线路数和线程数自动选择 serial/parallel。
        """
        ulm_lines_for_batch = []
        if ULM_AVAILABLE and ULMLine is not None:
            ulm_lines_for_batch = [
                line for line in self.transmission_lines.values()
                if isinstance(line, ULMLine)
            ]

        mode = getattr(self, 'ulm_batch_mode', 'auto')
        if len(ulm_lines_for_batch) < 2 or mode == 'off':
            self._ulm_batch = None
            self._ulm_batch_meta = []
            self._ulm_batch_line_index = {}
            self._line_inject_maps_nonbatch = list(self._line_inject_maps)
            self._ulm_batch_k_rows = self._ulm_batch_k_slots = self._ulm_batch_k_nodes = self._ulm_batch_k_valid = None
            self._ulm_batch_m_rows = self._ulm_batch_m_slots = self._ulm_batch_m_nodes = self._ulm_batch_m_valid = None
            self._ulm_batch_k_rows_v = self._ulm_batch_k_slots_v = self._ulm_batch_k_nodes_v = None
            self._ulm_batch_m_rows_v = self._ulm_batch_m_slots_v = self._ulm_batch_m_nodes_v = None
            if mode == 'off' and len(ulm_lines_for_batch) >= 2:
                logger.info("ULM batch 已关闭: %d 条线路走逐线 fallback", len(ulm_lines_for_batch))
            return

        use_parallel = self._choose_ulm_batch_parallel(len(ulm_lines_for_batch))
        self._ulm_batch = ULMBatchPack(ulm_lines_for_batch, parallel=use_parallel)
        self._ulm_batch.import_state_from_lines()
        self._ulm_batch.bind_line_fast_views()

        NL = self._ulm_batch.n_lines
        max_nc = self._ulm_batch.max_nc
        self._ulm_batch._vk_in_work = np.zeros((NL, max_nc), dtype=np.float64)
        self._ulm_batch._vm_in_work = np.zeros((NL, max_nc), dtype=np.float64)
        self._ulm_batch_line_index = {
            id(line): idx for idx, line in enumerate(ulm_lines_for_batch)
        }

        line_map_by_id = {id(item[0]): item for item in self._line_inject_maps}
        self._ulm_batch_meta = []

        k_rows: List[int] = []
        k_slots: List[int] = []
        k_nodes: List[int] = []
        m_rows: List[int] = []
        m_slots: List[int] = []
        m_nodes: List[int] = []

        for row, line in enumerate(ulm_lines_for_batch):
            _, k_idx, m_idx, nc, _is_multi, _has_full = line_map_by_id[id(line)]
            phase_slots = self._get_line_phase_slots(line, nc)
            if len(phase_slots) != nc:
                raise ValueError(
                    f"ULM batch 相索引数量错误: line={line.name}, "
                    f"nc={nc}, phase_slots={phase_slots}"
                )
            if np.any(phase_slots < 0) or np.any(phase_slots >= max_nc):
                raise ValueError(
                    f"ULM batch 相索引越界: line={line.name}, "
                    f"max_nc={max_nc}, phase_slots={phase_slots}"
                )
            self._ulm_batch_meta.append((line, k_idx, m_idx, nc, phase_slots))

            for i in range(nc):
                slot = int(phase_slots[i])
                k_rows.append(row); k_slots.append(slot); k_nodes.append(int(k_idx[i]))
                m_rows.append(row); m_slots.append(slot); m_nodes.append(int(m_idx[i]))

        self._ulm_batch_k_rows = np.asarray(k_rows, dtype=np.int64)
        self._ulm_batch_k_slots = np.asarray(k_slots, dtype=np.int64)
        self._ulm_batch_k_nodes = np.asarray(k_nodes, dtype=np.int64)
        self._ulm_batch_k_valid = self._ulm_batch_k_nodes >= 0
        self._ulm_batch_k_rows_v = self._ulm_batch_k_rows[self._ulm_batch_k_valid]
        self._ulm_batch_k_slots_v = self._ulm_batch_k_slots[self._ulm_batch_k_valid]
        self._ulm_batch_k_nodes_v = self._ulm_batch_k_nodes[self._ulm_batch_k_valid]

        self._ulm_batch_m_rows = np.asarray(m_rows, dtype=np.int64)
        self._ulm_batch_m_slots = np.asarray(m_slots, dtype=np.int64)
        self._ulm_batch_m_nodes = np.asarray(m_nodes, dtype=np.int64)
        self._ulm_batch_m_valid = self._ulm_batch_m_nodes >= 0
        self._ulm_batch_m_rows_v = self._ulm_batch_m_rows[self._ulm_batch_m_valid]
        self._ulm_batch_m_slots_v = self._ulm_batch_m_slots[self._ulm_batch_m_valid]
        self._ulm_batch_m_nodes_v = self._ulm_batch_m_nodes[self._ulm_batch_m_valid]

        ulm_batch_set = set(self._ulm_batch_line_index)
        self._line_inject_maps_nonbatch = [
            item for item in self._line_inject_maps
            if id(item[0]) not in ulm_batch_set
        ]
        logger.info(
            "ULM batch 已启用: %d 条线路, max_nc=%d, kernel=%s, mode=%s, record_line_history=%s",
            NL, max_nc, 'parallel' if use_parallel else 'serial', mode, self.record_line_history,
        )

    # UMEC 变压器
    # =========================================================================

    def add_UMEC_transformer(
        self, name: str, data: 'UMECTransformerData',
    ) -> 'UMECTransformer':
        """添加 UMEC 变压器,按多端口邮票法入网。"""
        self._ensure_unique_device_name(name, "UMEC transformer")
        if not UMEC_AVAILABLE:
            raise ImportError("UMEC 模块不可用,请确保 umec_transformer.py 可导入")

        xfmr = UMECTransformer(data, self.dt, self.verbose)
        self.transformers[name] = xfmr

        if data.nodes is not None:
            for phase_nodes in data.nodes:
                for (nf, nt) in phase_nodes:
                    self._update_node_count(nf, nt)
        self.mark_topology_changed(f"add UMEC transformer: {name}")

        if self.use_multiport_transformers:
            self._register_multiport_device(
                UMECTransformerDevice(name, xfmr)
            )

        logger.debug("添加 UMEC 变压器 %s: %d×%d 绕组, S=%.2f MVA",
                     name, xfmr.n_phases, xfmr.n_windings,
                     data.S_rated / 1e6)
        return xfmr

    # =========================================================================
    # 传输线辅助
    # =========================================================================

    @staticmethod
    def _get_line_Zc(
        line: TransmissionLineInterface, info: Optional[Dict] = None,
    ) -> float:
        """获取线路等效特性阻抗,按优先级查找 _Zc_equiv / info / G_eq。"""
        if info is None:
            info = line.get_info()

        if hasattr(line, '_Zc_equiv'):
            return line._Zc_equiv
        if 'Zc' in info:
            return info['Zc']
        if 'Zc_dc' in info:
            return info['Zc_dc']

        if hasattr(line, 'G_eq'):
            G_eq = line.G_eq
            if isinstance(G_eq, np.ndarray):
                if G_eq.ndim == 2 and G_eq.shape[0] > 0:
                    G_avg = np.mean(np.diag(G_eq))
                    if G_avg > 0:
                        return 1.0 / G_avg
            elif G_eq > 0:
                return 1.0 / G_eq

        G_diag = info.get('G_diagonal', [])
        if G_diag and len(G_diag) > 0 and G_diag[0] > 0:
            return 1.0 / G_diag[0]
        return 0.0

    @staticmethod
    def _get_line_tau(
        line: TransmissionLineInterface, info: Optional[Dict] = None,
    ) -> float:
        """获取线路等效传播时延。"""
        if info is None:
            info = line.get_info()

        if hasattr(line, '_tau_equiv'):
            return line._tau_equiv
        if 'tau' in info:
            return info['tau']

        time_delays = info.get('time_delays', [])
        if time_delays:
            if isinstance(time_delays, (list, np.ndarray)) and len(time_delays) > 0:
                return float(time_delays[0])
            return float(time_delays)
        return 0.0

    @staticmethod
    def _is_multiphase_line(line: TransmissionLineInterface) -> bool:
        if hasattr(line, '_is_multiphase'):
            return bool(getattr(line, '_is_multiphase'))

        if hasattr(line, 'nodes_k') and isinstance(
            line.nodes_k, (list, tuple, np.ndarray)
        ):
            return len(line.nodes_k) > 1

        return False

    @classmethod
    def _get_line_nodes(
        cls, line: TransmissionLineInterface,
    ) -> Tuple[List[int], List[int]]:
        """获取端口节点列表,兼容单相/多相。"""
        if cls._is_multiphase_line(line):
            return list(line.nodes_k), list(line.nodes_m)
        return [line.node_k], [line.node_m]

    @classmethod
    def _get_line_nc(cls, line: TransmissionLineInterface) -> int:
        if hasattr(line, 'nc'):
            return line.nc
        nodes_k, _ = cls._get_line_nodes(line)
        return len(nodes_k)

    @staticmethod
    def _get_line_phase_slots(
        line: TransmissionLineInterface,
        nc: int,
    ) -> np.ndarray:
        """返回线路节点相序到 batch 列号的映射。

        多相线路: [0, 1, 2, ...]
        单相 ULM: [0]
        """
        if ULM_AVAILABLE and ULMLine is not None and isinstance(line, ULMLine):
            if not getattr(line, '_is_multiphase', False):
                return np.array([int(getattr(line, '_sp_idx', 0))], dtype=int)

        return np.arange(nc, dtype=int)

    def _get_line_history_sources(
        self,
        line: TransmissionLineInterface,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """返回线路当前步历史电流源。

        对 batch ULM，直接从 batch 数组读取，避免 line 状态滞后。
        """
        batch = getattr(self, '_ulm_batch', None)
        batch_index = getattr(self, '_ulm_batch_line_index', {})

        if (
            ULM_AVAILABLE
            and ULMLine is not None
            and batch is not None
            and id(line) in batch_index
        ):
            row = batch_index[id(line)]

            if getattr(line, '_is_multiphase', False):
                nk_list, _ = self._get_line_nodes(line)
                nc = len(nk_list)
                return (
                    batch.I_hist_k_batch[row, :nc],
                    batch.I_hist_m_batch[row, :nc],
                )

            phase_idx = int(getattr(line, '_sp_idx', 0))
            return (
                float(batch.I_hist_k_batch[row, phase_idx]),
                float(batch.I_hist_m_batch[row, phase_idx]),
            )

        return line.I_hist_k, line.I_hist_m

    # =========================================================================
    # 传输线添加
    # =========================================================================

    def add_line(self, line: TransmissionLineInterface) -> None:
        """添加传输线(单相或多相)。"""
        self._ensure_unique_device_name(line.name, "transmission line")
        if self.compile_lines_on_add:
            line.initialize(self.dt)
        else:
            self._lines_compiled = False
        self.transmission_lines[line.name] = line

        nodes_k, nodes_m = self._get_line_nodes(line)
        self._update_node_count(nodes_k, nodes_m)
        self.mark_topology_changed(f"add transmission line: {line.name}")

        if self.verbose:
            info = line.get_info()
            nc = self._get_line_nc(line)
            Zc = self._get_line_Zc(line, info)
            tau = self._get_line_tau(line, info)
            logger.info(
                "添加传输线 %s: 类型=%s, 相数=%d, Zc≈%.2fΩ, τ≈%.2fμs",
                line.name, info.get('model_type', 'Unknown'),
                nc, Zc, tau * 1e6,
            )

    def add_bergeron_line(
        self, name: str, node_k: int, node_m: int,
        Zc: float, tau: float
    ) -> BergeronLine:
        """添加无损 Bergeron 单相传输线。"""
        line = BergeronLine(name, node_k, node_m, Zc, tau)
        self.add_line(line)
        # Register MultiPortDevice adapter (idempotent; activated by
        # use_multiport_lines flag in _build_MNA_matrix / _build_MNA_rhs).
        if self.use_multiport_lines:
            self._register_multiport_device(
                BergeronLineDevice(name, line, node_k, node_m)
            )
        return line

    def add_ULM_line(
        self,
        name: str,
        nodes_send: Union[int, List[int]],
        nodes_recv: Union[int, List[int]],
        *,
        length: Optional[float] = None,
        generate_fitulm: bool = False,
        fitulm_path=None,
        lcp_spec=None,
        cache_dir=".lcp_cache",
        force_recompute: bool = False,
    ) -> 'ULMLine':
        """Add a ULM transmission line with optional auto-generation.

        Parameters
        ----------
        name:
            Unique line name.
        nodes_send:
            Sending-end node(s).  Single int for single-phase, list for
            multi-phase (length must match the fitULM conductor count).
        nodes_recv:
            Receiving-end node(s).  Same rules as *nodes_send*.
        length:
            Line length in metres.  When *generate_fitulm* is True,
            *lcp_spec.length* is the authoritative source — omit
            *length* or pass the same value.  Required when
            *generate_fitulm* is False.
        generate_fitulm:
            When False (default), read from an existing *fitulm_path*.
            When True, generate a fitULM file via LCP from *lcp_spec*.
        fitulm_path:
            Path to an existing fitULM file.  Required when
            *generate_fitulm* is False.
        lcp_spec:
            :class:`~pylcp.LCPFitULMSpec` for automatic generation.
            Required when *generate_fitulm* is True.
        cache_dir:
            Directory for auto-generated fitULM cache files.
        force_recompute:
            When True, re-run LCP generation even if cached.

        Returns
        -------
        ULMLine
        """
        # -- length resolution --------------------------------------------
        if generate_fitulm:
            if lcp_spec is None:
                raise ValueError(
                    "lcp_spec is required when generate_fitulm=True"
                )
            lcp_length = float(lcp_spec.length)
            if length is not None:
                solver_length = float(length)
                if abs(solver_length - lcp_length) > 1e-9 * max(1.0, abs(lcp_length)):
                    raise ValueError(
                        f"length mismatch: add_ULM_line length={solver_length}, "
                        f"lcp_spec.length={lcp_length}. "
                        "When generate_fitulm=True, lcp_spec.length is the "
                        "authoritative length — omit length or pass the same value."
                    )
            length = lcp_length
        else:
            if length is None:
                raise ValueError(
                    "length is required when generate_fitulm=False"
                )
            length = float(length)

        spec = FitULMSpec(
            name=name,
            generate_fitulm=generate_fitulm,
            fitulm_path=Path(fitulm_path) if fitulm_path else None,
            lcp_spec=lcp_spec,
            cache_dir=Path(cache_dir),
            force_recompute=force_recompute,
        )
        resolver = FitULMResolver()
        resolved_path = resolver.resolve(spec)
        return self.add_ulm_line(
            name=name,
            nodes_k=nodes_send,
            nodes_m=nodes_recv,
            fitulm_file=str(resolved_path),
            length=length,
        )

    def add_ulm_line(
        self, name: str,
        nodes_k: Union[int, List[int]],
        nodes_m: Union[int, List[int]],
        fitulm_file: str, length: float,
    ) -> 'ULMLine':
        """添加 ULM 线路(单相/多相)。

        Parameters
        ----------
        nodes_k, nodes_m : int or list of int
            端口节点。FitULM 数据为单相时传单个 int；
            FitULM 数据为多相时必须传长度等于相数 nc 的节点列表。
        """
        self._ensure_unique_device_name(name, "ULM line")
        if not ULM_AVAILABLE:
            raise ImportError("ULM 模块不可用,请确保 ulm_transmission_line.py 可导入")

        nodes_k = self._as_node_list(nodes_k)
        nodes_m = self._as_node_list(nodes_m)

        if len(nodes_k) != len(nodes_m):
            raise ValueError(
                f"线路 {name} 的 k 端和 m 端节点数量不一致: "
                f"{len(nodes_k)} / {len(nodes_m)}"
            )

        fit_data = FitULMReader(fitulm_file).read()
        ulm_model = ULMModel(fit_data, length, self.dt, self.verbose)
        nc = ulm_model.nc

        if nc == 1:
            if len(nodes_k) != 1:
                raise ValueError(
                    f"线路 {name} 是单相 ULM 数据,但提供了 "
                    f"{len(nodes_k)} 个 k 端 / {len(nodes_m)} 个 m 端节点"
                )

            line = ULMLine(name, ulm_model, nodes_k[0], nodes_m[0])
            line.nodes_k = nodes_k
            line.nodes_m = nodes_m
        else:
            if len(nodes_k) != nc or len(nodes_m) != nc:
                raise ValueError(
                    f"线路 {name} 是 {nc} 相 ULM 数据,必须提供 {nc} 个 k 端和 {nc} 个 m 端节点, "
                    f"当前为 {len(nodes_k)} 个 k 端 / {len(nodes_m)} 个 m 端"
                )

            line = ULMLine(name, ulm_model)
            line.nodes_k = nodes_k
            line.nodes_m = nodes_m
            line.node_k = nodes_k[0]
            line.node_m = nodes_m[0]

        self._attach_ulm_equivalents(line, length)
        self.add_line(line)
        if self.use_multiport_lines:
            self._register_multiport_device(
                ULMLineDevice(name, line, nodes_k, nodes_m)
            )
        return line

    @staticmethod
    def _as_node_list(nodes: Union[int, List[int]]) -> List[int]:
        if isinstance(nodes, (int, np.integer)):
            return [int(nodes)]
        return [int(n) for n in nodes]

    @staticmethod
    def _attach_ulm_equivalents(line: 'ULMLine', length: float) -> None:
        """计算并缓存 ULM 线路的等效 Zc 与 τ,增强 get_info()。"""
        original_info = line.get_info()
        G_eq = line.G_eq

        if isinstance(G_eq, np.ndarray):
            G_diag = np.diag(G_eq) if G_eq.ndim == 2 else G_eq.flatten()
            G_avg = np.mean(G_diag[G_diag > 0]) if np.any(G_diag > 0) else 1e-6
            Zc_equiv = 1.0 / G_avg
        elif G_eq > 0:
            Zc_equiv = 1.0 / G_eq
        else:
            Zc_equiv = 300.0

        time_delays = original_info.get('time_delays', [])
        if time_delays:
            t_val = time_delays[0] if isinstance(time_delays, list) else time_delays
            tau_equiv = float(t_val)
        else:
            tau_equiv = original_info.get('tau', length / 3e8)

        line._Zc_equiv = Zc_equiv
        line._tau_equiv = tau_equiv

        original_get_info = line.get_info

        def enhanced_get_info() -> Dict[str, Any]:
            info = original_get_info()
            info.setdefault('Zc', Zc_equiv)
            info.setdefault('tau', tau_equiv)
            info['nc'] = EMTPSolver._get_line_nc(line)
            info['is_multiphase'] = EMTPSolver._is_multiphase_line(line)
            return info

        line.get_info = enhanced_get_info


    def get_transmission_line(
        self, name: str,
    ) -> Optional[TransmissionLineInterface]:
        return self.transmission_lines.get(name)

    # =========================================================================
    # MultiPortDevice 注册表与统一 dispatch（骨架，默认空注册表）
    # =========================================================================

    def _register_multiport_device(self, dev) -> None:
        """Register a :class:`MultiPortDevice` for unified dispatch."""
        self._multiport_devices.append(dev)
        self._stamping.mark_dirty()

    def _register_multiport_nodes(self) -> None:
        for dev in self._multiport_devices:
            dev.register_nodes(self._indexer)

    def _stamp_multiport_G(self, stamper) -> None:
        for dev in self._multiport_devices:
            if dev.contributes_G:
                dev.stamp_G(stamper, self._indexer)

    def _stamp_multiport_rhs(self, rhs, t: float) -> None:
        for dev in self._multiport_devices:
            dev.stamp_rhs(rhs, self._indexer, t)

    def _update_multiport_after_solve(self, V, t: float) -> None:
        for dev in self._multiport_devices:
            dev.update_after_solve(V, self._indexer, t)

    def _update_multiport_history(self, V, dt: float) -> None:
        for dev in self._multiport_devices:
            if dev.is_dynamic:
                dev.update_history(V, self._indexer, dt)

    def _check_multiport_rebuild_required(self, V, t: float) -> bool:
        changed = False
        for dev in self._multiport_devices:
            if dev.check_rebuild_required(V, self._indexer, t):
                changed = True
        if changed:
            self._stamping.mark_dirty()
        return changed

    # =========================================================================
    # 系统矩阵装配
    # =========================================================================

    def _build_MNA_matrix(self) -> sp.csc_matrix:
        """构建 MNA 增广稀疏矩阵 (CSC 格式)。

        委托 StampingEngine 处理 devices + VS；传输线与变压器贡献
        由 solver 在 begin/finish 之间插入。
        """
        n = self._indexer.n
        if n == 0:
            raise ValueError("电路中没有节点")

        if self._vs_list is None:
            self._vs_list = list(self.voltage_sources.values())
            self._vs_index_map = {
                vs.name: idx for idx, vs in enumerate(self._vs_list)
            }

        m = len(self._vs_list)
        self._mna_size = n + m

        eng = self._stamping
        stamper = eng.begin_G(n, m)

        # 1. branch devices
        eng.stamp_devices_G(stamper, self._devices)

        # 2. transmission lines (solver-owned, not yet device-ified)
        for line in self.transmission_lines.values():
            nk_list, nm_list = self._get_line_nodes(line)
            nc = len(nk_list)
            G_line = line.G_eq

            if not isinstance(G_line, np.ndarray):
                G_line = np.eye(nc) * G_line
            elif G_line.ndim == 1:
                G_line = np.diag(G_line)
            elif G_line.shape != (nc, nc):
                if G_line.shape[0] >= nc and G_line.shape[1] >= nc:
                    G_line = G_line[:nc, :nc]
                else:
                    G_line = np.eye(nc) * G_line[0, 0]

            for i, node_row in enumerate(nk_list):
                if node_row <= 0:
                    continue
                cr = self._indexer.to_compact(node_row)
                for j, node_col in enumerate(nk_list):
                    if node_col > 0:
                        stamper.add(cr, self._indexer.to_compact(node_col), G_line[i, j])
            for i, node_row in enumerate(nm_list):
                if node_row <= 0:
                    continue
                cr = self._indexer.to_compact(node_row)
                for j, node_col in enumerate(nm_list):
                    if node_col > 0:
                        stamper.add(cr, self._indexer.to_compact(node_col), G_line[i, j])

        # 3. UMEC transformers
        for xfmr in self.transformers.values():
            G_tf, _ = xfmr.get_norton_equivalent()
            port_nodes = xfmr.get_port_nodes()
            mp = len(port_nodes)
            for i in range(mp):
                nf_i, nt_i = port_nodes[i]
                cf_i = self._indexer.to_compact(nf_i)
                ct_i = self._indexer.to_compact(nt_i)
                for j in range(mp):
                    nf_j, nt_j = port_nodes[j]
                    cf_j = self._indexer.to_compact(nf_j)
                    ct_j = self._indexer.to_compact(nt_j)
                    g = G_tf[i, j]
                    if cf_i >= 0 and cf_j >= 0:
                        stamper.add(cf_i, cf_j, g)
                    if ct_i >= 0 and ct_j >= 0:
                        stamper.add(ct_i, ct_j, g)
                    if cf_i >= 0 and ct_j >= 0:
                        stamper.add(cf_i, ct_j, -g)
                    if ct_i >= 0 and cf_j >= 0:
                        stamper.add(ct_i, cf_j, -g)

        # 4. voltage sources
        eng.stamp_vs_G(stamper, self._vs_list)

        return eng.finish_G(stamper)

    def _build_MNA_rhs(self) -> np.ndarray:
        """构建 MNA 增广右端向量 [I; E]。

        opt3: 复用 RHS 缓冲区，并在 ULM batch 模式下直接从 batch 数组
        使用预编译索引表注入线路历史源，避免每步逐线 Python 对象遍历。
        """
        n = self._indexer.n
        m = len(self._vs_list) if self._vs_list else 0
        N = n + m

        rhs = getattr(self, '_rhs_buf', None)
        if rhs is None or rhs.shape[0] != N:
            rhs = np.zeros(N, dtype=np.float64)
            self._rhs_buf = rhs
        else:
            rhs.fill(0.0)

        # ---- 1. 支路历史源 ----
        for dev in self._devices:
            dev.stamp_rhs(rhs, self._indexer, self.time)

        # ---- 1b. MultiPortDevice history sources ----
        self._stamp_multiport_rhs(rhs, self.time)

        # ---- 2. 电流源 ----
        if self.pre_sample_sources and self._current_source_samples:
            step_idx = int(round(self.time / self.dt))
            for source in self.current_sources.values():
                I_s = float(
                    self._current_source_samples[source.name][step_idx]
                )
                cf = self._indexer.to_compact(source.node_from)
                ct = self._indexer.to_compact(source.node_to)
                if cf >= 0:
                    rhs[cf] -= I_s
                if ct >= 0:
                    rhs[ct] += I_s
        else:
            for source in self.current_sources.values():
                I_s = source.current_at(self.time)
                cf = self._indexer.to_compact(source.node_from)
                ct = self._indexer.to_compact(source.node_to)
                if cf >= 0:
                    rhs[cf] -= I_s
                if ct >= 0:
                    rhs[ct] += I_s

        # ---- 3. 传输线历史源 ----
        batch = getattr(self, '_ulm_batch', None)
        if batch is not None and self._ulm_batch_k_nodes_v is not None:
            # np.add.at 对重复节点安全；串接线路中接头节点会被多条线路同时注入。
            if self._ulm_batch_k_nodes_v.size:
                np.add.at(
                    rhs,
                    self._ulm_batch_k_nodes_v,
                    -batch.I_hist_k_batch[
                        self._ulm_batch_k_rows_v,
                        self._ulm_batch_k_slots_v,
                    ],
                )
            if self._ulm_batch_m_nodes_v.size:
                np.add.at(
                    rhs,
                    self._ulm_batch_m_nodes_v,
                    -batch.I_hist_m_batch[
                        self._ulm_batch_m_rows_v,
                        self._ulm_batch_m_slots_v,
                    ],
                )

            # 如有非 batch 线路，仍按 fallback 路径注入。
            line_iter = getattr(self, '_line_inject_maps_nonbatch', [])
        else:
            line_iter = getattr(self, '_line_inject_maps', [])

        for line, k_idx, m_idx, nc, _is_multi, _has_full in line_iter:
            I_hist_k, I_hist_m = self._get_line_history_sources(line)

            arr = np.asarray(I_hist_k)
            if arr.ndim == 0:
                if nc == 1 and k_idx[0] >= 0:
                    rhs[k_idx[0]] -= float(np.real(arr))
            else:
                vals = arr.real.ravel()
                limit = min(nc, len(k_idx), len(vals))
                for i in range(limit):
                    node_idx = k_idx[i]
                    if node_idx >= 0:
                        rhs[node_idx] -= float(vals[i])

            arr = np.asarray(I_hist_m)
            if arr.ndim == 0:
                if nc == 1 and m_idx[0] >= 0:
                    rhs[m_idx[0]] -= float(np.real(arr))
            else:
                vals = arr.real.ravel()
                limit = min(nc, len(m_idx), len(vals))
                for i in range(limit):
                    node_idx = m_idx[i]
                    if node_idx >= 0:
                        rhs[node_idx] -= float(vals[i])

        # ---- 4. UMEC 变压器历史源 ----
        for xfmr in self.transformers.values():
            _, I_hist_tf = xfmr.get_norton_equivalent()
            port_nodes = xfmr.get_port_nodes()
            for i, (nf_i, nt_i) in enumerate(port_nodes):
                cf_i = self._indexer.to_compact(nf_i)
                ct_i = self._indexer.to_compact(nt_i)
                if cf_i >= 0:
                    rhs[cf_i] -= I_hist_tf[i]
                if ct_i >= 0:
                    rhs[ct_i] += I_hist_tf[i]

        # ---- 5. 电压源激励 E ----
        if self._vs_list:
            if self.pre_sample_sources and self._voltage_source_samples:
                step_idx = int(round(self.time / self.dt))
                for k, vs in enumerate(self._vs_list):
                    rhs[n + k] = float(
                        self._voltage_source_samples[vs.name][step_idx]
                    )
            else:
                for k, vs in enumerate(self._vs_list):
                    rhs[n + k] = vs.voltage_at(self.time)

        return rhs

    def _build_system_matrix(self) -> Tuple[sp.csc_matrix, np.ndarray]:
        """(MNA, rhs) 对:MNA 矩阵按脏位缓存,rhs 每步重建。"""
        eng = self._stamping
        if eng.G_dirty or eng.cached_MNA is None:
            self._build_MNA_matrix()  # side-effect: caches via eng.finish_G
            self._cached_MNA = eng.cached_MNA
            self._stats['G_rebuilds'] = self._stats.get('G_rebuilds', 0) + 1
        else:
            self._cached_MNA = eng.cached_MNA
            self._stats['G_cache_hits'] = self._stats.get('G_cache_hits', 0) + 1

        if self.use_rhs_plan and (self._rhs_plan_dirty or self._rhs_plan is None):
            self._rhs_plan = self._compile_rhs_plan()
            self._rhs_plan_dirty = False

        rhs = (self._build_rhs_fast() if self.use_rhs_plan
               else self._build_MNA_rhs())

        return self._cached_MNA, rhs

    # =========================================================================
    # MNA 稀疏求解 (SuperLU)
    # =========================================================================

    def _solve_mna(
        self, MNA: sp.csc_matrix, rhs: np.ndarray,
    ) -> np.ndarray:
        """MNA sparse solve - delegates to StampingEngine."""
        self._active_mna_solver_name = "SuperLU(splu)"
        return self._stamping.solve(MNA, rhs, self._vs_list or [])

    # =========================================================================
    # 单步求解
    # =========================================================================

    def _solve_segmented(self) -> np.ndarray:
        """PSCAD 分段线性法求解。"""
        V = np.zeros(self._indexer.n)
        converged = False
        for seg_iter in range(self._MAX_SEG_ITER):
            MNA, rhs = self._build_system_matrix()
            V = self._solve_mna(MNA, rhs)

            if not self._seg_node_map:
                converged = True
                return V

            voltages = {}
            for name, (nf, nt) in self._seg_node_map.items():
                v_i = V[self._indexer.to_compact(nf)] if nf > 0 else 0.0
                v_j = V[self._indexer.to_compact(nt)] if nt > 0 else 0.0
                voltages[name] = v_i - v_j

            need_resolve, updates = self.seg_helper.check_all_segments(voltages)

            if not need_resolve:
                converged = True
                break

            for seg_name, (g_new, i_new) in updates.items():
                branch = self.branches[seg_name]
                branch.Geq = g_new
                branch.Ihist = i_new
                self.mark_topology_changed(f"nonlinear segment changed: {seg_name}")
                self._stats['segment_switches'] += 1

            self._stats['segment_resolves'] += 1
            if seg_iter + 1 > self._stats['max_seg_iter']:
                self._stats['max_seg_iter'] = seg_iter + 1

        if not converged:
            logger.warning(
                "Segmented nonlinear solver did not converge at t=%g after %d iterations",
                self.time,
                self._MAX_SEG_ITER,
            )

        return V

    def _solve_linear(self) -> np.ndarray:
        MNA, rhs = self._build_system_matrix()
        return self._solve_mna(MNA, rhs)

    def _solve_step(self) -> np.ndarray:
        """Single-step solve with unified nonlinear/LPM/UMEC resolve loop.

        Delegates the iterative re-solve orchestration to
        :class:`ResolveManager`.
        """

        def _check(V: np.ndarray) -> bool:
            def _mark_dirty(reason: str) -> None:
                self.mark_topology_changed(reason)
            return self._runtime.post_solve_resolve_check(
                V, self.time,
                self._lpm_elements, self._lpm_node_map,
                self.transformers,
                self._seg_node_map, self.seg_helper,
                self.branches, self._indexer,
                _mark_dirty, self._stats,
            )

        solve = self._solve_segmented if self._has_nonlinear else self._solve_linear
        return self._resolve_mgr.solve_with_resolve(
            solve, _check, self._stats, self.time, logger,
        )

    def _run_one_step(self, step_idx: int, n_steps: int, _t) -> None:
        """Execute a single time step (called by :class:`TimeStepper`)."""
        self.time = step_idx * self.dt

        # 1. switch events
        t0 = _t()
        if self._runtime.step_pre_solve(
            self.time, self._devices, set(self._lpm_elements),
        ):
            self.mark_topology_changed("switch event")
        t1 = _t()
        self._timing['switch_update'] += t1 - t0

        # 2. core solve
        V = self._solve_step()
        t2 = _t()
        self._timing['solve_step_total'] += t2 - t1

        # 3. branch V/I update (MUST precede probes)
        self._runtime.step_post_solve_V_I(
            V, self._devices, self._indexer,
            step_idx, n_steps,
            bool(getattr(self, 'record_branch_history', False)),
            self._branch_v_bufs, self._branch_i_bufs,
        )
        t3 = _t()
        self._timing['branch_update'] += t3 - t2

        # 4. probe / time / voltage recording (before history update
        #    so that L/C/SRL probes see current-step Ihist)
        self._record_probes(step_idx, V)
        self._time_array_buf[step_idx] = self.time
        if self._voltage_buf is not None:
            self._voltage_buf[:, step_idx] = V

        self._update_source_history()

        if self.record_source_history:
            for name, vs in self.voltage_sources.items():
                vs.current_history.append(vs.current)
                if name in self._vs_current_bufs:
                    self._vs_current_bufs[name][step_idx] = vs.current

        t_probe = _t()
        self._timing['probe_store'] += t_probe - t3

        # 5. transmission lines
        self._update_lines_combined(V)
        t4 = _t()
        self._timing['line_combined_update'] += t4 - t_probe

        # 6. branch reactive history (after probes, after lines)
        self._runtime.step_post_solve_history(self._devices)

        # 7. transformer history
        self._update_transformer_history(V)
        t5 = _t()
        self._timing['transformer_history'] += t5 - t4

        self._timing['data_store'] += _t() - t5

        self.step_count += 1
        self._stats['total_steps'] += 1

    # =========================================================================
    # 状态更新
    # =========================================================================

    def _update_source_history(self) -> None:
        if not getattr(self, 'record_source_history', False):
            return
        for source in self.current_sources.values():
            source.current_history.append(source.current_at(self.time))


    def _update_lines_combined(self, V: np.ndarray) -> None:
        """传输线状态更新 + 历史源更新。

        batch ULM:
          - RHS 历史源由 batch.I_hist_k/m_batch 作为权威来源
          - 不再手动 copy 回 line._I_hist_*，避免破坏 batch 视图
        """
        # ---- 批量路径 ----
        batch = getattr(self, '_ulm_batch', None)
        if batch is not None:
            vk_in = batch._vk_in_work
            vm_in = batch._vm_in_work

            vk_in.fill(0.0)
            vm_in.fill(0.0)

            # 预编译好的压缩扁平索引表，避免每个时间步 Python 双重循环和布尔切片。
            k_nodes = self._ulm_batch_k_nodes_v
            m_nodes = self._ulm_batch_m_nodes_v
            if k_nodes is not None and k_nodes.size:
                vk_in[self._ulm_batch_k_rows_v, self._ulm_batch_k_slots_v] = V[k_nodes]
            if m_nodes is not None and m_nodes.size:
                vm_in[self._ulm_batch_m_rows_v, self._ulm_batch_m_slots_v] = V[m_nodes]

            record_hist = bool(getattr(self, 'record_line_history', False))
            sync_each = bool(getattr(self, 'sync_line_state_each_step', False))
            batch.step(
                vk_in,
                vm_in,
                sync_lines=(sync_each or record_hist),
                record_history=record_hist,
            )

        # ---- 非批量路径 ----
        for line, k_idx, m_idx, nc, is_multi, has_full in self._line_inject_maps_nonbatch:
            vk = self._line_vk_bufs[line.name]
            vm = self._line_vm_bufs[line.name]

            for i in range(nc):
                vk[i] = V[k_idx[i]] if k_idx[i] >= 0 else 0.0
                vm[i] = V[m_idx[i]] if m_idx[i] >= 0 else 0.0

            if has_full:
                if nc == 1 and not is_multi:
                    line.full_step(vk[0], vm[0], record_history=self.record_line_history)
                else:
                    line.full_step(vk, vm, record_history=self.record_line_history)
            else:
                if nc == 1 and not is_multi:
                    try:
                        line.update_state(
                            vk[0], vm[0],
                            record_history=self.record_line_history,
                        )
                    except TypeError:
                        line.update_state(vk[0], vm[0])
                else:
                    try:
                        line.update_state(
                            vk, vm,
                            record_history=self.record_line_history,
                        )
                    except TypeError:
                        line.update_state(vk, vm)
                line.update_history_sources()


    def _update_transformer_history(self, V: np.ndarray) -> None:
        """UMEC 变压器历史源更新。"""
        for xfmr in self.transformers.values():
            V_ports = self._get_transformer_port_voltages(xfmr, V)
            xfmr.update_history(V_ports)

    def _get_transformer_port_voltages(
        self,
        xfmr: 'UMECTransformer',
        V: np.ndarray,
    ) -> np.ndarray:
        """Return UMEC transformer port voltages from the current MNA solution."""
        port_nodes = xfmr.get_port_nodes()
        V_ports = np.zeros(xfmr.m)
        for k, (nf, nt) in enumerate(port_nodes):
            v_f = V[self._indexer.to_compact(nf)] if nf > 0 else 0.0
            v_t = V[self._indexer.to_compact(nt)] if nt > 0 else 0.0
            V_ports[k] = v_f - v_t
        return V_ports

    def _is_empty_circuit(self) -> bool:
        return (
            not self.branches
            and not self.current_sources
            and not self.voltage_sources
            and not self.transmission_lines
            and not self.transformers
        )

    def estimate_result_memory_bytes(self) -> int:
        """Estimate result buffer memory usage in bytes before running.

        Covers the main arrays allocated during ``run()``: time vector, node
        voltage history (if enabled), probes, line/branch history buffers,
        and source current history.  The estimate assumes float64 storage.
        """
        n_steps = int(round(self.finish_time / self.dt)) + 1
        F64 = 8

        total = n_steps * F64  # time_array

        if self.record_all_node_voltages:
            total += self._indexer.n * n_steps * F64

        n_vp = len(self.voltage_probes)
        if n_vp:
            total += n_vp * n_steps * F64

        n_bcp = len(self.branch_current_probes)
        if n_bcp:
            total += n_bcp * n_steps * F64

        if self.record_line_history:
            for line in self.transmission_lines.values():
                nk_list, _ = self._get_line_nodes(line)
                nc = len(nk_list)
                total += 4 * nc * n_steps * F64  # Ik, Im, Vk, Vm

        if self.record_branch_history:
            total += 2 * len(self.branches) * n_steps * F64  # V, I

        if self.record_source_history:
            total += len(self.voltage_sources) * n_steps * F64

        return total

    def _pre_sample_sources(self, n_steps: int) -> None:
        """Pre-sample independent sources on the simulation time grid.

        When ``pre_sample_sources`` is enabled, scalar source functions are
        evaluated once for the entire grid and stored in flat arrays.  This
        removes per-step Python function-call overhead at the cost of
        evaluating all sources upfront.

        .. note::
           Pre-sampling changes the *timing* of source function evaluation.
           Users whose callables depend on external state should keep
           ``pre_sample_sources=False``.
        """
        self._current_source_samples.clear()
        self._voltage_source_samples.clear()

        t_arr = np.arange(n_steps, dtype=np.float64) * self.dt

        if self.current_sources:
            for name, src in self.current_sources.items():
                self._current_source_samples[name] = np.array(
                    [src.current_at(float(ti)) for ti in t_arr],
                    dtype=np.float64,
                )

        if self.voltage_sources:
            for name, vs in self.voltage_sources.items():
                self._voltage_source_samples[name] = np.array(
                    [vs.voltage_at(float(ti)) for ti in t_arr],
                    dtype=np.float64,
                )

    def _compile_rhs_plan(self) -> RHSPlan:
        """Pre-compile topological index arrays for fast RHS assembly.

        Builds flat arrays of node indices for reactive branches,
        current sources, and transformer ports so the per-step RHS
        construction avoids iterating Python device objects.
        """
        plan = RHSPlan()

        # ---- reactive branch history sources ----
        dyn_names = []
        dyn_nf = []
        dyn_nt = []
        dyn_types = []
        for name, branch in self.branches.items():
            et = branch.element_type
            if et in (ElementType.INDUCTOR, ElementType.CAPACITOR,
                       ElementType.SERIES_RL, ElementType.NONLINEAR_RESISTOR):
                dyn_names.append(name)
                dyn_nf.append(self._indexer.to_compact(branch.node_from))
                dyn_nt.append(self._indexer.to_compact(branch.node_to))
                if et == ElementType.NONLINEAR_RESISTOR:
                    dyn_types.append("NR")
                elif et == ElementType.SERIES_RL:
                    dyn_types.append("SRL")
                else:
                    dyn_types.append("LC")
            # R, SW have no Ihist — skip

        plan.dyn_branch_names = dyn_names
        plan.dyn_branch_nf_idx = np.array(dyn_nf, dtype=int)
        plan.dyn_branch_nt_idx = np.array(dyn_nt, dtype=int)
        plan.dyn_branch_type = dyn_types

        # ---- current sources ----
        is_names = []
        is_nf = []
        is_nt = []
        for name, source in self.current_sources.items():
            is_names.append(name)
            is_nf.append(self._indexer.to_compact(source.node_from))
            is_nt.append(self._indexer.to_compact(source.node_to))

        plan.isource_names = is_names
        plan.isource_nf_idx = np.array(is_nf, dtype=int)
        plan.isource_nt_idx = np.array(is_nt, dtype=int)

        # ---- transformer ports ----
        for name, xfmr in self.transformers.items():
            plan.xfmr_names.append(name)
            port_nodes = xfmr.get_port_nodes()
            nf_arr = np.array([self._indexer.to_compact(nf) for nf, _ in port_nodes],
                              dtype=int)
            nt_arr = np.array([self._indexer.to_compact(nt) for _, nt in port_nodes],
                              dtype=int)
            plan.xfmr_port_nf_idx.append(nf_arr)
            plan.xfmr_port_nt_idx.append(nt_arr)

        return plan

    def _build_rhs_fast(self) -> np.ndarray:
        """Build the MNA RHS vector using the pre-compiled RHSPlan.

        This path avoids iterating Python device objects; it walks flat
        index arrays and looks up scalar values directly.
        """
        n = self._indexer.n
        m = len(self._vs_list) if self._vs_list else 0
        N = n + m
        plan = self._rhs_plan

        rhs = getattr(self, '_rhs_buf', None)
        if rhs is None or rhs.shape[0] != N:
            rhs = np.zeros(N, dtype=np.float64)
            self._rhs_buf = rhs
        else:
            rhs.fill(0.0)

        # ---- 1. 支路历史源 (flat index arrays) ----
        n_dyn = len(plan.dyn_branch_names)
        for k in range(n_dyn):
            br = self.branches[plan.dyn_branch_names[k]]
            if plan.dyn_branch_type[k] == "NR":
                i_eq = getattr(br, 'Ihist', 0.0)
            else:
                i_eq = br.Ihist
            if i_eq == 0.0:
                continue
            nf_idx = plan.dyn_branch_nf_idx[k]
            nt_idx = plan.dyn_branch_nt_idx[k]
            if nf_idx >= 0:
                rhs[nf_idx] -= i_eq
            if nt_idx >= 0:
                rhs[nt_idx] += i_eq

        # ---- 2. 电流源 ----
        if self.pre_sample_sources and self._current_source_samples:
            step_idx = int(round(self.time / self.dt))
            n_is = len(plan.isource_names)
            for k in range(n_is):
                I_s = float(
                    self._current_source_samples[plan.isource_names[k]][step_idx]
                )
                if I_s == 0.0:
                    continue
                nf_idx = plan.isource_nf_idx[k]
                nt_idx = plan.isource_nt_idx[k]
                if nf_idx >= 0:
                    rhs[nf_idx] -= I_s
                if nt_idx >= 0:
                    rhs[nt_idx] += I_s
        else:
            n_is = len(plan.isource_names)
            for k in range(n_is):
                source = self.current_sources[plan.isource_names[k]]
                I_s = source.current_at(self.time)
                if I_s == 0.0:
                    continue
                nf_idx = plan.isource_nf_idx[k]
                nt_idx = plan.isource_nt_idx[k]
                if nf_idx >= 0:
                    rhs[nf_idx] -= I_s
                if nt_idx >= 0:
                    rhs[nt_idx] += I_s

        # ---- 3. 传输线历史源 (reuse existing ULM batch path) ----
        batch = getattr(self, '_ulm_batch', None)
        if batch is not None and self._ulm_batch_k_nodes_v is not None:
            if self._ulm_batch_k_nodes_v.size:
                np.add.at(
                    rhs, self._ulm_batch_k_nodes_v,
                    -batch.I_hist_k_batch[
                        self._ulm_batch_k_rows_v, self._ulm_batch_k_slots_v,
                    ],
                )
            if self._ulm_batch_m_nodes_v.size:
                np.add.at(
                    rhs, self._ulm_batch_m_nodes_v,
                    -batch.I_hist_m_batch[
                        self._ulm_batch_m_rows_v, self._ulm_batch_m_slots_v,
                    ],
                )
            line_iter = getattr(self, '_line_inject_maps_nonbatch', [])
        else:
            line_iter = getattr(self, '_line_inject_maps', [])

        for line, k_idx, m_idx, nc, _is_multi, _has_full in line_iter:
            I_hist_k, I_hist_m = self._get_line_history_sources(line)
            arr = np.asarray(I_hist_k)
            if arr.ndim == 0:
                if nc == 1 and k_idx[0] >= 0:
                    rhs[k_idx[0]] -= float(np.real(arr))
            else:
                vals = arr.real.ravel()
                limit = min(nc, len(k_idx), len(vals))
                for i in range(limit):
                    if k_idx[i] >= 0:
                        rhs[k_idx[i]] -= float(vals[i])
            arr = np.asarray(I_hist_m)
            if arr.ndim == 0:
                if nc == 1 and m_idx[0] >= 0:
                    rhs[m_idx[0]] -= float(np.real(arr))
            else:
                vals = arr.real.ravel()
                limit = min(nc, len(m_idx), len(vals))
                for i in range(limit):
                    if m_idx[i] >= 0:
                        rhs[m_idx[i]] -= float(vals[i])

        # ---- 4. UMEC 变压器历史源 (flat index arrays) ----
        for x_idx, name in enumerate(plan.xfmr_names):
            xfmr = self.transformers[name]
            _, I_hist_tf = xfmr.get_norton_equivalent()
            nf_arr = plan.xfmr_port_nf_idx[x_idx]
            nt_arr = plan.xfmr_port_nt_idx[x_idx]
            for i in range(len(nf_arr)):
                if nf_arr[i] >= 0:
                    rhs[nf_arr[i]] -= I_hist_tf[i]
                if nt_arr[i] >= 0:
                    rhs[nt_arr[i]] += I_hist_tf[i]

        # ---- 5. 电压源激励 E ----
        if self._vs_list:
            if self.pre_sample_sources and self._voltage_source_samples:
                step_idx = int(round(self.time / self.dt))
                for k, vs in enumerate(self._vs_list):
                    rhs[n + k] = float(
                        self._voltage_source_samples[vs.name][step_idx]
                    )
            else:
                for k, vs in enumerate(self._vs_list):
                    rhs[n + k] = vs.voltage_at(self.time)

        return rhs

    def validate_probes(self) -> None:
        """Validate probe references without mutating circuit topology."""
        for name, probe in self.voltage_probes.items():
            for key in ("node_pos", "node_neg"):
                node = int(probe[key])
                if node > 0 and node not in self._node_set:
                    raise ValueError(
                        f"电压探针 {name!r} 引用的节点 {node} 不存在"
                    )

        for name, probe in self.branch_current_probes.items():
            branch_name = probe["branch_name"]
            if branch_name not in self.branches:
                raise ValueError(
                    f"支路电流探针 {name!r} 引用的支路不存在: {branch_name}"
                )

    def validate_circuit(self, strict: bool = True) -> ValidationReport:
        """Run topology and parameter checks before time stepping.

        Parameters
        ----------
        strict : bool
            If True (default), raise RuntimeError when errors are found.
            If False, return the ValidationReport without raising.

        Returns
        -------
        ValidationReport
            Collected issues with severity levels.
        """
        issues: List[ValidationIssue] = []

        def _add(severity: str, code: str, message: str,
                 nodes=None, branches=None) -> None:
            issues.append(ValidationIssue(
                severity=severity, code=code, message=message,
                related_nodes=nodes or [], related_branches=branches or [],
            ))

        # ---- 1. basic parameter checks (must pass before everything else) ----
        if self.dt <= 0:
            _add("error", "E001", f"dt 必须为正,当前为 {self.dt}")
        if self.finish_time < 0:
            _add("error", "E002",
                 f"finish_time 不能为负,当前为 {self.finish_time}")

        # Basic param errors prevent memory estimation and topology checks.
        if any(i.severity == "error" for i in issues):
            return self._finalize_validation(issues, strict)

        # ---- 2. probe validation ----
        for name, probe in self.voltage_probes.items():
            for key in ("node_pos", "node_neg"):
                node = int(probe[key])
                if node > 0 and node not in self._node_set:
                    _add("error", "E003",
                         f"电压探针 {name!r} 引用的节点 {node} 不存在",
                         nodes=[node])
        for name, probe in self.branch_current_probes.items():
            branch_name = probe["branch_name"]
            if branch_name not in self.branches:
                _add("error", "E004",
                     f"支路电流探针 {name!r} 引用的支路不存在: {branch_name}",
                     branches=[branch_name])

        # ---- 3. empty / node-less circuits ----
        if self._is_empty_circuit():
            return self._finalize_validation(issues, strict)

        if self._indexer.n <= 0:
            _add("error", "E005", "电路中没有非地节点")
            return self._finalize_validation(issues, strict)

        # ---- 4. memory warnings (dt is now guaranteed positive) ----
        max_ext_id = max(self._node_set) if self._node_set else 0
        if max_ext_id > 10 * len(self._node_set):
            _add("warning", "W001",
                 f"Sparse external integer node IDs detected "
                 f"(max={max_ext_id}, unique={len(self._node_set)}). "
                 "NodeIndexer compacts external IDs, so MNA size is not increased, "
                 "but sparse IDs can make models harder to read and debug. "
                 "Consider using named nodes or compact external numbering.",
                 nodes=[max_ext_id])

        mem_bytes = self.estimate_result_memory_bytes()
        if self.record_all_node_voltages and self._indexer.n > 0:
            n_steps = int(round(self.finish_time / self.dt)) + 1
            _add("info", "I001",
                 f"record_all_node_voltages=True will allocate "
                 f"{self._indexer.n}×{n_steps} voltage matrix "
                 f"(~{mem_bytes / 1e6:.1f} MB). "
                 "For large networks, set record_all_node_voltages=False "
                 "and use probes instead.")
        if (self.max_result_memory_mb is not None
                and mem_bytes > self.max_result_memory_mb * 1e6):
            _add("warning", "W002",
                 f"Estimated result memory {mem_bytes / 1e6:.1f} MB exceeds "
                 f"limit {self.max_result_memory_mb} MB. "
                 "Disable record_all_node_voltages or reduce probes.")

        # ---- 5. branch-level checks ----
        adjacency: Dict[int, set] = {
            int(n): set() for n in self._node_set if n > 0
        }
        grounded: set = set()

        vs_parent: Dict[int, int] = {}

        def vs_find(node: int) -> int:
            node = int(node)
            vs_parent.setdefault(node, node)
            while vs_parent[node] != node:
                vs_parent[node] = vs_parent[vs_parent[node]]
                node = vs_parent[node]
            return node

        def vs_union(nf: int, nt: int, source_name: str) -> None:
            root_f = vs_find(nf)
            root_t = vs_find(nt)
            if root_f == root_t:
                _add("error", "E006",
                     "Ideal voltage-source loop detected while adding "
                     f"{source_name!r}. Break the loop with impedance or "
                     "replace part of it with an equivalent source.",
                     branches=[source_name])
            else:
                vs_parent[root_t] = root_f

        def add_matrix_edge(nf: int, nt: int) -> None:
            nf = int(nf)
            nt = int(nt)
            if nf > 0:
                adjacency.setdefault(nf, set())
            if nt > 0:
                adjacency.setdefault(nt, set())
            if nf > 0 and nt > 0:
                adjacency[nf].add(nt)
                adjacency[nt].add(nf)
            elif nf > 0:
                grounded.add(nf)
            elif nt > 0:
                grounded.add(nt)

        for branch in self.branches.values():
            if branch.node_from == branch.node_to:
                _add("error", "E007",
                     f"支路 {branch.name} 的两端不能是同一节点: "
                     f"{branch.node_from}",
                     nodes=[branch.node_from], branches=[branch.name])
            if branch.element_type == ElementType.RESISTOR and branch.value <= 0:
                _add("error", "E008",
                     f"电阻 {branch.name} 的 R 必须为正",
                     branches=[branch.name])
            if branch.element_type == ElementType.INDUCTOR and branch.value <= 0:
                _add("error", "E009",
                     f"电感 {branch.name} 的 L 必须为正",
                     branches=[branch.name])
            if branch.element_type == ElementType.CAPACITOR and branch.value <= 0:
                _add("error", "E010",
                     f"电容 {branch.name} 的 C 必须为正",
                     branches=[branch.name])
            if branch.element_type == ElementType.SWITCH:
                if branch.R_closed <= 0 or branch.R_open <= 0:
                    _add("error", "E011",
                         f"开关 {branch.name} 的 R_closed/R_open 必须为正",
                         branches=[branch.name])
            if branch.element_type in (
                ElementType.RESISTOR,
                ElementType.INDUCTOR,
                ElementType.CAPACITOR,
                ElementType.SERIES_RL,
                ElementType.SWITCH,
                ElementType.NONLINEAR_RESISTOR,
            ):
                add_matrix_edge(branch.node_from, branch.node_to)

        # ---- 6. voltage source checks (loop detection via union-find) ----
        for vs in self.voltage_sources.values():
            if vs.node_pos == vs.node_neg:
                _add("error", "E012",
                     f"电压源 {vs.name} 的正负端不能是同一节点",
                     branches=[vs.name])
            vs_union(vs.node_pos, vs.node_neg, vs.name)
            add_matrix_edge(vs.node_pos, vs.node_neg)

        # ---- 7. transmission line nodes ----
        for line in self.transmission_lines.values():
            nk_list, nm_list = self._get_line_nodes(line)
            for node in list(nk_list) + list(nm_list):
                if node > 0:
                    adjacency.setdefault(int(node), set())
                    grounded.add(int(node))

        # ---- 8. transformer port nodes ----
        for xfmr in self.transformers.values():
            for nf, nt in xfmr.get_port_nodes():
                add_matrix_edge(nf, nt)

        # ---- 9. current source nodes (track for floating-node diagnostics) ----
        for source in self.current_sources.values():
            if source.node_from > 0:
                adjacency.setdefault(int(source.node_from), set())
            if source.node_to > 0:
                adjacency.setdefault(int(source.node_to), set())

        # ---- 10. floating component detection ----
        visited = set()
        floating_components = []
        for start in sorted(adjacency):
            if start in visited:
                continue
            stack = [start]
            component = set()
            is_grounded = False
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                if node in grounded:
                    is_grounded = True
                for nxt in adjacency.get(node, ()):
                    if nxt not in visited:
                        stack.append(nxt)
            if not is_grounded:
                floating_components.append(sorted(component))

        if floating_components:
            _add("error", "E013",
                 "Circuit has floating node groups without a ground reference: "
                 f"{floating_components}. Add a ground reference, shunt path, "
                 "or explicitly inspect the topology.",
                 nodes=[n for grp in floating_components for n in grp])

        return self._finalize_validation(issues, strict)

    @staticmethod
    def _finalize_validation(
        issues: List[ValidationIssue], strict: bool,
    ) -> ValidationReport:
        """Build a ValidationReport and optionally raise on errors."""
        report = ValidationReport(issues=issues)
        if strict and report.has_errors:
            error_msgs = "\n".join(
                f"[{i.code}] {i.message}" for i in report.errors()
            )
            raise RuntimeError(
                f"Circuit validation failed with {len(report.errors())} "
                f"error(s):\n{error_msgs}"
            )
        return report

    def reset_dynamic_state(self) -> None:
        """Reset all time-domain device state before a fresh run()."""
        self.time = 0.0
        self.step_count = 0
        self.time_array = []
        self.voltage_results = {}
        self._actual_steps = 0

        for dev in self._devices:
            dev.reset_state()

        for dev in self._multiport_devices:
            dev.reset_state()

        self.seg_helper.reset_all()
        for name, model in getattr(self.seg_helper, 'elements', {}).items():
            if name not in self.branches:
                continue
            model.reset_to_segment(0)
            g0, i0 = model.get_norton_equivalent(0.0)
            self.branches[name].Geq = g0
            self.branches[name].Ihist = i0

        for name, lpm in self._lpm_elements.items():
            lpm.reset()
        self._lpm_flashover_log.clear()

        for source in self.current_sources.values():
            source.current_history.clear()
        for vs in self.voltage_sources.values():
            vs.current = 0.0
            vs.current_history.clear()

        for xfmr in self.transformers.values():
            if hasattr(xfmr, 'reset_state'):
                xfmr.reset_state()
            else:
                if hasattr(xfmr, 'I_hist'):
                    xfmr.I_hist[:] = 0.0
                if hasattr(xfmr, 'V_prev'):
                    xfmr.V_prev[:] = 0.0
                if hasattr(xfmr, 'I_prev'):
                    xfmr.I_prev[:] = 0.0
                if hasattr(xfmr, 'flux'):
                    xfmr.flux[:] = 0.0

        # Force transmission-line buffers and ULM batch views to be rebuilt.
        self._lines_compiled = False
        self._ulm_batch = None
        self._ulm_batch_meta = []
        self._ulm_batch_line_index = {}

    def run(self) -> None:
        """运行仿真（含预编译优化与计时探针）。

        Switch events are evaluated on the fixed simulation grid.
        An event scheduled at ``t_event`` is applied at the first
        step satisfying ``t >= t_event``, i.e. the effective event
        time is ``ceil(t_event / dt) * dt``.
        """
        _t = _perf_time.perf_counter
        t_run_start = _t()

        report = self.validate_circuit()

        if self.verbose and report.has_warnings:
            for w in report.warnings():
                logger.warning("[%s] %s", w.code, w.message)

        # 重置求解器和所有动态元件状态，保证同一个 solver 可重复 run()。
        self._stats = self._fresh_stats()
        self._timing = defaultdict(float)
        self.reset_dynamic_state()
        self._reset_caches()
        self._is_running = True

        if self.verbose:
            self._log_run_header()

        # 预分配输出 — 委托给 ResultStore，并将旧属性别名化
        n_steps = int(round(self.finish_time / self.dt)) + 1
        self._init_result_store(n_steps)

        # Empty circuits are useful as a timing-grid smoke test and should not
        # enter MNA assembly, where there is intentionally no matrix to solve.
        if self._is_empty_circuit():
            self._time_array_buf[:] = np.arange(n_steps, dtype=np.float64) * self.dt
            self._actual_steps = n_steps
            self.step_count = n_steps
            self.time = self._time_array_buf[-1] if n_steps else 0.0
            self.time_array = self._time_array_buf
            self.voltage_results = {}
            self._stats['total_steps'] = n_steps
            self._timing['init'] = _t() - t_run_start
            self._timing['run_total'] = self._timing['init']
            self._results_valid = True
            if self.verbose:
                self.print_timing_report()
                logger.info("空电路仿真完成,共 %d 步", self.step_count)
                self.print_solver_statistics()
            return

        # ---- 预编译传输线注入映射 (6-tuple) ----
        self._line_inject_maps = []
        self._line_vk_bufs = {}
        self._line_vm_bufs = {}

        for line in self.transmission_lines.values():
            nk_list, nm_list = self._get_line_nodes(line)
            nc = len(nk_list)
            is_multi = self._is_multiphase_line(line)
            has_full = hasattr(line, 'full_step')

            k_idx = np.array([self._indexer.to_compact(n) for n in nk_list], dtype=int)
            m_idx = np.array([self._indexer.to_compact(n) for n in nm_list], dtype=int)

            self._line_inject_maps.append((line, k_idx, m_idx, nc, is_multi, has_full))
            self._line_vk_bufs[line.name] = np.zeros(nc, dtype=np.float64)
            self._line_vm_bufs[line.name] = np.zeros(nc, dtype=np.float64)

        # PSCAD v4.5+ 风格：在电路图/编译阶段并行求解各传输段，
        # 然后再创建 ULM batch 运行时。
        self.compile_transmission_lines(max_workers=self.line_compile_workers)
        self._build_ulm_batch_runtime()

        # ---- 预检测定时开关 ----
        self._has_timed_switches = any(
            b.element_type == ElementType.SWITCH
            and (b.t_close >= 0 or b.t_open >= 0)
            and name not in self._lpm_elements
            for name, b in self.branches.items()
        )

        # MNA: 预建有序电压源列表(供 _build_MNA_matrix 和 _build_MNA_rhs 使用)
        self._vs_list = list(self.voltage_sources.values())
        self._vs_index_map = {
            vs.name: idx for idx, vs in enumerate(self._vs_list)
        }

        # ---- 源预采样 (可选优化) ----
        if self.pre_sample_sources:
            self._pre_sample_sources(n_steps)

        # Lock the node indexer so no new external ids can sneak in during
        # the main loop.  From here on all MNA dimensions use _compact_n.
        self._indexer.freeze()
        self._compact_n = self._indexer.n

        t_init_end = _t()
        self._timing['init'] = t_init_end - t_run_start

        # 主循环 — 委托给 TimeStepper
        self._stepper.run(self, n_steps, self._timing)

        # 截断处理浮点时间累积误差 — 委托给 ResultStore.finalize
        self._actual_steps = n_steps

        if self._result_store is not None:
            # Existing write paths alias into ResultStore arrays directly,
            # so _steps_written stays 0.  Set it before finalize.
            self._result_store._steps_written = n_steps
            self._result_store.finalize(self._indexer)

        # Sync legacy attributes from ResultStore (backward compat)
        self.time_array = self._time_array_buf[:n_steps]
        if self._voltage_buf is not None:
            self.voltage_results = {
                self._indexer.to_external(c): self._voltage_buf[c, :n_steps]
                for c in range(self._indexer.n)
            }
        else:
            self.voltage_results = {}

        t_run_end = _t()
        self._timing['run_total'] = t_run_end - t_run_start

        # 自动打印计时报告
        if self.verbose:
            self.print_timing_report()

        if self.verbose:
            logger.info("仿真完成,共 %d 步", self.step_count)
            self.print_solver_statistics()

        self._results_valid = True

    def _reset_caches(self) -> None:
        """重置所有矩阵缓存。"""
        self._stamping.mark_dirty()
        self._active_mna_solver_name = _SPARSE_SOLVER_NAME
        self._vs_list = None
        self._vs_index_map = None
        self._rhs_buf = None

    def _init_result_store(self, n_steps: int) -> None:
        """Create :class:`ResultStore` and alias legacy buffer attributes.

        After this call every pre-existing buffer reference
        (``_time_array_buf``, ``_voltage_buf``, ``_branch_v_bufs``,
        ``_vs_current_bufs``, probe data arrays, etc.) points into the
        ``ResultStore`` so existing write paths work unchanged.
        """
        vp_names = list(self.voltage_probes.keys())
        cp_names = list(self.branch_current_probes.keys())

        self._result_store = ResultStore(
            n_nodes=self._indexer.n,
            n_steps=n_steps,
            record_node_voltage=self.record_all_node_voltages,
            vs_names=list(self.voltage_sources.keys()),
            record_branch_history=bool(self.record_branch_history),
            branch_names=list(self.branches.keys()),
            voltage_probe_names=vp_names,
            branch_current_probe_names=cp_names,
        )

        # Alias legacy attributes so all existing per-step write paths
        # (step_post_solve_V_I, _record_probes, run() body, etc.)
        # transparently route into the ResultStore.
        rs = self._result_store
        self._time_array_buf = rs.time
        self._voltage_buf = rs.voltage
        self._vs_current_bufs = rs.vs_current
        self._branch_v_bufs = rs.branch_v
        self._branch_i_bufs = rs.branch_i
        self._voltage_probe_data = rs.voltage_probe_data
        self._branch_current_probe_data = rs.branch_current_probe_data

        # Rebuild probe-index metadata (previously in _init_probe_storage).
        self._voltage_probe_names = vp_names
        self._branch_current_probe_names = cp_names
        self._voltage_probe_index = {n: i for i, n in enumerate(vp_names)}
        self._branch_current_probe_index = {n: i for i, n in enumerate(cp_names)}

    def _log_run_header(self) -> None:
        """仿真启动时的概要日志。"""
        m_vs = len(self.voltage_sources)
        logger.info(
            "EMTP 仿真 (MNA + %s): dt=%.3fμs, t_end=%.1fμs, "
            "节点=%d, MNA维度=%d (n=%d + m_vs=%d), "
            "支路=%d, IS=%d, VS=%d, TL=%d, 变压器=%d",
            _SPARSE_SOLVER_NAME,
            self.dt * 1e6, self.finish_time * 1e6,
            self.num_nodes, self.num_nodes + m_vs, self.num_nodes, m_vs,
            len(self.branches), len(self.current_sources),
            len(self.voltage_sources), len(self.transmission_lines),
            len(self.transformers),
        )
        if self._vs_node_set:
            logger.info("电压源节点: %s", sorted(self._vs_node_set))
        if self._seg_node_map:
            logger.info("分段线性元件: %s", list(self._seg_node_map))
        if self._lpm_elements:
            logger.info("LPM 绝缘子: %s", list(self._lpm_elements))

    # =========================================================================
    # 结果获取
    # =========================================================================

    def get_time(self, unit: str = 's') -> np.ndarray:
        """时间数组。unit ∈ {'s','ms','us','ns'}。"""
        self._require_run_completed()
        t = np.asarray(self.time_array)
        return {'ms': t * 1e3, 'us': t * 1e6,
                'ns': t * 1e9}.get(unit, t.copy())

    def get_node_voltage(
        self, node: Union[str, int], unit: str = 'V',
    ) -> np.ndarray:
        """获取节点电压历史。支持字符串节点名或整数节点号。"""
        node_id = self._resolve_existing_node(node)
        self._require_run_completed()
        if not self.record_all_node_voltages:
            raise RuntimeError(
                "Full node voltage history was not recorded. Set "
                "record_all_node_voltages=True before run(), or register a "
                "voltage probe with add_voltage_probe() and use get_probe()."
            )
        if node_id == 0:
            return np.zeros(self._actual_steps)
        if node_id not in self.voltage_results:
            label = self._node_label(node_id)
            raise ValueError(f"节点 {label} 不存在")
        V = np.asarray(self.voltage_results[node_id])
        return {'kV': V / 1e3, 'mV': V * 1e3}.get(unit, V.copy())

    def get_branch_current(self, name: str, unit: str = 'A') -> np.ndarray:
        if name not in self.branches:
            raise ValueError(f"支路 {name} 不存在")
        self._require_run_completed()
        if hasattr(self, '_branch_i_bufs') and name in self._branch_i_bufs:
            actual = getattr(self, '_actual_steps', len(self._branch_i_bufs[name]))
            I = self._branch_i_bufs[name][:actual].copy()
        else:
            I = np.array(self.branches[name].current_history)
        if I.size == 0:
            raise RuntimeError(
                f"Branch current history for {name!r} was not recorded. "
                "Set record_branch_history=True before run(), or register a "
                "branch current probe with add_branch_current_probe()."
            )
        return {'kA': I / 1e3, 'mA': I * 1e3}.get(unit, I)

    def get_branch_voltage(self, name: str, unit: str = 'V') -> np.ndarray:
        if name not in self.branches:
            raise ValueError(f"支路 {name} 不存在")
        self._require_run_completed()
        if hasattr(self, '_branch_v_bufs') and name in self._branch_v_bufs:
            actual = getattr(self, '_actual_steps', len(self._branch_v_bufs[name]))
            V = self._branch_v_bufs[name][:actual].copy()
        else:
            V = np.array(self.branches[name].voltage_history)
        if V.size == 0:
            raise RuntimeError(
                f"Branch voltage history for {name!r} was not recorded. "
                "Set record_branch_history=True before run()."
            )
        return {'kV': V / 1e3, 'mV': V * 1e3}.get(unit, V)

    def get_source_current(self, name: str) -> np.ndarray:
        if name not in self.current_sources:
            raise ValueError(f"电流源 {name} 不存在")
        self._require_run_completed()
        data = np.array(self.current_sources[name].current_history)
        if data.size == 0:
            raise RuntimeError(
                f"Current source history for {name!r} was not recorded. "
                "Set record_source_history=True before run()."
            )
        return data

    # ---- 电压源结果 ----

    def get_vs_current(self, name: str, unit: str = 'A') -> np.ndarray:
        """电压源电流历史(正方向:正端→外部电路→负端)。"""
        if name not in self.voltage_sources:
            raise ValueError(f"电压源 {name} 不存在")
        self._require_run_completed()
        if hasattr(self, '_vs_current_bufs') and name in self._vs_current_bufs:
            I = self._vs_current_bufs[name].copy()
        else:
            I = np.array(self.voltage_sources[name].current_history)
        if I.size == 0:
            raise RuntimeError(
                f"Voltage source current history for {name!r} was not recorded. "
                "Set record_source_history=True before run()."
            )
        return {'kA': I / 1e3, 'mA': I * 1e3}.get(unit, I)

    def get_vs_voltage(self, name: str, unit: str = 'V') -> np.ndarray:
        """电压源激励电压历史。"""
        if name not in self.voltage_sources:
            raise ValueError(f"电压源 {name} 不存在")
        self._require_run_completed()
        vs = self.voltage_sources[name]
        V = np.array([vs.voltage_at(t) for t in self.time_array])
        return {'kV': V / 1e3, 'mV': V * 1e3}.get(unit, V)



    # ---- 传输线结果 ----

    def _line_history(
        self, name: str, attr: str, unit_map: Dict[str, float],
        unit: str, phase: Optional[int],
    ) -> np.ndarray:
        if name not in self.transmission_lines:
            raise ValueError(f"传输线 {name} 不存在")
        self._require_run_completed()
        data = np.array(getattr(self.transmission_lines[name], attr))
        if data.size == 0:
            raise RuntimeError(
                f"Transmission line history {attr!r} for {name!r} was not recorded. "
                "Set record_line_history=True before run()."
            )
        if phase is not None and data.ndim > 1:
            data = data[:, phase]
        scale = unit_map.get(unit, 1.0)
        return data * scale if scale != 1.0 else data

    def get_line_current_k(
        self, name: str, unit: str = 'A', phase: Optional[int] = None,
    ) -> np.ndarray:
        return self._line_history(
            name, 'I_k_history', {'kA': 1e-3, 'mA': 1e3}, unit, phase,
        )

    def get_line_current_m(
        self, name: str, unit: str = 'A', phase: Optional[int] = None,
    ) -> np.ndarray:
        return self._line_history(
            name, 'I_m_history', {'kA': 1e-3, 'mA': 1e3}, unit, phase,
        )

    def get_line_voltage_k(
        self, name: str, unit: str = 'V', phase: Optional[int] = None,
    ) -> np.ndarray:
        return self._line_history(
            name, 'V_k_history', {'kV': 1e-3, 'mV': 1e3}, unit, phase,
        )

    def get_line_voltage_m(
        self, name: str, unit: str = 'V', phase: Optional[int] = None,
    ) -> np.ndarray:
        return self._line_history(
            name, 'V_m_history', {'kV': 1e-3, 'mV': 1e3}, unit, phase,
        )

    def get_line_info(self, name: str) -> Dict[str, Any]:
        if name not in self.transmission_lines:
            raise ValueError(f"传输线 {name} 不存在")
        line = self.transmission_lines[name]
        info = line.get_info()
        info.setdefault('Zc', self._get_line_Zc(line, info))
        info.setdefault('tau', self._get_line_tau(line, info))
        return info

    def get_transformer_info(self, name: str) -> Dict[str, Any]:
        if name not in self.transformers:
            raise ValueError(f"UMEC 变压器 {name} 不存在")
        return self.transformers[name].get_info()

    # =========================================================================
    # 统计与报告
    # =========================================================================

    def get_solver_statistics(self) -> Dict[str, Any]:
        stats = dict(self._stats)

        total = stats['total_steps']
        stats['segment_switch_ratio'] = (stats['segment_switches'] / total
                                          if total else 0.0)
        stats['segment_resolve_ratio'] = (stats['segment_resolves'] / total
                                           if total else 0.0)

        stats['segmented_elements'] = list(self._seg_node_map)
        stats['num_segmented_elements'] = len(self._seg_node_map)
        stats['lpm_elements'] = list(self._lpm_elements)
        stats['num_lpm_elements'] = len(self._lpm_elements)
        stats['lpm_flashover_log'] = list(self._lpm_flashover_log)
        stats['lpm_flashovers'] = len(self._lpm_flashover_log)
        stats['sparse_solver'] = _SPARSE_SOLVER_NAME
        stats['mna_size'] = self.num_nodes + len(self.voltage_sources)
        return stats

    # =========================================================================
    # Snapshot / Resume
    # =========================================================================

    def save_snapshot(
        self, path, *, config=None, notes: str = "",
    ) -> None:
        """Save the current solver state to *path*.

        Creates a directory with metadata.json, branches.json, lines.json,
        lpm.json and arrays.npz.  Restore with :meth:`load_snapshot`.
        """
        from emtp.snapshot.serializer import save_snapshot as _save
        _save(self, path, config=config, notes=notes)

    def load_snapshot(self, path, *, strict: bool = True) -> None:
        """Restore dynamic state from a previously-saved snapshot directory.

        The solver must already have the correct topology (branches, lines,
        transformers).  Only dynamic state is restored.
        """
        from emtp.snapshot.restore import load_snapshot_into_solver
        load_snapshot_into_solver(self, path, strict=strict)

    def run_until(self, t_end: float, *, reset_state: bool = False) -> None:
        """Run the simulation from the current time to *t_end*.

        When *reset_state* is ``False`` (default), preserves existing
        dynamic state (branch histories, LPM leader length, etc.) and
        continues from the solver's current time.

        When *reset_state* is ``True``, resets all dynamic state before
        running (equivalent to a fresh :meth:`run`).
        """
        old_finish = self.finish_time
        old_time = self.time

        try:
            self.finish_time = t_end

            if reset_state:
                self.reset_dynamic_state()
                self._stats = self._fresh_stats()
                self._timing = defaultdict(float)
                self._reset_caches()
            else:
                # Clear caches so MNA rebuilds on next solve
                self._reset_caches()
                self.mark_topology_changed("run_until resume")

            self._is_running = True

            n_steps = int(round((t_end - self.time) / self.dt))
            if n_steps <= 0:
                return

            # Allocate fresh ResultStore for the new segment
            self._init_result_store(n_steps)
            self._vs_list = list(self.voltage_sources.values())
            self._vs_index_map = {
                vs.name: idx for idx, vs in enumerate(self._vs_list)
            }

            # Ensure lines are compiled (needed for first run_until call
            # and after load_snapshot).  Safe to call — it skips if already
            # compiled unless force=True.
            self.compile_transmission_lines(max_workers=self.line_compile_workers)
            self._build_ulm_batch_runtime()

            # Rebuild line inject maps (normally done in run() init)
            self._line_inject_maps = []
            self._line_vk_bufs = {}
            self._line_vm_bufs = {}
            for line in self.transmission_lines.values():
                nk_list, nm_list = self._get_line_nodes(line)
                nc = len(nk_list)
                is_multi = self._is_multiphase_line(line)
                has_full = hasattr(line, 'full_step')
                k_idx = np.array(
                    [self._indexer.to_compact(n) for n in nk_list], dtype=int,
                )
                m_idx = np.array(
                    [self._indexer.to_compact(n) for n in nm_list], dtype=int,
                )
                self._line_inject_maps.append(
                    (line, k_idx, m_idx, nc, is_multi, has_full),
                )
                self._line_vk_bufs[line.name] = np.zeros(nc, dtype=np.float64)
                self._line_vm_bufs[line.name] = np.zeros(nc, dtype=np.float64)

            # Sync _line_inject_maps_nonbatch so the non-batch path in
            # _update_lines_combined picks up all lines (Bergeron, etc.).
            self._line_inject_maps_nonbatch = list(self._line_inject_maps)

            if not self._is_empty_circuit():
                self._indexer.freeze()
                self._compact_n = self._indexer.n
                self._stepper.run(self, n_steps, self._timing)

            # Finalize results
            if self._result_store is not None:
                self._result_store._steps_written = n_steps
                self._result_store.finalize(self._indexer)

            self._actual_steps = n_steps
            self.time_array = self._time_array_buf[:n_steps]
            if self._voltage_buf is not None:
                self.voltage_results = {
                    self._indexer.to_external(c): self._voltage_buf[c, :n_steps]
                    for c in range(self._indexer.n)
                }
            self._results_valid = True
            self.time = t_end

        finally:
            self.finish_time = old_finish

    def print_solver_statistics(self) -> None:
        """打印求解统计(用户侧报告)。"""
        stats = self.get_solver_statistics()
        sep = "-" * 50

        print(sep)
        print("求解统计")
        print(sep)
        print(f"  求解方法       : MNA + {stats.get('sparse_solver', 'N/A')}")
        print(f"  MNA 维度       : {stats.get('mna_size', 'N/A')}")
        print(f"  总步数         : {stats['total_steps']}")
        print(f"  分段线性元件数 : {stats['num_segmented_elements']}")
        print(f"  段切换次数     : {stats['segment_switches']} "
              f"({stats['segment_switch_ratio']*100:.2f}%)")
        print(f"  重解次数       : {stats['segment_resolves']} "
              f"({stats['segment_resolve_ratio']*100:.2f}%)")
        print(f"  最大段迭代数   : {stats['max_seg_iter']}")

        g_rebuilds = stats.get('G_rebuilds', 0)
        g_hits = stats.get('G_cache_hits', 0)
        total_calls = g_rebuilds + g_hits
        if total_calls > 0:
            print(f"  G 重建 / 命中  : {g_rebuilds} / {g_hits} "
                  f"(命中率 {g_hits / total_calls * 100:.1f}%)")

        if stats['num_lpm_elements'] > 0:
            print(f"  LPM 绝缘子数   : {stats['num_lpm_elements']}")
            print(f"  LPM 重解次数   : {stats.get('lpm_resolves', 0)}")
            print(f"  闪络事件       : {stats['lpm_flashovers']}")
            for event in stats['lpm_flashover_log']:
                print(f"    {event['name']}: t={event['time_us']:.2f}μs, "
                      f"V={event['voltage_kV']:.1f}kV")
        print(sep)

    def print_timing_report(self) -> None:
        """打印模块级性能剖析报告。"""
        T = self._timing
        total = T.get('run_total', 1e-12)
        n_steps = self._stats.get('total_steps', 1) or 1
        sep = "=" * 66

        print(f"\n{sep}")
        solver_name = 'SuperLU(splu)'
        print(f"  EMTP 求解器 (MNA + {solver_name}) · 模块级性能剖析")
        print(sep)
        print(f"  仿真步数       :  {n_steps:>12,d}")
        print(f"  节点数         :  {self.num_nodes:>12d}")
        print(f"  MNA 维度       :  {self.num_nodes + len(self.voltage_sources):>12d}")
        print(f"  线性求解器     :  {solver_name:>12s}")
        print(f"  dt             :  {self.dt * 1e6:>12.3f} μs")
        print(f"  仿真时长       :  {self.finish_time * 1e6:>12.1f} μs")
        print(f"  总运行时间     :  {total:>12.3f} s")
        print(f"  平均每步耗时   :  {total / n_steps * 1e6:>12.2f} μs")
        print("-" * 66)
        fmt = "  {:<40s} {:>8.4f}   {:>6.2f}%   {:>8.2f}"
        hdr = "  {:<40s} {:>8s}   {:>7s}   {:>8s}"
        print(hdr.format("模块", "耗时(s)", "占比(%)", "每步(μs)"))
        print("-" * 66)

        rows = [
            ('init',                   '初始化 (预分配+重置)'),
            ('switch_update',          '开关状态更新'),
            ('solve_step_total',       '核心求解 (solve_step)'),
            ('branch_update',          '支路量更新'),
            ('line_combined_update',   '传输线更新 (并行batch/串行fallback)'),
            ('branch_history_update',  '支路历史源更新'),
            ('transformer_history',    '变压器历史源更新'),
            ('probe_store',            '轻量探针记录'),
            ('data_store',             '数据存储'),
        ]

        accounted = 0.0
        for key, label in rows:
            v = T.get(key, 0.0)
            accounted += v
            pct = v / total * 100 if total > 0 else 0
            per_step = v / n_steps * 1e6
            print(fmt.format(label, v, pct, per_step))

        other = max(total - accounted, 0)
        pct_o = other / total * 100 if total > 0 else 0
        print("-" * 66)
        print(fmt.format("其他 (循环控制等)", other, pct_o, other / n_steps * 1e6))
        print(fmt.format("合计", total, 100.0, total / n_steps * 1e6))

        # TOP-3
        vals = [(label, T.get(key, 0.0)) for key, label in rows]
        vals.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  >>> TOP-3 耗时模块:")
        for i, (label, v) in enumerate(vals[:3]):
            pct = v / total * 100
            print(f"      {i+1}. {label:<20s} {v:.4f}s  ({pct:.1f}%)")
        print(sep)

    def get_timing_report(self) -> Dict[str, Any]:
        """返回计时数据字典（供程序化分析）。"""
        T = dict(self._timing)
        n_steps = self._stats.get('total_steps', 1) or 1
        T['_meta'] = {
            'total_steps': n_steps,
            'dt': self.dt,
            'finish_time': self.finish_time,
            'num_nodes': self.num_nodes,
            'mna_size': self.num_nodes + len(self.voltage_sources),
            'sparse_solver': 'SuperLU(splu)',
            'run_total': T.get('run_total', 0),
            'avg_step_us': T.get('run_total', 0) / n_steps * 1e6,
        }
        return T

    def print_circuit_summary(self) -> None:
        """打印电路结构摘要(用户侧报告)。"""
        sep = "-" * 60
        print(sep)
        print("电路摘要")
        print(sep)
        print(f"  节点数         : {self.num_nodes}")
        print(f"  支路数         : {len(self.branches)}")
        print(f"  电流源         : {len(self.current_sources)}")
        print(f"  电压源         : {len(self.voltage_sources)}")
        print(f"  传输线         : {len(self.transmission_lines)}")
        print(f"  UMEC 变压器    : {len(self.transformers)}")
        print(f"  分段线性元件   : {len(self._seg_node_map)}")
        print(f"  LPM 绝缘子     : {len(self._lpm_elements)}")
        print(f"  命名节点数     : {len(self.nodes)}")

        if self._vs_node_set:
            vs_nodes = sorted(self._vs_node_set)
            m_vs = len(self.voltage_sources)
            print(f"  MNA 维度       : {self.num_nodes + m_vs} "
                  f"(n={self.num_nodes} + m_vs={m_vs})")
            print(f"  电压源节点     : {vs_nodes}")
            print(f"  稀疏求解器     : {_SPARSE_SOLVER_NAME}")

        if self.branches:
            print("\n支路:")
            for name, b in self.branches.items():
                mark = ""
                if name in self._seg_node_map:
                    mark = " [分段线性]"
                elif name in self._lpm_elements:
                    lpm = self._lpm_elements[name]
                    mark = (f" [LPM d={lpm.config.gap_length:.3f}m, "
                            f"E0={lpm.E0_eff:.0f}kV/m]")
                print(f"  {name}: {b.element_type.value} "
                      f"({b.node_from}-{b.node_to}){mark}")

        if self.current_sources:
            print("\n电流源:")
            for name, s in self.current_sources.items():
                print(f"  {name}: ({s.node_from}-{s.node_to})")

        if self.voltage_sources:
            print("\n电压源:")
            for name, vs in self.voltage_sources.items():
                print(f"  {name}: (+){vs.node_pos}—(-){vs.node_neg}, "
                      f"V(0)={vs.voltage_at(0.0):.2f}V")

        if self.transmission_lines:
            print("\n传输线:")
            for name, line in self.transmission_lines.items():
                info = line.get_info()
                nc = self._get_line_nc(line)
                nodes_k, nodes_m = self._get_line_nodes(line)
                Zc = self._get_line_Zc(line, info)
                tau = self._get_line_tau(line, info)
                mtype = info.get('model_type', 'Unknown')
                if nc > 1:
                    print(f"  {name}: {mtype} ({nc}相)")
                    print(f"    k端={nodes_k}, m端={nodes_m}")
                    print(f"    Zc≈{Zc:.1f}Ω, τ≈{tau*1e6:.1f}μs")
                else:
                    print(f"  {name}: {mtype} ({nodes_k[0]}-{nodes_m[0]}), "
                          f"Zc={Zc:.1f}Ω, τ={tau*1e6:.1f}μs")

        if self.transformers:
            print("\nUMEC 变压器:")
            for tname, xfmr in self.transformers.items():
                info = xfmr.get_info()
                print(f"  {tname}: {info['S_rated']/1e6:.2f} MVA, "
                      f"{info['n_phases']}相×{info['n_windings']}绕组")
                for w in range(info['n_windings']):
                    print(f"    绕组 #{w+1}: V_ph={info['V_phase'][w]/1e3:.3f}kV, "
                          f"N={info['N'][w]:.4f}")
                print(f"    端口: {xfmr.get_port_nodes()}")
        print(sep)


__all__ = [
    'EMTPSolver', 'VoltageSource', 'NodeBook',
    'LPMInsulatorType', 'LPMConfig', 'InsulatorFlashoverLPM',
    'UMECTransformer', 'UMECTransformerData',
    'WindingType', 'create_umec_transformer_3ph_bank',
    'create_lightning_current_source', 'create_standard_twoexpf_current_source',
    'TWOEXPFCurrentSource', 'HEIDLERFCurrentSource',
]
