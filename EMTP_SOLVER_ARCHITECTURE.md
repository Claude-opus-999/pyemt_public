# EMTP 电磁暂态求解器 — 架构文档 v3.0

## 1. 项目概述

`emtp_v0.2` 是一个基于 Python 的电磁暂态（EMTP）仿真求解器，使用修正节点分析法（MNA）进行电路求解，集成了多相传输线、非线性元件、UMEC 变压器和绝缘子闪络模型。

**仓库**: `github.com/Claude-opus-999/pyemt_public`
**当前 commit**: `cf8b7dc` (159 tests, 0 failures)
**Python**: 3.12+ | **依赖**: numpy, scipy

### 1.1 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `emtp/solver.py` | 3525 | **主求解器** — canonical EMTPSolver 实现 |
| `emtp/circuit.py` | 73 | CircuitModel — 元件数据容器 |
| `emtp/nodes.py` | — | NodeIndexer / NodeBook — 节点管理 |
| `emtp/types.py` | 188 | Branch / CurrentSource / VoltageSource / RHSPlan 等数据类型 |
| `emtp/stamping.py` | 157 | COOStamper / StampingEngine — MNA G 矩阵装配 |
| `emtp/sparse_solver.py` | 84 | SparseLinearSolver — SuperLU 稀疏求解 |
| `emtp/validation.py` | — | Circuit 校验 |
| `emtp/results/__init__.py` | 87 | 结果 helper 函数（单位缩放、节点电压/支路电流读取） |
| `emtp/results/store.py` | 168 | ResultStore — 预分配结果缓冲区管理 |
| `emtp/runtime/__init__.py` | 172 | DynamicDeviceRuntime — 每步状态管理 |
| `emtp/runtime/resolve.py` | 129 | ResolveManager + ResolveEvent — 重解循环 |
| `emtp/runtime/stepper.py` | 41 | TimeStepper — 主循环 |
| `emtp/assembly/__init__.py` | 5 | 装配模块 |
| `emtp/assembly/mna.py` | 108 | MNAAssembler — 系统矩阵/RHS 装配 |
| `emtp/devices/base.py` | 47 | Device Protocol — 二端元件抽象接口 |
| `emtp/devices/multiport.py` | 83 | MultiPortDevice Protocol — 多端口元件抽象接口 |
| `emtp/devices/resistor.py` | — | ResistorDevice |
| `emtp/devices/inductor.py` | — | InductorDevice |
| `emtp/devices/capacitor.py` | — | CapacitorDevice |
| `emtp/devices/switch.py` | — | SwitchDevice |
| `emtp/devices/series_rl.py` | — | SeriesRLDevice |
| `emtp/devices/nonlinear.py` | — | NonlinearResistorDevice (MOA) |
| `emtp/devices/lpm.py` | — | LPMFlashoverDevice (绝缘子闪络) |
| `emtp/lines/bergeron.py` | 81 | BergeronLineDevice — Bergeron 线路 MultiPort adapter |
| `emtp/lines/ulm.py` | 130 | ULMLineDevice — ULM 线路 MultiPort adapter |
| `emtp/transformers/umec.py` | 113 | UMECTransformerDevice — UMEC 变压器 MultiPort adapter |

**外部底层模型**（可选依赖）:

| 文件 | 说明 |
|------|------|
| `transmission_line_emtp_v2.py` | Bergeron 传输线底层模型 |
| `ulm_transmission_line_PARA.py` | ULM 频率相关线路模型（含并行 batch） |
| `umec_transformer.py` | UMEC 多端口变压器模型 |
| `nonlinear_models_pscad.py` | PSCAD 风格分段 MOA + LPM 绝缘子闪络 |
| `atp_lightning_current_generator_simplified.py` | ATP 兼容雷电电流源 |

**兼容入口**:

| 文件 | 行数 | 说明 |
|------|------|------|
| `emtp_solver_v3.py` | 144 | Legacy compat shim — 转发到 `emtp.solver` |

---

## 2. 分层架构

求解器采用**五层架构**，从底层数据结构到顶层门面逐步组合：

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMTPSolver (门面层)                             │
│  add_* / validate / run / get_* / probes / print                 │
│                                                                    │
│  组合: CircuitModel + MNAAssembler + SparseLinearSolver           │
│        + DynamicDeviceRuntime + ResolveManager + TimeStepper      │
│        + ResultStore + Device/MultiPortDevice                     │
└──────┬────────────┬────────────┬─────────────┬────────────────────┘
       │            │            │             │
       ▼            ▼            ▼             ▼
┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  Circuit  │ │  MNA     │ │  Sparse  │ │  TimeStepper │
│  Model    │ │ Assembler│ │  Linear  │ │              │
│           │ │          │ │  Solver  │ │  主循环      │
│ 元件容器  │ │ G/RHS    │ │          │ │              │
│ 节点管理  │ │ 装配     │ │ SuperLU  │ │  每步调度    │
└───────────┘ └──────────┘ └──────────┘ └──────────────┘
       │            │            │             │
       └────────────┼────────────┼─────────────┘
                    │            │
                    ▼            ▼
          ┌──────────────┐ ┌──────────────────┐
          │  Device (7)  │ │ MultiPortDevice   │
          │              │ │ (3 adapters)      │
          │ R/L/C/SW/    │ │                  │
          │ SRL/MOA/LPM  │ │ Bergeron/ULM/UMEC│
          └──────────────┘ └──────────────────┘
                    │            │
                    ▼            ▼
          ┌──────────────────────────────────┐
          │    DynamicDeviceRuntime          │
          │    + ResolveManager              │
          │    + ResultStore                 │
          └──────────────────────────────────┘
```

### 2.1 各层职责

| 层 | 模块 | 职责 |
|----|------|------|
| **数据** | `types.py`, `nodes.py`, `circuit.py` | Branch / VoltageSource / NodeIndexer / CircuitModel |
| **协议** | `devices/base.py`, `devices/multiport.py` | Device / MultiPortDevice 抽象接口 |
| **实现** | `devices/`, `lines/`, `transformers/` | 7 个二端设备 + 3 个多端口 adapter |
| **装配** | `stamping.py`, `assembly/mna.py`, `sparse_solver.py` | COO 累加 → CSC 矩阵 → SuperLU 求解 |
| **运行时** | `runtime/`, `results/store.py` | 状态管理、重解循环、主循环、结果存储 |
| **门面** | `solver.py` | EMTPSolver 统一 API |

---

## 3. MNA 修正节点分析

### 3.1 数学原理

EMTP 求解器使用**修正节点分析法**（Modified Nodal Analysis）构建电路方程。

对于含 n 个节点和 m 个理想电压源的电路，构建 (n+m)×(n+m) 增广系统：

```
    ┌       ┐ ┌     ┐   ┌   ┐
    │ G   B │ │  v  │   │ I │
    │       │ │     │ = │   │
    │ C   D │ │ i_s │   │ E │
    └       ┘ └     ┘   └   ┘
```

- **G** (n×n): 节点导纳矩阵 — 所有二端元件和传输线的 Norton 等效电导
- **B** (n×m): 电压源关联矩阵 — 每列包含 +1（正端）和 -1（负端）
- **C** (m×n): = Bᵀ
- **D** (m×m): 零矩阵
- **v**: 节点电压向量（求解目标）
- **i_s**: 电压源电流（求解副产物）
- **I**: 独立电流源 + 历史源等效注入电流
- **E**: 电压源设定值

### 3.2 元件 Norton 等效

每个动态元件使用**隐式梯形法**（trapezoidal integration）离散化为 Norton 等效电路：

| 元件 | Geq | Ihist |
|------|-----|-------|
| 电阻 R | 1/R | 0 |
| 电感 L | Δt/(2L) | -(i(t-Δt) + (Δt/(2L))·v(t-Δt)) |
| 电容 C | 2C/Δt | i(t-Δt) + (2C/Δt)·v(t-Δt) |
| 串联 RL | G_L/(1+R·G_L) | I_L_hist/(1+R·G_L) |
| 开关 | 1/R_closed 或 1/R_open | 0 |
| 传输线 | 1/Zc (两端) | I_hist_k, I_hist_m |
| UMEC | G_tf (端口导纳矩阵) | I_hist_tf (端口历史电流) |

### 3.3 稀疏求解

矩阵以 `scipy.sparse.csc_matrix` 格式存储，使用 `scipy.sparse.linalg.splu`（SuperLU 后端）进行 LU 分解。矩阵仅在拓扑变化时重建，否则复用缓存的 LU 分解。

---

## 4. Device 协议体系

求解器定义了两级元件协议：

### 4.1 Device（二端元件）

位于 `emtp/devices/base.py`，适用于所有传统二端支路元件：

```python
class Device(Protocol):
    name: str                # 元件名
    _branch: Branch          # 关联的 Branch 数据结构

    def stamp_G(self, stamper, indexer):     # G 矩阵 stamp
    def stamp_rhs(self, rhs, indexer, t):    # RHS 历史源注入
    def update_branch_quantities(self, V, indexer):  # 求解后更新 V/I
    def update_history(self, dt):            # 历史源递推
    def reset_state(self):                   # 复位

    @property
    def is_dynamic(self) -> bool:            # 是否有动态状态
    @property
    def element_kind(self) -> str:           # 元件类型标签
```

**已有实现**（7 个）：

| 类 | 文件 | 元件类型 |
|----|------|---------|
| `ResistorDevice` | `resistor.py` | 电阻 R |
| `InductorDevice` | `inductor.py` | 电感 L（梯形离散） |
| `CapacitorDevice` | `capacitor.py` | 电容 C（梯形离散） |
| `SwitchDevice` | `switch.py` | 定时开关 |
| `SeriesRLDevice` | `series_rl.py` | 串联 RL（无中间节点） |
| `NonlinearResistorDevice` | `nonlinear.py` | 分段 MOA 避雷器 |
| `LPMFlashoverDevice` | `lpm.py` | 先导发展法绝缘子闪络 |

### 4.2 MultiPortDevice（多端口元件）

位于 `emtp/devices/multiport.py`，适用于传输线和变压器等无法映射到单一 Branch 的复杂模型：

```python
class MultiPortDevice(Protocol):
    name: str

    @property
    def ports(self) -> tuple[tuple[int, int], ...]:  # 端口列表
    @property
    def contributes_G(self) -> bool:                  # 是否贡献 G
    @property
    def is_dynamic(self) -> bool:                     # 是否有历史状态

    def register_nodes(self, indexer):                # 注册节点
    def stamp_G(self, stamper, indexer):              # G 矩阵 stamp
    def stamp_rhs(self, rhs, indexer, t):             # RHS 注入
    def update_after_solve(self, V, indexer, t):      # 求解后读取端口量
    def update_history(self, V, indexer, dt):         # 历史源递推
    def check_rebuild_required(self, V, indexer, t):  # 是否需要重建矩阵
    def reset_state(self):                            # 复位
```

**已有 adapter**（3 个）：

| 类 | 文件 | 底层模型 |
|----|------|---------|
| `BergeronLineDevice` | `lines/bergeron.py` | `transmission_line_emtp_v2.BergeronLine` |
| `ULMLineDevice` | `lines/ulm.py` | `ulm_transmission_line_PARA.ULMLine` |
| `UMECTransformerDevice` | `transformers/umec.py` | `umec_transformer.UMECTransformer` |

**Protocol 与 adapter 的关系**：adapter 实现 Protocol 的全部方法，但内部仍然调用底层模型的现有逻辑。这种方式无需重写物理模型即可获得统一 dispatch 能力。

**当前状态**：adapter 已验证协议正确性（Matrix/RHS stamping 与 legacy 路径等价），通过 `use_multiport_lines` / `use_multiport_transformers` feature flags 控制是否注册。默认关闭，待充分验证后切换。

---

## 5. 运行时系统

### 5.1 DynamicDeviceRuntime — 每步状态管理

`emtp/runtime/__init__.py:17`

```python
class DynamicDeviceRuntime:
    def __init__(self, dt: float)

    # 每步调用顺序:
    def step_pre_solve(t, devices, lpm_names) -> bool           # 1. 定时开关
    def step_post_solve_V_I(V, devices, indexer, ...)           # 3. 更新支路 V/I
    def step_post_solve_history(devices)                        # 6. 历史源递推
    def post_solve_resolve_check(V, t, lpm, transformers, ...)  # 重解检测
```

### 5.2 ResolveManager — 重解循环

`emtp/runtime/resolve.py:48`

```python
class ResolveManager:
    def __init__(self, max_iter: int = 5)

    # 布尔接口（当前使用）
    def solve_with_resolve(solve_fn, check_fn, stats, t) -> V

    # 事件接口（新增，可替换布尔接口）
    def solve_with_resolve_events(solve_fn, event_check_fn, stats, t) -> V

    @property
    def last_events -> list[ResolveEvent]
```

重解循环处理以下场景：

```
solve → check LPM flashover? → mark dirty → rebuild G → re-solve
     → check UMEC saturation? → mark dirty → rebuild G → re-solve
     → check MOA segment switch? → update Geq/Ihist → rebuild G → re-solve
     → check MultiPortDevice.check_rebuild_required()
     → converged → return V
```

### 5.3 ResolveEvent — 统一事件

`emtp/runtime/resolve.py:18`

```python
@dataclass
class ResolveEvent:
    source: str                    # "LPM" | "MOA" | "UMEC" | "multiport"
    device_name: str               # 触发事件的设备名
    reason: str                    # "flashover" | "segment_switch" | "saturation"
    requires_matrix_rebuild: bool  # 是否需要重建 G 矩阵
    severity: str                  # "info" | "warning"
```

每次重解迭代中，所有触发源的事件被收集为 `list[ResolveEvent]`，统一记录和日志输出。

### 5.4 TimeStepper — 主循环

`emtp/runtime/stepper.py:14`

```python
class TimeStepper:
    def run(self, solver, n_steps, timing) -> None:
        for step_idx in range(n_steps):
            solver._run_one_step(step_idx, n_steps, _t)
        # Post-loop: export ULM batch state
```

`EMTPSolver.run()` 现为主循环委托 3 行：

```python
self._stepper.run(self, n_steps, self._timing)
```

### 5.5 每步执行顺序（关键！）

```
step_pre_solve(t)               ← 1. 定时开关事件
    ↓
_solve_step()                   ← 2. 求解 + 非线性/LPM/UMEC 重解
    ↓
step_post_solve_V_I(V)          ← 3. 支路电压/电流更新
    ↓
_record_probes(step_idx, V)     ← 4. 探针记录（在历史更新之前！）
_time_array_buf[step_idx] = t
_voltage_buf[:, step_idx] = V
    ↓
_update_lines_combined(V)       ← 5. 传输线状态更新
    ↓
step_post_solve_history()       ← 6. 支路历史源递推
    ↓
_update_transformer_history(V)  ← 7. 变压器历史源递推
```

探针必须在历史更新**之前**记录：这是因为探针需要看到当前步的 Ihist（梯形法的"当前"历史源），而非下一步的 Ihist。

---

## 6. 结果系统

### 6.1 ResultStore — 预分配缓冲区

`emtp/results/store.py:15`

```python
class ResultStore:
    def __init__(self, n_nodes, n_steps, *,
                 record_node_voltage, vs_names,
                 record_branch_history, branch_names,
                 voltage_probe_names, branch_current_probe_names)

    # 缓冲区
    time: np.ndarray                    # (n_steps,)
    voltage: np.ndarray | None          # (n_nodes, n_steps)
    vs_current: dict[str, np.ndarray]   # 电压源电流
    branch_v / branch_i: dict           # 支路电压/电流
    voltage_probe_data: np.ndarray      # 探针数据
    branch_current_probe_data: np.ndarray

    def record_step(step_idx, t, V, *, probe_values)  # 记录一步
    def finalize(indexer)                              # 裁截 + 构建 voltage_results
```

### 6.2 接入策略：别名化

`EMTPSolver._init_result_store()` 创建 `ResultStore` 后，将 solver 的旧 buffer 属性**别名化**为 `ResultStore` 的内部数组：

```python
self._time_array_buf = rs.time           # 别名
self._voltage_buf = rs.voltage           # 别名
self._vs_current_bufs = rs.vs_current    # 别名
# ... 所有旧属性指向 ResultStore 内部缓冲区
```

**效果**：所有现有写入路径（`step_post_solve_V_I`、`_record_probes`、`run()` 循环体）无需修改，通过别名透明写入 `ResultStore`。

### 6.3 结果查询 API

所有 getter 委托到 `emtp/results/__init__.py` 中的独立函数：

```python
# 用户 API          → 委托到
get_time(unit)       → emtp.results.scale_values
get_node_voltage(n)  → emtp.results.node_voltage_from_solution
get_branch_voltage(n)→ emtp.results.branch_voltage_from_solution
get_branch_current(n)→ emtp.results.branch_current_from_solution
get_voltage_probe(n) → emtp.results.scale_probe_values
get_branch_current_probe(n) → emtp.results.scale_probe_values
```

---

## 7. 电路校验

`EMTPSolver.validate_circuit(strict=True)` 在 `run()` 开始时自动调用，检查：

| 检查项 | 错误码 | 说明 |
|--------|--------|------|
| dt > 0, finish_time ≥ 0 | E001/E002 | 基本参数 |
| 探针引用节点存在 | E003/E004 | 探针校验 |
| 电路无节点 | E005 | 空电路 |
| 电压源环路 | E006 | Union-Find 检测 |
| 同一节点短路 | E007 | 拓扑 |
| R/L/C/SW 参数为正 | E008-E011 | 参数校验 |
| 电压源自环 | E012 | 拓扑 |
| 浮空节点 | E013 | BFS 连通分量检测 |

校验返回 `ValidationReport`，`strict=True` 时发现 error 直接抛 `RuntimeError`。

---

## 8. 元件添加 API

### 8.1 基本无源元件

```python
solver = EMTPSolver(dt=1e-6, finish_time=100e-6)

# 电阻
solver.add_R("r1", node_from=1, node_to=0, R=10.0)
solver.add_resistor("r1", 1, 0, R=10.0)   # alias

# 电感（隐式梯形法离散，可选并联阻尼 Rp）
solver.add_L("l1", 1, 2, L=1e-3, Rp=None)
solver.add_inductor("l1", 1, 2, L=1e-3)

# 电容
solver.add_C("c1", 2, 0, C=1e-6, Rp=None)
solver.add_capacitor("c1", 2, 0, C=1e-6)

# 串联 RL（无中间节点，缩减矩阵维度）
solver.add_series_RL("rl1", 1, 2, R=0.1, L=1e-3)

# 开关
solver.add_SW("sw1", 1, 2, t_close=10e-6, t_open=50e-6)
solver.add_switch("sw1", 1, 2, t_close=10e-6)
```

### 8.2 电源

```python
# 独立电流源（支持函数、常数、LightningWaveform、ATP 雷电源）
solver.add_IS("is1", 1, 0, lambda t: np.sin(2*np.pi*50*t))
solver.add_current_source("is1", 1, 0, 5.0)   # 常数 5A

# 理想电压源（MNA 增广方程）
solver.add_VS("vs1", node_pos=1, node_neg=0, voltage_func=lambda t: 100.0)
solver.add_voltage_source("vs1", 1, 0, 100.0)

# ATP 雷电电流源
solver.add_lightning_IS("lightning", 1, 0, model="heidlerf",
                         peak=30e3, T1=1.2e-6, T2=50e-6)
```

### 8.3 传输线

```python
# Bergeron 无损传输线
solver.add_bergeron_line("line1", node_k=1, node_m=3, Zc=300.0, tau=10e-6)

# ULM 频率相关传输线
solver.add_ulm_line("ulm1", nodes_k=1, nodes_m=3, fitulm_file="cable.fit", length=100.0)

# 通用线路接口
solver.add_line(line_interface_instance)
```

### 8.4 非线性元件

```python
# MOA 避雷器（从 V-I 数据文件）
solver.add_MOA_from_file("moa1", 1, 0, file_path="moa.data",
                          rated_voltage=100e3, voltage_is_pu=True)

# LPM 绝缘子闪络
solver.add_insulator_LPM("lpm1", 1, 2, gap_length=0.5,
                          k=1e-6, E0=600.0, R_arc=1.0, R_open=1e9)
```

### 8.5 UMEC 变压器

```python
data = create_umec_transformer_3ph_bank(...)
solver.add_UMEC_transformer("T1", data)
```

### 8.6 探针

```python
# 电压探针（可在 run 前随时添加）
solver.add_voltage_probe("vp1", node_pos=2, node_neg=0)

# 支路电流探针
solver.add_branch_current_probe("ip1", branch_name="r1")
```

---

## 9. 主循环详解

```
run()
  │
  ├─ validate_circuit()                    ← 拓扑 + 参数校验
  ├─ reset_dynamic_state()                 ← 复位所有元件状态
  ├─ _reset_caches()                       ← 清空 LU 缓存
  │
  ├─ _init_result_store(n_steps)           ← 创建 ResultStore + 别名化
  │
  ├─ [空电路快速路径]
  │
  ├─ compile_transmission_lines()          ← PSCAD 风格并行编译
  ├─ _build_ulm_batch_runtime()            ← ULM batch 运行时
  ├─ _indexer.freeze()                     ← 锁定节点映射
  │
  ├─ _stepper.run(self, n_steps, timing)   ← ★ TimeStepper 主循环
  │     │
  │     └─ for step_idx in range(n_steps):
  │           ├─ solver._run_one_step(step_idx, n_steps, _t)
  │           │     ├─ 1. step_pre_solve(t)          开关事件
  │           │     ├─ 2. _solve_step()              solve + resolve
  │           │     ├─ 3. step_post_solve_V_I(V)     支路 V/I 更新
  │           │     ├─ 4. _record_probes()            探针记录
  │           │     ├─ 5. _update_lines_combined(V)   传输线更新
  │           │     ├─ 6. step_post_solve_history()   支路历史递推
  │           │     └─ 7. _update_transformer_history 变压器历史递推
  │           │
  │     └─ export_model_state_to_lines()   ← ULM batch 状态导出
  │
  ├─ ResultStore.finalize(indexer)         ← 裁截 + voltage_results
  ├─ 同步旧属性（向后兼容）
  │
  └─ print_timing_report()                 ← 计时统计
```

### 9.1 `_solve_step()` 内部

```
_solve_step()
  │
  └─ ResolveManager.solve_with_resolve(solve_fn, check_fn, ...)
       │
       └─ for resolve_round in range(MAX_SEG_ITER=5):
             ├─ solve_fn()                ← _solve_segmented 或 _solve_linear
             │     └─ _build_system_matrix() → _build_MNA_matrix + _build_MNA_rhs
             │     └─ _solve_mna(MNA, rhs) → StampingEngine.solve → SuperLU
             │
             ├─ check_fn(V)               ← post_solve_resolve_check
             │     ├─ LPM flashover?
             │     ├─ UMEC saturation?
             │     ├─ MOA segment change?
             │     └─ MultiPortDevice.check_rebuild_required?
             │
             ├─ no change → return V
             └─ changed  → mark_topology_changed → rebuild G → loop
```

---

## 10. 依赖关系

```
emtp/solver.py  (canonical EMTPSolver, 3525 行)
  ├── emtp/circuit.py                   CircuitModel 数据容器
  ├── emtp/nodes.py                     NodeBook, NodeIndexer
  ├── emtp/types.py                     Branch, VoltageSource, CurrentSource, RHSPlan, ...
  ├── emtp/stamping.py                  COOStamper, StampingEngine
  ├── emtp/sparse_solver.py             SparseLinearSolver (SuperLU)
  ├── emtp/validation.py                电路校验
  │
  ├── emtp/runtime/__init__.py          DynamicDeviceRuntime
  ├── emtp/runtime/resolve.py           ResolveManager, ResolveEvent
  ├── emtp/runtime/stepper.py           TimeStepper
  │
  ├── emtp/results/__init__.py          结果 helper 函数 (scale_values, ...)
  ├── emtp/results/store.py             ResultStore
  │
  ├── emtp/devices/base.py              Device Protocol
  ├── emtp/devices/multiport.py         MultiPortDevice Protocol
  ├── emtp/devices/{resistor,inductor,capacitor,switch,series_rl,nonlinear,lpm}.py
  │
  ├── emtp/lines/bergeron.py            BergeronLineDevice adapter
  ├── emtp/lines/ulm.py                 ULMLineDevice adapter
  ├── emtp/transformers/umec.py         UMECTransformerDevice adapter
  │
  ├── emtp/assembly/mna.py              MNAAssembler skeleton
  │
  ├── numpy                             (数值计算)
  ├── scipy.sparse / scipy.sparse.linalg (CSC 稀疏矩阵 + SuperLU)
  │
  ├── [可选] transmission_line_emtp_v2.py              Bergeron 底层模型
  ├── [可选] ulm_transmission_line_PARA.py             ULM 底层模型 + batch
  ├── [可选] umec_transformer.py                       UMEC 底层模型
  ├── [可选] nonlinear_models_pscad.py                 MOA + LPM
  └── [可选] atp_lightning_current_generator_simplified.py  雷电源

emtp_solver_v3.py  (144 行 legacy compat shim)
  └── from emtp.solver import EMTPSolver
      └── 重导出所有 emtp.* 符号以便旧导入路径兼容
```

---

## 11. 导入约定

### 11.1 推荐导入（新代码）

```python
from emtp import EMTPSolver
from emtp.nodes import NodeBook, NodeIndexer
from emtp.types import Branch, VoltageSource, ElementType
from emtp.devices import Device, MultiPortDevice
from emtp.lines import BergeronLineDevice, ULMLineDevice
from emtp.transformers import UMECTransformerDevice
```

### 11.2 兼容导入（旧代码继续工作）

```python
from emtp_solver_v3 import EMTPSolver          # 返回同一个类
from emtp_solver_v3 import NodeBook, NodeIndexer
```

### 11.3 Identity 保证

```python
from emtp import EMTPSolver as A
from emtp.solver import EMTPSolver as B
from emtp_solver_v3 import EMTPSolver as C
assert A is B is C   # ← 永远为 True
```

### 11.4 包结构保证

```python
import emtp.runtime
# emtp.runtime.__file__ → .../emtp/runtime/__init__.py  (package, not single file)

import emtp.results
# emtp.results.__file__ → .../emtp/results/__init__.py  (package, not single file)
```

`emtp/runtime.py` 和 `emtp/results.py` 作为单文件模块的旧形式**已永久移除**。

---

## 12. 配置选项

```python
solver = EMTPSolver(
    # 时间参数
    dt=1e-6,                      # 时间步长 (s)
    finish_time=100e-6,           # 仿真结束时间 (s)

    # 输出控制
    verbose=True,                 # 打印计时和统计
    record_all_node_voltages=True,# 记录全部节点电压（大型网络建议 False）
    record_branch_history=False,  # 记录支路 V/I 历史
    record_source_history=False,  # 记录电压源电流历史
    record_line_history=False,    # 记录传输线端口历史

    # 性能优化
    pre_sample_sources=False,     # 预采样独立源（减少每步函数调用）
    use_rhs_plan=False,           # 预编译 RHS 拓扑（极速路径）
    line_compile_workers=None,    # 传输线并行编译线程数
    compile_lines_on_add=False,   # 添加时立即编译线路

    # ULM batch
    ulm_batch_mode="auto",        # "auto" | "parallel" | "serial" | "off"
    ulm_batch_parallel_threshold_factor=2,

    # 稳定性
    allow_singular_regularization=False,  # 奇异矩阵自动正则化
    max_result_memory_mb=None,            # 结果内存上限告警

    # Multiport dispatch (experimental)
    use_multiport_lines=False,            # Bergeron/ULM 通过 MultiPortDevice
    use_multiport_transformers=False,     # UMEC 通过 MultiPortDevice
)
```

---

## 13. 求解器统计信息

`run()` 结束后可用：

```python
solver.print_timing_report()        # 计时分解（init/switch/solve/branch/line/...）
solver.print_solver_statistics()    # G 重建次数、缓存命中率、重解次数

solver._stats  # {
#   'total_steps': 101,
#   'segment_switches': 0,
#   'segment_resolves': 0,
#   'max_seg_iter': 0,
#   'lpm_resolves': 0,
#   'lpm_flashovers': 0,
#   'lpm_extinctions': 0,
#   'transformer_saturation_resolves': 0,
#   'transformer_saturation_switches': 0,
#   'G_rebuilds': 1,
#   'G_cache_hits': 100,
# }
```

---

## 14. 测试

### 14.1 测试矩阵（159 个）

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|---------|
| `test_solver_regression.py` | ~38 | API 回归：getter、probe、validate、pre_sample、rhs_plan |
| `test_p5_basic_physics.py` | 4 | RC/RL/SRL 解析解验证 |
| `test_trapezoidal_rlc.py` | 3 | 梯形法数值验证 |
| `test_basic_mna.py` | 2 | MNA 基本求解 |
| `test_switches.py` | 1 | 定时开关 |
| `test_nodes.py` | 2 | NodeBook/NodeIndexer |
| `test_moa_segments.py` | 1 | MOA 段切换 |
| `test_lpm_flashover.py` | 2 | LPM 闪络 |
| `test_p5_lpm_validation.py` | 4 | LPM 物理验证 |
| `test_p5_moa_validation.py` | 4 | MOA 物理验证 |
| `test_bergeron_line.py` | 2 | Bergeron 线路 smoke |
| `test_p5_bergeron_reflection.py` | 3 | Bergeron 反射/开路/短路 |
| `test_ulm_smoke.py` | 2 | ULM 线路 smoke |
| `test_p5_ulm_validation.py` | 3 | ULM 物理验证 |
| `test_umec_transformer.py` | 1 | UMEC Norton 等效 |
| `test_p5_umec_validation.py` | 4 | UMEC 物理验证 |
| `test_p5_tower_validation.py` | 3 | 杆塔（skip，缺数据） |
| `test_multiport_contract.py` | 7 | MultiPortDevice 协议 |
| `test_bergeron_adapter.py` | 12 | Bergeron adapter 等价性 |
| `test_ulm_umec_adapters.py` | 12 | ULM/UMEC adapter smoke |
| `test_multiport_registry.py` | 6 | MultiPort 注册表 skeleton |
| `test_result_store.py` | 11 | ResultStore 独立测试 |
| `test_import_canonical_paths.py` | 8 | import 路径规范 |
| `test_circuit_model.py` | 5 | CircuitModel 容器 |
| `test_mna_assembler.py` | 4 | MNAAssembler 装配 |

### 14.2 运行测试

```bash
# 全部测试（排除缺外部依赖的 test_tower_case_p1.py）
pytest tests/ -q --ignore=tests/test_tower_case_p1.py

# 特定模块
pytest tests/test_multiport_contract.py -v
pytest tests/test_p5_basic_physics.py -v
```

---

## 15. 架构迁移历程

| 版本 | Commit | 关键变更 |
|------|--------|---------|
| v0.1 | `75f307e` | P3/P4/P5 模块化：Device 协议、emtp 包、物理验证 |
| v0.2.0 | `d439b80` | Solver 迁移：emtp/solver.py canonical、去重、MultiPortDevice、ResolveManager、ResultStore（131 tests） |
| v0.2.1 | `f42404b` | PR-10~17：ResultStore 接入、Multiport registry、Bergeron/ULM/UMEC adapter 注册、ResolveEvent、MNAAssembler（154 tests） |
| v0.2.2 | `cf8b7dc` | PR-18~19：TimeStepper 主循环、CircuitModel 容器（159 tests） |

### 15.1 从原型到可维护架构

```
原型阶段:
  emtp_solver_v3.py (4580 行单体) + 散落外部模块 + 无统一接口

第一阶段 (v0.2.0):
  emtp/solver.py canonical + emtp_solver_v3.py shim
  Device Protocol + 7 实现
  MultiPortDevice Protocol + 3 adapter
  DynamicDeviceRuntime 去重
  ResolveManager / ResultStore

第二阶段 (v0.2.1～v0.2.2):
  ResultStore 真实接入 solver
  MultiPortDevice 注册表 skeleton
  Bergeron/ULM/UMEC adapter 验证等价性
  ResolveEvent 事件体系
  TimeStepper 主循环
  CircuitModel 数据容器
  MNAAssembler 骨架
```

### 15.2 下一步路线

- [ ] MultiPortDevice 正式切换：`use_multiport_lines=True` 下删除旧 Bergeron/ULM/UMEC 特判路径
- [ ] MNAAssembler 深度集成：接管 `_build_MNA_matrix` / `_build_MNA_rhs`
- [ ] CircuitModel 深度集成：`add_*` 方法委托到 CircuitModel
- [ ] 统一 ResolveEvent 替换布尔 check_fn
- [ ] ULMBatchMultiPortDevice batch adapter

---

## 16. 快速开始

```python
from emtp import EMTPSolver

# 1. 创建求解器
solver = EMTPSolver(dt=1e-6, finish_time=100e-6)

# 2. 搭建电路
solver.add_VS("vs", 1, 0, lambda t: 100.0)       # 100V 电压源
solver.add_R("r1", 1, 2, 10.0)                    # 10Ω 电阻
solver.add_L("l1", 2, 0, 1e-3)                    # 1mH 电感

# 3. 加探针
solver.add_voltage_probe("v_l1", 2, 0)
solver.add_branch_current_probe("i_r1", "r1")

# 4. 运行
solver.run()

# 5. 读取结果
t = solver.get_time("us")
v_load = solver.get_voltage_probe("v_l1", "V")
i_load = solver.get_branch_current_probe("i_r1", "A")

print(f"V_load max: {v_load.max():.2f}V")
print(f"I_load max: {i_load.max():.2f}A")

# 6. 统计
solver.print_timing_report()
solver.print_solver_statistics()
```
