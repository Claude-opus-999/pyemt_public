# EMTP 电磁暂态求解器 — 架构文档

## 1. 项目概述

`emtp_v0.2` 是一个基于 Python 的电磁暂态（EMTP）仿真求解器，使用修正节点分析法（MNA）进行电路求解，集成了多相传输线、非线性元件、UMEC 变压器和绝缘子闪络模型。

### 1.1 文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `emtp_solver_v3.py` | 176 KB / 4580 行 | **主求解器**（核心） |
| `emtp_components_series_rl_only.py` | 4 KB / 129 行 | 基础数据结构（Branch, ElementType, CurrentSource） |
| `transmission_line_emtp_v2.py` | 10 KB | Bergeron 传输线模型 |
| `ulm_transmission_line_PARA.py` | 96 KB | ULM 通用线路模型（含并行 batch） |
| `umec_transformer.py` | 22 KB | UMEC 多端口变压器模型 |
| `nonlinear_models_pscad.py` | 28 KB | PSCAD 风格分段非线性（MOA 避雷器）、LPM 绝缘子闪络 |
| `atp_lightning_current_generator_simplified.py` | 34 KB | ATP 兼容雷电电流源（Heidler/双指数） |
| `emtp_plotting.py` | 4 KB | 辅助绘图 |
| `tests/test_solver_regression.py` | 657 行 / 56 个测试 | 回归 + 单元测试 |

---

## 2. 分层架构

求解器采用 **四层引擎架构**，从底层数据结构到顶层门面逐步组合：

```
┌──────────────────────────────────────────────────────────────┐
│                    EMTPSolver (门面层)                        │
│  ~2900 行：add_* / validate / run / get_* / probes / print   │
│                                                              │
│  组合三个引擎 + 探针管理 + 校验 + 结果 API                      │
└──────┬──────────────────┬──────────────────┬─────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐
│StampingEngine│  │SparseLinear   │  │DynamicDeviceRuntime  │
│              │  │   Solver      │  │                      │
│ G 矩阵装配    │  │               │  │ 每步状态管理           │
│ RHS 缓冲复用  │  │ LU 分解缓存    │  │ · 开关定时事件        │
│ COO 生命周期  │  │ 奇异正则化     │  │ · 支路 V/I 更新       │
│              │  │               │  │ · 历史源递推           │
└──────┬───────┘  └───────────────┘  │ · 非线性/LPM/UMEC     │
       │                             │   收敛检测             │
       │                             └──────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│          7 个 Device 实现类              │
│                                        │
│  ResistorDevice    InductorDevice       │
│  CapacitorDevice   SwitchDevice         │
│  SeriesRLDevice    NonlinearResistorDevice │
│  LPMFlashoverDevice                    │
│                                        │
│  每个类封装自身物理：                      │
│  stamp_G / stamp_rhs /                  │
│  update_branch_quantities /             │
│  update_history / reset_state           │
└────────────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  NodeIndexer    │
        │                 │
        │ ext ↔ compact   │
        │ GND → -1        │
        │ freeze()        │
        └────────────────┘
```

---

## 3. 第零层：基础数据结构

### 3.1 NodeIndexer（L184-252）

**职责**：将任意外部整数节点 ID 映射到紧凑的 [0, n) 索引区间。

```python
idx = NodeIndexer()
idx.register(1)     # → 0
idx.register(5)     # → 1
idx.register(9999)  # → 2
idx.n               # → 3 (矩阵维度，不是 9999)
idx.freeze()        # 锁定，防止运行时新增节点
```

关键设计：
- 外部节点 0（GND）始终映射到哨兵值 `COMPACT_GND = -1`
- `register()` 幂等：重复注册同一节点不产生副作用
- `freeze()` 后注册新节点抛出 `RuntimeError`
- 在主循环前调用 `freeze()`，之后所有 MNA 维度使用 `idx.n`

### 3.2 NodeBook（L254-377）

字符串节点名 → 整数 ID 的自动分配器。支持 `reserve()` 手动绑定和 `alias()` 别名。

### 3.3 VoltageSource（L387-408）

理想电压源，带有 `node_pos`/`node_neg` 端点、`voltage_at(t)` 波形函数和 `current` 状态。

### 3.4 ElementType / Branch / CurrentSource（emtp_components_series_rl_only.py）

基础枚举和 dataclass：
- `ElementType` 枚举：RESISTOR, INDUCTOR, CAPACITOR, SWITCH, SERIES_RL, NONLINEAR_RESISTOR, 等
- `Branch`：二端支路 dataclass，含 `node_from/node_to`, `Geq/Ihist`（Norton 等效），`voltage/current` 状态
- `CurrentSource`：独立电流源，含 `current_at(t)` 方法

---

## 4. 第一层：设备抽象（Device Protocol）

### 4.1 Device Protocol（L479-517）

所有支路元件必须实现的接口：

| 方法 | 职责 | 调用时机 |
|---|---|---|
| `stamp_G(stamper, indexer)` | 将 Norton 电导向 COO 累加器写入 | MNA 矩阵装配 |
| `stamp_rhs(rhs, indexer, t)` | 将历史源电流写入 RHS 向量 | 每步 RHS 构建 |
| `update_branch_quantities(V, indexer)` | 从 MNA 解向量 V 计算支路电压/电流 | 求解后、探针前 |
| `update_history(dt)` | 递推历史源（梯形法则） | 探针后、每步末尾 |
| `reset_state()` | 清除所有动态状态 | run() 开始前 |
| `is_dynamic` (property) | 是否贡献历史项 | RHSPlan 编译 |
| `element_kind` (property) | 元件类型标签 | 诊断/统计 |

使用 `Protocol` + `@runtime_checkable` 而非 ABC，零运行时开销。

### 4.2 COOStamper（L519-544）

COO 三元组累加器，将稀疏矩阵装配与格式转换隔离：

```python
stamper = COOStamper(N)     # N = n_compact + m_vs
stamper.add(row, col, val)  # 累加一个贡献
A_csc = stamper.tocsc()     # 转为 CSC 供 SuperLU
```

### 4.3 七种 Device 实现

| 类 | element_kind | is_dynamic | 物理特点 |
|---|---|---|---|
| `ResistorDevice` (L574) | `R` | False | 纯电阻：Geq = 1/R，无历史项 |
| `InductorDevice` (L628) | `L` | True | 隐式梯形：Geq = Δt/(2L)，Ihistₖ₊₁ = Ihistₖ + 2·Geq·vₖ |
| `CapacitorDevice` (L697) | `C` | True | 隐式梯形：Geq = 2C/Δt，Ihistₖ₊₁ = -Ihistₖ - 2·Geq·vₖ |
| `SwitchDevice` (L766) | `SW` | False | 定时开关：G = 1/R_open 或 1/R_closed，含 `update_timed_state(t)` |
| `SeriesRLDevice` (L862) | `SRL` | True | 串联 RL（无内部节点）：Geq = G_L/(1+R·G_L) |
| `NonlinearResistorDevice` (L934) | `NR` | True | PSCAD 分段 MOA：Geq/Ihist 由 seg_helper 外部管理 |
| `LPMFlashoverDevice` (L1006) | `LPM` | False | CIGRE 先导发展法闪络开关：状态由 LPM 模型驱动 |

每个 Device 类在构造时创建对应的 `Branch` 对象（向后兼容旧 API），暴露 `_branch` 属性供 solver 访问。

---

## 5. 第二层：引擎模块

### 5.1 StampingEngine（L1083-1190）

**职责**：MNA 稀疏矩阵装配 + RHS 构建 + 稀疏求解委托。

#### G 矩阵装配（开放/闭合模式）

```python
eng = StampingEngine(indexer, allow_singular_regularization=True)

# 开放 → solver 插入线/变压器贡献 → 闭合
stamper = eng.begin_G(n_compact=3, n_vs=1)          # COOStamper(4)
eng.stamp_devices_G(stamper, devices)                # 支路电导
# [solver 在此插入 line.G_eq 和 transformer.G_tf 贡献]
eng.stamp_vs_G(stamper, vs_list)                     # B/C 分块
A_csc = eng.finish_G(stamper)                        # → CSC + 缓存 + bump matrix_id
```

#### 矩阵缓存

| 属性 | 说明 |
|---|---|
| `G_dirty` | 拓扑变更时置 True，触发重装配 |
| `cached_MNA` | 最近一次装配的 CSC 矩阵 |
| `matrix_id` | 每次装配递增，供 `SparseLinearSolver` 做缓存键 |

#### 稀疏求解

```python
V = eng.solve(MNA, rhs, vs_list)
# → 内部委托 SparseLinearSolver.solve(A, b, matrix_id, n)
# → V = x[:n]，VS 电流回写到 vs.current
```

#### mark_dirty

`mark_dirty()` 清除 G 缓存和 LU 缓存；在 `finish_G()` 中自动清除 LU（因 matrix_id 变化）。

### 5.2 SparseLinearSolver（L1192-1245）

**职责**：纯线性代数层，独立于 MNA/Device 概念。

```python
solver = SparseLinearSolver(allow_singular_regularization=False)
x = solver.solve(A, b, matrix_id=0, n_compact=3)
# 首次调用：LU 分解 + 缓存
# 同一 matrix_id：复用 LU 缓存，仅做回代
# 不同 matrix_id：重分解
solver.invalidate()  # 强制清除 LU 缓存
```

关键行为：
- 矩阵奇异 + `allow_reg=False` → `RuntimeError`（含诊断信息）
- 矩阵奇异 + `allow_reg=True` → 在节点电压块添加 `1e-12` 正则项后重试
- LU 分解使用 `scipy.sparse.linalg.splu`（SuperLU 后端）

### 5.3 DynamicDeviceRuntime（L1247-1396）

**职责**：封装每时间步的状态管理操作。

#### step_pre_solve(t, devices, lpm_names) → bool

遍历 `SwitchDevice` 实例检查定时事件。跳过 LPM 控制的开关（LPM 由物理模型驱动，不由定时器驱动）。

```python
if self._runtime.step_pre_solve(t, devices, lpm_names):
    self.mark_topology_changed("switch event")
```

#### step_post_solve_V_I(V, devices, indexer, step_idx, n_steps, ...)

更新支路电压/电流。**必须在探针记录之前调用**，否则探针将读到错误的值。

```python
self._runtime.step_post_solve_V_I(V, devices, indexer, step_idx, n_steps, ...)
# 此后立即记录探针
self._record_probes(step_idx, V)
```

#### step_post_solve_history(devices)

递推 L/C/SRL 历史源（梯形法则）。**必须在探针记录之后调用**，因为 `CapacitorDevice.update_history` 会翻转 `Ihist` 符号。

#### post_solve_resolve_check(V, t, ...) → bool

**统一的三合一收敛检测**，替代旧的三层嵌套（segmented → LPM → UMEC）：

1. **LPM 闪络检测**：检查所有 LPM 绝缘子的先导长度是否超过间隙 → 闭合开关
2. **UMEC 饱和检测**：检查变压器铁芯是否切换饱和段 → 更新电导矩阵
3. **分段非线性检测**：检查 MOA 避雷器是否越过 v-i 分段点 → 更新 Geq/Ihist

返回 `True` 表示电路拓扑或参数改变，需要重新求解。调用者将此放入统一循环：

```python
for resolve_round in range(MAX_ITER):
    V = self._solve_segmented()  # 或 _solve_linear
    if not self._runtime.post_solve_resolve_check(V, t, ...):
        break  # 收敛
else:
    logger.warning("未收敛")
```

---

## 6. 第三层：EMTPSolver 门面

### 6.1 构造与组合（L1399-1640）

```python
solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=True)
```

`__init__` 按顺序初始化：
1. **参数存储**：`dt`, `finish_time`, `verbose`, `record_*` 开关
2. **元件容器**：`branches`, `current_sources`, `voltage_sources`, `transmission_lines`, `transformers`, `lines`
3. **节点管理**：`num_nodes`, `_node_set`, `_vs_node_set`
4. **核心对象**：
   - `self.nodes = NodeBook(start=1)` — 命名节点管理
   - `self._indexer = NodeIndexer()` — compact 索引
   - `self._runtime = DynamicDeviceRuntime(self.dt)` — 运行时
   - `self._stamping = StampingEngine(self._indexer, ...)` — 矩阵装配 + 求解
5. **探针**：`voltage_probes`, `branch_current_probes`
6. **非线性/LPM/统计/计时/ULM batch**

### 6.2 公共 API：添加元件

| 方法 | 创建的 Device | 说明 |
|---|---|---|
| `add_R(name, nf, nt, R)` | `ResistorDevice` | 电阻 |
| `add_L(name, nf, nt, L, Rp)` | `InductorDevice` | 电感（隐式梯形） |
| `add_C(name, nf, nt, C, Rp)` | `CapacitorDevice` | 电容（隐式梯形） |
| `add_SW(name, nf, nt, ...)` | `SwitchDevice` | 定时开关 |
| `add_switch(...)` | → `add_SW` | 别名 |
| `add_series_RL(name, nf, nt, R, L)` | `SeriesRLDevice` | 串联 RL |
| `add_IS(name, nf, nt, func)` | `CurrentSource` | 独立电流源 |
| `add_VS(name, nf, nt, func)` | `VoltageSource` | 理想电压源 |
| `add_lightning_IS(...)` | `CurrentSource` + 雷电波形 | ATP 雷电源 |
| `add_MOA_from_file(...)` | `NonlinearResistorDevice` | 分段 MOA |
| `add_insulator_LPM(...)` | `LPMFlashoverDevice` | 绝缘子闪络 |
| `add_bergeron_line(...)` | `TransmissionLineInterface` | Bergeron 传输线 |
| `add_ulm_line(...)` | `ULMLine` | ULM 通用线路 |
| `add_UMEC_transformer(...)` | `UMECTransformer` | UMEC 变压器 |

每个 `add_*` 方法按统一模式操作：
```python
def add_R(self, name, node_from, node_to, R):
    self._ensure_unique_device_name(name, "resistor")
    node_from = self._resolve_node(node_from)
    node_to = self._resolve_node(node_to)
    dev = ResistorDevice(name, node_from, node_to, R)
    self.branches[name] = dev._branch       # 向后兼容
    self._update_node_count(node_from, node_to)  # 更新 NodeIndexer
    self._devices.append(dev)               # 加入 Device 列表
    self.mark_topology_changed(f"add resistor: {name}")
```

### 6.3 探针 API

| 方法 | 说明 |
|---|---|
| `add_voltage_probe(name, node_pos, node_neg)` | 注册电压差探针 |
| `add_branch_current_probe(name, branch_name)` | 注册支路电流探针 |
| `get_probe(name, unit)` | 统一读取探针波形 |
| `list_probes()` | 列出已注册探针 |

### 6.4 结果 API

| 方法 | 返回 | 说明 |
|---|---|---|
| `get_time(unit)` | `ndarray` | 时间序列 |
| `get_node_voltage(node, unit)` | `ndarray` | 节点电压（支持 GND=0） |
| `get_branch_current(name, unit)` | `ndarray` | 支路电流 |
| `get_branch_voltage(name, unit)` | `ndarray` | 支路电压 |
| `get_source_current(name)` | `ndarray` | 电流源输出 |
| `get_vs_current(name, unit)` | `ndarray` | 电压源电流 |
| `get_vs_voltage(name, unit)` | `ndarray` | 电压源电压 |
| `get_solver_statistics()` | `dict` | 求解统计（G 重建次数、分段切换等） |
| `get_timing_report()` | `dict` | 各阶段计时 |
| `print_solver_statistics()` | — | 打印统计 |
| `print_timing_report()` | — | 打印计时 |
| `print_circuit_summary()` | — | 打印电路摘要 |

### 6.5 主循环（run() 方法，L3943-4170）

```python
def run(self):
    # 0. 校验 + 重置
    report = self.validate_circuit()
    self.reset_dynamic_state()
    self._reset_caches()

    # 1. 预分配输出数组（compact 维度）
    n_steps = int(round(finish_time / dt)) + 1
    self._voltage_buf = np.zeros((self._indexer.n, n_steps))

    # 2. 预编译传输线注入映射 + ULM batch
    self._build_ulm_batch_runtime()

    # 3. 锁定节点索引
    self._indexer.freeze()
    self._compact_n = self._indexer.n

    # 4. 时间步循环
    for step_idx in range(n_steps):
        t = step_idx * self.dt

        # 4a. 开关定时事件
        if self._runtime.step_pre_solve(t, self._devices, lpm_set):
            self.mark_topology_changed()

        # 4b. 统一求解 + 非线性/LPM/UMEC 收敛循环
        V = self._solve_step()

        # 4c. 支路 V/I 更新（必须在探针之前）
        self._runtime.step_post_solve_V_I(V, ...)

        # 4d. 探针 + 电压记录
        self._record_probes(step_idx, V)

        # 4e. 传输线更新
        self._update_lines_combined(V)

        # 4f. 支路历史源递推（必须在探针之后）
        self._runtime.step_post_solve_history(self._devices)

        # 4g. 变压器历史源递推
        self._update_transformer_history(V)

    # 5. 后处理：截断 + voltage_results 字典（外部 ID 键）
    self.voltage_results = {
        self._indexer.to_external(c): self._voltage_buf[c, :]
        for c in range(self._indexer.n)
    }
```

### 6.6 _solve_step 统一收敛循环（L3320-3360）

```python
def _solve_step(self):
    for resolve_round in range(self._MAX_SEG_ITER):
        V = self._solve_segmented()   # 含 nonlinear 内循环
              # 或 self._solve_linear()  # 无非线性时

        if not self._runtime.post_solve_resolve_check(
            V, t, lpm_elements, ..., transformers, ..., seg_helper, ...
        ):
            break  # 三个触发器均未激活 → 收敛

    return V
```

三段触发器统一处理：
1. `_solve_segmented` 内循环处理 PSCAD 分段非线性
2. `post_solve_resolve_check` 同时检查 LPM 闪络 + UMEC 饱和 + 分段边界

### 6.7 校验（validate_circuit）

`validate_circuit(strict=True)` 执行多层检查：
1. **参数校验**：dt > 0, finish_time >= 0
2. **探针校验**：引用的节点/支路是否存在
3. **空电路**：无支路 + 无源 → 跳过 MNA
4. **节点校验**：非地节点数 > 0
5. **内存警告**：估计结果缓存大小
6. **稀疏节点 ID 警告**：max(ext_id) > 10 × unique_nodes
7. **支路校验**：R/L/C 正值、电压源自环、浮空电流源检测（并查集连通性）

返回 `ValidationReport`，含 `errors()` / `warnings()` / `has_errors` / `has_warnings`。

---

## 7. 数据流全景

```
用户 API 调用
    │
    ├─ add_R/L/C/SW/...  ──→ 创建 Device ──→ branches[name] = dev._branch
    │                                       ──→ _devices.append(dev)
    │                                       ──→ _indexer.register(node_id)
    │
    └─ run()
         │
         ├─ validate_circuit()    检查电路完整性
         ├─ reset_dynamic_state()  for dev: dev.reset_state()
         ├─ _indexer.freeze()     锁定节点映射
         │
         └─ 主循环 [step_idx]
              │
              ├─ step_pre_solve(t)          开关定时事件 → mark_dirty?
              │
              ├─ _solve_step()              ←── 收敛循环 ──┐
              │   ├─ _build_MNA_matrix()      G 矩阵装配     │
              │   │   ├─ begin_G              COOStamper     │
              │   │   ├─ stamp_devices_G      dev→stamper   │
              │   │   ├─ stamp_lines_G        line→stamper  │
              │   │   ├─ stamp_xfmrs_G        xfmr→stamper  │
              │   │   ├─ stamp_vs_G           vs→stamper    │
              │   │   └─ finish_G             → CSC + 缓存   │
              │   ├─ _build_MNA_rhs()          RHS 向量     │
              │   ├─ _solve_mna(MNA, rhs)     SuperLU 求解  │
              │   └─ post_solve_resolve_check │ LPM+UMEC+NL │
              │                                └── 需要重解? ─┘
              │
              ├─ step_post_solve_V_I(V)      支路 V/I 更新
              ├─ _record_probes(V)           探针记录
              ├─ _update_lines_combined(V)   传输线状态更新
              ├─ step_post_solve_history()   历史源递推 (L/C/SRL)
              └─ _update_transformer_history(V)  变压器历史
```

---

## 8. 测试覆盖

**文件**：`tests/test_solver_regression.py`（657 行，56 个测试用例）

### 8.1 测试分类

| 测试类 | 用例数 | 覆盖内容 |
|---|---|---|
| `SolverRegressionTests` | 40 | 完整 solver 回归：基本元件、开关、探针、预采样、RHSPlan、校验、记忆估算 |
| `NodeIndexerTests` | 10 | compact 映射：注册、冻结、幂等、GND、KeyError、solver 集成 |
| `SparseLinearSolverTests` | 6 | LU 缓存、奇异检测、正则化、invalidate |

### 8.2 关键回归用例

| 用例 | 验证内容 |
|---|---|
| `test_run_uses_integer_steps_and_includes_finish_time` | 步数计算 + 时间终点 |
| `test_capacitor_probe_includes_parallel_damping_current` | 电容并联阻尼电流（关键：Ihist 时序） |
| `test_run_resets_dynamic_state` | 重复 run() 一致性 |
| `test_timed_switch_events_are_consumed_once` | 开关事件 + G_rebuilds 统计 |
| `test_rhs_plan_matches_legacy_rhs` | RHSPlan 快路径与普通路径等价 |
| `test_large_integer_node_warning` | 稀疏 ID 警告 |
| `test_validate_circuit_strict_mode_raises_on_errors` | 严格校验 |

---

## 9. 设计决策与约束

### 9.1 Compact Node Index

外部节点 ID（如 9999）不再直接作为 MNA 矩阵下标。`NodeIndexer` 将外部 ID 映射到 [0, n) 紧凑区间。对于雷电仿真场景（节点 ID 可能到 10000+ 但实际独立节点 ~50），矩阵维度从 ~10000×10000 降至 ~50×50。

### 9.2 Device Protocol 替代 ElementType 分发

旧代码在 6 个位置维护 `if et == ElementType.X` 分支。新架构中每个 Device 类封装自己的物理逻辑，消除所有分支点。

### 9.3 探针记录时序（关键约束）

每步操作顺序不能颠倒：
```
step_post_solve_V_I → _record_probes → step_post_solve_history
```

因为 `CapacitorDevice.update_history()` 翻转 Ihist 符号（`Ihist = -Ihist - 2Geq·v`），探针必须在翻转前读取当前步值。

### 9.4 三合一收敛循环

旧的三层嵌套（segmented 内循环 → LPM 触发 → UMEC 触发）统一为一个 `post_solve_resolve_check` 调用，在统一的外层循环中处理所有三种收敛触发器。

### 9.5 传输线和变压器

传输线和变压器暂未迁移到 Device 协议。它们在 `_build_MNA_matrix` 和主循环中保持独立的 stamping/更新路径。这是故意的设计选择——它们使用多端口模型且需要 solver 专用的数据结构（ULM batch、端口节点映射）。

### 9.6 向后兼容

- `self.branches[name]` 字典继续维护，每个 Device 通过 `_branch` 属性暴露 Branch 对象
- `voltage_results` 字典键使用外部节点 ID（通过 `to_external()` 反查）
- `self.num_nodes` 保留为外部 ID 上界（仅用于诊断和打印）

---

## 10. 依赖关系

```
emtp_solver_v3.py
  ├── emtp_components_series_rl_only.py  (Branch, ElementType, CurrentSource, LineData)
  ├── numpy                               (数值计算)
  ├── scipy.sparse / scipy.sparse.linalg  (稀疏矩阵 + SuperLU)
  ├── [可选] transmission_line_emtp_v2.py     (Bergeron 线)
  ├── [可选] ulm_transmission_line_PARA.py     (ULM 通用线 + batch)
  ├── [可选] umec_transformer.py               (UMEC 变压器)
  ├── [可选] nonlinear_models_pscad.py          (MOA + LPM)
  └── [可选] atp_lightning_current_generator_simplified.py  (雷电源)
```

---

## 11. 性能特征

| 优化点 | 实现方式 |
|---|---|
| MNA 矩阵缓存 | `StampingEngine` 按脏位复用 CSC 矩阵，仅在拓扑变化时重装配 |
| LU 分解缓存 | `SparseLinearSolver` 按 `matrix_id` 复用分解结果 |
| RHS 缓冲区 | `_rhs_buf` 每步 fill(0) 复用，避免 np.zeros 分配 |
| RHSPlan 快路径 | 预编译平面索引数组，避免每步 Python 对象遍历 |
| ULM batch | 多条 ULM 线路的 Numba 并行核（`ulm_transmission_line_PARA.py`） |
| 预采样 | `pre_sample_sources=True` 时在仿真前预计算源波形数组 |
| 电压/探针缓冲 | 预分配 `(compact_n, n_steps)` 矩阵，每步列写入 |
