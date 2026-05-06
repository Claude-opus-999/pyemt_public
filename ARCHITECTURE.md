# PyEMTP 架构文档

版本 `v0.4.0` · 2026-05-06 · **445 passed, 3 skipped**

---

## 目录

1. [整体分层架构](#整体分层架构)
2. [根目录文件清单](#根目录文件清单)
3. [Layer 0: 外部物理库](#layer-0-外部物理库)
4. [Layer 1: 核心求解器 (emtp/)](#layer-1-核心求解器-emtp)
5. [Layer 2: 模块化子包](#layer-2-模块化子包)
6. [Layer 3: 高层管线](#layer-3-高层管线)
7. [Layer 4: LCP 线路常数计算](#layer-4-lcp-线路常数计算)
8. [数据流](#数据流)
9. [测试体系](#测试体系)
10. [版本历程](#版本历程)
11. [已知技术债](#已知技术债)

---

## 整体分层架构

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: 线路常数计算 (LCP)           ← v0.3.2 新增      │
│  LCP/  (物理引擎)  +  pylcp/  (Python 包装层)             │
├──────────────────────────────────────────────────────────┤
│  Layer 3: 高层管线 (v0.3 新增)                             │
│  config/ builders/ snapshot/ export/                       │
│  case_runner.py result_bundle.py result_db.py run_id.py    │
├──────────────────────────────────────────────────────────┤
│  Layer 2: 模块化子包                                       │
│  devices/ assembly/ runtime/ results/                      │
│  lines/ (fitulm_resolver, bergeron, ulm)                   │
│  transformers/ sources/ nonlinear/                         │
├──────────────────────────────────────────────────────────┤
│  Layer 1: 核心求解器 (emtp/)                                │
│  solver.py types.py nodes.py circuit.py                    │
│  sparse_solver.py stamping.py validation.py                │
├──────────────────────────────────────────────────────────┤
│  Layer 0: 外部物理库 (顶层 .py 文件)                        │
│  transmission_line_emtp_v2.py   ulm_transmission_line_PARA.py │
│  nonlinear_models_pscad.py      umec_transformer.py          │
│  atp_lightning_current_generator_simplified.py               │
└──────────────────────────────────────────────────────────┘
```

**依赖方向**: Layer 4 → Layer 3 → Layer 2 → Layer 1 → Layer 0（单向，上层依赖下层，绝不反向）

---

## 根目录文件清单

```
emtp_v0.2/
├── README.md                                     # 项目文档
├── CLAUDE.md                                     # Claude Code 行为指南
├── ARCHITECTURE.md                               # 架构文档（本文件）
├── API_MIGRATION.md                              # 旧→新 API 迁移说明
├── DIRECTION_CONVENTIONS.md                      # 符号/单位/stamping 约定
├── .gitignore
│
├── atp_lightning_current_generator_simplified.py  # Layer 0: 雷电电流源 (1058 行)
├── transmission_line_emtp_v2.py                   # Layer 0: Bergeron 传输线 (381 行)
├── ulm_transmission_line_PARA.py                  # Layer 0: ULM 频变传输线 (2540 行)
├── nonlinear_models_pscad.py                      # Layer 0: MOA/LPM 非线性 (765 行)
├── umec_transformer.py                            # Layer 0: UMEC 变压器 (766 行)
│
├── LCP/                                           # Layer 4: 线路常数物理引擎 (12 .py)
├── pylcp/                                         # Layer 4: LCP Python 包装层 (10 .py)
├── emtp/                                          # 主求解器包 (63 .py) ← v0.4.0 新增 registry/probes/rhs/kernel
├── tests/                                         # 测试套件 (51 .py, 445 passed)
└── cases/templates/                               # JSON 工况模板 (4 个)
```

---

## v0.3.1–v0.3.2 已删除的旧内容

| 文件/目录 | 删除原因 |
|-----------|---------|
| `emtp_solver_v3.py` | 旧入口垫片 — 22 个引用文件已全部迁移到 `from emtp import EMTPSolver` |
| `emtp_components_series_rl_only.py` | 旧类型垫片 — solver.py 三层 try/except 导入链已清理 |
| `emtp_plotting.py` | 死代码 — 全项目无任何 import 引用 |
| `test_lasted/` | 旧测试 — 6 个遗留脚本，与 tests/ 功能重叠 |
| `validation/` | 空框架 — cases/golden_results 子目录均空，仅 4 个工具脚本 |
| `EMTP_SOLVER_ARCHITECTURE.md` | 旧文档 — 大量引用已删除的 emtp_solver_v3.py |
| `P3_P4_P5_IMPLEMENTATION_REPORT.md` | 历史报告 — 描述旧→新架构迁移，已无参考价值 |

---

## Layer 0: 外部物理库

五个大型自包含模块。仅依赖 numpy/scipy/numba/stdlib，相互无交叉依赖，不依赖 `emtp/` 包内任何模块。由 `emtp/` wrapper 子包通过 try/except + `None` 回退模式按需导入。

| 文件 | 行数 | 用途 | 核心导出 |
|------|------|------|---------|
| `transmission_line_emtp_v2.py` | 381 | 无损 Bergeron 恒参数传输线 | `BergeronLine`, `DelayBuffer`, `TransmissionLineInterface` |
| `ulm_transmission_line_PARA.py` | 2540 | ULM 频变传输线 + Numba JIT | `FitULMData`, `FitULMReader`, `ULMModel`, `ULMLine`, `ULMBatchPack` |
| `nonlinear_models_pscad.py` | 765 | PSCAD 分段 MOA + CIGRE LPM 闪络 | `SegmentedMOAResistor`, `InsulatorFlashoverLPM`, `SegmentedSolverHelper` |
| `umec_transformer.py` | 766 | UMEC 三相变压器（含饱和） | `UMECTransformer`, `UMECTransformerData`, `UMECSaturationModel` |
| `atp_lightning_current_generator_simplified.py` | 1058 | ATP 兼容雷电电流源（双指数 + Heidler） | `TWOEXPFCurrentSource`, `HEIDLERFCurrentSource`, `LightningWaveform` |

---

## Layer 1: 核心求解器 (emtp/)

`EMTPSolver` 是 MNA 瞬态仿真的主类。Layer 1 是 emtp/ 包内的根级模块。

```python
from emtp import EMTPSolver
# 或
from emtp.case_runner import run_case
```

| 文件 | 行数 | 角色 |
|------|------|------|
| `emtp/__init__.py` | 17 | 惰性导出 `EMTPSolver`（`__getattr__`），避免循环导入 |
| `emtp/solver.py` | ~3670 | **主求解器** — MNA 瞬态仿真引擎 Facade。v0.4.0 新增 5 个子模块委托：`self.registry` / `self.probe_manager` / `self.rhs_engine` / `self.kernel` / `self.event_runtime` |
| `emtp/types.py` | ~120 | 共享类型 — `ElementType`, `Branch`, `VoltageSource`, `CurrentSource`, `LineData`, `ValidationIssue/Report`, `RHSPlan` |
| `emtp/nodes.py` | ~80 | `NodeIndexer`（紧凑整数→稀疏矩阵行映射）+ `NodeBook`（命名节点注册） |
| `emtp/circuit.py` | ~60 | `CircuitModel` dataclass — 独立于求解器的电路拓扑容器 |
| `emtp/sparse_solver.py` | ~80 | `SparseLinearSolver` — SuperLU 封装，LU 分解缓存 |
| `emtp/stamping.py` | ~120 | `COOStamper`（三元组累加器）+ `StampingEngine`（MNA 装配生命周期） |
| `emtp/validation.py` | ~40 | 拓扑/参数/内存校验 → `ValidationReport` |

### 关键关系

- `solver.py` 单向导入 Layer 0 库和 Layer 2 子包模块（v0.3.1 已清理历史遗留的三层 try/except 回退链）
- `emtp/__init__.py` 惰性导出 `EMTPSolver`（避免了 `__init__` 阶段触发 solver 内所有 Layer 0 导入）
- v0.4.0: solver.py 新增 5 个子模块（registry/probes/rhs/kernel/event_runtime），均为 thin wrapper，内部分发到 solver 已有方法；后续可逐步将内部逻辑迁入子模块

### v0.4.0 新增子模块

| 模块 | 文件 | PR | 角色 |
|------|------|----|------|
| `emtp/registry/` | `simulation_registry.py`, `records.py` | PR2 | 统一对象注册中心（shadow mode），拓扑/数值版本计数器 |
| `emtp/probes/` | `probe_manager.py` | PR3 | `ProbeManager` — 探针注册/采样，`ProbeSpec` 不可变描述 |
| `emtp/rhs/` | `rhs_engine.py` | PR4 | `RHSEngine` — RHS 构建/预采样/RHSPlan 失效 |
| `emtp/kernel/` | `mna_kernel.py` | PR5 | `MNAKernel` — G 矩阵生命周期/LU 求解/mark_dirty |
| `emtp/runtime/` | `event_runtime.py` (新增) | PR6 | `EventRuntime` — 每步编排 wrapper（开关/求解/分支更新/历史推进） |

---

## Layer 2: 模块化子包

### devices/ — 分支元件物理实现

每个 Device 实现统一协议：`stamp_G` · `stamp_rhs` · `update_branch_quantities` · `update_history`

| 文件 | 实现 |
|------|------|
| `base.py` | `Device` Protocol — 二端元件抽象接口 |
| `multiport.py` | `MultiPortDevice` Protocol — 多端口元件接口（Bergeron/ULM/UMEC） |
| `resistor.py` | `ResistorDevice` — 纯电阻，恒定电导，无历史项 |
| `inductor.py` | `InductorDevice` — 梯形法：`Geq = dt/(2L)`，可选并联阻尼 |
| `capacitor.py` | `CapacitorDevice` — 梯形法：`Geq = 2C/dt` |
| `switch.py` | `SwitchDevice` — 定时开/关，Ron/Roff 电导，触发拓扑重建 |
| `series_rl.py` | `SeriesRLDevice` — 串联 RL，无内部节点的二端实现 |
| `nonlinear.py` | `NonlinearResistorDevice` — 分段 MOA，`SegmentedSolverHelper` 管理电导切换 |
| `lpm.py` | `LPMFlashoverDevice` — CIGRE 先导发展法闪络开关 |

### assembly/ — MNA 矩阵装配

| 文件 | 角色 |
|------|------|
| `mna.py` | `MNAAssembler` — 构建 (n+m)×(n+m) 增广系统矩阵 G 和 RHS |

### runtime/ — 每步求解编排

| 文件 | 角色 |
|------|------|
| `__init__.py` | `DynamicDeviceRuntime` — 开关事件/分支V-I更新/历史推进/非线性重解检查 |
| `resolve.py` | `ResolveManager` + `ResolveEvent` — 统一的 MOA/LPM/UMEC 重解循环 |
| `stepper.py` | `TimeStepper` — 主时间步循环，委托每步物理给 solver |
| `event_runtime.py` | **v0.4.0 新增** — `EventRuntime`：每步完整流程 wrapper（开关→求解→分支更新→历史推进） |

### results/ — 结果存储

| 文件 | 角色 |
|------|------|
| `store.py` | `ResultStore` — 预分配缓冲区：时间/节点电压/电压源电流/探针波形 |
| `__init__.py` | 工具函数：`scale_probe_values`, `node_voltage_from_solution` 等 |

### lines/ — 传输线适配器 + fitULM 解析

| 文件 | 角色 |
|------|------|
| `fitulm_resolver.py` | **v0.3.2 新增** — `FitULMSpec` + `FitULMResolver`：外部文件校验 / LCP 自动生成 / 内容 hash 缓存 / `cache_dir` 传播 |
| `bergeron.py` | `BergeronLineDevice` — Bergeron 线 `MultiPortDevice` 适配器 |
| `ulm.py` | `ULMLineDevice` — ULM 线 `MultiPortDevice` 适配器 |

### transformers/ · sources/ · nonlinear/

| 文件 | 角色 |
|------|------|
| `transformers/umec.py` | `UMECTransformerDevice` — UMEC 多端口适配器，饱和驱动矩阵重建 |
| `sources/__init__.py` | 从 `atp_lightning_current_generator_simplified` try/except 导出 `TWOEXPFCurrentSource`, `HEIDLERFCurrentSource`, `LightningWaveform` |
| `nonlinear/__init__.py` | 从 `nonlinear_models_pscad` try/except 导出 `SegmentedMOAResistor`, `InsulatorFlashoverLPM` |

---

## Layer 3: 高层管线

### config/ — JSON 工况配置

| 文件 | 角色 |
|------|------|
| `schema.py` | `CaseConfig` + `SimulationOptions` dataclass |
| `loader.py` | `load_case_config()` — 加载/合并默认值/验证 |
| `validator.py` | `validate_case_config()` — dt/finish_time 正值、有效元件类型、唯一名称 |
| `defaults.py` | `SUPPORTED_ELEMENTS`, `SUPPORTED_SOURCES`, `SUPPORTED_PROBES`, `DEFAULT_SIMULATION` |

### builders/ — 配置→求解器构建

| 文件 | 角色 |
|------|------|
| `solver_builder.py` | `build_solver_from_config()` — 从 CaseConfig 创建并配置 EMTPSolver |
| `element_builder.py` | `add_element_to_solver()` — 按 `kind` 分发元件到 solver 方法。v0.4.0 新增 `ulm_line` kind |
| `source_builder.py` | `add_source_to_solver()` — 分发电源 |
| `probe_builder.py` | `add_probe_to_solver()` — 分发探针 |

### snapshot/ — 状态快照

| 文件 | 角色 |
|------|------|
| `schema.py` | `SnapshotMetadata` dataclass |
| `serializer.py` | `save_snapshot()` — 序列化分支/线路/变压器状态到目录 |
| `restore.py` | `load_snapshot_into_solver()` — 从目录恢复状态 |
| `hashing.py` | `compute_config_hash()` + `compute_topology_hash()` — SHA-256 |

### export/ — 结果导出

| 文件 | 角色 |
|------|------|
| `waveform_exporter.py` | `export_waveforms_npz()` — NPZ + 元数据 JSON + stride 降采样；`read_waveform_chunk()` — 分块读取 |
| `metrics_exporter.py` | `export_metrics_json()` — 标量指标 → JSON |
| `csv_exporter.py` | `export_waveforms_csv()` — 1-D 波形 → CSV |

### 根级管线模块

| 文件 | 行数 | 角色 |
|------|------|------|
| `emtp/case_runner.py` | 313 | `run_case()` — 全流程入口：加载→构建→模拟→收集→导出→入库 |
| `emtp/result_bundle.py` | 39 | `ResultBundle` dataclass — 结构化输出容器 |
| `emtp/result_db.py` | 196 | `ResultDatabase` — SQLite 运行历史 + 指标 + 波形信号 |
| `emtp/run_id.py` | 13 | `make_run_id()` — 时戳 + UUID 去重 ID |

---

## Layer 4: LCP 线路常数计算

v0.3.2 新增。分两层：`LCP/` 是物理引擎（底层算法），`pylcp/` 是 Python 包装层（面向 EMTP 集成）。

### LCP/ — 线路常数物理引擎 (12 .py)

```
LCP/
├── __init__.py                           # 包入口
├── cable_model.py                        # 电缆 Z/Y (Ametani 1980)
├── ulm_atp_zy_deri_semlyen.py            # 架空线 Z/Y (Deri-Semlyen)
├── vectfit3.py                           # Vector Fitting v1.3.1 引擎
├── vf_core.py                            # VF 适配层 → VectorFitResult
├── vector_fitting_v411_independent.py    # ULM 完整拟合 v4.11
└── test/                                 # 案例/验证脚本
    ├── pscad_reader.py                   # PSCAD 输出文件读取器
    ├── ulm_ohl_calculation_deri_semlyen.py  # 架空线完整案例
    ├── ulm_three_core_cable_v2 (1).py    # 三芯管型电缆案例
    ├── ulm_cable_calculation.py          # 多回铠装电缆案例
    └── test0304.py                       # 架空线 PSCAD 对比
```

**模块依赖链**:

```
vectfit3.py          ← VF 底层引擎，无 LCP 内依赖
  ↑
vf_core.py           ← VF 适配层，导入 .vectfit3
  ↑
vector_fitting_v411_independent.py  ← ULM 完整拟合 + fitULM 读写，导入 .vf_core
```

`cable_model.py` 和 `ulm_atp_zy_deri_semlyen.py` 为 Z/Y 计算引擎，各自独立，无 LCP 内依赖。

**核心 API**（均在 `vector_fitting_v411_independent.py`）:

| 函数 | 用途 |
|------|------|
| `ulm_complete_fitting(freq, Z, Y, length, ...)` | Z/Y → VF → ULM 参数 |
| `write_fitULM(result, filepath, ...)` | 序列化 fitULM 文本文件 |
| `verify_fitULM_file(filepath)` | 校验 fitULM 文件完整性 |
| `read_fitULM_header(filepath)` | 读取 fitULM 头部元数据 |
| `IterativePoleFindingConfig` | VF 配置 dataclass |

### pylcp/ — LCP Python 包装层 (10 .py)

```
pylcp/
├── __init__.py              # 统一导出
├── specs.py                 # LCPLineType 枚举 + LCPFitULMSpec dataclass
├── exceptions.py            # LCPError / LCPInputError / LCPFittingError / ...
├── validation.py            # validate_frequency_vector() / validate_zy_matrices()
├── cache.py                 # compute_cache_key() / get_cache_path()
├── lcp_fitulm_generator.py  # LCPFitULMGenerator — Z/Y → VF → fitULM 全链路
└── generation/
    ├── __init__.py
    ├── ohl_deri_semlyen.py          # 架空线 Z/Y
    ├── pipe_type_cable.py           # 管型电缆 Z/Y (兼容 2D/3D P_matrix)
    └── multi_armored_cable.py       # 多回铠装电缆 Z/Y (块对角 Y 组装)
```

### cache.py — 内容 hash 缓存

```python
# 缓存路径: .lcp_cache/{name}_{hash}.fitULM
# hash 覆盖：schema_version + pylcp_version + lcp_version
#           + line_type + length + freq_hash
#           + geometry_config + soil_config + vf_config
#           + precision + use_freq_dependent + enforce_passivity
```

- `compute_cache_key(spec)` — 对所有影响 fitULM 结果的因素做 SHA-256 取前 16 位
- `get_cache_path(spec)` — 返回 `{cache_dir}/{name}_{key}.fitULM`
- 版本字段 (`schema_version=2`, `pylcp_version`, `lcp_version`) 确保升级后旧缓存自动失效
- `_get_output_path()` 无条件使用外层 `FitULMSpec.cache_dir`（覆盖 lcp_spec 默认值）
- `lcp_spec.output_path` 显式设置时优先级最高

### fitulm_resolver.py — fitULM 文件校验与解析

```
FitULMResolver.resolve(spec)
  │
  ├─ _resolve_external_file() → _verify_fitulm() → path
  └─ _resolve_from_lcp()
       ├─ _get_output_path() → cache_dir 传播 → get_cache_path()
       ├─ 缓存命中 + 未过期 → _verify_fitulm() → return
       └─ 否则 → LCPFitULMGenerator.generate() → _verify_fitulm() → return
```

关键修复 (v0.3.3):
- `_verify_fitulm()` 只捕获 `ImportError`（LCP 不可用），不吞 `Exception`
- `_resolve_external_file()` 也调用 `_verify_fitulm()`，坏文件不会抵达 solver
- `_get_output_path()` 无条件用外层 `cache_dir` 覆盖 lcp_spec 默认值

### ULM 线路接入 — 两条路径

**路径 A：外部 fitULM 文件**（已有文件，直接读取）

```python
solver.add_ULM_line(
    name="line1",
    nodes_send=[1, 2, 3], nodes_recv=[101, 102, 103],
    length=5000.0,
    generate_fitulm=False,
    fitulm_path="models/cable14.fitULM",
)
```

**路径 B：LCP 自动生成**（从几何参数生成 fitULM，length 可省略）

```python
from pylcp import LCPLineType, LCPFitULMSpec

lcp_spec = LCPFitULMSpec(
    line_type=LCPLineType.OHL_DERI_SEMLYEN,
    name="ohl_line", length=20000.0,
    freq=np.logspace(0, 5, 201),
    geometry_config=line_geometry,
)

# length 可省略 — 自动使用 lcp_spec.length
solver.add_ULM_line(
    name="ohl_line",
    nodes_send=[1, 2], nodes_recv=[101, 102],
    generate_fitulm=True,
    lcp_spec=lcp_spec,
)
```

**length 语义**:

| 模式 | length 参数 | 行为 |
|------|------------|------|
| `generate_fitulm=True` | 省略 | 使用 `lcp_spec.length` |
| `generate_fitulm=True` | 与 `lcp_spec.length` 一致 | 通过 |
| `generate_fitulm=True` | 与 `lcp_spec.length` 不一致 | `ValueError("length mismatch")` |
| `generate_fitulm=False` | 必须显式传入 | 直接使用 |

**内部链路**:

```
solver.add_ULM_line(...)
        │
        ├─ length 一致性校验 (generate_fitulm=True 时强制)
        │
        ▼
FitULMResolver.resolve(spec)
        │
  ┌─────┴──────┐
  │ 外部文件      │  LCP 自动生成
  │ verify→path   │   │
  └──────┬──────┘   ▼
         │    LCPFitULMGenerator.generate()
         │      ├── compute_zy()     → Z/Y 矩阵
         │      ├── ulm_complete_fitting() → VF 拟合
         │      └── write_fitULM()   → fitULM 文件 + verify
         │              │
         │              └── get_cache_path(spec)  → {name}_{hash}.fitULM
         │
         ▼
  solver.add_ulm_line(fitulm_file=path)
         │
         ▼
  FitULMReader → ULMModel → ULMLine → EMTP 时域仿真
```

---

## 数据流

### run_case() 全链路

```
run_case("cases/templates/rc_step.json")
  │
  ├─ 1. load_case_config()          config/loader.py
  │     ├─ JSON → CaseConfig        config/schema.py
  │     └─ validate_case_config()   config/validator.py
  │
  ├─ 2. build_solver_from_config()  builders/solver_builder.py
  │     ├─ add_element_to_solver()  builders/element_builder.py
  │     ├─ add_source_to_solver()   builders/source_builder.py
  │     └─ add_probe_to_solver()    builders/probe_builder.py
  │
  ├─ 3. solver.run()                solver.py
  │     ├─ DynamicDeviceRuntime     runtime/__init__.py
  │     ├─ ResolveManager           runtime/resolve.py
  │     ├─ TimeStepper              runtime/stepper.py
  │     ├─ MNAAssembler             assembly/mna.py
  │     ├─ SparseLinearSolver       sparse_solver.py
  │     └─ ResultStore              results/store.py
  │
  ├─ 4. _collect_metrics()          case_runner.py
  ├─ 5. _collect_waveforms()        case_runner.py
  │
  ├─ 6. export_waveforms_npz()      export/waveform_exporter.py
  ├─ 7. export_metrics_json()       export/metrics_exporter.py
  ├─ 8. [可选] export_waveforms_csv() export/csv_exporter.py
  │
  └─ 9. ResultDatabase              result_db.py
        ├─ insert_run()
        ├─ insert_metrics()
        ├─ insert_signals()
        └─ update_run_done()
```

### LCP 自动生成链路

```
LCPFitULMSpec (line_type, length, freq, geometry_config, ...)
  │
  ▼
LCPFitULMGenerator.generate()
  │
  ├─ 1. validate_frequency_vector()    pylcp/validation.py
  │
  ├─ 2. compute_zy()                    pylcp/generation/
  │     ├─ OHL:      compute_ohl_zy()     → Z/Y via Deri-Semlyen
  │     ├─ PIPE:     compute_pipe_type_cable_zy() → Z/Y via Ametani
  │     │              └─ _potential_to_admittance()  2D/3D P_matrix 兼容
  │     └─ ARMORED:  compute_multi_armored_cable_zy() → Z/Y
  │                    └─ _compute_multi_armored_admittance()  块对角 Y
  │
  ├─ 3. validate_zy_matrices()          pylcp/validation.py
  │
  ├─ 4. ulm_complete_fitting()          LCP/vector_fitting_v411_independent.py
  │     ├─ compute_ulm_parameters()      Stage 1: ZY → Yc/H/γ/τ/Ti (NR eigensolver)
  │     └─ perform_ulm_fitting()         Stage 2: VF on tr(Yc) + H modes
  │
  └─ 5. write_fitULM() + verify_fitULM_file()
        ↓
     .fitULM 文件 → solver.add_ulm_line() 读取
```

---

## 测试体系

```
445 passed, 3 skipped
```

```
tests/
├── test_basic_mna.py               # MNA 基本装配
├── test_trapezoidal_rlc.py         # 梯形法 RLC
├── test_switches.py                # 开关元件
├── test_nodes.py                   # 节点管理
│   ...
│
├── test_case_config.py             # 配置加载/验证
├── test_snapshot.py                # 快照保存/恢复
├── test_export_and_db.py           # 导出 + 数据库
├── test_product_kernel_loop.py     # run_case → export → db 闭环
├── test_solver_regression.py       # 求解器回归 (38 tests)
│
├── test_baseline_lcp_emtp.py       # ★ LCP 模块可达性 + fitULM API + 语法检查
├── test_pr1_fitulm_resolver.py     # ★ FitULMResolver + add_ULM_line 全接口
│
├── pylcp_tests/                    # ★ LCP 集成测试
│   ├── test_pr2_generation.py       # Z/Y 生成 + P_matrix 2D/3D + Y 块对角
│   ├── test_pr3_generator.py        # LCPFitULMGenerator 管线
│   ├── test_cache.py                # 内容 hash 缓存 + 版本字段 + cache_dir 传播
│   └── test_pr67_integration.py     # 缓存复用 + E2E 求解器仿真
│
└── refactor_safety/                # ★ v0.4.0 重构安全网 (136 tests)
    ├── test_public_api_contract.py  # 74 方法 + 属性存在 + 8 调用模式
    ├── test_import_boundaries.py    # Layer 隔离 / solver→Layer0 禁止 (xfail)
    ├── test_waveform_regression.py  # RC/RL/开关/Bergeron 标量不变量
    ├── test_registry_consistency.py # 双写一致性/版本号/去重
    ├── test_probe_manager.py        # 注册/索引/采样/向后兼容
    ├── test_rhs_engine.py           # RHS 构建/预采样等效
    ├── test_mna_kernel.py           # G 重建/LU 求解/dirty 检测
    ├── test_event_runtime.py        # 步进编排/开关事件
    └── test_element_builder_ulm.py  # Builder ulm_line 集成
```

---

## 版本历程

| 版本 | Commit | 关键变更 |
|------|--------|---------|
| v0.1 | `75f307e` | P3/P4/P5 模块化：Device 协议、emtp 包、物理验证 |
| v0.2.0 | `d439b80` | Solver 迁移：emtp/solver.py canonical、MultiPortDevice、ResolveManager |
| v0.2.1 | `f42404b` | PR-10~17：ResultStore、Multiport registry、Bergeron/ULM/UMEC adapter |
| v0.2.2 | `cf8b7dc` | PR-18~19：TimeStepper 主循环、CircuitModel 容器 |
| v0.3.0 | `6d77ab8` | Case/Config 层、Snapshot/Resume、降采样导出、SQLite |
| v0.3.1 | `52b87f8` | Bugfix: run_id 字符串路径；PR1: 删除旧 API 垫片 |
| v0.3.1 | `a487e0f` | Cleanup: 删除死代码/旧测试/空框架 (-10,771 行) |
| v0.3.2 | `866e210` | **LCP 集成**: fitULM 自动生成, solver.add_ULM_line(), pylcp 包 |
| v0.3.2 | `200d879`→`56f3d43` | **P0 修复 x6**: 语法检查 / verify 不吞异常 / hash 缓存 / length 一致性 / P_matrix 2D-3D / Y 块对角 |
| v0.3.3 | `19acfa0`→`735cfdf` | **严格验收 x2**: cache key 版本字段 + cache_dir 传播 / length 默认 None 语义 |
| v0.4.0 | `8278832`→`07ac052` | **重构 PR0–PR7**: 安全网 (136 tests) + registry/probes/rhs/kernel/event_runtime 子模块 + element_builder ulm_line |

---

## 已知技术债（v0.4.0 后）

PR2–PR7 已建立 thin wrapper 层，每个子模块有了自己的家。后续深度重构方向：

| # | 问题 | 状态 |
|---|------|------|
| 1 | `SimulationRegistry` 从 shadow mode 升级为唯一真相源 | ✅ 框架已建，双写进行中 |
| 2 | `ProbeManager` 接管 ResultStore 的探针分配 | ✅ 框架已建，采样逻辑仍在 solver |
| 3 | `RHSEngine` 内部化 source_sampler / RHSPlan 编译 | ✅ wrapper 已建，内部仍委托 solver |
| 4 | `MNAKernel` 接管 layout / topology signature / 诊断 | ✅ wrapper 已建，内部仍委托 solver |
| 5 | `EventRuntime` 三步接口（pre_step / post_solve_check / commit_step） | ✅ wrapper 已建，设备接口待统一 |
| 6 | `MultiPortDevice` 全量接管线路和变压器 | ✅ element_builder 已支持 ulm_line |
| 7 | `solver.py` 不再直接 import Layer 0 物理模型 | ⏳ 仍在 solver.py 中（xfail 标记） |
| 8 | LCP test/ 中案例脚本仍用旧 import | ⏳ 待迁移 |

---

## v0.4.0 重构总结

```
PR 1 ✅ (v0.3.1)  删除旧 API
PR 0 ✅ (v0.4.0)  重构安全网 — 136 个保护测试
PR 2 ✅ (v0.4.0)  SimulationRegistry — 统一对象注册 (shadow mode)
PR 3 ✅ (v0.4.0)  ProbeManager — 探针注册/采样
PR 4 ✅ (v0.4.0)  RHSEngine — RHS 构建 wrapper
PR 5 ✅ (v0.4.0)  MNAKernel — G 矩阵/LU 求解 wrapper
PR 6 ✅ (v0.4.0)  EventRuntime — 每步编排 wrapper
PR 7 ✅ (v0.4.0)  element_builder ulm_line + SUPPORTED_ELEMENTS
```

**架构红线**（后续修改强制遵守）:

1. `solver.py` 不能直接 import Layer 0 物理模型
2. `solver.py` 不能直接构造 G/RHS
3. 新增物理模型不能修改 `solver.py`，只能新增 Device/MultiPortDevice adapter 和 builder
