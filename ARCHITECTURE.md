# PyEMTP 架构梳理

版本 `v0.3.1` · PR1+cleanup · 2026-05-06

---

## 整体分层架构

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 高层管线 (v0.3 新增)                            │
│  config/ builders/ snapshot/ export/                      │
│  case_runner.py result_bundle.py result_db.py run_id.py   │
├─────────────────────────────────────────────────────────┤
│  Layer 2: 模块化子包                                      │
│  devices/ assembly/ runtime/ results/                     │
│  lines/ transformers/ sources/ nonlinear/                 │
├─────────────────────────────────────────────────────────┤
│  Layer 1: 核心求解器 (emtp/)                               │
│  solver.py types.py nodes.py circuit.py                   │
│  sparse_solver.py stamping.py validation.py               │
├─────────────────────────────────────────────────────────┤
│  Layer 0: 外部物理库 (顶层 .py)                            │
│  transmission_line_emtp_v2.py  ulm_transmission_line_PARA.py │
│  nonlinear_models_pscad.py  umec_transformer.py            │
│  atp_lightning_current_generator_simplified.py             │
└─────────────────────────────────────────────────────────┘
```

**依赖方向**: Layer 3 → Layer 2 → Layer 1 → Layer 0（单向，上层依赖下层）

---

## Layer 0: 外部物理库（顶层 .py 文件）

五个体积大、独立自足的物理模型库。**仅依赖 numpy/scipy/numba/stdlib，相互之间无交叉依赖**，也不依赖 `emtp/` 包内的任何模块。

| 文件 | 行数 | 功能 | 导出 |
|------|------|------|------|
| `transmission_line_emtp_v2.py` | 381 | 无损 Bergeron 恒参数传输线模型 | `BergeronLine`, `DelayBuffer`, `TransmissionLineInterface`, `LineCalculator`, `TransmissionLineFactory` |
| `ulm_transmission_line_PARA.py` | 2540 | ULM 频变传输线，含 Numba JIT 加速 | `FitULMData`, `FitULMReader`, `ULMModel`, `ULMLine`, `ULMBatchPack` |
| `nonlinear_models_pscad.py` | 765 | PSCAD 风格分段 MOA 电阻 + CIGRE LPM 闪络 | `SegmentedMOAResistor`, `InsulatorFlashoverLPM`, `SegmentedSolverHelper`, `NonlinearResistorModel` |
| `umec_transformer.py` | 766 | UMEC 三相变压器模型（含饱和） | `UMECTransformer`, `UMECTransformerData`, `UMECSaturationModel`, `WindingType` |
| `atp_lightning_current_generator_simplified.py` | 1058 | ATP 兼容雷电电流源（双指数 + Heidler） | `TWOEXPFCurrentSource`, `HEIDLERFCurrentSource`, `LightningWaveform`, 工厂函数 |

**状态**: 这五个文件是项目的物理核心，必须保留。它们由 `emtp/` 内的 wrapper 子包按需导入（try/except + `None` 回退）。

---

## Layer 1: 核心求解器

`emtp/` 包内的根级模块，构成 MNA 瞬态仿真引擎的主体。

| 文件 | 行数 | 角色 |
|------|------|------|
| `emtp/solver.py` | ~3640 | **主求解器** — `EMTPSolver` 类。完整的 MNA 瞬态仿真引擎，包含多相线路、非线性 MOA、UMEC 变压器、LPM 闪络。是系统中最大的单体文件 |
| `emtp/types.py` | ~120 | 共享数据类型 — `ElementType` 枚举, `Branch`/`VoltageSource`/`CurrentSource` dataclass, `LineData`, `ValidationIssue`/`ValidationReport`, `RHSPlan` |
| `emtp/nodes.py` | ~80 | `NodeIndexer`（紧凑整数索引映射）和 `NodeBook`（命名节点注册表） |
| `emtp/circuit.py` | ~60 | `CircuitModel` dataclass — 分支/设备/电源/线路/变压器/节点的独立数据容器，与求解器解耦 |
| `emtp/sparse_solver.py` | ~80 | `SparseLinearSolver` — 封装 `scipy.sparse.linalg.splu` (SuperLU)，带 LU 分解缓存 |
| `emtp/stamping.py` | ~120 | `COOStamper`（三元组累加器，构建稀疏 G 矩阵）+ `StampingEngine`（管理装配生命周期） |
| `emtp/validation.py` | ~40 | 拓扑/参数/内存校验，返回 `ValidationReport` |

**关键关系**:
- `solver.py` 通过 `emtp.types`、`emtp.sources` 等子包导入所需符号（v0.3.1 已清理历史遗留的三层 try/except 导入回退链）
- `emtp/__init__.py` 通过 `__getattr__` 惰性导出 `EMTPSolver`，避免循环导入

---

## Layer 2: 模块化子包

### devices/ — 分支元件物理实现

| 文件 | 角色 |
|------|------|
| `base.py` | `Device` Protocol — 二端元件的抽象接口：`stamp_G`, `stamp_rhs`, `update_branch_quantities`, `update_history` |
| `multiport.py` | `MultiPortDevice` Protocol — 多端口元件的抽象接口（Bergeron/ULM/UMEC） |
| `resistor.py` | `ResistorDevice` — 纯电阻，无历史项，恒定电导 |
| `inductor.py` | `InductorDevice` — 梯形法离散：Geq = dt/(2L)，可选的并联阻尼 |
| `capacitor.py` | `CapacitorDevice` — 梯形法离散：Geq = 2C/dt |
| `switch.py` | `SwitchDevice` — 定时开/关，Ron/Roff 电导模型，触发拓扑重建 |
| `series_rl.py` | `SeriesRLDevice` — 串联 RL，无内部节点的二端实现 |
| `nonlinear.py` | `NonlinearResistorDevice` — PSCAD 风格分段 MOA，由 `SegmentedSolverHelper` 管理电导切换 |
| `lpm.py` | `LPMFlashoverDevice` — CIGRE 先导发展法闪络开关 |

### assembly/ — MNA 矩阵装配

| 文件 | 角色 |
|------|------|
| `mna.py` | `MNAAssembler` — 构建 (n+m)×(n+m) 增广 MNA 系统矩阵 G 和 RHS 向量 |

### runtime/ — 每步状态管理与重解循环

| 文件 | 角色 |
|------|------|
| `__init__.py` | `DynamicDeviceRuntime` — 管理求解前的开关事件、求解后的分支 V/I 更新、历史项推进、LPM/UMEC/非线性后求解重解检查 |
| `resolve.py` | `ResolveManager` + `ResolveEvent` — 统一的非线性/LPM/UMEC 重解循环，检测拓扑变化并触发矩阵重建 |
| `stepper.py` | `TimeStepper` — 抽取的主时间步循环编排器，将每步物理通过 `_run_one_step` 委托回求解器 |

### results/ — 结果检索与存储

| 文件 | 角色 |
|------|------|
| `store.py` | `ResultStore` — 预分配缓冲区：时间数组、节点电压矩阵、电压源电流缓冲、探针存储 |
| `__init__.py` | 辅助函数：`scale_probe_values`, `node_voltage_from_solution`, `branch_current_from_solution` 等 |

### lines/ — 传输线适配器

| 文件 | 角色 |
|------|------|
| `bergeron.py` | `BergeronLineDevice` — `MultiPortDevice` 适配器，将 Bergeron 线的对地端口和历史电流注入接入 MNA |
| `ulm.py` | `ULMLineDevice` — `MultiPortDevice` 适配器，将频变 ULM 线接入 MNA |

### transformers/ — 变压器适配器

| 文件 | 角色 |
|------|------|
| `umec.py` | `UMECTransformerDevice` — `MultiPortDevice` 适配器，支持饱和驱动的矩阵重建 |

### sources/ — 雷电电流源

| 文件 | 角色 |
|------|------|
| `__init__.py` | 从 `atp_lightning_current_generator_simplified.py` 做 try/except 导入，导出 `TWOEXPFCurrentSource`, `HEIDLERFCurrentSource`, 工厂函数 |

### nonlinear/ — 非线性元件 wrapper

| 文件 | 角色 |
|------|------|
| `__init__.py` | 从 `nonlinear_models_pscad.py` 做 try/except 导入，导出 `SegmentedMOAResistor`, `InsulatorFlashoverLPM`, `SegmentedSolverHelper` |

---

## Layer 3: 高层管线（v0.3 新增）

### config/ — JSON 工况配置

| 文件 | 角色 |
|------|------|
| `schema.py` | `CaseConfig` + `SimulationOptions` dataclass，定义完整配置模式 |
| `loader.py` | `load_case_config()` — 加载 + 合并默认值 + 验证 JSON 工况文件 |
| `validator.py` | `validate_case_config()` — 校验 dt/finish_time 正值、有效元件类型、唯一名称 |
| `defaults.py` | `SUPPORTED_ELEMENTS`, `SUPPORTED_SOURCES`, `SUPPORTED_PROBES`, `DEFAULT_SIMULATION` |

### builders/ — 配置→求解器构建

| 文件 | 角色 |
|------|------|
| `solver_builder.py` | `build_solver_from_config()` — 从 `CaseConfig` 创建并配置 `EMTPSolver` |
| `element_builder.py` | `add_element_to_solver()` — 按 `kind` 键分发元件 |
| `source_builder.py` | `add_source_to_solver()` — 分发电源 |
| `probe_builder.py` | `add_probe_to_solver()` — 分发探针 |

### snapshot/ — 状态保存/恢复

| 文件 | 角色 |
|------|------|
| `schema.py` | `SnapshotMetadata` dataclass |
| `serializer.py` | `save_snapshot()` — 将求解器动态状态序列化到快照目录 |
| `restore.py` | `load_snapshot_into_solver()` — 从快照目录恢复分支/线路/变压器动态状态 |
| `hashing.py` | `compute_config_hash()` + `compute_topology_hash()` — SHA-256 哈希，用于快照完整性校验和拓扑变更检测 |

### export/ — 结果导出

| 文件 | 角色 |
|------|------|
| `waveform_exporter.py` | `export_waveforms_npz()` — NPZ + 波形元数据 JSON + stride 降采样；`read_waveform_chunk()` — 分块读取（前端友好） |
| `metrics_exporter.py` | `export_metrics_json()` — 标量指标 → JSON |
| `csv_exporter.py` | `export_waveforms_csv()` — 1-D 波形 → CSV |

### 根级管线模块

| 文件 | 行数 | 角色 |
|------|------|------|
| `emtp/case_runner.py` | 313 | `run_case()` — 高层入口：加载→构建→模拟→收集→导出→入库 |
| `emtp/result_bundle.py` | 39 | `ResultBundle` dataclass — 结构化输出容器 |
| `emtp/result_db.py` | 196 | `ResultDatabase` — SQLite 运行历史、指标、波形信号记录 |
| `emtp/run_id.py` | 13 | `make_run_id()` — 生成唯一运行 ID |

---

## v0.3.1 已删除的历史遗留内容

PR1 + cleanup 已从仓库中删除以下内容：

| 文件/目录 | 类型 | 删除原因 |
|-----------|------|---------|
| `emtp_solver_v3.py` | 兼容垫片 | 旧入口，全部 22 个引用文件已迁移到 `from emtp import EMTPSolver` |
| `emtp_components_series_rl_only.py` | 兼容垫片 | `solver.py` 三层 try/except 导入链已清理，符号直接来自 `emtp.types` |
| `emtp_plotting.py` | 死代码 | 全项目无 import，功能已被 `ResultStore` 替代 |
| `test_lasted/` | 旧测试 | 6 个遗留验证脚本，与 `tests/` 重复，使用旧 API |
| `validation/` | 空框架 | 仅 4 个工具脚本，`cases/` 和 `golden_results/` 子目录均空 |
| `EMTP_SOLVER_ARCHITECTURE.md` | 旧文档 | 被本文档取代，大量引用已删除文件 |
| `P3_P4_P5_IMPLEMENTATION_REPORT.md` | 历史报告 | 描述旧→新架构迁移过程，已无参考价值 |

---

## 根目录文件清单

```
emtp_v0.2/
├── README.md                                    # 项目文档
├── CLAUDE.md                                    # Claude Code 行为指南
├── ARCHITECTURE.md                              # 架构文档
├── API_MIGRATION.md                             # API 迁移指南
├── DIRECTION_CONVENTIONS.md                     # 符号/单位/stamping 约定
├── .gitignore
│
├── atp_lightning_current_generator_simplified.py # ✅ Layer 0: 雷电电流源
├── transmission_line_emtp_v2.py                  # ✅ Layer 0: Bergeron 线
├── ulm_transmission_line_PARA.py                 # ✅ Layer 0: ULM 线
├── nonlinear_models_pscad.py                     # ✅ Layer 0: MOA + LPM
├── umec_transformer.py                           # ✅ Layer 0: UMEC 变压器
│
├── emtp/                                         # ✅ 主包（54 个 .py 文件）
├── tests/                                        # ✅ 测试套件（248 passed, 3 skipped）
└── cases/templates/                              # ✅ JSON 工况模板（4 个）
```

---

## 数据流：run_case() 全链路

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
  ├─ 3. solver.run()                solver.py (TimeStepper)
  │     ├─ DynamicDeviceRuntime     runtime/__init__.py
  │     ├─ ResolveManager           runtime/resolve.py
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

---

## 已知技术债（剩余）

PR1 + cleanup 已解决之前记录的全部 9 条技术债。剩余工作进入 PR2–PR7 架构升级计划：

| # | 问题 | 文件 | 计划 PR |
|---|------|------|---------|
| 1 | `solver.py` 是 ~3640 行单体文件 | `emtp/solver.py` | PR2–PR7 |
| 2 | 仿真对象状态分散在 solver 的多个并行容器中 | `emtp/solver.py` | PR2 |
| 3 | 探针和结果记录逻辑嵌入 solver | `emtp/solver.py` | PR3 |
| 4 | RHS 构建、电源预采样未独立 | `emtp/solver.py` | PR4 |
| 5 | G 矩阵装配、LU 缓存与求解逻辑嵌入 solver | `emtp/solver.py` | PR5 |
| 6 | 非线性重解/开关事件分散在不同设备中 | `emtp/solver.py`, `devices/` | PR6 |
| 7 | 线路/变压器在 solver 中有特殊处理分支 | `emtp/solver.py` | PR7 |

---

## 下一步架构升级 (PR2–PR7)

参见 PR 拆分规划文档（未写入仓库）。核心方向：

```
PR 1 ✅  清理旧 API
PR 2    建立 SimulationRegistry，统一仿真对象状态
PR 3    迁出 ProbeManager / ResultStore
PR 4    抽出 RHS Engine
PR 5    抽出 MNAKernel（矩阵装配、缓存、求解）
PR 6    重构 Runtime / Resolve（统一事件与重解）
PR 7    MultiPortDevice 全量接管线路和变压器
```
