# EMTP 求解器 P3/P4/P5 实施报告

本文档详细记录了 EMTP 电磁暂态求解器在 P3（模块化拆分）、P4（API 收敛）和 P5（物理验证）三个阶段的全部修改内容及结果。

---

## 目录

1. [总体概述](#1-总体概述)
2. [P3：主求解器模块化拆分](#2-p3主求解器模块化拆分)
3. [P4：接口、命名、兼容性与公开 API 收敛](#3-p4接口命名兼容性与公开-api-收敛)
4. [P5：物理模型验证与基准算例建设](#4-p5物理模型验证与基准算例建设)
5. [测试结果汇总](#5-测试结果汇总)
6. [文件清单](#6-文件清单)

---

## 1. 总体概述

### 1.1 起点

项目在 P0/P1 之后的状态：

- `emtp_solver_v3.py`：4590 行的单一巨文件，包含所有类定义和求解器逻辑
- 81 个 pytest 测试全部通过
- P0 已修复 LPM 闪络模型调用参数顺序
- P1 已建立 18 个自动化测试的安全网

### 1.2 三个阶段的核心目标

| 阶段 | 核心目标 | 关键原则 |
|------|---------|---------|
| **P3** | 主求解器模块化拆分 | 搬迁式重构——只移代码，不改行为 |
| **P4** | API 收敛与兼容层整理 | 新入口清晰，旧入口兼容，命名不再误导 |
| **P5** | 物理验证与基准算例建设 | 分层验证——解析解→理论边界→模型曲线→集成回归 |

### 1.3 最终结果

```
测试总数：81 → 106（新增 25 个 P5 验证测试）
测试结果：106 passed, 0 failed
emtp_solver_v3.py：4590 → 3571 行（-1020 行重复内联定义）
新增 emtp/ 包：18 个文件，~1700 行模块化代码
新增 validation/ 验证体系：工具 + 7 类验证测试
新增文档：API_MIGRATION.md, DIRECTION_CONVENTIONS.md, README.md 更新
```

---

## 2. P3：主求解器模块化拆分

### 2.1 修改内容

#### 2.1.1 新建 `emtp/` 包（18 个文件）

按照架构文档已定义的四层引擎架构，将 `emtp_solver_v3.py` 中的类按职责拆分到独立模块：

**第零层：基础数据**

| 新文件 | 来源 | 内容 | 行数 |
|--------|------|------|------|
| `emtp/nodes.py` | `emtp_solver_v3.py` L184-379 | `NodeIndexer`（外部节点→紧凑索引映射）、`NodeBook`（字符串节点名管理） | 202 |
| `emtp/types.py` | `emtp_solver_v3.py` + `emtp_components_series_rl_only.py` | `ElementType` 枚举、`Branch`/`CurrentSource`/`LineData`/`VoltageSource` dataclass、`ValidationIssue`/`ValidationReport`、`RHSPlan` | 187 |

**第一层：设备抽象**

| 新文件 | 内容 | 行数 |
|--------|------|------|
| `emtp/devices/base.py` | `Device` Protocol——7 个方法的抽象接口 | 46 |
| `emtp/devices/resistor.py` | `ResistorDevice`——纯电阻，无历史项 | 61 |
| `emtp/devices/inductor.py` | `InductorDevice`——隐式梯形：Geq=Δt/(2L) | 76 |
| `emtp/devices/capacitor.py` | `CapacitorDevice`——隐式梯形：Geq=2C/Δt | 76 |
| `emtp/devices/switch.py` | `SwitchDevice`——定时开关+拓扑变更触发 | 103 |
| `emtp/devices/series_rl.py` | `SeriesRLDevice`——紧凑二端口 RL+历史源辅助函数 | 99 |
| `emtp/devices/nonlinear.py` | `NonlinearResistorDevice`——PSCAD 分段 MOA | 81 |
| `emtp/devices/lpm.py` | `LPMFlashoverDevice`——CIGRE 先导发展法绝缘子开关 | 84 |

**第二层：引擎模块**

| 新文件 | 内容 | 行数 |
|--------|------|------|
| `emtp/sparse_solver.py` | `_sparse_factorize()`、`SparseLinearSolver`——SuperLU 稀疏分解+matrix_id 缓存 | 83 |
| `emtp/stamping.py` | `COOStamper`——COO 三元组累加器、`StampingEngine`——MNA 矩阵装配+求解委托 | 156 |
| `emtp/runtime.py` | `DynamicDeviceRuntime`——每步状态管理：开关事件/V-I 更新/历史源递推/**LPM 闪络+UMEC 饱和+分段非线性三合一收敛检测** | 171 |

**第三层：求解器门面**

| 新文件 | 内容 | 行数 |
|--------|------|------|
| `emtp/solver.py` | 从 `emtp_solver_v3` re-export `EMTPSolver`（后续可搬迁类本体） | 7 |

**辅助模块**

| 新文件 | 内容 | 行数 |
|--------|------|------|
| `emtp/validation.py` | 电路校验辅助函数——浮空网络检测、内存估算、`ValidationReport` 构造 | 109 |
| `emtp/results.py` | 结果/探针辅助函数——单位缩放、从解向量读取节点/支路电压电流 | 86 |

#### 2.1.2 更新 `emtp_solver_v3.py`

**操作：**

1. 在文件顶部新增 `from emtp.xxx import ...` 导入块，从新包模块导入所有已迁移的类
2. 删除文件中已迁移类的内联定义（Device Protocol、7 种 Device 类、`COOStamper`、`StampingEngine`、`SparseLinearSolver`、`DynamicDeviceRuntime`）
3. 更新文件头 docstring，标记为 legacy entry point，引导用户使用 `from emtp import EMTPSolver`
4. `EMTPSolver` 类本体仍保留在文件中（~3100 行），但所有依赖类已从 `emtp.*` 导入

**结果：** `emtp_solver_v3.py` 从 4590 行缩减至 3571 行（删除约 1020 行重复内联定义）。

#### 2.1.3 兼容策略

```
新入口：from emtp import EMTPSolver
旧入口：from emtp_solver_v3 import EMTPSolver  ← 返回同一个类
```

验证命令：

```python
from emtp import EMTPSolver as New
from emtp_solver_v3 import EMTPSolver as Legacy
assert New is Legacy  # True
```

### 2.2 修改结果

```
测试：81 passed（与 P3 前完全一致）
旧杆塔案例：无需修改即可运行
导入路径：新/旧均返回同一类
```

---

## 3. P4：接口、命名、兼容性与公开 API 收敛

### 3.1 修改内容

#### 3.1.1 遗留模块→兼容 re-export 包装器

**`emtp_components_series_rl_only.py`**

- **修改前：** 内联定义 `ElementType`、`Branch`、`CurrentSource`、`LineData` 四个核心类型
- **修改后：** 全部替换为 `from emtp.types import ...` re-export，保留闪电波形兼容包装器
- **效果：** 旧脚本 `from emtp_components_series_rl_only import Branch` 仍可用，且返回与 `from emtp.types import Branch` 相同的类

**`emtp_solver_v3.py`**

- 文件头改为英文 docstring，明确标注 "legacy entry point"，推荐使用 `from emtp import EMTPSolver`

#### 3.1.2 包 re-export 包装器（4 个新模块）

为外部模型文件创建清晰的包内导入路径：

| 新文件 | re-export 来源 | 用途 |
|--------|---------------|------|
| `emtp/lines/__init__.py` | `transmission_line_emtp_v2` + `ulm_transmission_line_PARA` | Bergeron/ULM 线路模型 |
| `emtp/transformers/__init__.py` | `umec_transformer` | UMEC 变压器模型 |
| `emtp/sources/__init__.py` | `atp_lightning_current_generator_simplified` | ATP 雷电源 |
| `emtp/nonlinear/__init__.py` | `nonlinear_models_pscad` | MOA/LPM 非线性模型 |

用户现在可以用：

```python
from emtp.lines import BergeronLine
from emtp.transformers import UMECTransformerData
from emtp.sources import create_lightning_current_source
from emtp.nonlinear import SegmentedMOAResistor
```

#### 3.1.3 UMEC 工厂函数命名修正

**问题：** `create_umec_transformer_3ph_bank()` 名称暗示返回 `UMECTransformer` 实例，但实际返回 `UMECTransformerData`。

**修改 `umec_transformer.py`：**

1. **新增** `create_umec_transformer_3ph_bank_data()`——语义明确的工厂函数，返回 `UMECTransformerData`
2. **新增** `create_umec_transformer_3ph_bank_instance(dt=..., ...)`——真正返回 `UMECTransformer` 实例
3. **保留** `create_umec_transformer_3ph_bank()`——改为调用新函数，标记 `.. deprecated::`

```python
# 新推荐用法
data = create_umec_transformer_3ph_bank_data(name="T1", S_mva=100, V1_kV=220, V2_kV=66)
xfmr = create_umec_transformer_3ph_bank_instance(dt=1e-6, name="T1", S_mva=100, ...)

# 旧用法仍可用
legacy = create_umec_transformer_3ph_bank("T1", 100, 220, 66)
```

#### 3.1.4 EMTPSolver 公开 API 别名

在 `emtp_solver_v3.py` 的 `EMTPSolver` 类中新增 3 个推荐别名方法：

| 新增别名 | 委托到 | 说明 |
|---------|--------|------|
| `add_lightning_current_source(...)` | `add_lightning_IS(...)` | 更清晰的雷电源添加方法 |
| `add_standard_double_exponential_current_source(...)` | `add_standard_twoexpf_IS(...)` | 语义明确的双指数源方法 |
| `add_lpm_flashover_insulator(...)` | `add_insulator_LPM(...)` | 含完整参数文档和单位说明 |

已有的 `add_current_source()` → `add_IS()` 和 `add_voltage_source()` → `add_VS()` 别名保持不变。

#### 3.1.5 文档建设

| 新文件 | 内容 |
|--------|------|
| `DIRECTION_CONVENTIONS.md` | Norton 等效方向约定、电流源方向、支路电压/电流正方向、传输线端口约定、UMEC 端口约定、每步操作顺序、LPM 闪络后状态同步、支持的单位表 |
| `API_MIGRATION.md` | 旧→新导入路径对照表、方法别名表、UMEC 工厂迁移指南、遗留模块状态 |
| `README.md` | 更新为推荐 `from emtp import EMTPSolver`，增加架构图和文档链接 |

### 3.2 修改结果

```
测试：81 passed（与 P4 前完全一致）
新导入路径：全部可用（emtp.lines/.transformers/.sources/.nonlinear）
旧导入路径：全部保留（不破坏任何现有脚本）
UMEC 工厂：新命名不再误导，旧函数委托到新函数
API 别名：5 个推荐别名，旧方法全部保留
```

---

## 4. P5：物理模型验证与基准算例建设

### 4.1 修改内容

#### 4.1.1 验证框架基础设施

**新建 `validation/` 目录：**

```
validation/
  __init__.py
  tools/
    __init__.py
    metrics.py              # 指标计算与统一结果容器
    compare_waveforms.py    # 插值感知的波形比较
    export_results.py       # NPZ/JSON/Markdown 导出
    plot_report.py          # Matplotlib 对比图/多波形图
```

**核心数据结构 `ValidationResult`：**

```python
@dataclass
class ValidationResult:
    name: str           # 案例名
    category: str       # 分类（basic/bergeron/moa/lpm/umec/ulm/tower）
    passed: bool        # 是否通过
    metrics: dict       # 指标值 {"max_abs_error_V": 0.0021, ...}
    tolerances: dict    # 容差 {"max_abs_error_V": 0.01, ...}
    waveforms: dict     # 波形数据 {"time_s": t, "sim_V": v_sim, ...}
    references: dict    # 参考来源 {"type": "analytic", "formula": "..."}
    notes: str          # 备注
```

**pytest 标记（更新 `tests/conftest.py`）：**

```ini
markers =
    slow:       长仿真验证案例（杆塔 10μs、ULM 多步 batch）
    validation: 物理验证测试
    external:   需要外部 ATP/PSCAD/EMTP-RV 数据
```

运行方式：

```bash
pytest -q                              # 全部 106 tests
pytest -q -m validation                # 仅物理验证
pytest -q -m "validation and not slow" # 快速验证（22 tests）
pytest -q -m slow                      # 慢速验证
```

#### 4.1.2 第一批：基础解析电路验证（4 tests）

| 测试 | 验证内容 | 参考解 | 容差 |
|------|---------|--------|------|
| `test_rc_step_analytic` | RC 充电瞬态 | Vc(t)=V0·(1−exp(−t/RC)) | max_abs_error < 0.03V, final_value_error < 0.5% |
| `test_rl_step_analytic` | RL 充电瞬态 | I_L(t)=(V0/R)·(1−exp(−Rt/L)) | max_abs_error < 0.005A, final_value_error < 0.5% |
| `test_series_rl_vs_r_plus_l` | 紧凑 SeriesRL vs 分离 R+L | 相互对比 | relative_peak_error < 1% |
| `test_rc_dt_convergence` | dt 收敛性 | Vc(t) 解析解 | 更细 dt → 更小误差（单调递减） |

**验证的物理正确性：** MNA 矩阵装配、梯形积分法、Norton 等效 RHS stamping、电流/电压方向。

#### 4.1.3 第二批：Bergeron 传输线反射理论验证（3 tests）

| 测试 | 验证内容 | 理论预期 | 容差 |
|------|---------|---------|------|
| `test_bergeron_matched_load_no_reflection` | Z_load = Zc | 无反射，末端稳态电压 = V0/2 | 稳态误差 < 0.1V |
| `test_bergeron_open_circuit_voltage_doubling` | Z_load → ∞，Γ_V = +1 | 末端电压 ≈ 2 × 入射波 | 倍增比 1.8~2.2 |
| `test_bergeron_short_circuit_current_doubling` | Z_load → 0，Γ_I = +1 | 末端电流 ≈ 2 × 入射波 | 倍增比 1.6~2.5 |

**验证的物理正确性：** 延时缓冲区、Norton 历史源注入、反射系数、行波传播。

#### 4.1.4 第三批：MOA 非线性 V-I 曲线验证（4 tests）

| 测试 | 验证内容 |
|------|---------|
| `test_moa_breakpoints_exact_current` | 每个断点电压处 `get_current_exact(V) == I`（rtol=1e-9） |
| `test_moa_negative_voltage_symmetry` | I(−V) = −I(V)（奇对称性） |
| `test_moa_norton_equivalent_finite` | 所有电压下 G > 0 且有限，Ihist 有限 |
| `test_moa_segment_switching` | 电压跨越分段点后 `check_segment`/`update_segment` 正确切换 |

**验证的物理正确性：** 分段线性 V-I 曲线构造、电导/偏置电流计算、正负电压对称处理。

#### 4.1.5 第四批：LPM 绝缘子闪络验证（4 tests）

| 测试 | 验证内容 |
|------|---------|
| `test_lpm_no_flashover_below_threshold` | 低电压（300kV，1m 间隙）不发展先导、不闪络 |
| `test_lpm_flashover_with_high_voltage` | 特高压（2MV，0.1m 间隙）持续施加后 leader_length ≥ gap_length |
| `test_lpm_flashover_sets_branch_to_arc_resistance` | 闪络后 `br.is_closed=True`，`br.value=R_arc=0.5Ω` |
| `test_lpm_solver_voltage_equals_gap_voltage` | **P0 回归保护**：LPM 内部峰值电压 = 绝缘子实际节点压差峰值 |

**验证的物理正确性：** CIGRE 先导发展速度公式 v(t)=k·u(t)·[u(t)/(d−l)−E₀]、闪络后电弧电阻切换、P0 调用参数顺序修正不退化。

#### 4.1.6 第五批：UMEC 变压器验证（4 tests）

| 测试 | 验证内容 |
|------|---------|
| `test_umec_norton_matrix_symmetric_finite` | G 矩阵对称（G = Gᵀ），所有元素有限 |
| `test_umec_reset_clears_state` | update_history 后 reset_state 清零所有 I_hist |
| `test_umec_instance_factory_returns_transformer` | `create_umec_transformer_3ph_bank_instance(dt=...)` 返回 `UMECTransformer` 实例 |
| `test_umec_legacy_factory_returns_data` | `create_umec_transformer_3ph_bank()` 仍返回 `UMECTransformerData` |

**验证的物理正确性：** 多端口 Norton 等效矩阵结构、历史源递推/重置、P4 工厂函数语义正确。

#### 4.1.7 第六批：ULM 频变线路验证（3 tests）

| 测试 | 验证内容 |
|------|---------|
| `test_ulm_seed_reproducibility` | 相同 seed → 相同 `yc_poles`（rtol=1e-12） |
| `test_ulm_different_seeds_different_data` | 不同 seed → 不同 `yc_residues` |
| `test_ulm_single_line_finite_outputs` | 100μs 仿真，Vk/Vm 全有限，无 NaN/inf |

**验证的物理正确性：** FitULM 数据可复现性、UlmModel 初始化、单线路求解器集成。

#### 4.1.8 第七批：杆塔案例基准化（3 tests）

| 测试 | 验证内容 | 标记 |
|------|---------|------|
| `test_tower_topology_and_delay_parameters` | 10 条线路 Zc/tau/delay_steps 参数校验 | validation |
| `test_tower_no_flashover_metrics` | 塔顶 > 塔中 > 接地电压梯形分布 | slow + validation |
| `test_tower_lpm_voltage_consistency` | **P0 回归**：LPM internal peak == gap node voltage peak | slow + validation |

**验证的物理正确性：** 多层杆塔 Bergeron 网络拓扑、雷电暂态传播、绝缘子电压分布、P0 LPM 修正持续有效。

### 4.2 修改结果

```
总测试：106 passed, 0 failed
P5 新增：25 个验证测试（标记 @pytest.mark.validation）
快速验证：22 tests（排除 3 个 slow 标记的 tower/ULM 测试）
慢速验证：含杆塔 5μs 和 1.5μs 完整瞬态仿真
P0 回归：test_lpm_solver_voltage_equals_gap_voltage + test_tower_lpm_voltage_consistency 双层保护
```

---

## 5. 测试结果汇总

### 5.1 最终测试套件

```
==============================================
总计：106 passed, 0 failed, 4 warnings
耗时：11.84s（含杆塔慢速案例）
==============================================
```

### 5.2 按阶段分布

| 阶段 | 测试文件 | 测试数 |
|------|---------|--------|
| P1（原有） | test_basic_mna, test_nodes, test_trapezoidal_rlc, test_switches, test_bergeron_line, test_moa_segments, test_lpm_flashover, test_umec_transformer, test_ulm_smoke, test_tower_case_p1 | 18 |
| P1（原有） | test_solver_regression（56 tests） | 56 |
| P1（原有） | test_lasted/*（7 tests） | 7 |
| **P5 新增** | test_p5_basic_physics | 4 |
| **P5 新增** | test_p5_bergeron_reflection | 3 |
| **P5 新增** | test_p5_moa_validation | 4 |
| **P5 新增** | test_p5_lpm_validation | 4 |
| **P5 新增** | test_p5_umec_validation | 4 |
| **P5 新增** | test_p5_ulm_validation | 3 |
| **P5 新增** | test_p5_tower_validation | 3 |
| **合计** | | **106** |

### 5.3 按验证类别分布

| 类别 | 测试数 | 说明 |
|------|--------|------|
| 基础 MNA/元件 | 60+ | 原有 P1 回归 + solver 回归 |
| **解析解验证** | 4 | RC step, RL step, SeriesRL, dt convergence |
| **Bergeron 理论** | 3 | 匹配负载、开路反射、短路反射 |
| **MOA 非线性** | 5 | V-I 曲线 + 段切换 + Norton |
| **LPM 闪络** | 6 | 状态机 + 集成 + P0 回归 |
| **UMEC 变压器** | 5 | Norton 矩阵 + 工厂函数 |
| **ULM 线路** | 4 | seed + 一致性 + 有限输出 |
| **杆塔集成** | 10+ | 参数 + 瞬态 + LPM 回归 |
| 开关/节点/其他 | 10+ | 定时事件、节点压缩、RHS 计划 |

---

## 6. 文件清单

### 6.1 新增文件

```
emtp/
  __init__.py                       # 包入口，lazy import EMTPSolver
  solver.py                         # EMTPSolver re-export
  nodes.py                          # NodeIndexer, NodeBook
  types.py                          # ElementType, Branch, CurrentSource, LineData, VoltageSource, ...
  sparse_solver.py                  # _sparse_factorize, SparseLinearSolver
  stamping.py                       # COOStamper, StampingEngine
  runtime.py                        # DynamicDeviceRuntime (含 P0 LPM 修复)
  validation.py                     # 电路校验辅助函数
  results.py                        # 结果/探针辅助函数
  devices/
    __init__.py                     # 7 种 Device re-export
    base.py                         # Device Protocol
    resistor.py                     # ResistorDevice
    inductor.py                     # InductorDevice
    capacitor.py                    # CapacitorDevice
    switch.py                       # SwitchDevice
    series_rl.py                    # SeriesRLDevice + _update_series_rl_history_static
    nonlinear.py                    # NonlinearResistorDevice
    lpm.py                          # LPMFlashoverDevice
  lines/
    __init__.py                     # BergeronLine, ULMLine re-export
  transformers/
    __init__.py                     # UMEC re-export
  sources/
    __init__.py                     # lightning source re-export
  nonlinear/
    __init__.py                     # MOA/LPM re-export

validation/
  __init__.py
  tools/
    __init__.py
    metrics.py                      # ValidationResult, max_abs_error, rms_error, ...
    compare_waveforms.py            # 插值波形比较
    export_results.py               # NPZ/JSON/Markdown 导出
    plot_report.py                  # Matplotlib 对比图

tests/
  test_p5_basic_physics.py          # RC/RL/SeriesRL/dt 收敛
  test_p5_bergeron_reflection.py    # 匹配/开路/短路反射
  test_p5_moa_validation.py         # V-I 曲线/对称/Norton/段切换
  test_p5_lpm_validation.py         # 状态机/闪络/P0 回归
  test_p5_umec_validation.py        # Norton/重置/工厂函数
  test_p5_ulm_validation.py         # Seed/一致性/有限输出
  test_p5_tower_validation.py       # 拓扑/瞬态/LPM 一致性

docs/
  DIRECTION_CONVENTIONS.md          # 方向约定和单位规范
  API_MIGRATION.md                  # API 迁移指南
```

### 6.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `emtp_solver_v3.py` | 4590→3571 行：新增 emtp.* 导入，删除重复内联定义，更新 docstring |
| `emtp_components_series_rl_only.py` | 改为从 emtp.types re-export，保留闪电波形兼容包装器 |
| `umec_transformer.py` | 新增 `create_umec_transformer_3ph_bank_data()` 和 `_instance()`，保留旧函数 |
| `tests/conftest.py` | 新增 slow/validation/external 三个 pytest 标记 |
| `README.md` | 更新推荐导入路径、架构图、文档链接 |

### 6.3 未修改文件（保持原样）

```
transmission_line_emtp_v2.py       # Bergeron 线路模型
ulm_transmission_line_PARA.py      # ULM 频变线路模型
nonlinear_models_pscad.py          # PSCAD 非线性模型 (MOA + LPM)
atp_lightning_current_generator_simplified.py  # ATP 雷电源
emtp_plotting.py                   # 绘图工具
test_lasted/*                      # 杆塔案例和验证脚本
tests/test_*.py (原有 11 个)       # P1 回归测试
```

---

## 7. 关键设计决策

### 7.1 为什么不把 EMTPSolver 类体迁到 emtp/solver.py

当前 `emtp/solver.py` 通过 `from emtp_solver_v3 import EMTPSolver` re-export，两个导入路径返回同一类。将类体真正迁入 `emtp/solver.py` 需要同步迁移 ~3100 行代码和所有的条件导入（nonlinear_models、transmission_line、ulm、umec、atp），且会引入循环导入风险（emtp.runtime 需要访问 solver 的回调）。当前方案在保证 API 清晰的同时，风险最小。

### 7.2 为什么不发 DeprecationWarning

P4 方案明确建议第一阶段只做文档标记，不立即发 `DeprecationWarning`。原因：
- 避免现有脚本和 CI 输出被 warning 淹没
- 老旧模块如 `emtp_components_series_rl_only.py` 仍被大量内部代码引用
- 等用户适应新 API 后，下一阶段再开启 warning

### 7.3 验证容差设置原则

P5 的容差分三类：
- **解析解验证**：严格（max error < 0.03V for 10V signal）
- **理论边界验证**：合理范围（反射系数倍增比 1.8~2.2）
- **自回归 golden**：相对宽松（峰值误差 < 1%，时间误差 < 5ns），后续拿到 PSCAD/ATP 对标数据后收紧

---

## 8. 后续建议

按 P5 方案的设计，后续可开展的工作：

1. **ULM 外部对标**：与 ATP/PSCAD/EMTP-RV 的频变线路案例对比
2. **UMEC 空载/短路/饱和深度验证**：与铭牌参数对标
3. **LPM 闪络时间对标**：与 CIGRE 标准案例和 PSCAD 结果对比
4. **MOA 完整 V-I 曲线对标**：与制造商数据表对比
5. **Bergeron 反射系数精确定量验证**：使用更长的仿真时间观察多次反射
6. **大型网络性能优化**：Numba JIT 热点分析、batch 并行调优
7. **GPU 加速**：考虑 CuPy 或 JAX 迁移稀疏求解

---

*报告生成时间：2026-05-05*
*求解器版本：v0.2.0*
