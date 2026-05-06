# API Migration Guide

v0.3.1 已完成旧 API 清理。`emtp_solver_v3.py` 和 `emtp_components_series_rl_only.py` 已删除。

---

## 唯一入口

| 用途 | 导入 |
|------|------|
| 求解器 | `from emtp import EMTPSolver` |
| 高层管线 | `from emtp.case_runner import run_case` |
| 节点管理 | `from emtp import NodeBook, NodeIndexer` |
| 类型 | `from emtp.types import Branch, CurrentSource, ElementType, LineData, ...` |
| Bergeron 线 | `from emtp.lines import BergeronLine` |
| ULM 线 | `from emtp.lines import ULMLine` |
| UMEC 变压器 | `from emtp.transformers import UMECTransformer` |
| 雷电电流源 | `from emtp.sources import ...` |
| 非线性元件 | `from emtp.nonlinear import SegmentedMOAResistor` |

---

## Solver Methods

| Old method | Recommended alias | Notes |
|-----------|------------------|-------|
| `add_IS(...)` | `add_current_source(...)` | Same behavior |
| `add_VS(...)` | `add_voltage_source(...)` | Same behavior |
| `add_insulator_LPM(...)` | `add_lpm_flashover_insulator(...)` | Clearer name; adds unit docs |
| `add_lightning_IS(...)` | `add_lightning_current_source(...)` | Same behavior |
| `add_standard_twoexpf_IS(...)` | `add_standard_double_exponential_current_source(...)` | Same behavior |

---

## UMEC Factory Functions

| Old function | New function | Returns |
|-------------|-------------|---------|
| `create_umec_transformer_3ph_bank(...)` | `create_umec_transformer_3ph_bank_data(...)` | `UMECTransformerData` |
| _(new)_ | `create_umec_transformer_3ph_bank_instance(dt=..., ...)` | `UMECTransformer` |

The old function `create_umec_transformer_3ph_bank()` returns `UMECTransformerData`, not a transformer instance. The name was misleading. Use `create_umec_transformer_3ph_bank_data()` for clarity.

---

## Direction & Unit Conventions

See [DIRECTION_CONVENTIONS.md](DIRECTION_CONVENTIONS.md) for the full specification of:

- Branch voltage/current direction
- Norton equivalent RHS stamping
- Independent current source direction
- Transmission line port conventions
- UMEC port conventions
- LPM insulator voltage convention
- Per-step operation order
- Supported output units

---

## 模块状态

| 模块 | 状态 |
|--------|--------|
| `emtp_solver_v3.py` | ❌ 已删除（v0.3.1） |
| `emtp_components_series_rl_only.py` | ❌ 已删除（v0.3.1） |
| `transmission_line_emtp_v2.py` | ✅ 活跃 — Bergeron 模型 |
| `ulm_transmission_line_PARA.py` | ✅ 活跃 — ULM 模型 |
| `umec_transformer.py` | ✅ 活跃 — UMEC 模型 |
| `nonlinear_models_pscad.py` | ✅ 活跃 — 非线性模型 |
| `atp_lightning_current_generator_simplified.py` | ✅ 活跃 — 雷电电流源 |
| `emtp_plotting.py` | ❌ 死代码 — 未被任何文件导入 |
