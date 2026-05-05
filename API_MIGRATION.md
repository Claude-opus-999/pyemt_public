# API Migration Guide

This guide helps you migrate from the legacy flat-module imports to the new `emtp` package structure introduced in P3/P4.

The legacy imports continue to work — no existing scripts need to change immediately.

---

## Quick Reference

| Old import | New import |
|-----------|-----------|
| `from emtp_solver_v3 import EMTPSolver` | `from emtp import EMTPSolver` |
| `from emtp_components_series_rl_only import Branch` | `from emtp.types import Branch` |
| `from emtp_components_series_rl_only import CurrentSource` | `from emtp.types import CurrentSource` |
| `from emtp_components_series_rl_only import ElementType` | `from emtp.types import ElementType` |
| `from transmission_line_emtp_v2 import BergeronLine` | `from emtp.lines import BergeronLine` |
| `from ulm_transmission_line_PARA import ULMLine` | `from emtp.lines import ULMLine` |
| `from umec_transformer import UMECTransformer` | `from emtp.transformers import UMECTransformer` |
| `from atp_lightning_current_generator_simplified import ...` | `from emtp.sources import ...` |
| `from nonlinear_models_pscad import SegmentedMOAResistor` | `from emtp.nonlinear import SegmentedMOAResistor` |

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

## Legacy Module Status

| Module | Status |
|--------|--------|
| `emtp_solver_v3.py` | Legacy — re-exports from `emtp` package |
| `emtp_components_series_rl_only.py` | Legacy — re-exports from `emtp.types` |
| `transmission_line_emtp_v2.py` | Active — Bergeron model (unchanged) |
| `ulm_transmission_line_PARA.py` | Active — ULM model (unchanged) |
| `umec_transformer.py` | Active — UMEC model (enhanced with new factories) |
| `nonlinear_models_pscad.py` | Active — Nonlinear models (unchanged) |
| `atp_lightning_current_generator_simplified.py` | Active — Lightning sources (unchanged) |
| `emtp_plotting.py` | Active — Plotting utilities (unchanged) |
