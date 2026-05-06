# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PyEMTP is a Python electromagnetic transients simulation solver using Modified Nodal Analysis (MNA). It integrates multi-phase transmission lines, nonlinear components (MOA arresters, LPM flashover), UMEC transformers, and lightning current sources.

**Stack**: Python 3.12+, numpy, scipy (sparse SuperLU). Optional: numba (ULM batch).

## Commands

```bash
# Run all tests (skip slow tower case)
pytest tests/ -q --ignore=tests/test_tower_case_p1.py

# Run a single test file
pytest tests/test_pr1_fitulm_resolver.py -v

# Run LCP-specific tests
pytest tests/pylcp_tests/ -v
pytest tests/test_baseline_lcp_emtp.py -v

# Syntax check new/modified modules
python -m py_compile emtp/lines/fitulm_resolver.py
python -m py_compile pylcp/*.py
python -m py_compile pylcp/generation/*.py
```

## Architecture (four-layer, top-down)

```
Layer 3: High-level pipeline
  config/   builders/   snapshot/   export/
  case_runner.py   result_bundle.py   result_db.py   run_id.py

Layer 2: Modular subpackages
  devices/   assembly/   runtime/   results/
  lines/   transformers/   sources/   nonlinear/

Layer 1: Core solver (emtp/)
  solver.py (~3660 lines — the monolith)
  types.py   nodes.py   circuit.py   sparse_solver.py   stamping.py   validation.py

Layer 0: Standalone physics libraries (top-level .py files)
  transmission_line_emtp_v2.py   ulm_transmission_line_PARA.py
  nonlinear_models_pscad.py      umec_transformer.py
  atp_lightning_current_generator_simplified.py
```

**Dependency direction**: Layer 3 → 2 → 1 → 0 (never reverse).

**Two entry points**:
- `from emtp import EMTPSolver` — programmatic API (solver.py)
- `from emtp.case_runner import run_case` — JSON config-driven pipeline

**LCP integration** (v0.3.2+): `LCP/` contains line-constants physics (cable Z/Y, overhead line Z/Y, Vector Fitting engine). `pylcp/` wraps it for solver integration via `solver.add_ULM_line()` with two modes:
- External file: `generate_fitulm=False, fitulm_path="file.fitULM"`
- Auto-generation: `generate_fitulm=True, lcp_spec=LCPFitULMSpec(...)` — length defaults to `lcp_spec.length`

**Key files for ULM/LCP flow**: `emtp/lines/fitulm_resolver.py` (FitULMSpec + FitULMResolver), `pylcp/lcp_fitulm_generator.py` (LCPFitULMGenerator), `pylcp/cache.py` (content-hash cache keys), `pylcp/generation/_soil.py` (shared soil params).

## MNA sign conventions

- Branch current positive direction: `node_from → node_to`
- Branch voltage: `V(node_from) - V(node_to)`
- Norton equivalent: `i = Geq · v + Ihist`
- RHS stamping: `rhs[pos] -= Ihist`, `rhs[neg] += Ihist`
- Current source injection: `rhs[pos] -= I`, `rhs[neg] += I`
- Ground node is 0 — never write to `rhs[0]` or matrix row/col for node 0

## Key patterns

- **Device protocol**: Every two-terminal element implements `stamp_G`, `stamp_rhs`, `update_branch_quantities`, `update_history`. Defined in `emtp/devices/base.py`.
- **MultiPortDevice protocol**: Multi-port elements (Bergeron, ULM, UMEC) implement `stamp_G`, `stamp_rhs`, `update_from_solution`, `advance_history`, `get_resolve_event`. Defined in `emtp/devices/multiport.py`.
- **Optional Layer 0 imports**: All Layer 0 libraries are imported with try/except, falling back to `None` stubs. This allows the solver to run without all physics libraries present.
- **fitULM verification**: `_verify_fitulm()` checks existence, non-empty, and runs `verify_fitULM_file()` if LCP is available. Only catches `ImportError` (LCP missing) — real errors propagate. Both external-file and LCP-generated paths go through this check.
- **Cache key**: `compute_cache_key()` in `pylcp/cache.py` hashes geometry, soil, frequency, VF config, and pylcp/LCP version fields. Path: `{cache_dir}/{name}_{hash}.fitULM`. Outer `FitULMSpec.cache_dir` always overrides `lcp_spec.cache_dir`.

## Behavioral guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
