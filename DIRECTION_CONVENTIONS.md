# Direction and Sign Conventions

This document defines the electrical sign conventions used throughout the EMTP solver. All device implementations, MNA stamping, probes, and result queries follow these rules.

---

## 1. Branch Voltage and Current

```
    node_from o──────[ branch ]──────o node_to
                  i_branch →
```

**Branch voltage:**

```
v_branch = V(node_from) - V(node_to)
```

**Branch current** (positive direction):

```
i_branch flows from node_from to node_to
```

---

## 2. Norton Equivalent

Every dynamic branch is represented at each time step by a Norton equivalent:

```
i_branch = Geq · v_branch + Ihist
```

| Term | Meaning |
|------|---------|
| `Geq` | Equivalent conductance (S) |
| `Ihist` | History current source (A) |
| `v_branch` | Branch voltage (V) |
| `i_branch` | Branch current (A) |

**RHS stamping:**

```
rhs[node_from] -= Ihist
rhs[node_to]   += Ihist
```

This convention is used by:

- `InductorDevice`
- `CapacitorDevice`
- `SeriesRLDevice`
- `NonlinearResistorDevice`
- `LPMFlashoverDevice`
- Transmission line Norton current sources
- UMEC port Norton current sources

---

## 3. Independent Current Source

```
    node_from o─────→ I ─────o node_to
```

**Positive direction:** `node_from` → `node_to`

**RHS stamping:**

```
rhs[node_from] -= I(t)
rhs[node_to]   += I(t)
```

**Example** — injecting +100 kA from ground into node 1:

```python
solver.add_IS("I1", node_from=0, node_to=1, current_func=lambda t: 100e3)
```

---

## 4. Ideal Voltage Source

```
    node_pos o───[ +  VS  - ]───o node_neg
              ←─── i_vs ────
```

**MNA augmented system:** row `n + k` enforces:

```
V(node_pos) - V(node_neg) = voltage_func(t)
```

**VS current** is read from the MNA solution vector:

```python
vs.current = -x[n + k]   # positive: node_pos → external circuit → node_neg
```

---

## 5. Nonlinear Resistor (MOA)

The segmented nonlinear resistor follows the same branch conventions:

```
v_branch = V(node_from) - V(node_to)
i_branch = nonlinear_model.get_current(v_branch)
```

For symmetric V-I curves (MOA arresters), segment detection checks `abs(v)` and applies:

```
i = sign(v) · segment_current(abs(v))
```

The Norton equivalent is updated by `SegmentedSolverHelper` during segment switches.

---

## 6. Transmission Lines

```
    node_k o──────[ line ]──────o node_m
       I_k →                    → I_m
```

**Port currents** (injected into nodes):

- `I_hist_k`: current injected into `node_k`
- `I_hist_m`: current injected into `node_m`

**RHS stamping:**

```
rhs[node_k] -= I_hist_k
rhs[node_m] -= I_hist_m
```

**Port voltages** (read from MNA solution):

```
V_k = V(node_k)
V_m = V(node_m)
```

---

## 7. UMEC Transformer Ports

Each port is a pair `(node_from, node_to)`. Port voltage:

```
V_port[k] = V(node_from_k) - V(node_to_k)
```

Port Norton current sources are stamped into the RHS following the same convention as dynamic branches:

```
rhs[node_from] -= I_hist_port[k]
rhs[node_to]   += I_hist_port[k]
```

---

## 8. LPM Insulator Flashover

The LPM insulator voltage is the branch voltage across the switch device:

```
v_gap = V(node_from) - V(node_to)
```

The LPM model tracks leader propagation using **kV** internally. The solver provides `v_gap` in **V** and the LPM model handles the conversion.

After flashover, the branch resistance switches from `R_open` to `R_arc`:

```python
br.is_closed = bool(lpm.is_flashed_over)
br.value = lpm.R_current      # R_arc after flashover
br.Geq = lpm.G_current        # 1 / R_current
```

---

## 9. Per-Step Operation Order

The solver main loop executes operations in this fixed order:

```
1. step_pre_solve(t)          — timed switch events → mark_dirty if needed
2. _solve_step()              — MNA assemble → SuperLU solve → resolve check
3. step_post_solve_V_I(V)     — update branch V/I (MUST be before probes)
4. _record_probes(V)          — record voltage/current probes
5. _update_lines_combined(V)  — transmission line state update
6. step_post_solve_history()  — advance L/C/SRL history sources
7. _update_transformer_history(V) — advance UMEC history sources
```

**Critical:** steps 3–4–5–6–7 must remain in this order. `CapacitorDevice.update_history()` flips `Ihist` sign, so probes must read the *current-step* `Ihist` before the flip.

---

## 10. Unit Conventions

| Quantity | Base unit | Supported output units |
|----------|-----------|------------------------|
| Time | s | s, ms, us, µs, ns |
| Voltage | V | V, kV, mV, MV |
| Current | A | A, kA, mA |
| Length (LPM gaps) | m | m, mm, cm |
| Power (transformers) | VA | MVA, kVA |

Unit conversion is handled by the result-query methods:

```python
solver.get_time("us")
solver.get_node_voltage(1, unit="kV")
solver.get_branch_current("R1", unit="kA")
```
