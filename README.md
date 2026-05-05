# PyEMTP — Python Electromagnetic Transients Program

## Recommended Import

```python
from emtp import EMTPSolver
```

## Legacy Import (still supported)

```python
from emtp_solver_v3 import EMTPSolver
```

## Quick Start

```python
from emtp import EMTPSolver

solver = EMTPSolver(dt=1e-6, finish_time=100e-6)
solver.add_VS("Vs", 1, 0, 10.0)
solver.add_R("R1", 1, 0, 100.0)
solver.add_voltage_probe("V1", 1, 0)
solver.run()

t = solver.get_time("us")
v = solver.get_voltage_probe("V1", "V")
```

## Internal Architecture

```
emtp.solver.EMTPSolver
  ├── emtp.nodes          (NodeIndexer, NodeBook)
  ├── emtp.types          (ElementType, Branch, CurrentSource, ...)
  ├── emtp.devices        (R, L, C, Switch, SeriesRL, Nonlinear, LPM)
  ├── emtp.stamping       (COOStamper, StampingEngine)
  ├── emtp.sparse_solver  (SuperLU wrapper)
  ├── emtp.runtime        (DynamicDeviceRuntime)
  ├── emtp.validation     (circuit validation)
  └── emtp.results        (result helpers)
```

## Documentation

- [API Migration Guide](API_MIGRATION.md) — old → new import paths
- [Direction Conventions](DIRECTION_CONVENTIONS.md) — sign, unit, and stamping conventions
- [Solver Architecture](EMTP_SOLVER_ARCHITECTURE.md) — detailed internal design
