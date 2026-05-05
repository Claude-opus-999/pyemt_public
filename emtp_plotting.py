"""
Lightweight probe registration and plotting utilities for EMTPSolver.

This module keeps probe usage and probe plotting in one file.

Typical usage
-------------
from emtp_probes import (
    add_voltage_probe,
    add_branch_current_probe,
    plot_voltage_probes,
    plot_current_probes,
)

add_voltage_probe(solver, "V_Cp_0km", 1, 0)
add_branch_current_probe(solver, "I_R_C1k_Splus", "R_C1k_S+")

solver.run()

plot_voltage_probes(solver, ["V_Cp_0km"], unit="kV")
plot_current_probes(solver, ["I_R_C1k_Splus"], unit="kA")
"""

from __future__ import annotations

from typing import Optional, Sequence



def _pretty_label(name: str) -> str:
    """Convert internal probe names to readable plot labels."""
    return str(name).replace("_", " ")


# ---------------------------------------------------------------------------
# Probe registration wrappers
# ---------------------------------------------------------------------------

def add_voltage_probe(
    solver,
    name: str,
    node_pos,
    node_neg=0,
) -> None:
    """Register a voltage probe on solver."""
    solver.add_voltage_probe(name, node_pos, node_neg)


def add_branch_current_probe(
    solver,
    name: str,
    branch_name: str,
) -> None:
    """Register a normal branch-current probe on solver."""
    solver.add_branch_current_probe(name, branch_name)


# ---------------------------------------------------------------------------
# Probe plotting
# ---------------------------------------------------------------------------

def plot_probes(
    solver,
    probes: Sequence[str],
    *,
    unit: str = "kV",
    time_unit: str = "us",
    title: str = "Probe Waveforms",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (9, 5),
) -> None:
    """
    Plot selected probe waveforms.

    Parameters
    ----------
    solver:
        EMTPSolver object after solver.run().
    probes:
        Probe names registered by add_voltage_probe() or
        add_branch_current_probe().
    unit:
        Output unit, for example "V", "kV", "A", "kA".
    time_unit:
        Time unit: "s", "ms", "us", or "ns".
    """
    import matplotlib.pyplot as plt

    t = solver.get_time(time_unit)

    plt.figure(figsize=figsize)

    for probe in probes:
        y = solver.get_probe(probe, unit=unit)
        plt.plot(t, y, label=_pretty_label(probe))

    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel(ylabel or f"Value ({unit})")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_voltage_probes(
    solver,
    probes: Sequence[str],
    *,
    unit: str = "kV",
    time_unit: str = "us",
    title: str = "Voltage Probes",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (9, 5),
) -> None:
    """Plot selected voltage probes."""
    plot_probes(
        solver,
        probes,
        unit=unit,
        time_unit=time_unit,
        title=title,
        ylabel=ylabel or f"Voltage ({unit})",
        save_path=save_path,
        show=show,
        figsize=figsize,
    )


def plot_current_probes(
    solver,
    probes: Sequence[str],
    *,
    unit: str = "kA",
    time_unit: str = "us",
    title: str = "Current Probes",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[float, float] = (9, 5),
) -> None:
    """Plot selected branch-current probes."""
    plot_probes(
        solver,
        probes,
        unit=unit,
        time_unit=time_unit,
        title=title,
        ylabel=ylabel or f"Current ({unit})",
        save_path=save_path,
        show=show,
        figsize=figsize,
    )


__all__ = [
    "add_voltage_probe",
    "add_branch_current_probe",
    "plot_probes",
    "plot_voltage_probes",
    "plot_current_probes",
]
