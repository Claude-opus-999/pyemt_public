"""Build an EMTPSolver instance from a CaseConfig."""

from emtp import EMTPSolver

from .element_builder import add_element_to_solver
from .source_builder import add_source_to_solver
from .probe_builder import add_probe_to_solver


def build_solver_from_config(config):
    """Create and configure an :class:`EMTPSolver` from *config*.

    Parameters
    ----------
    config : CaseConfig
        Fully-loaded case configuration.

    Returns
    -------
    EMTPSolver
        Solver instance with all elements, sources and probes added.
    """
    sim = config.simulation

    solver = EMTPSolver(
        dt=sim.dt,
        finish_time=sim.finish_time,
        verbose=sim.verbose,
        record_all_node_voltages=sim.record_all_node_voltages,
        record_line_history=sim.record_line_history,
        record_branch_history=sim.record_branch_history,
        record_source_history=sim.record_source_history,
        pre_sample_sources=sim.pre_sample_sources,
        use_rhs_plan=sim.use_rhs_plan,
    )

    for element in config.elements:
        add_element_to_solver(solver, element)

    for source in config.sources:
        add_source_to_solver(solver, source)

    for probe in config.probes:
        add_probe_to_solver(solver, probe)

    return solver
