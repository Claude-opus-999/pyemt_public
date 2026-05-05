"""Case-config validation."""

from .defaults import SUPPORTED_ELEMENTS, SUPPORTED_SOURCES, SUPPORTED_PROBES


def validate_case_config(config) -> None:
    """Validate a :class:`CaseConfig`; raise ``ValueError`` on failure."""

    sim = config.simulation

    if sim.dt <= 0:
        raise ValueError("simulation.dt must be positive")
    if sim.finish_time <= 0:
        raise ValueError("simulation.finish_time must be positive")
    if sim.output_stride < 1:
        raise ValueError("simulation.output_stride must be >= 1")
    if sim.probe_stride < 1:
        raise ValueError("simulation.probe_stride must be >= 1")

    # -- elements ------------------------------------------------------------
    element_names: set = set()
    for elem in config.elements:
        kind = elem.get("kind", "")
        if kind not in SUPPORTED_ELEMENTS:
            raise ValueError(
                f"Unsupported element kind: {kind!r}. "
                f"Supported: {sorted(SUPPORTED_ELEMENTS)}"
            )
        name = elem.get("name", "")
        if not name:
            raise ValueError("Every element must have a non-empty 'name'")
        if name in element_names:
            raise ValueError(f"Duplicate element name: {name!r}")
        element_names.add(name)

    # -- sources -------------------------------------------------------------
    for src in config.sources:
        kind = src.get("kind", "")
        if kind not in SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source kind: {kind!r}. "
                f"Supported: {sorted(SUPPORTED_SOURCES)}"
            )

    # -- probes --------------------------------------------------------------
    for probe in config.probes:
        kind = probe.get("kind", "")
        if kind not in SUPPORTED_PROBES:
            raise ValueError(
                f"Unsupported probe kind: {kind!r}. "
                f"Supported: {sorted(SUPPORTED_PROBES)}"
            )
