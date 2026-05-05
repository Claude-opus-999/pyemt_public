"""Default simulation options and supported element/source/probe kinds."""

SUPPORTED_ELEMENTS = {
    "resistor",
    "inductor",
    "capacitor",
    "series_rl",
    "switch",
    "bergeron_line",
    "lpm_insulator",
    "moa_arrester",
    "umec_transformer",
}

SUPPORTED_SOURCES = {
    "current",
    "voltage",
    "standard_double_exponential_current",
    "lightning_current",
}

SUPPORTED_PROBES = {
    "voltage",
    "branch_current",
}

DEFAULT_SIMULATION = {
    "verbose": False,
    "record_all_node_voltages": False,
    "record_line_history": True,
    "record_branch_history": True,
    "record_source_history": True,
    "pre_sample_sources": True,
    "use_rhs_plan": True,
    "output_stride": 1,
    "probe_stride": 1,
}
