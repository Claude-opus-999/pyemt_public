"""Case / Config layer — JSON-based simulation configuration.

Usage::

    from emtp.config import load_case_config, CaseConfig, SimulationOptions
    config = load_case_config("cases/templates/rc_step.json")
"""

from .schema import CaseConfig, SimulationOptions      # noqa: F401
from .loader import load_case_config                     # noqa: F401
from .validator import validate_case_config              # noqa: F401
