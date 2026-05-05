"""Deterministic, sortable, unique run IDs."""

from datetime import datetime, timezone
from uuid import uuid4


def make_run_id(prefix: str = "run") -> str:
    """Return a unique run ID like ``rc_step_20260505_143022_a1b2c3d4``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:8]
    prefix = prefix.replace(" ", "_").replace("/", "_")
    return f"{prefix}_{ts}_{short}"
