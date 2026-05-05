"""Export scalar metrics to JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def export_metrics_json(
    metrics: Dict[str, Any],
    result_dir: str | Path,
    *,
    filename: str = "metrics.json",
) -> Path:
    """Write a metrics dict to *result_dir / filename*.

    Non-JSON-serializable values are converted to strings.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    safe: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)

    path = result_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)
    return path
