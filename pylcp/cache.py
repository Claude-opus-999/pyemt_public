"""Content-hash cache key for LCP-generated fitULM files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .specs import LCPFitULMSpec


def _get_pylcp_version() -> str:
    try:
        import pylcp
        return str(getattr(pylcp, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _get_lcp_version() -> str:
    try:
        import LCP
        return str(getattr(LCP, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _array_hash(arr: np.ndarray) -> str:
    """SHA-256 hash of the raw float64 bytes of an array."""
    x = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(x.tobytes()).hexdigest()[:16]


def _normalize(obj) -> str:
    """Stable JSON-serializable representation of an object."""
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(obj)


def compute_cache_key(spec: LCPFitULMSpec) -> str:
    """Content-based cache key for a :class:`LCPFitULMSpec`.

    Two specs with identical geometry, soil, frequency, length, VF config,
    precision, and pylcp/LCP versions produce the same key.
    """
    payload = {
        "schema_version": 2,
        "pylcp_version": _get_pylcp_version(),
        "lcp_version": _get_lcp_version(),
        "line_type": spec.line_type.value,
        "name": spec.name,
        "length": float(spec.length),
        "freq_hash": _array_hash(spec.freq),
        "geometry_config": _normalize(getattr(spec, "geometry_config", None)),
        "soil_config": _normalize(getattr(spec, "soil_config", None)),
        "vf_config": _normalize(getattr(spec, "vf_config", None)),
        "precision": getattr(spec, "precision", 16),
        "use_freq_dependent": str(getattr(spec, "use_freq_dependent", "auto")),
        "enforce_passivity": bool(getattr(spec, "enforce_passivity", True)),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def get_cache_path(spec: LCPFitULMSpec) -> Path:
    """Auto-generated cache file path for *spec*."""
    key = compute_cache_key(spec)
    return Path(spec.cache_dir) / f"{spec.name}_{key}.fitULM"
