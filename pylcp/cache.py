"""Content-hash cache key for LCP-generated fitULM files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .specs import LCPFitULMSpec


def _array_hash(arr: np.ndarray) -> str:
    """SHA-256 hash of the raw float64 bytes of an array."""
    x = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(x.tobytes()).hexdigest()[:16]


def _stable_repr(obj) -> str:
    """Stable JSON-serializable representation of an object."""
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(obj)


def compute_cache_key(spec: LCPFitULMSpec) -> str:
    """Content-based cache key for a :class:`LCPFitULMSpec`.

    Two specs with identical geometry, soil, frequency, length, VF config,
    and precision produce the same key.
    """
    payload = {
        "line_type": spec.line_type.value,
        "name": spec.name,
        "length": float(spec.length),
        "freq_hash": _array_hash(spec.freq),
        "geometry": _stable_repr(spec.geometry_config),
        "soil": _stable_repr(spec.soil_config),
        "vf": _stable_repr(spec.vf_config),
        "precision": spec.precision,
        "use_freq_dependent": spec.use_freq_dependent,
        "enforce_passivity": spec.enforce_passivity,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def get_cache_path(spec: LCPFitULMSpec) -> Path:
    """Auto-generated cache file path for *spec*."""
    key = compute_cache_key(spec)
    return Path(spec.cache_dir) / f"{spec.name}_{key}.fitULM"
