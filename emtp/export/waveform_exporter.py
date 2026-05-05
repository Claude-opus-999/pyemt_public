"""Export time-series waveforms to NPZ with rich metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def export_waveforms_npz(
    waveforms: Dict[str, object],
    result_dir: str | Path,
    *,
    stride: int = 1,
    signal_specs: Optional[dict] = None,
    flatten: bool = False,
) -> Path:
    """Write waveforms to ``waveforms.npz`` and ``waveform_metadata.json``.

    Parameters
    ----------
    waveforms:
        Dict mapping signal name → ndarray or list.
    result_dir:
        Output directory (created if needed).
    stride:
        Downsampling factor.  ``stride=10`` keeps every 10th sample.
    signal_specs:
        Optional ``{name: {kind, unit}}`` hints derived from config probes.
    flatten:
        When ``True``, ravel multi-dimensional signals (legacy behaviour).
        When ``False``, raise ``ValueError`` for ndim > 2 signals.

    Returns
    -------
    Path
        Path to the NPZ file.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    specs = signal_specs or {}
    arrays = {}
    metadata: dict = {"signals": [], "stride": stride}

    for name, values in waveforms.items():
        arr = np.asarray(values)
        original_shape = list(arr.shape)

        # -- downsample along last axis (time) --------------------------------
        if arr.ndim == 1:
            arr_ds = arr[::stride]
        elif arr.ndim == 2:
            arr_ds = arr[..., ::stride]
        else:
            if flatten:
                arr_ds = arr.ravel()[::stride]
            else:
                raise ValueError(
                    f"Waveform {name!r} has unsupported shape {arr.shape}; "
                    "pass flatten=True or export components separately."
                )

        arrays[name] = arr_ds

        spec = specs.get(name, {})
        metadata["signals"].append({
            "name": name,
            "kind": spec.get("kind", _infer_signal_kind(name)),
            "unit": spec.get("unit", _infer_signal_unit(name)),
            "length": int(arr_ds.shape[-1]) if arr_ds.ndim >= 1 else 0,
            "shape": list(arr_ds.shape),
            "original_shape": original_shape,
            "flattened": bool(flatten and arr.ndim > 1),
            "min": float(np.nanmin(arr_ds)) if arr_ds.size else 0.0,
            "max": float(np.nanmax(arr_ds)) if arr_ds.size else 0.0,
            "peak_abs": float(np.nanmax(np.abs(arr_ds))) if arr_ds.size else 0.0,
        })

    np.savez_compressed(result_dir / "waveforms.npz", **arrays)

    with (result_dir / "waveform_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return result_dir / "waveforms.npz"


def collect_waveform_metadata(result_dir: str | Path) -> dict:
    """Read waveform_metadata.json from *result_dir*."""
    with (Path(result_dir) / "waveform_metadata.json").open(encoding="utf-8") as f:
        return json.load(f)


def read_waveform_chunk(
    result_dir: str | Path,
    signal: str,
    start: int = 0,
    count: int = 1000,
) -> dict:
    """Read a chunk of a waveform signal from an NPZ result directory.

    Returns ``{"signal", "start", "count", "time", "values"}``.
    """
    result_dir = Path(result_dir)
    npz_path = result_dir / "waveforms.npz"

    with np.load(npz_path) as data:
        time_full = data.get("time_s", np.array([]))
        values_full = data.get(signal, np.array([]))

    end = min(start + count, len(values_full))
    return {
        "signal": signal,
        "start": start,
        "count": end - start,
        "time": time_full[start:end].tolist(),
        "values": values_full[start:end].tolist(),
    }


# ---------------------------------------------------------------------------
# Signal kind / unit inference
# ---------------------------------------------------------------------------

def _infer_signal_kind(name: str) -> str:
    lower = name.lower()
    if lower in {"time", "time_s", "t"}:
        return "time"
    if lower.startswith("v_") or "voltage" in lower:
        return "voltage"
    if lower.startswith("i_") or "current" in lower:
        return "current"
    if "leader" in lower:
        return "leader_length"
    return "other"


def _infer_signal_unit(name: str) -> str:
    lower = name.lower()
    if lower in {"time", "time_s"}:
        return "s"
    if lower.endswith("_kv"):
        return "kV"
    if lower.endswith("_v"):
        return "V"
    if lower.endswith("_ka"):
        return "kA"
    if lower.endswith("_a"):
        return "A"
    if lower.endswith("_mm"):
        return "mm"
    return ""
