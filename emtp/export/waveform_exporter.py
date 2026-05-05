"""Export time-series waveforms to NPZ with metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def export_waveforms_npz(
    waveforms: Dict[str, object],
    result_dir: str | Path,
    *,
    stride: int = 1,
) -> Path:
    """Write waveforms to ``waveforms.npz`` and ``waveform_metadata.json``.

    Parameters
    ----------
    waveforms:
        Dict mapping signal name → 1-D ndarray or list.
    result_dir:
        Output directory (created if needed).
    stride:
        Downsampling factor.  ``stride=10`` keeps every 10th sample.

    Returns
    -------
    Path
        Path to the NPZ file.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata: Dict[str, object] = {
        "signals": [],
        "stride": stride,
    }

    for name, values in waveforms.items():
        arr = np.asarray(values).ravel()
        arr_ds = arr[::stride]

        arrays[name] = arr_ds
        metadata["signals"].append({
            "name": name,
            "length": int(arr_ds.shape[0]),
            "min": float(np.nanmin(arr_ds)) if len(arr_ds) else 0.0,
            "max": float(np.nanmax(arr_ds)) if len(arr_ds) else 0.0,
            "peak_abs": float(np.nanmax(np.abs(arr_ds))) if len(arr_ds) else 0.0,
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
