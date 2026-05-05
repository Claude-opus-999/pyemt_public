"""Export waveform signals to CSV."""

import csv
from pathlib import Path

import numpy as np


def export_waveforms_csv(
    waveforms: dict,
    result_dir: str | Path,
    *,
    filename: str = "probes.csv",
    stride: int = 1,
) -> Path:
    """Write 1-D waveform signals to a CSV file.

    The first column is ``time_s``; subsequent columns are every other
    1-D signal in *waveforms*.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if "time_s" not in waveforms:
        raise ValueError("waveforms must contain 'time_s' for CSV export")

    time = np.asarray(waveforms["time_s"])[::stride]

    # Collect 1-D signal names (exclude time)
    one_d_names = []
    for n, v in waveforms.items():
        if n == "time_s":
            continue
        arr = np.asarray(v)
        if arr.ndim == 1:
            one_d_names.append(n)

    path = result_dir / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + one_d_names)

        for idx in range(len(time)):
            row = [time[idx]]
            for n in one_d_names:
                vals = np.asarray(waveforms[n])[::stride]
                row.append(vals[idx] if idx < len(vals) else "")
            writer.writerow(row)

    return path
