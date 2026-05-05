"""ResultDatabase — lightweight SQLite store for simulation run history.

Usage::

    db = ResultDatabase("results.sqlite")
    db.insert_run("run_001", "rc_step", "running", Path("runs/run_001"))
    db.update_run_done("run_001", elapsed_s=0.5)
    db.insert_metrics("run_001", {"V_cap_peak": 1.05})
    runs = db.list_recent_runs(10)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS simulation_runs (
    id TEXT PRIMARY KEY,
    case_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    result_dir TEXT,
    config_path TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    elapsed_s REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL,
    text_value TEXT,
    unit TEXT,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
);

CREATE TABLE IF NOT EXISTS waveform_signals (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT,
    unit TEXT,
    length INTEGER,
    min_value REAL,
    max_value REAL,
    peak_abs REAL,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES simulation_runs(id)
);
"""


class ResultDatabase:
    """Manage a SQLite database of simulation runs and their results."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # -- runs ----------------------------------------------------------------

    def insert_run(
        self,
        run_id: str,
        case_name: str,
        status: str,
        result_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO simulation_runs
               (id, case_name, status, result_dir, config_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id, case_name, status,
                str(result_dir) if result_dir else None,
                str(config_path) if config_path else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def update_run_done(self, run_id: str, elapsed_s: float) -> None:
        self.conn.execute(
            """UPDATE simulation_runs
               SET status='done', finished_at=?, elapsed_s=?
               WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), elapsed_s, run_id),
        )
        self.conn.commit()

    def update_run_failed(self, run_id: str, error: str) -> None:
        self.conn.execute(
            """UPDATE simulation_runs
               SET status='failed', finished_at=?, error=?
               WHERE id=?""",
            (datetime.now(timezone.utc).isoformat(), error, run_id),
        )
        self.conn.commit()

    def list_recent_runs(self, limit: int = 20) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM simulation_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM simulation_runs WHERE id=?", (run_id,),
        ).fetchone()
        return dict(row) if row else None

    # -- metrics -------------------------------------------------------------

    def insert_metrics(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        units: Optional[Dict[str, str]] = None,
    ) -> None:
        units = units or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.conn.execute(
                    """INSERT OR REPLACE INTO run_metrics
                       (run_id, key, value, unit)
                       VALUES (?, ?, ?, ?)""",
                    (run_id, key, float(value), units.get(key)),
                )
            else:
                self.conn.execute(
                    """INSERT OR REPLACE INTO run_metrics
                       (run_id, key, text_value, unit)
                       VALUES (?, ?, ?, ?)""",
                    (run_id, key, str(value), units.get(key)),
                )
        self.conn.commit()

    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value, text_value, unit FROM run_metrics WHERE run_id=?",
            (run_id,),
        ).fetchall()
        result = {}
        for r in rows:
            result[r["key"]] = (
                r["value"] if r["value"] is not None else r["text_value"]
            )
        return result

    # -- waveform signals ---------------------------------------------------

    def insert_signals(
        self, run_id: str, signals: List[Dict[str, Any]],
    ) -> None:
        for sig in signals:
            self.conn.execute(
                """INSERT OR REPLACE INTO waveform_signals
                   (run_id, name, kind, unit, length, min_value, max_value, peak_abs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    sig.get("name", ""),
                    sig.get("kind"),
                    sig.get("unit"),
                    sig.get("length"),
                    sig.get("min_value"),
                    sig.get("max_value"),
                    sig.get("peak_abs"),
                ),
            )
        self.conn.commit()

    def get_signals(self, run_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM waveform_signals WHERE run_id=?", (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
