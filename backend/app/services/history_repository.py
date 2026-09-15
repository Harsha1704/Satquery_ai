"""SQLite-backed analysis history for the release-candidate workspace."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


class AnalysisHistoryRepository:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self):
        with self._lock, self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    job_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    intent TEXT,
                    targets_json TEXT NOT NULL DEFAULT '[]',
                    before_year INTEGER,
                    after_year INTEGER,
                    sensor TEXT,
                    status TEXT NOT NULL,
                    changed_area_km2 REAL,
                    changed_share_pct REAL,
                    evidence_quality TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    report_url TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_analysis_history_completed ON analysis_history(completed_at DESC)")
            con.commit()

    @staticmethod
    def _float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def record(self, job) -> None:
        data = job.model_dump(mode="json") if hasattr(job, "model_dump") else dict(job)
        result = data.get("result") or {}
        parsed = (data.get("plan") or {}).get("parsed") or {}
        stats = result.get("statistics") or {}
        layer = stats.get("task_layer") or {}
        validation = stats.get("validation") or {}
        provenance = result.get("provenance") or {}
        imagery = provenance.get("imagery") or []
        sensor = " / ".join(dict.fromkeys(str(x.get("collection_id") or "") for x in imagery if x.get("collection_id"))) or None
        years = parsed.get("years") or []
        values = (
            data.get("job_id"),
            str(parsed.get("query") or ""),
            str(parsed.get("intent") or ""),
            json.dumps(parsed.get("targets") or []),
            int(years[0]) if len(years) > 0 else None,
            int(years[1]) if len(years) > 1 else None,
            sensor,
            str(data.get("status") or "completed"),
            self._float(layer.get("affected_area_km2")),
            self._float(layer.get("changed_percentage")),
            str(validation.get("quality") or "Not reported"),
            data.get("started_at"),
            data.get("completed_at"),
            f"/api/v1/jobs/{data.get('job_id')}/report",
        )
        with self._lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO analysis_history (
                    job_id, query, intent, targets_json, before_year, after_year,
                    sensor, status, changed_area_km2, changed_share_pct,
                    evidence_quality, started_at, completed_at, report_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    changed_area_km2=excluded.changed_area_km2,
                    changed_share_pct=excluded.changed_share_pct,
                    evidence_quality=excluded.evidence_quality,
                    completed_at=excluded.completed_at,
                    sensor=excluded.sensor
                """,
                values,
            )
            con.commit()

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT * FROM analysis_history ORDER BY completed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            try:
                item["targets"] = json.loads(item.pop("targets_json") or "[]")
            except Exception:
                item["targets"] = []
            output.append(item)
        return output
