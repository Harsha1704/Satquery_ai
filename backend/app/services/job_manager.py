"""Bounded background job manager with persisted progress and cancellation.

Phase 7 RC1.1 hardening notes
-----------------------------
Windows can temporarily deny ``os.replace`` on a JSON state file while virus
scanners, indexers, or another reader have the destination open. Job progress
must not fail just because its persistence file is briefly locked.

The manager therefore keeps active state in memory as the authoritative state
for the current process and persists snapshots with unique temporary files,
retry/backoff, and best-effort durability. A transient persistence problem is
logged but never converts a successful analysis into a failed analysis.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from query_engine.policy import QueryError

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_STAGE_PERCENT = {
    "queued": 2,
    "preparing": 10,
    "retrieving_imagery": 22,
    "preprocessing": 34,
    "analyzing": 52,
    "generating_evidence": 70,
    "validating": 83,
    "finalizing": 93,
    "completed": 100,
    "failed": 100,
    "cancelled": 100,
}


class AnalysisJobManager:
    def __init__(self, service, history_repository=None, max_workers: int = 2, max_queue: int = 6):
        self.service = service
        self.history = history_repository
        self.max_workers = max(1, int(max_workers))
        self.max_queue = max(0, int(max_queue))
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="satquery-analysis")
        self._capacity = threading.BoundedSemaphore(self.max_workers + self.max_queue)
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._futures = {}
        # In-process source of truth for jobs owned by this server process.
        # Disk remains the recovery/audit snapshot, but disk contention must not
        # break a running analysis.
        self._states: dict[str, dict[str, Any]] = {}
        self._recover_orphaned_jobs()

    def _state_path(self, job_id: str) -> Path:
        return self.service.job_root / job_id / "state.json"

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, Any], *, attempts: int = 12) -> bool:
        """Persist JSON atomically with Windows-friendly retry/backoff.

        A unique temp file prevents writers from contending on ``state.tmp``.
        ``os.replace`` is retried because Windows may transiently return
        ``WinError 5`` when the destination is inspected by another process.
        Returns False only after exhausting retries; callers keep the live
        in-memory state and log the degraded persistence condition.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, allow_nan=False)
        last_exc: Exception | None = None

        for attempt in range(max(1, attempts)):
            tmp = path.parent / f".{path.name}.{uuid4().hex}.tmp"
            try:
                with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        # Some filesystems do not support fsync; atomic replace
                        # still gives the required consistency semantics.
                        pass
                os.replace(str(tmp), str(path))
                return True
            except PermissionError as exc:
                last_exc = exc
                # 20ms .. 240ms; long enough for Windows scanners/indexers to
                # release a transient handle without materially delaying jobs.
                time.sleep(0.02 * (attempt + 1))
            except OSError as exc:
                last_exc = exc
                # Sharing violations can surface as generic OSError depending
                # on the Python/Windows build. Retry a small number of times.
                time.sleep(0.02 * (attempt + 1))
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

        logger.error("Could not persist job state %s after %s attempts: %s", path, attempts, last_exc)
        return False

    def _recover_orphaned_jobs(self) -> None:
        """Mark non-terminal persisted jobs from a previous process as failed.

        In-memory workers cannot survive a process restart; leaving them as
        forever-running would be dishonest. Durable distributed workers can
        replace this policy in a multi-node deployment.
        """
        root = self.service.job_root
        if not root.exists():
            return
        for path in root.glob("ana_*/state.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("status") in {"queued", "running"}:
                    state.update({
                        "status": "failed",
                        "stage": "failed",
                        "progress": 100,
                        "error_code": "server_restarted",
                        "message": "The server restarted while this analysis was active. Re-run the analysis; no partial result was promoted as final.",
                        "updated_at": _now(),
                    })
                    self._atomic_json_write(path, state)
            except Exception:
                logger.exception("Could not reconcile persisted job state %s", path)

    def _write(self, state: dict[str, Any]) -> bool:
        path = self._state_path(state["job_id"])
        ok = self._atomic_json_write(path, state)
        if not ok:
            # The live job remains available from memory. Surface this only as
            # an internal durability signal, not as an analysis failure.
            state["state_persistence_degraded"] = True
        return ok

    def _read_disk(self, job_id: str) -> dict[str, Any]:
        path = self._state_path(job_id)
        if not path.is_file():
            # Backward compatibility for synchronous completed jobs.
            try:
                job = self.service.get_job(job_id)
                return {
                    "job_id": job_id,
                    "status": job.status,
                    "stage": job.status,
                    "progress": 100,
                    "message": "Analysis completed." if job.status == "completed" else (job.error or "Analysis failed."),
                    "created_at": job.started_at,
                    "updated_at": job.completed_at,
                    "job": job.model_dump(mode="json"),
                }
            except Exception as exc:
                raise QueryError("job_not_found", "Job not found.", 404) from exc
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A state file should always be complete because writes are atomic.
            # If external tooling interferes, fail with a stable API error rather
            # than leaking an implementation exception.
            raise QueryError("job_state_unavailable", "Job state is temporarily unavailable. Retry shortly.", 503) from exc

    def _read(self, job_id: str) -> dict[str, Any]:
        if not job_id.startswith("ana_") or len(job_id) != 36:
            raise QueryError("job_not_found", "Job not found.", 404)
        with self._lock:
            live = self._states.get(job_id)
            if live is not None:
                return copy.deepcopy(live)
        return self._read_disk(job_id)

    def _append_event(self, state: dict[str, Any]) -> None:
        event_path = self.service.job_root / state["job_id"] / "events.jsonl"
        event = {
            "timestamp": state.get("updated_at") or _now(),
            "job_id": state.get("job_id"),
            "status": state.get("status"),
            "stage": state.get("stage"),
            "progress": state.get("progress"),
            "message": state.get("message"),
            "error_code": state.get("error_code"),
        }
        try:
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        except OSError:
            # Observability must not make the user analysis fail.
            logger.exception("Could not append lifecycle event for %s", event["job_id"])
        logger.info(
            "job=%s status=%s stage=%s progress=%s",
            event["job_id"], event["status"], event["stage"], event["progress"],
        )

    def _update(self, job_id: str, **changes) -> dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self._states.get(job_id)) if job_id in self._states else self._read_disk(job_id)
            state.update(changes)
            state["updated_at"] = _now()
            if "stage" in changes and "progress" not in changes:
                state["progress"] = _STAGE_PERCENT.get(str(changes["stage"]), state.get("progress", 0))

            # Update memory before persistence. This ensures active-job polling
            # and cancellation remain correct even if Windows temporarily locks
            # the snapshot file.
            self._states[job_id] = copy.deepcopy(state)
            self._write(state)
            # _write may add the degraded flag.
            self._states[job_id] = copy.deepcopy(state)

            if any(key in changes for key in ("status", "stage", "message", "error_code")):
                self._append_event(state)
            return copy.deepcopy(state)

    def submit(self, request, plan, execution_context: dict[str, Any], public_plan: dict[str, Any], aoi_summary: dict[str, Any]) -> dict[str, Any]:
        if not self._capacity.acquire(blocking=False):
            raise QueryError(
                "analysis_queue_full",
                "SatQuery is at its configured analysis capacity. Wait for an active job to finish and try again.",
                429,
            )
        job_id = "ana_" + uuid4().hex
        cancel = threading.Event()
        state = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": _STAGE_PERCENT["queued"],
            "message": "Analysis queued.",
            "created_at": _now(),
            "updated_at": _now(),
            "cancel_requested": False,
            "plan_fingerprint": public_plan.get("fingerprint"),
            "execution_plan": public_plan,
            "aoi": aoi_summary,
            "job": None,
        }
        with self._lock:
            self._cancel[job_id] = cancel
            self._states[job_id] = copy.deepcopy(state)
            self._write(state)
            self._states[job_id] = copy.deepcopy(state)
            self._append_event(state)
            self._futures[job_id] = self._pool.submit(
                self._run, job_id, cancel, request, plan, execution_context
            )
        return copy.deepcopy(state)

    def _run(self, job_id, cancel, request, plan, execution_context):
        try:
            if cancel.is_set():
                self._update(job_id, status="cancelled", stage="cancelled", message="Analysis cancelled before execution.")
                return
            self._update(job_id, status="running", stage="preparing", message="Preparing the approved analysis job.")

            def progress(stage: str, percent: int | None = None, message: str | None = None):
                if cancel.is_set():
                    raise QueryError("analysis_cancelled", "Analysis was cancelled by the user.", 409)
                self._update(
                    job_id,
                    status="running",
                    stage=stage,
                    progress=int(percent if percent is not None else _STAGE_PERCENT.get(stage, 0)),
                    message=message or stage.replace("_", " ").title(),
                )

            job, http_status = self.service.analyze(
                request,
                plan=plan,
                execution_context=execution_context,
                job_id=job_id,
                progress_cb=progress,
                cancel_check=cancel.is_set,
            )
            terminal = "completed" if job.status == "completed" else ("cancelled" if job.error_code == "analysis_cancelled" else "failed")
            state = self._update(
                job_id,
                status=terminal,
                stage=terminal,
                progress=100,
                http_status=http_status,
                message=("Analysis completed." if terminal == "completed" else (job.error or "Analysis did not complete.")),
                job=job.model_dump(mode="json"),
            )
            if terminal == "completed" and self.history is not None:
                try:
                    self.history.record(job)
                except Exception:
                    logger.exception("Failed to persist analysis history for %s", job_id)
            return state
        except QueryError as exc:
            terminal = "cancelled" if exc.code == "analysis_cancelled" else "failed"
            self._update(job_id, status=terminal, stage=terminal, progress=100, error_code=exc.code, message=str(exc))
        except Exception:
            logger.exception("Background analysis failed for %s", job_id)
            self._update(job_id, status="failed", stage="failed", progress=100, error_code="internal_error", message="Analysis failed; inspect server logs with the job ID.")
        finally:
            self._capacity.release()
            with self._lock:
                self._futures.pop(job_id, None)

    def get(self, job_id: str) -> dict[str, Any]:
        return self._read(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        state = self._read(job_id)
        if state.get("status") in {"completed", "failed", "cancelled"}:
            return state
        with self._lock:
            event = self._cancel.get(job_id)
            if event is None:
                # Persisted job from a prior server process cannot be killed by
                # this process; mark the request and expose the limitation.
                return self._update(job_id, cancel_requested=True, message="Cancellation requested; active worker ownership is unavailable after server restart.")
            event.set()
        return self._update(job_id, cancel_requested=True, message="Cancellation requested. SatQuery will stop at the next safe checkpoint.")
