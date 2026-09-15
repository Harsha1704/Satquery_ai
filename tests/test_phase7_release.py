"""Phase 7 model-free release-candidate tests."""
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.history_repository import AnalysisHistoryRepository
from backend.app.services.job_manager import AnalysisJobManager
from query_engine.capabilities import enforce_worldwide_capability
from query_engine.policy import QueryError
from query_engine.schemas import AnalysisPlan, AnalysisRequest, Inputs, Intent, ParsedQuery


class FakeExecutor:
    timeout = 10

    def execute(self, plan, request, inputs, job_dir, progress_cb=None, cancel_check=None):
        if progress_cb:
            progress_cb("analyzing", 52, "Controlled model-free analysis.")
        evidence = job_dir / "artifacts" / "result.png"
        evidence.write_bytes(b"evidence")
        return {"answer": "controlled result", "statistics": {}, "imagery_provenance": []}


def test_capability_contract_rejects_object_counting():
    try:
        enforce_worldwide_capability("Count every car in this city between 2020 and 2026")
    except QueryError as exc:
        assert exc.code == "unsupported_aoi_analysis"
    else:
        raise AssertionError("Object-level counting must not be silently coerced into temporal change.")


def test_capability_contract_allows_supported_language():
    enforce_worldwide_capability("Has greenery reduced here since 2021?")
    enforce_worldwide_capability("Show urban expansion between 2020 and 2026")
    enforce_worldwide_capability("What changed in this area between 2020 and 2026?")


def test_release_routes_and_security_headers():
    with TemporaryDirectory() as td:
        root = Path(td)
        inputs = root / "inputs"
        inputs.mkdir()
        service = AnalysisService(inputs, root / "jobs", executor=FakeExecutor())
        repo = AnalysisHistoryRepository(root / "history.sqlite3")
        app = create_app(service=service, history_repository=repo)
        client = TestClient(app)
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        assert response.json()["unlimited_capabilities"] is False
        assert response.headers["x-content-type-options"] == "nosniff"
        assert client.get("/api/v1/history").status_code == 200
        paths = app.openapi()["paths"]
        assert "/api/v1/map/jobs" in paths
        assert "/api/v1/map/jobs/{job_id}" in paths


def test_bounded_job_manager_persists_history():
    with TemporaryDirectory() as td:
        root = Path(td)
        inputs = root / "inputs"
        inputs.mkdir()
        (inputs / "sample.png").write_bytes(b"input")
        service = AnalysisService(inputs, root / "jobs", executor=FakeExecutor())
        repo = AnalysisHistoryRepository(root / "history.sqlite3")
        manager = AnalysisJobManager(service, repo, max_workers=1, max_queue=1)
        request = AnalysisRequest(query="What is visible?", inputs=Inputs(image_path="sample.png"))
        plan = AnalysisPlan(
            parsed=ParsedQuery(query=request.query, intent=Intent.VQA),
            tools=["single_image_vqa"],
            source="local",
        )
        state = manager.submit(request, plan, {}, {"fingerprint": "controlled"}, {})
        deadline = time.time() + 5
        while time.time() < deadline:
            state = manager.get(state["job_id"])
            if state["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert state["status"] == "completed"
        rows = repo.list()
        assert rows and rows[0]["job_id"] == state["job_id"]


def test_api_request_limit_blocks_oversized_contract(monkeypatch):
    monkeypatch.setenv("SATQUERY_MAX_REQUEST_BYTES", "16")
    with TemporaryDirectory() as td:
        root = Path(td)
        inputs = root / "inputs"
        inputs.mkdir()
        service = AnalysisService(inputs, root / "jobs", executor=FakeExecutor())
        repo = AnalysisHistoryRepository(root / "history.sqlite3")
        client = TestClient(create_app(service=service, history_repository=repo))
        response = client.post("/api/v1/analysis", content=b"x" * 64, headers={"content-type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error_code"] == "request_too_large"


def test_running_job_can_be_cancelled():
    class SlowExecutor:
        timeout = 10
        def execute(self, plan, request, inputs, job_dir, progress_cb=None, cancel_check=None):
            for _ in range(100):
                if cancel_check and cancel_check():
                    raise QueryError("analysis_cancelled", "Analysis was cancelled by the user.", 409)
                time.sleep(0.01)
            return {"answer": "unexpected", "statistics": {}, "imagery_provenance": []}

    with TemporaryDirectory() as td:
        root = Path(td)
        inputs = root / "inputs"
        inputs.mkdir()
        (inputs / "sample.png").write_bytes(b"input")
        service = AnalysisService(inputs, root / "jobs", executor=SlowExecutor())
        manager = AnalysisJobManager(service, None, max_workers=1, max_queue=0)
        request = AnalysisRequest(query="What is visible?", inputs=Inputs(image_path="sample.png"))
        plan = AnalysisPlan(
            parsed=ParsedQuery(query=request.query, intent=Intent.VQA),
            tools=["single_image_vqa"],
            source="local",
        )
        state = manager.submit(request, plan, {}, {"fingerprint": "controlled"}, {})
        time.sleep(0.05)
        manager.cancel(state["job_id"])
        deadline = time.time() + 3
        while time.time() < deadline:
            state = manager.get(state["job_id"])
            if state["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert state["status"] == "cancelled"


def test_job_state_persistence_tolerates_transient_replace_lock(monkeypatch):
    import os
    with TemporaryDirectory() as td:
        root = Path(td)
        inputs = root / "inputs"
        inputs.mkdir()
        (inputs / "sample.png").write_bytes(b"input")
        service = AnalysisService(inputs, root / "jobs", executor=FakeExecutor())
        manager = AnalysisJobManager(service, None, max_workers=1, max_queue=0)

        real_replace = os.replace
        calls = {"count": 0}
        def flaky_replace(src, dst):
            if calls["count"] < 2:
                calls["count"] += 1
                raise PermissionError(5, "simulated Windows sharing violation")
            return real_replace(src, dst)
        monkeypatch.setattr(os, "replace", flaky_replace)

        request = AnalysisRequest(query="What is visible?", inputs=Inputs(image_path="sample.png"))
        plan = AnalysisPlan(
            parsed=ParsedQuery(query=request.query, intent=Intent.VQA),
            tools=["single_image_vqa"],
            source="local",
        )
        state = manager.submit(request, plan, {}, {"fingerprint": "controlled"}, {})
        deadline = time.time() + 5
        while time.time() < deadline:
            state = manager.get(state["job_id"])
            if state["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert state["status"] == "completed"
        assert calls["count"] == 2
