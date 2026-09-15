"""Model-free API/adapter contracts. Run with unittest; no model downloads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.main import create_app
from backend.app.services.analysis_service import AnalysisService
from query_engine.executor import Executor
from query_engine.legacy_adapter import execute_legacy
from query_engine.planner import QueryPlanner
from query_engine.policy import QueryError, resolve_input
from query_engine.schemas import AOI, AnalysisRequest
from query_engine.tool_registry import validate_plan

RECTANGLE = {"type": "Polygon", "coordinates": [[[77, 12], [77.01, 12], [77.01, 12.01], [77, 12.01], [77, 12]]]}
HISTORICAL = {"query": "Has vegetation decreased in this region since 2020?", "aoi": RECTANGLE}
SINGLE = {"query": "What is in this image?", "inputs": {"image_path": "sample.png"}}


class FakeExecutor:
    def execute(self, plan, request, inputs, job_dir):
        assert Path(inputs["image_path"]).is_relative_to(job_dir)
        evidence = job_dir / "outputs" / "evidence"
        evidence.mkdir(parents=True)
        (evidence / "result.png").write_bytes(b"evidence")
        return {"answer": request.query, "confidence": 0.99, "statistics": {"count": 2}}


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "sample.png").write_bytes(b"input")
        self.service = AnalysisService(self.inputs, self.root / "jobs", executor=FakeExecutor())
        self.client = TestClient(create_app(self.service))

    def test_health_does_not_import_specialists(self):
        code = "from backend.app.main import app; import sys; assert 'torch' not in sys.modules; assert 'ai.router.orchestrator' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
        self.assertEqual(self.client.get("/api/v1/health").json()["model_readiness"], "not_checked")

    def test_default_paths_and_openapi(self):
        with patch.dict(os.environ, {}, clear=True):
            service = create_app().state.analysis_service
        self.assertEqual(service.job_root, Path(__file__).resolve().parents[1] / "outputs" / "jobs")
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/v1/analysis", response.json()["paths"])

    def test_worker_timeout_is_reported(self):
        class HangingProcess:
            returncode = None
            def __init__(self, *args, **kwargs):
                self.done = False
            def poll(self):
                return -15 if self.done else None
            def terminate(self):
                self.done = True
                self.returncode = -15
            def wait(self, timeout=None):
                self.done = True
                self.returncode = -15
                return self.returncode
            def kill(self):
                self.done = True
                self.returncode = -9

        self.service.executor = Executor(timeout=1)
        with patch("query_engine.executor.subprocess.Popen", HangingProcess):
            response = self.client.post("/api/v1/analysis", json=SINGLE)
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error_code"], "execution_timeout")

    def test_historical_plan_and_relative_dates(self):
        response = self.client.post("/api/v1/analysis/plan", json=HISTORICAL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["parsed"]["years"], [2020, datetime.now(timezone.utc).year])
        request = AnalysisRequest(query="Has urban development expanded during the last five years?", aoi=RECTANGLE)
        years = QueryPlanner().plan(request).parsed.years
        self.assertEqual(years[1] - years[0], 5)

    def test_malformed_aoi(self):
        for coordinates in ([], [[[77, 12]]], [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValidationError):
                AOI(coordinates=coordinates)
        response = self.client.post("/api/v1/analysis", json={**HISTORICAL, "aoi": {"coordinates": []}})
        self.assertEqual(response.status_code, 422)

    def test_invalid_requests_do_not_execute(self):
        cases = [
            {"query": " "},
            {**SINGLE, "tools": ["os.system"]},
            {**SINGLE, "inputs": {"image_path": "../outside.png"}},
            {**SINGLE, "inputs": {}},
            {**SINGLE, "inputs": {"image_path": "sample.png", "sar_path": "sample.png"}},
            {**HISTORICAL, "inputs": SINGLE["inputs"]},
            {**HISTORICAL, "before_year": 2025},
            {**HISTORICAL, "before_year": 2025, "after_year": 2020},
            {**HISTORICAL, "before_year": 2020, "after_year": 2999},
            {**HISTORICAL, "query": "What is visible in this image?"},
            {**SINGLE, "query": "xyzzy plugh"},
        ]
        with patch.object(self.service.executor, "execute") as execute:
            for body in cases:
                with self.subTest(body=body):
                    self.assertEqual(self.client.post("/api/v1/analysis", json=body).status_code, 422)
            execute.assert_not_called()

    def test_absolute_and_sibling_paths_rejected(self):
        for value in ("C:\\private\\secret.png", "../inputs-other/file.png", str(self.inputs / "sample.png")):
            with self.subTest(value=value), self.assertRaises(QueryError):
                resolve_input(self.inputs, value)

    def test_registry_rejects_tampered_plan(self):
        plan = QueryPlanner().plan(AnalysisRequest(**SINGLE))
        with self.assertRaises(QueryError):
            validate_plan(plan.model_copy(update={"tools": ["os.system"]}))

    def test_completed_result_is_persisted_without_generic_confidence(self):
        response = self.client.post("/api/v1/analysis", json=SINGLE)
        self.assertEqual(response.status_code, 200)
        job = response.json()
        self.assertEqual(job["status"], "completed")
        self.assertIsNone(job["result"]["confidence"]["model"])
        self.assertEqual(job["result"]["evidence"], ["outputs/evidence/result.png"])
        trace = job["result"]["execution_trace"]
        self.assertEqual(trace["analysis_id"], job["job_id"])
        self.assertEqual(trace["task"], job["plan"]["parsed"]["intent"])
        self.assertTrue(trace["selected_tools"])
        self.assertTrue(trace["steps"])
        self.assertEqual(self.client.get("/api/v1/jobs/" + job["job_id"]).json(), job)
        restarted = AnalysisService(self.inputs, self.root / "jobs")
        self.assertEqual(restarted.get_job(job["job_id"]).status, "completed")
        self.assertNotIn(str(self.root), response.text)

    def test_published_artifact_allows_the_flask_map_origin(self):
        """Evidence images are fetched by the Flask UI on :5000 from the API on :8000."""
        job = self.client.post("/api/v1/analysis", json=SINGLE).json()
        response = self.client.get(
            f"/api/v1/jobs/{job['job_id']}/artifacts/outputs/evidence/result.png",
            headers={"Origin": "http://127.0.0.1:5000"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5000")
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")

    def test_concurrent_jobs_have_distinct_inputs_and_evidence(self):
        def run(_):
            return self.service.analyze(AnalysisRequest(**SINGLE))[0]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = list(pool.map(run, range(2)))
        self.assertNotEqual(jobs[0].job_id, jobs[1].job_id)
        for job in jobs:
            self.assertTrue((self.service.job_root / job.job_id / "outputs/evidence/result.png").is_file())

    def test_failed_job_preserves_id_and_error(self):
        with patch.object(self.service.executor, "execute", side_effect=QueryError("imagery_not_configured", "Configure imagery.", 503)):
            response = self.client.post("/api/v1/analysis", json=HISTORICAL)
        self.assertEqual(response.status_code, 503)
        job = response.json()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(self.service.get_job(job["job_id"]).error_code, "imagery_not_configured")

    def test_adapter_uses_approved_intent(self):
        plan = QueryPlanner().plan(AnalysisRequest(**SINGLE))
        from ai.router.orchestrator import SatQueryOrchestrator
        with patch.object(SatQueryOrchestrator, "_execute_single_image_vqa", return_value={"success": True, "answer": "Vegetation"}) as vqa:
            result = execute_legacy(plan, {"image_path": str(self.inputs / "sample.png")})
        self.assertTrue(result["execution"]["success"])
        vqa.assert_called_once()

    def test_explicit_pair_preserves_change_routing(self):
        request = AnalysisRequest(query="find buildings", inputs={"before_path": "a.png", "after_path": "b.png"})
        plan = QueryPlanner().plan(request)
        self.assertEqual(plan.parsed.intent.value, "change_detection")
        self.assertIsNone(plan.parsed.routing_confidence)

    def test_missing_job(self):
        self.assertEqual(self.client.get("/api/v1/jobs/ana_" + "0" * 32).status_code, 404)

    def test_real_worker_missing_gee_configuration(self):
        self.service.executor = Executor(timeout=30)
        with patch.dict(os.environ, {"GEE_SERVICE_ACCOUNT_EMAIL": "", "GEE_SERVICE_ACCOUNT_KEY": ""}):
            response = self.client.post("/api/v1/analysis", json=HISTORICAL)
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["error_code"], "imagery_not_configured")

    def test_real_multispectral_worker(self):
        from query_engine.runtime import configure_raster_runtime
        configure_raster_runtime()
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        image_path = self.inputs / "spectral.tif"
        with rasterio.open(image_path, "w", driver="GTiff", width=16, height=16, count=12,
                           dtype="float32", crs="EPSG:4326", transform=from_origin(77, 12, 0.001, 0.001)) as raster:
            raster.write(np.full((12, 16, 16), 0.25, dtype="float32"))
        self.service.executor = Executor(timeout=60)
        response = self.client.post("/api/v1/analysis", json={"query": "Calculate NDVI for this image", "inputs": {"image_path": "spectral.tif"}})
        if response.status_code != 200:
            job_dir = self.service.job_root / response.json()["job_id"]
            self.fail((job_dir / "execution.log").read_text(encoding="utf-8"))
        self.assertEqual(response.json()["status"], "completed")
        self.assertTrue(response.json()["result"]["statistics"])


if __name__ == "__main__":
    unittest.main()
