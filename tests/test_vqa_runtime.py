import importlib
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai.router.orchestrator import SatQueryOrchestrator

with patch.dict(os.environ, {"SATQUERY_PRELOAD_VQA": "0"}):
    frontend = importlib.import_module("frontend.app")


class VQARuntimeTests(unittest.TestCase):
    def test_startup_and_first_request_share_one_instance(self):
        loading = threading.Event()
        release = threading.Event()
        model = Mock(
            MODEL_NAME="test-model", device="cpu",
            loaded_from_cache=False, model_load_ms=10.0,
        )
        model.answer.return_value = {
            "success": True, "answer": "yes", "model": "test-model",
            "device": "cpu", "cache_hit": True,
            "timing_ms": {"total": 0.1}, "metadata": {},
        }

        def construct(**kwargs):
            loading.set()
            self.assertTrue(release.wait(5))
            return model

        constructor = Mock(side_effect=construct)
        orchestrator = SatQueryOrchestrator()
        with patch.dict("sys.modules", {"ai.vqa": SimpleNamespace(SingleImageVQA=constructor)}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                startup = pool.submit(orchestrator.preload_vqa)
                try:
                    self.assertTrue(loading.wait(5))
                    request = pool.submit(
                        orchestrator._execute_single_image_vqa,
                        str(Path(__file__)), "Are there roads?",
                    )
                finally:
                    release.set()
                self.assertTrue(startup.result(timeout=5)["success"])
                result = request.result(timeout=5)
            constructor.assert_called_once_with(device="auto")
        self.assertTrue(result["success"])
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["timing_ms"], {"total": 0.1})
        self.assertEqual(result["execution_summary"]["timing_ms"], result["timing_ms"])

    def test_optional_warmup_uses_the_existing_model(self):
        orchestrator = SatQueryOrchestrator()
        model = Mock(MODEL_NAME="test-model", device="cpu", loaded_from_cache=True, model_load_ms=0)
        model.warmup.return_value = {"success": True, "warmup_ms": 1}
        orchestrator.single_image_vqa = model
        with patch.dict("sys.modules", {"ai.vqa": SimpleNamespace(SingleImageVQA=Mock())}):
            orchestrator.preload_vqa()
            model.warmup.assert_not_called()
            self.assertTrue(orchestrator.preload_vqa(warmup=True)["warmup"]["success"])
            model.warmup.assert_called_once()

    def test_runtime_status_reports_ready_and_preload_failure(self):
        with patch.object(frontend, "MODEL_RUNTIME_STATE", {"vqa": {"status": "not_started"}}):
            with patch.object(frontend.orchestrator, "preload_vqa") as preload:
                preload.return_value = {"success": True, "model": "test-model", "device": "cpu"}
                with patch.dict(os.environ, {"SATQUERY_BLIP_WARMUP": "1"}):
                    frontend.preload_competition_models()
                preload.assert_called_once_with(warmup=True)
                response = frontend.app.test_client().get("/api/runtime-status")
                self.assertEqual(response.status_code, 200)
                state = response.get_json()["models"]["vqa"]
                self.assertEqual(state["status"], "ready")
                self.assertEqual(state["device"], "cpu")
                self.assertGreaterEqual(state["load_ms"], 0)
                preload.side_effect = RuntimeError("Model unavailable")
                frontend.preload_competition_models()
                state = frontend.app.test_client().get("/api/runtime-status").get_json()["models"]["vqa"]
                self.assertEqual(state["status"], "error")
                self.assertEqual(state["error"], "Model unavailable")

    def test_failed_warmup_is_not_reported_ready(self):
        with patch.object(frontend, "MODEL_RUNTIME_STATE", {"vqa": {}}):
            with patch.object(frontend.orchestrator, "preload_vqa", return_value={
                "success": True, "warmup": {"success": False},
            }):
                frontend.preload_competition_models()
                self.assertEqual(frontend.MODEL_RUNTIME_STATE["vqa"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
