"""Run legacy code in a cancellable child process with job-local isolation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from query_engine.policy import QueryError
from query_engine.tool_registry import validate_plan


class Executor:
    def __init__(self, timeout: int = 900):
        self.timeout = int(timeout)

    @staticmethod
    def _terminate(process: subprocess.Popen):
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def execute(self, plan, request, inputs, job_dir: Path, progress_cb=None, cancel_check=None):
        validate_plan(plan)
        progress_cb = progress_cb or (lambda *_args, **_kwargs: None)
        cancel_check = cancel_check or (lambda: False)
        payload = {
            "plan": plan.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "inputs": inputs,
        }
        (job_dir / "request.json").write_text(json.dumps(payload), encoding="utf-8")
        root = str(Path(__file__).resolve().parents[1])
        bootstrap = "import sys; sys.path.insert(0, sys.argv[1]); from query_engine.worker import main; main()"
        progress_path = job_dir / "worker-progress.json"
        last_progress = None
        started = time.monotonic()

        with (job_dir / "execution.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", bootstrap, root],
                cwd=job_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                while process.poll() is None:
                    if cancel_check():
                        self._terminate(process)
                        raise QueryError("analysis_cancelled", "Analysis was cancelled by the user.", 409)
                    if time.monotonic() - started > self.timeout:
                        self._terminate(process)
                        raise QueryError("execution_timeout", "Analysis exceeded the configured worker time limit.", 504)
                    if progress_path.is_file():
                        try:
                            current = json.loads(progress_path.read_text(encoding="utf-8"))
                            signature = (current.get("stage"), current.get("percent"), current.get("message"))
                            if signature != last_progress:
                                last_progress = signature
                                progress_cb(
                                    str(current.get("stage") or "analyzing"),
                                    int(current.get("percent") or 50),
                                    str(current.get("message") or "Running analysis."),
                                )
                        except (OSError, ValueError, TypeError):
                            pass
                    time.sleep(0.25)
            finally:
                if process.poll() is None:
                    self._terminate(process)

        if process.returncode != 0 or not (job_dir / "worker-result.json").is_file():
            raise QueryError("worker_failed", "Analysis worker failed; inspect the job execution log.", 500)
        output = json.loads((job_dir / "worker-result.json").read_text(encoding="utf-8"))
        if "error_code" in output:
            raise QueryError(output["error_code"], output["error"], output.get("status_code", 500))
        return output
