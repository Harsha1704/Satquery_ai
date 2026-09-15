"""SatQuery FastAPI application factory — Phase 7 release candidate.

All middleware, routes, worker limits and persistence are registered inside
``create_app`` so tests and the running server share the same topology.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.analysis import router as analysis_router
from backend.app.api.v1.capabilities import router as capabilities_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.history import router as history_router
from backend.app.api.v1.map import router as map_router
from backend.app.api.v1.map_context import router as map_context_router
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.history_repository import AnalysisHistoryRepository
from backend.app.services.job_manager import AnalysisJobManager
from query_engine.executor import Executor
from query_engine.policy import QueryError


def _cors_origins():
    configured = os.environ.get("SATQUERY_CORS_ORIGINS")
    if configured:
        return [x.strip() for x in configured.split(",") if x.strip()]
    return [
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]


def create_app(service=None, history_repository=None):
    app = FastAPI(
        title="SatQuery AI",
        version="1.0.0-rc1",
        description="Natural-language geospatial intelligence and evidence-grounded temporal analysis API.",
    )
    root = Path(__file__).resolve().parents[2]
    timeout = int(os.environ.get("SATQUERY_ANALYSIS_TIMEOUT", "900"))
    job_root = Path(os.environ.get("SATQUERY_JOB_ROOT", str(root / "outputs" / "jobs")))
    service = service or AnalysisService(
        input_root=Path(os.environ.get("SATQUERY_INPUT_ROOT", str(root / "frontend" / "uploads"))),
        job_root=job_root,
        executor=Executor(timeout=max(60, int(timeout * 0.75))),
        pipeline_timeout_seconds=timeout,
    )
    app.state.analysis_service = service
    app.state.history_repository = history_repository or AnalysisHistoryRepository(
        Path(os.environ.get("SATQUERY_HISTORY_DB", str(root / "outputs" / "satquery_history.sqlite3")))
    )
    app.state.job_manager = AnalysisJobManager(
        service=service,
        history_repository=app.state.history_repository,
        max_workers=int(os.environ.get("SATQUERY_MAX_WORKERS", "2")),
        max_queue=int(os.environ.get("SATQUERY_MAX_QUEUE", "6")),
    )

    @app.exception_handler(QueryError)
    async def query_error(request: Request, exc: QueryError):
        return JSONResponse(status_code=exc.status_code, content={"error_code": exc.code, "error": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "invalid_request",
                "error": "Request validation failed.",
                "details": [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()],
            },
        )

    max_request_bytes = int(os.environ.get("SATQUERY_MAX_REQUEST_BYTES", str(2 * 1024 * 1024)))

    @app.middleware("http")
    async def release_candidate_guards(request: Request, call_next):
        # The analysis API accepts compact JSON contracts, not arbitrary file
        # uploads. Large imagery is server-controlled or handled by the Flask
        # upload workflow, so rejecting oversized API bodies is intentional.
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > max_request_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"error_code": "request_too_large", "error": "Request body exceeds the configured API limit."},
                    )
            except ValueError:
                pass
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Range"],
        expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
    )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(capabilities_router, prefix="/api/v1")
    app.include_router(history_router, prefix="/api/v1")
    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(map_router, prefix="/api/v1")
    app.include_router(map_context_router)
    return app


app = create_app()
