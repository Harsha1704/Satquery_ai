from __future__ import annotations

from fastapi import APIRouter, Request

from query_engine.capabilities import public_capability_contract

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(request: Request):
    payload = public_capability_contract()
    payload["execution"] = {
        "job_model": "bounded-background-workers",
        "cancellation": True,
        "persistent_history": True,
        "artifact_policy": "published-artifacts-only",
        "source_plan_consistency": "enforced-by-plan-fingerprint-and-provenance",
        "exact_polygon_statistics": True,
    }
    return payload
