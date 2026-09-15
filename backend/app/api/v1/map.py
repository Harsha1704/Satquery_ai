from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.app.api.v1.map_context import AOIFeature, build_canonical_map_plan

router = APIRouter(prefix="/map", tags=["map"])


class MapPlanRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    aoi: Dict[str, Any]
    plan_fingerprint: Optional[str] = Field(default=None, min_length=8, max_length=128)


@router.get("/health")
def map_phase_health() -> Dict[str, Any]:
    return {
        "service": "satquery-map",
        "status": "ok",
        "workspace": "geospatial-intelligence",
        "execution": "enabled",
        "planning": "canonical",
        "aoi_statistics": "exact-polygon-when-polygon-selected",
    }


@router.post("/plan")
def plan_map_query(request: MapPlanRequest) -> Dict[str, Any]:
    feature = AOIFeature.model_validate(request.aoi)
    analysis_request, _plan, public_plan, aoi_summary = build_canonical_map_plan(request.query, feature)
    return {
        "request_id": "mapplan_" + uuid4().hex[:12],
        "status": "planned",
        "query": analysis_request.query,
        "aoi": {
            **aoi_summary,
            "feature": feature.model_dump(mode="json"),
            "analysis_geometry": analysis_request.aoi.model_dump(mode="json") if analysis_request.aoi else None,
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
        "plan": public_plan,
        "plan_fingerprint": public_plan["fingerprint"],
        "execution_started": False,
        "next_stage": "execute_exact_approved_plan",
    }



@router.post("/jobs", status_code=202)
def submit_map_job(body: MapPlanRequest, request: Request) -> Dict[str, Any]:
    """Submit an approved map analysis to the bounded background worker pool."""
    feature = AOIFeature.model_validate(body.aoi)
    analysis_request, plan, public_plan, aoi_summary = build_canonical_map_plan(body.query, feature)

    if body.plan_fingerprint and body.plan_fingerprint != public_plan["fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "The executable imagery plan changed after preview. Review the refreshed source/date plan and run again; "
                "SatQuery will not silently execute a different plan."
            ),
        )

    execution_context = {
        "plan_fingerprint": public_plan["fingerprint"],
        "imagery": public_plan.get("execution_imagery") or {},
        "aoi": {
            "statistics_geometry": aoi_summary.get("statistics_geometry"),
            "retrieval_geometry": aoi_summary.get("retrieval_geometry"),
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
    }
    state = request.app.state.job_manager.submit(
        analysis_request,
        plan=plan,
        execution_context=execution_context,
        public_plan=public_plan,
        aoi_summary={
            **aoi_summary,
            "feature": feature.model_dump(mode="json"),
            "analysis_geometry": analysis_request.aoi.model_dump(mode="json") if analysis_request.aoi else None,
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
    )
    return state


@router.get("/jobs/{job_id}")
def get_map_job(job_id: str, request: Request) -> Dict[str, Any]:
    return request.app.state.job_manager.get(job_id)


@router.delete("/jobs/{job_id}")
def cancel_map_job(job_id: str, request: Request) -> Dict[str, Any]:
    return request.app.state.job_manager.cancel(job_id)


@router.post("/analyze")
def analyze_map_query(body: MapPlanRequest, request: Request, response: Response) -> Dict[str, Any]:
    """Execute the same canonical plan previewed by the map workspace.

    The selected polygon remains the analysis/statistics footprint. Only its
    rectangular envelope is used for efficient imagery retrieval. A supplied
    plan fingerprint acts as an optimistic-concurrency guard: if data
    availability or the request changed between preview and execution, the API
    rejects the stale plan rather than silently running something different.
    """
    feature = AOIFeature.model_validate(body.aoi)
    analysis_request, plan, public_plan, aoi_summary = build_canonical_map_plan(body.query, feature)

    if body.plan_fingerprint and body.plan_fingerprint != public_plan["fingerprint"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "The executable imagery plan changed after preview. Review the refreshed source/date plan and run again; "
                "SatQuery will not silently execute a different plan."
            ),
        )

    execution_context = {
        "plan_fingerprint": public_plan["fingerprint"],
        "imagery": public_plan.get("execution_imagery") or {},
        "aoi": {
            "statistics_geometry": aoi_summary.get("statistics_geometry"),
            "retrieval_geometry": aoi_summary.get("retrieval_geometry"),
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
    }
    job, status_code = request.app.state.analysis_service.analyze(
        analysis_request,
        plan=plan,
        execution_context=execution_context,
    )
    response.status_code = status_code
    return {
        "request_id": "maprun_" + uuid4().hex[:12],
        "query": analysis_request.query,
        "aoi": {
            **aoi_summary,
            "feature": feature.model_dump(mode="json"),
            "analysis_geometry": analysis_request.aoi.model_dump(mode="json") if analysis_request.aoi else None,
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
        "execution_plan": public_plan,
        "plan_fingerprint": public_plan["fingerprint"],
        "job": job.model_dump(mode="json"),
    }
