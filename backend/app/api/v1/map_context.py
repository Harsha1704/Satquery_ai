from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from query_engine.capabilities import enforce_worldwide_capability
from query_engine.policy import QueryError

router = APIRouter(prefix="/api/v1/map/context", tags=["map-context"])


class GeoJSONGeometry(BaseModel):
    type: Literal["Point", "Polygon", "MultiPolygon", "LineString"]
    coordinates: Any


class AOIFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any] = Field(default_factory=dict)
    bbox: Optional[List[float]] = None


class MapClientContext(BaseModel):
    source: Optional[str] = None
    map_center: Optional[Dict[str, float]] = None
    zoom: Optional[float] = None


class MapContextRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    aoi: AOIFeature
    client: Optional[MapClientContext] = None


def _walk_coordinate_pairs(value: Any) -> List[List[float]]:
    pairs: List[List[float]] = []
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        pairs.append([float(value[0]), float(value[1])])
        return pairs
    if isinstance(value, (list, tuple)):
        for item in value:
            pairs.extend(_walk_coordinate_pairs(item))
    return pairs


def _validate_aoi(feature: AOIFeature) -> Dict[str, Any]:
    pairs = _walk_coordinate_pairs(feature.geometry.coordinates)
    if not pairs:
        raise HTTPException(status_code=422, detail="AOI geometry does not contain valid coordinate pairs.")

    for lon, lat in pairs:
        if not (-180.0 <= lon <= 180.0):
            raise HTTPException(status_code=422, detail=f"Invalid longitude in AOI: {lon}")
        if not (-90.0 <= lat <= 90.0):
            raise HTTPException(status_code=422, detail=f"Invalid latitude in AOI: {lat}")

    west = min(pair[0] for pair in pairs)
    south = min(pair[1] for pair in pairs)
    east = max(pair[0] for pair in pairs)
    north = max(pair[1] for pair in pairs)

    if feature.geometry.type in {"Polygon", "MultiPolygon"} and (west == east or south == north):
        raise HTTPException(status_code=422, detail="AOI polygon has zero spatial extent.")

    if feature.geometry.type not in {"Polygon", "Point"}:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{feature.geometry.type} AOIs are not executable in the current worldwide workflow. "
                "Use a polygon/rectangle or point selection."
            ),
        )

    if feature.geometry.type == "Polygon":
        rings = feature.geometry.coordinates
        if not isinstance(rings, list) or not rings or len(rings[0]) < 4:
            raise HTTPException(status_code=422, detail="AOI polygon is malformed.")
        if rings[0][0] != rings[0][-1]:
            raise HTTPException(status_code=422, detail="AOI polygon ring must be closed.")

    return {
        "geometry_type": feature.geometry.type,
        "bbox": [west, south, east, north],
        "coordinate_count": len(pairs),
        "statistics_geometry": "original_polygon" if feature.geometry.type == "Polygon" else "point_buffer_rectangle",
        "retrieval_geometry": "bounding_rectangle",
    }


def _planning_rectangle(bbox: List[float]) -> Dict[str, Any]:
    west, south, east, north = bbox
    if west == east:
        west, east = west - 0.005, east + 0.005
    if south == north:
        south, north = south - 0.005, north + 0.005
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def _analysis_geometry(feature: AOIFeature, aoi_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return the authoritative analysis footprint.

    Polygon selections remain polygons. A point has no area, so it receives the
    existing small rectangular footprint. Imagery retrieval may still use the
    polygon bounds, but area statistics must use this geometry.
    """
    if feature.geometry.type == "Polygon":
        return feature.geometry.model_dump()
    if feature.geometry.type == "Point":
        return _planning_rectangle(aoi_summary["bbox"])
    raise HTTPException(status_code=422, detail="Unsupported AOI geometry for analysis.")


def _human_sensor_label(spec: Dict[str, Any]) -> str:
    label = str(spec.get("label") or spec.get("id") or "Earth observation imagery")
    upper = label.upper()
    if "SENTINEL-2" in upper or "S2" in upper:
        return "Sentinel-2 MSI"
    if "LANDSAT 5" in upper:
        return "Landsat 5 TM"
    if "LANDSAT 7" in upper:
        return "Landsat 7 ETM+"
    if "LANDSAT 8" in upper or "LANDSAT 9" in upper:
        return "Landsat 8/9 OLI"
    return label


def _resolve_executable_imagery_plan(analysis_request, plan) -> Dict[str, Any]:
    """Resolve the exact EO plan the executor is capable of using.

    This intentionally uses the same ``gee_temporal.plan_temporal_fetch``
    resolver as execution. UI labels therefore come from a real availability
    check rather than a separate heuristic such as "flood -> Sentinel-1".
    """
    if getattr(plan, "source", None) != "earth_engine":
        return {"source": "local", "label": "Local imagery"}

    years = list(getattr(getattr(plan, "parsed", None), "years", None) or [])
    if len(years) != 2:
        raise QueryError("invalid_years", "Worldwide temporal analysis requires two valid years.", 422)

    import gee_temporal as gee

    temporal = gee.plan_temporal_fetch(analysis_request.aoi.bounds, int(years[0]), int(years[1]))
    before_label = _human_sensor_label(temporal.before_spec)
    after_label = _human_sensor_label(temporal.after_spec)
    label = before_label if before_label == after_label else f"{before_label} + {after_label}"

    q = str(getattr(plan.parsed, "query", "") or "").lower()
    warnings: List[str] = []
    route = "optical_temporal_change"
    if any(token in q for token in ("flood", "flooded", "inundation", "waterlogging")):
        warnings.append(
            "Flood wording was recognized, but this executable build currently uses the validated optical water-change route. "
            "Sentinel-1 SAR is not advertised unless a SAR executor is actually enabled."
        )

    return {
        "source": "earth_engine",
        "provider": "Google Earth Engine",
        "route": route,
        "label": label,
        "before": {
            "year": int(years[0]),
            "platform": before_label,
            "collection_id": temporal.before_spec.get("id"),
            "collection_label": temporal.before_spec.get("label"),
            "scene_count": int(temporal.before_count),
        },
        "after": {
            "year": int(years[1]),
            "platform": after_label,
            "collection_id": temporal.after_spec.get("id"),
            "collection_label": temporal.after_spec.get("label"),
            "scene_count": int(temporal.after_count),
        },
        "target_scale_m": float(temporal.target_scale_m),
        "grid": {"width_px": int(temporal.width_px), "height_px": int(temporal.height_px)},
        "cross_sensor": bool(temporal.cross_sensor),
        "warnings": warnings,
    }


def _source_details(executable_imagery: Dict[str, Any], plan: Any) -> Dict[str, Any]:
    label = executable_imagery.get("label") or "Earth observation imagery"
    scale = executable_imagery.get("target_scale_m")
    targets = set(str(x).lower() for x in (getattr(getattr(plan, "parsed", None), "targets", None) or []))
    query = str(getattr(getattr(plan, "parsed", None), "query", "") or "").lower()
    if "vegetation" in targets or "ndvi" in query:
        reason = "The resolved multispectral scenes provide red and near-infrared bands required for NDVI temporal analysis."
    elif "built_up" in targets or "urban" in query or "construction" in query:
        reason = "The resolved multispectral scenes provide NIR and SWIR evidence required for NDBI-based built-up surface analysis."
    elif "water" in targets or any(x in query for x in ("water", "flood", "river", "lake")):
        reason = "The executable optical route provides green/NIR water-index evidence for temporal surface-water change."
    else:
        reason = "The resolver selected imagery that is actually available for both requested dates on a common comparison grid."
    return {
        "platform": label,
        "sensor": "Multispectral Earth observation" if executable_imagery.get("source") == "earth_engine" else "Local imagery",
        "resolution": f"{scale:g} m" if isinstance(scale, (int, float)) else "Source dependent",
        "reason": reason,
        "provider": executable_imagery.get("provider"),
    }


def _flatten_industry_plan(plan: Any, executable_imagery: Dict[str, Any]) -> Dict[str, Any]:
    parsed = getattr(plan, "parsed", None)
    parsed_dict = parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else {}
    warnings = list(getattr(plan, "warnings", []) or []) + list(executable_imagery.get("warnings") or [])
    return {
        "planner": "query_engine.planner.QueryPlanner",
        "intent": parsed_dict.get("intent", "unknown"),
        "routing_confidence": parsed_dict.get("routing_confidence"),
        "targets": parsed_dict.get("targets", []),
        "operation": parsed_dict.get("operation", "analyze"),
        "years": parsed_dict.get("years", []),
        "change_direction": parsed_dict.get("change_direction", "none"),
        "transition_from": parsed_dict.get("transition_from", "none"),
        "transition_to": parsed_dict.get("transition_to", "none"),
        "required_tools": list(getattr(plan, "tools", []) or []),
        "input_configuration": (
            plan.input_configuration.model_dump(mode="json")
            if getattr(plan, "input_configuration", None) is not None else None
        ),
        "execution_steps": [
            {"tool": step.tool, "dependencies": step.dependencies, "expected_output": step.expected_output}
            for step in (getattr(plan, "steps", []) or [])
        ],
        "capability": (
            plan.capability.model_dump(mode="json")
            if getattr(plan, "capability", None) is not None else None
        ),
        "source": getattr(plan, "source", "local"),
        "recommended_source": executable_imagery.get("label") or "Local imagery",
        "source_details": _source_details(executable_imagery, plan),
        "execution_imagery": executable_imagery,
        "warnings": warnings,
    }


def _plan_fingerprint(query: str, feature: AOIFeature, plan: Any, executable_imagery: Dict[str, Any]) -> str:
    payload = {
        "query": query.strip(),
        "geometry": feature.geometry.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "imagery": executable_imagery,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _translate_planning_error(exc: Exception) -> HTTPException:
    if isinstance(exc, QueryError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    try:
        import gee_temporal as gee
        if isinstance(exc, gee.RoiTooLargeError):
            return HTTPException(status_code=413, detail=str(exc))
        if isinstance(exc, gee.NoImageryError):
            return HTTPException(status_code=422, detail=str(exc))
        if isinstance(exc, gee.GeeNotConfiguredError):
            return HTTPException(status_code=503, detail="Earth Engine is not configured for this server.")
    except Exception:
        pass
    return HTTPException(status_code=503, detail=f"Executable imagery planning failed: {exc}")


def build_canonical_map_plan(query: str, feature: AOIFeature) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    """Build the one plan used for preview and execution.

    Validation failures are returned directly. There is deliberately no fallback
    planner that can claim success after the canonical planner rejected a job.
    """
    from query_engine.planner import QueryPlanner
    from query_engine.schemas import AOI, AnalysisRequest

    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    try:
        enforce_worldwide_capability(clean_query)
    except QueryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    aoi_summary = _validate_aoi(feature)
    geometry = _analysis_geometry(feature, aoi_summary)
    try:
        analysis_request = AnalysisRequest(
            query=clean_query,
            aoi=AOI.model_validate(geometry),
            project_id="map_workspace",
        )
        plan = QueryPlanner().plan(analysis_request)
        executable_imagery = _resolve_executable_imagery_plan(analysis_request, plan)
    except HTTPException:
        raise
    except Exception as exc:
        raise _translate_planning_error(exc) from exc

    public_plan = _flatten_industry_plan(plan, executable_imagery)
    fingerprint = _plan_fingerprint(clean_query, feature, plan, executable_imagery)
    public_plan["fingerprint"] = fingerprint
    return analysis_request, plan, public_plan, aoi_summary


@router.get("/health")
def map_context_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "satquery-map-context",
        "workspace": "geospatial-intelligence",
        "planner": "canonical-query-engine",
        "source_preview": "live-executable-imagery-plan",
        "fallback_planner": False,
    }


@router.post("")
def receive_map_context(request: MapContextRequest) -> Dict[str, Any]:
    analysis_request, _plan, public_plan, aoi_summary = build_canonical_map_plan(request.query, request.aoi)
    return {
        "request_id": f"map_{uuid4().hex[:12]}",
        "accepted": True,
        "query": analysis_request.query,
        "aoi": {
            **aoi_summary,
            "feature": request.aoi.model_dump(mode="json"),
            "analysis_geometry": analysis_request.aoi.model_dump(mode="json") if analysis_request.aoi else None,
            "retrieval_bbox": analysis_request.aoi.bounds if analysis_request.aoi else None,
        },
        "plan": public_plan,
        "plan_fingerprint": public_plan["fingerprint"],
        "client": request.client.model_dump() if request.client else None,
        "execution_started": False,
        "next_stage": "execute_exact_approved_plan",
    }
