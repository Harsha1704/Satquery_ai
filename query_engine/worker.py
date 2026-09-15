"""Private worker entry point; never accepts callable names from an API client."""
import json
import math
from pathlib import Path
import traceback

from query_engine.legacy_adapter import execute_legacy
from query_engine.policy import QueryError
from query_engine.schemas import AnalysisPlan, AnalysisRequest
from query_engine.tool_registry import validate_plan


def _progress(stage: str, percent: int, message: str) -> None:
    path = Path("worker-progress.json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stage": stage, "percent": int(percent), "message": message}), encoding="utf-8")
    tmp.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(payload):
    _progress("preparing", 12, "Validating the approved worker plan.")
    plan = AnalysisPlan.model_validate(payload["plan"])
    request = AnalysisRequest.model_validate(payload["request"])
    validate_plan(plan)
    from query_engine.runtime import configure_raster_runtime
    configure_raster_runtime()
    inputs = payload["inputs"]
    before = after = temporal = None
    if plan.source == "earth_engine":
        _progress("retrieving_imagery", 22, "Resolving and retrieving Earth-observation imagery.")
        import gee_temporal as gee
        if not gee.is_configured():
            raise QueryError("imagery_not_configured", "Earth Engine dependencies and service-account configuration are required.", 503)
        try:
            bounds = request.aoi.bounds
            first, last = plan.parsed.years
            temporal = gee.plan_temporal_fetch(bounds, first, last)
            dimensions = (temporal.width_px, temporal.height_px)
            exact_geometry = request.aoi.coordinates if request.aoi is not None else None
            _progress("retrieving_imagery", 25, f"Retrieving imagery for {first}.")
            before = gee.fetch_year_composite(
                bounds, first, Path.cwd() / "imagery", spec=temporal.before_spec,
                dimensions=dimensions, statistics_geometry=exact_geometry,
            )
            _progress("retrieving_imagery", 30, f"Retrieving imagery for {last}.")
            after = gee.fetch_year_composite(
                bounds, last, Path.cwd() / "imagery", spec=temporal.after_spec,
                dimensions=dimensions, statistics_geometry=exact_geometry,
            )
            _progress("preprocessing", 38, "Aligned temporal scenes are ready on a common comparison grid.")
            inputs = {"before_path": str(before.path), "after_path": str(after.path)}
        except gee.GeeNotConfiguredError as exc:
            raise QueryError("imagery_not_configured", "Earth Engine configuration could not be initialized.", 503) from exc
        except gee.RoiTooLargeError as exc:
            raise QueryError("aoi_too_large", "Select a smaller area for historical analysis.", 413) from exc
        except gee.NoImageryError as exc:
            raise QueryError("no_imagery", "No usable imagery was found for this area and time range.") from exc
    # Local, explicit before/after inputs use the deterministic temporal
    # engine directly. It retains the existing legacy adapter for specialist
    # flows that do not yet have an equivalent temporal implementation.
    if plan.source == "local" and plan.parsed.intent.value == "change_detection" and {"before_path", "after_path"}.issubset(inputs):
        from ai.temporal import TemporalPair, TemporalChangeEngine, TemporalChangeVQA
        try:
            years = list(plan.parsed.years or [])
            result = TemporalChangeEngine().analyze(TemporalPair(
                Path(inputs["before_path"]), Path(inputs["after_path"]),
                str(years[0]) if years else None, str(years[1]) if len(years) > 1 else None,
            ))
            from query_engine.result_validator import validate_temporal_result
            validate_temporal_result(result)
            artifact = Path("artifacts") / "temporal_change_polygons.geojson"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(result["change_polygons"], allow_nan=False), encoding="utf-8")
            answer = TemporalChangeVQA().answer(request.query, result)["answer"]
            return json_safe({
                "success": True, "answer": answer,
                "statistics": {"temporal": result, "change": result["statistics"]},
                "imagery_provenance": [
                    {"source_id": f"src_{result['provenance']['before_hash'][:16]}", "role": "before"},
                    {"source_id": f"src_{result['provenance']['after_hash'][:16]}", "role": "after"},
                ],
                "temporal_result": result,
                "evidence": [str(artifact)],
                "limitations": ["Learned change inference was not executed; this result uses deterministic spectral change fusion."],
            })
        except ValueError as exc:
            code = str(exc).split(":", 1)[0]
            raise QueryError(code.lower(), str(exc), 422) from exc
    _progress("analyzing", 52, "Running the task-specific GeoAI analysis.")
    output = execute_legacy(plan, inputs)
    _progress("analyzing", 64, "Core inference completed; validating worker output.")
    execution = output.get("execution", {})
    if not execution.get("success", False):
        print(json.dumps(json_safe(output), default=str))
        raise QueryError("analysis_failed", "The legacy analysis failed; inspect the job execution log.", 500)
    stats = execution.get("statistics") if isinstance(execution.get("statistics"), dict) else {}
    summary = execution.get("execution_summary") if isinstance(execution.get("execution_summary"), dict) else {}
    visual = execution.get("visual_change_assessment") if isinstance(execution.get("visual_change_assessment"), dict) else {}
    semantic = execution.get("semantic_change_assessment") if isinstance(execution.get("semantic_change_assessment"), dict) else {}
    quality_gate = semantic.get("quality_gate") if isinstance(semantic.get("quality_gate"), dict) else {}
    change_metrics = {
        "structural_change_percentage": execution.get("structural_change_percentage"),
        "structural_change_level": summary.get("structural_change_level"),
        "primary_evidence": execution.get("primary_evidence") or summary.get("primary_evidence"),
        "rgb_mean_absolute_difference": visual.get("mean_absolute_rgb_difference") or summary.get("rgb_mean_absolute_difference"),
        "rgb_pixels_difference_gt_20_pct": visual.get("pixels_difference_gt_20_pct") or summary.get("rgb_pixels_difference_gt_20_pct"),
        "semantic_quality_score": quality_gate.get("score") or summary.get("semantic_quality_score"),
        "semantic_reliable": summary.get("semantic_reliable"),
    }
    stats["change"] = {k: v for k, v in change_metrics.items() if v is not None}
    if temporal is not None:
        import gee_temporal as gee
        execution["answer"] = gee.build_change_narrative(before, after, execution.get("answer"), cross_sensor=temporal.cross_sensor)
        stats["imagery"] = {"before": before.stats, "after": after.stats}
        execution["imagery_provenance"] = [
            {"year": c.year, "collection_id": c.collection_id, "image_count": c.image_count, "scale_m": c.scale_m}
            for c in (before, after)
        ]
    execution["statistics"] = stats
    return json_safe(execution)


def main():
    try:
        payload = json.loads(Path("request.json").read_text(encoding="utf-8"))
        result = run(payload)
    except QueryError as exc:
        traceback.print_exc()
        result = {"error_code": exc.code, "error": str(exc), "status_code": exc.status_code}
    except Exception:
        traceback.print_exc()
        result = {"error_code": "worker_failed", "error": "Analysis worker failed; inspect the job execution log.", "status_code": 500}
    Path("worker-result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
