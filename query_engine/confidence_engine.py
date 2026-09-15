"""Truthful, quality-gated confidence reporting."""
from __future__ import annotations

from typing import Any

from query_engine.schemas import Confidence, InputConfiguration


_LEVELS = ((0.90, "VERY_HIGH"), (0.70, "HIGH"), (0.40, "MODERATE"), (0.0, "LOW"))


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def _component(value: float | None, source: str, details: dict | None = None) -> dict:
    return {"value": value, "source": source if value is not None else "UNAVAILABLE", "calibrated": False, "details": details or {}}


def _level(value: float | None) -> str | None:
    return next((label for lower, label in _LEVELS if value is not None and value >= lower), None)


def _input_scores(configuration: InputConfiguration | None) -> tuple[float | None, float | None, list[str]]:
    if configuration is None:
        return None, None, ["Input metadata was not available for confidence assessment."]
    if configuration.image_count == 0 and configuration.kind.value == "aoi_temporal":
        return 1.0, None, []
    values = [image.modality_confidence for image in configuration.images if image.modality_confidence is not None]
    modality = round(sum(values) / len(values), 2) if values else None
    if any(issue.get("level") == "error" for issue in configuration.validation_issues):
        return 0.0, modality, ["Input validation reported incompatible raster metadata."]
    return (1.0 if configuration.images else None), modality, []


def build_confidence(*, routing: float | None, validation: dict | None,
                     input_configuration: InputConfiguration | None = None,
                     model_confidence: float | None = None,
                     model_source: str = "UNAVAILABLE") -> Confidence:
    """Build reliability metadata without fabricating a model probability.

    Overall reliability starts from the weakest available critical component
    (input, evidence, spatial). Supporting signals may add at most 0.05, and a
    spatial score below 0.60 is a hard ceiling for map-linked conclusions.
    """
    validation = validation or {}
    warnings: list[str] = []
    routing = _number(routing)
    input_score, modality_score, input_warnings = _input_scores(input_configuration)
    warnings.extend(input_warnings)
    status = str(validation.get("status") or "").lower()
    evidence = {"good": 0.85, "moderate": 0.60, "review": 0.35}.get(status)
    alignment = validation.get("alignment") or {}
    alignment_status = str(alignment.get("status") or "").lower()
    alignment_score = 1.0 if alignment_status == "passed" else (0.45 if alignment_status == "warning" else None)
    coverage_raw = validation.get("valid_coverage_pct")
    coverage = _number(float(coverage_raw) / 100) if coverage_raw is not None else None
    spatial_candidates = [value for value in (alignment_score, coverage) if value is not None]
    spatial = min(spatial_candidates) if spatial_candidates else None
    if coverage is not None and coverage < 0.85:
        warnings.append(f"Only {coverage * 100:.1f}% valid AOI coverage was reported.")
    if alignment_status == "warning":
        warnings.append("Spatial alignment warning limits map-linked reliability.")
    if status == "review":
        warnings.append("Evidence validation requires review; it is not calibrated model accuracy.")
    model_confidence = _number(model_confidence)
    if model_confidence is None:
        warnings.append("Model confidence is unavailable; successful execution was not treated as a model probability.")
    critical = [value for value in (input_score, evidence, spatial) if value is not None]
    overall = None
    if critical:
        weakest = min(critical)
        supporting = [value for value in (routing, modality_score, coverage) if value is not None]
        uplift = min(0.05, max(0.0, ((sum(supporting) / len(supporting)) - weakest) * 0.10)) if supporting else 0.0
        overall = round(min(1.0, weakest + uplift), 2)
        if spatial is not None and spatial < 0.60:
            overall = min(overall, spatial)
    provenance = {
        "routing_confidence": _component(routing, "ROUTER"),
        "input_confidence": _component(input_score, "QUALITY_GATE"),
        "modality_confidence": _component(modality_score, "METADATA"),
        "model_confidence": _component(model_confidence, model_source),
        "evidence_confidence": _component(evidence, "QUALITY_GATE", {"validation_status": status or None}),
        "spatial_confidence": _component(spatial, "QUALITY_GATE", {"alignment": alignment_status or None, "valid_coverage_pct": coverage_raw}),
        "data_quality_confidence": _component(coverage, "QUALITY_GATE", {"valid_coverage_pct": coverage_raw}),
        "overall_confidence": _component(overall, "HEURISTIC", {"type": "SYSTEM_RELIABILITY_SCORE"}),
    }
    return Confidence(
        routing=routing, model=model_confidence, data_quality=coverage, alignment_quality=alignment_score,
        evidence_strength=evidence, overall=overall,
        method="quality_gated_weakest_critical_component" if overall is not None else None,
        routing_confidence=routing, input_confidence=input_score, modality_confidence=modality_score,
        model_confidence=model_confidence, evidence_confidence=evidence, spatial_confidence=spatial,
        data_quality_confidence=coverage, overall_confidence=overall,
        confidence_available=overall is not None,
        confidence_method="quality_gated_weakest_critical_component" if overall is not None else None,
        confidence_level=_level(overall), overall_type="SYSTEM_RELIABILITY_SCORE" if overall is not None else "UNAVAILABLE",
        confidence_provenance=provenance, confidence_warnings=warnings,
    )
