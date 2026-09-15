"""Conservative confidence fusion; absent specialist scores remain absent."""
from __future__ import annotations

from query_engine.schemas import Confidence


def build_confidence(*, routing: float | None, validation: dict | None) -> Confidence:
    validation = validation or {}
    quality = str(validation.get("status") or "").lower()
    evidence = {"good": 0.85, "moderate": 0.60, "review": 0.35}.get(quality)
    alignment = validation.get("alignment") or {}
    alignment_score = 1.0 if alignment.get("status") == "passed" else None
    components = [value for value in (routing, evidence, alignment_score) if value is not None]
    overall = sum(components) / len(components) if components else None
    return Confidence(
        routing=routing,
        evidence_strength=evidence,
        alignment_quality=alignment_score,
        overall=overall,
        method="mean_of_available_routing_and_evidence_components" if overall is not None else None,
    )
