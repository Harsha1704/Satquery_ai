"""Small contract-level validation before a result is published."""
from __future__ import annotations

import math

from query_engine.policy import QueryError


def validate_result(answer, statistics: dict) -> None:
    if not isinstance(answer, str) or not answer.strip():
        raise QueryError("invalid_result", "The analysis returned no usable answer.", 500)
    def visit(value):
        if isinstance(value, dict):
            for nested in value.values(): visit(nested)
        elif isinstance(value, list):
            for nested in value: visit(nested)
        elif isinstance(value, float) and not math.isfinite(value):
            raise QueryError("invalid_result", "The analysis returned non-finite statistics.", 500)
    visit(statistics)


def validate_temporal_result(result: dict) -> None:
    """Validate temporal facts before a map/report publishes them."""
    required = ("result_id", "effective_roi", "statistics", "change_polygons", "provenance")
    missing = [key for key in required if key not in result]
    if missing:
        raise QueryError("result_validation_failed", f"Temporal result is missing {', '.join(missing)}.", 500)
    roi = result["effective_roi"]
    if not roi.get("crs") or len(roi.get("bounds") or []) != 4:
        raise QueryError("result_validation_failed", "Temporal result has no valid effective ROI CRS/bounds.", 500)
    stats = result["statistics"]
    for key in ("changed_percentage", "unchanged_percentage", "changed_area_m2"):
        value = stats.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise QueryError("result_validation_failed", f"Invalid temporal statistic {key}.", 500)
    if not 0 <= stats["changed_percentage"] <= 100 or not 0 <= stats["unchanged_percentage"] <= 100:
        raise QueryError("result_validation_failed", "Temporal percentages must be in [0, 100].", 500)
    for feature in result["change_polygons"].get("features", []):
        if feature.get("type") != "Feature" or not feature.get("geometry"):
            raise QueryError("result_validation_failed", "Invalid change polygon.", 500)
