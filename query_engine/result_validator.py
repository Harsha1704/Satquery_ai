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
