"""Executable capability contract for SatQuery worldwide AOI analysis.

This module is intentionally independent from the legacy language router.  It
answers one question before planning: can the currently deployed executor
truthfully perform the requested *kind* of analysis?  Natural-language freedom
is preserved, but unsupported object-level tasks are never coerced into a
change workflow.
"""
from __future__ import annotations

import re
from typing import Any

from query_engine.policy import QueryError

WORLDWIDE_WORKFLOWS: tuple[dict[str, Any], ...] = (
    {
        "id": "vegetation_change",
        "label": "Vegetation change",
        "phrases": ["vegetation", "greenery", "green cover", "crop", "forest", "ndvi"],
        "primary_evidence": "NDVI temporal change",
        "source_policy": "Sentinel-2 when available; Landsat for historical coverage",
    },
    {
        "id": "built_up_change",
        "label": "Urban / built-up change",
        "phrases": ["urban", "built-up", "built up", "construction", "development", "ndbi"],
        "primary_evidence": "NDBI temporal change",
        "source_policy": "Sentinel-2 when available; Landsat for historical coverage",
    },
    {
        "id": "water_change",
        "label": "Water / flood extent change",
        "phrases": ["water", "lake", "river", "reservoir", "flood", "inundation", "ndwi"],
        "primary_evidence": "Optical water-index temporal change",
        "source_policy": "Validated optical temporal route; SAR is advertised only when a SAR executor is enabled",
    },
    {
        "id": "general_temporal_change",
        "label": "General temporal change",
        "phrases": ["what changed", "compare", "difference", "change"],
        "primary_evidence": "Multi-evidence temporal comparison",
        "source_policy": "Best executable Earth-observation source for both dates",
    },
)

_OBJECT_NOUNS = (
    "car", "cars", "vehicle", "vehicles", "truck", "trucks", "bus", "buses",
    "motorcycle", "motorcycles", "person", "people", "pedestrian", "pedestrians",
    "aircraft", "airplane", "airplanes", "plane", "planes", "ship", "ships",
    "boat", "boats", "building", "buildings", "house", "houses",
)
_OBJECT_ACTION = re.compile(r"\b(?:count|counting|how many|number of|detect|find|locate|identify|track)\b", re.I)
_OBJECT_NOUN = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in _OBJECT_NOUNS) + r")\b", re.I)


def enforce_worldwide_capability(query: str) -> None:
    """Reject unsupported object-level requests before the legacy router runs."""
    q = re.sub(r"\s+", " ", str(query or "")).strip()
    if _OBJECT_NOUN.search(q) and (_OBJECT_ACTION.search(q) or re.search(r"\bevery\s+", q, re.I)):
        raise QueryError(
            "unsupported_aoi_analysis",
            "This worldwide AOI executor does not currently support validated object-level counting or detection "
            "for individual cars, people, buildings, ships, aircraft, or similar objects. You can ask natural-language "
            "questions about vegetation change, urban/built-up change, water/flood extent change, or general temporal "
            "change between two dates.",
            422,
        )


def public_capability_contract() -> dict[str, Any]:
    return {
        "version": "2026-09-15",
        "mode": "worldwide_temporal_aoi",
        "natural_language": True,
        "free_form": True,
        "unlimited_capabilities": False,
        "supported_workflows": list(WORLDWIDE_WORKFLOWS),
        "unsupported_examples": [
            "Count every car in this city",
            "Locate every person in this AOI",
            "Count individual buildings from medium-resolution EO imagery",
        ],
        "truthfulness_policy": "Unsupported tasks are rejected; they are never silently mapped to a different workflow.",
    }
