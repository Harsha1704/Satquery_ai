# ai/execution/confidence.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ConfidenceResult:
    score: float
    percentage: float
    level: str
    confidence_type: str
    calibrated: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceManager:
    """
    Standard confidence formatter for SatQuery AI.

    Important:
    - Does NOT pretend every model confidence is calibrated.
    - Preserves the semantic meaning of each score.
    - Converts all scores into one consistent 0..1 representation.
    """

    CALIBRATED_TYPES = {
        "probability",
        "calibrated_probability",
    }

    RELATIVE_TYPES = {
        "relative_tile_relevance",
        "similarity",
        "relative_similarity",
        "ranking_score",
    }

    HEURISTIC_TYPES = {
        "heuristic",
        "rule_based",
        "routing_confidence",
    }

    def normalize(
        self,
        score: Optional[float],
        confidence_type: str = "model_score",
    ) -> ConfidenceResult:

        if score is None:
            score = 0.0

        score = float(score)

        if score > 1.0 and score <= 100.0:
            score /= 100.0

        score = max(0.0, min(1.0, score))

        confidence_type = (
            confidence_type or "model_score"
        ).strip().lower()

        calibrated = (
            confidence_type in self.CALIBRATED_TYPES
        )

        level = self._level(score)

        interpretation = self._interpretation(
            score=score,
            confidence_type=confidence_type,
            calibrated=calibrated,
        )

        return ConfidenceResult(
            score=round(score, 4),
            percentage=round(score * 100.0, 2),
            level=level,
            confidence_type=confidence_type,
            calibrated=calibrated,
            interpretation=interpretation,
        )

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.85:
            return "high"

        if score >= 0.65:
            return "moderate"

        if score >= 0.40:
            return "low"

        return "very_low"

    def _interpretation(
        self,
        score: float,
        confidence_type: str,
        calibrated: bool,
    ) -> str:

        percentage = round(score * 100.0, 2)

        if calibrated:
            return (
                f"Calibrated model probability: "
                f"{percentage}%."
            )

        if confidence_type in self.RELATIVE_TYPES:
            return (
                f"Relative relevance score: "
                f"{percentage}%. "
                f"This is not a calibrated probability."
            )

        if confidence_type in self.HEURISTIC_TYPES:
            return (
                f"Heuristic confidence score: "
                f"{percentage}%. "
                f"This is rule-based and is not a calibrated probability."
            )

        return (
            f"Model-derived confidence score: "
            f"{percentage}%. "
            f"Calibration has not been established."
        )