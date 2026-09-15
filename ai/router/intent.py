# ai/router/intent.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Intent(str, Enum):
    SEMANTIC_ANALYSIS = "semantic_analysis"
    SINGLE_IMAGE_VQA = "single_image_vqa"
    OBJECT_DETECTION = "object_detection"
    MULTISPECTRAL_ANALYSIS = "multispectral_analysis"
    SAR_ANALYSIS = "sar_analysis"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    CHANGE_DETECTION = "change_detection"
    UNKNOWN = "unknown"
    TEXT_GUIDED_GROUNDING = "text_guided_grounding"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    matched_keywords: List[str]

    target: str = "overall"
    targets: List[str] = field(default_factory=list)

    operation: str = "analyze"
    change_direction: str = "none"

    transition_from: str = "none"
    transition_to: str = "none"

    years: List[int] = field(default_factory=list)
    original_query: str = ""

    def to_dict(self):
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "matched_keywords": self.matched_keywords,
            "target": self.target,
            "targets": self.targets,
            "operation": self.operation,
            "change_direction": self.change_direction,
            "years": self.years,
            "original_query": self.original_query,
            "transition_from": self.transition_from,
            "transition_to": self.transition_to,
        }


__all__ = [
    "Intent",
    "IntentResult",
]