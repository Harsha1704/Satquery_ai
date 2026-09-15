# ai/execution/result_adapter.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .summary import ExecutionSummaryBuilder


class ResultAdapter:
    """
    Adds standardized confidence and execution metadata
    to existing SatQuery result dictionaries without
    changing their original fields.
    """

    def __init__(self):
        self.summary_builder = ExecutionSummaryBuilder()

    def standardize(
        self,
        result: Dict[str, Any],
        *,
        query: str,
        intent: str,
        tools: Optional[List[str]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        default_confidence_type: str = "model_score",
    ) -> Dict[str, Any]:

        standardized = dict(result)

        raw_confidence = result.get(
            "confidence",
            result.get(
                "score",
                result.get(
                    "routing_confidence",
                    0.0,
                ),
            ),
        )

        confidence_type = result.get(
            "confidence_type",
            default_confidence_type,
        )

        evidence = result.get(
            "evidence",
            result.get(
                "evidence_path",
                result.get(
                    "visual_evidence"
                ),
            ),
        )

        model = result.get(
            "model",
            result.get(
                "model_name"
            ),
        )

        device = result.get("device")

        limitations = result.get(
            "limitations",
            [],
        )

        if isinstance(limitations, str):
            limitations = [limitations]

        success = bool(
            result.get(
                "success",
                True,
            )
        )

        summary = self.summary_builder.build(
            query=query,
            intent=intent,
            success=success,
            tools=tools,
            confidence=raw_confidence,
            confidence_type=confidence_type,
            inputs=inputs,
            outputs=self._compact_outputs(result),
            validation=validation,
            model=model,
            device=device,
            evidence=evidence,
            limitations=limitations,
            message=result.get(
                "message",
                result.get(
                    "answer"
                ),
            ),
        )

        standardized["confidence_details"] = (
            summary["confidence"]
        )

        standardized["execution_summary"] = summary

        return standardized

    @staticmethod
    def _compact_outputs(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        skip = {
            "execution_summary",
            "confidence_details",
        }

        compact = {}

        for key, value in result.items():

            if key in skip:
                continue

            if key in {
                "mask",
                "array",
                "image",
                "heatmap",
                "feature_map",
            }:
                continue

            compact[key] = value

        return compact