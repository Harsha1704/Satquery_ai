# ai/execution/summary.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .confidence import ConfidenceManager


class ExecutionSummaryBuilder:
    """
    Produces one consistent observable execution summary
    for all SatQuery AI operations.
    """

    def __init__(self):
        self.confidence_manager = ConfidenceManager()

    def build(
        self,
        *,
        query: str,
        intent: str,
        success: bool,
        tools: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        confidence_type: str = "model_score",
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        evidence: Optional[Any] = None,
        limitations: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:

        confidence_result = (
            self.confidence_manager.normalize(
                confidence,
                confidence_type,
            )
        )

        execution_steps = []

        if validation is not None:
            execution_steps.append(
                {
                    "step": 1,
                    "name": "input_validation",
                    "status": (
                        "passed"
                        if validation.get("valid", False)
                        else "failed"
                    ),
                }
            )

        execution_steps.append(
            {
                "step": len(execution_steps) + 1,
                "name": "query_interpretation",
                "status": "completed",
                "intent": intent,
            }
        )

        if tools:
            execution_steps.append(
                {
                    "step": len(execution_steps) + 1,
                    "name": "tool_selection",
                    "status": "completed",
                    "tools": tools,
                }
            )

        execution_steps.append(
            {
                "step": len(execution_steps) + 1,
                "name": "analysis_execution",
                "status": (
                    "completed"
                    if success
                    else "failed"
                ),
            }
        )

        if evidence:
            execution_steps.append(
                {
                    "step": len(execution_steps) + 1,
                    "name": "evidence_generation",
                    "status": "completed",
                }
            )

        return {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),

            "query": query,

            "intent": intent,

            "success": bool(success),

            "message": message,

            "model": model,

            "device": device,

            "tools_used": tools or [],

            "inputs": inputs or {},

            "outputs": outputs or {},

            "validation": validation,

            "confidence": confidence_result.to_dict(),

            "evidence": evidence,

            "limitations": limitations or [],

            "execution_steps": execution_steps,
        }