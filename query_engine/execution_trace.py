"""Public, non-reasoning execution trace construction."""
from __future__ import annotations

from query_engine.schemas import ExecutionTrace, TraceStep


def completed_trace(*, analysis_id: str, plan, duration_ms: int, warnings: list[str]) -> ExecutionTrace:
    """Summarize executed stages without exposing prompts or hidden reasoning."""
    steps = []
    for step in plan.steps:
        status = "success"
        if step.tool == "input.validator" and plan.input_configuration is None:
            status = "skipped"
        steps.append(TraceStep(
            step_id=step.step_id, tool=step.tool, status=status,
            output=step.expected_output,
        ))
    return ExecutionTrace(
        analysis_id=analysis_id,
        task=plan.parsed.intent.value,
        input_configuration=plan.input_configuration,
        selected_tools=plan.tools,
        steps=steps,
        warnings=warnings,
        duration_ms=max(0, duration_ms),
    )
