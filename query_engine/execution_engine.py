"""Dependency validation for approved plans.

Specialist inference remains in the isolated legacy worker for compatibility;
this engine owns the plan-level invariant that no step can run before a
declared dependency has completed.
"""
from __future__ import annotations

from query_engine.policy import QueryError


def validate_step_graph(plan) -> None:
    available: set[str] = set()
    ids: set[str] = set()
    for step in plan.steps:
        if step.step_id in ids:
            raise QueryError("invalid_plan", f"Duplicate execution step {step.step_id!r}.", 500)
        ids.add(step.step_id)
        missing = set(step.dependencies) - available
        if missing:
            raise QueryError(
                "invalid_plan",
                f"Step {step.step_id!r} has unmet dependencies: {', '.join(sorted(missing))}.",
                500,
            )
        available.add(step.step_id)
