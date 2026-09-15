"""Adapters preserve the approved plan without modifying the legacy engine."""
from query_engine.schemas import AnalysisPlan


class ApprovedPlanner:
    def __init__(self, plan: AnalysisPlan):
        self.approved = plan.parsed

    def plan(self, query):
        from ai.router.intent import Intent, IntentResult

        p = self.approved
        return IntentResult(
            intent=Intent(p.intent.value), confidence=p.routing_confidence or 0.0,
            matched_keywords=[], target=p.targets[0] if p.targets else "overall",
            targets=p.targets, operation=p.operation, years=p.years,
            change_direction=p.change_direction, transition_from=p.transition_from,
            transition_to=p.transition_to, original_query=p.query,
        )

    def required_tools(self, intent):
        from ai.router.planner import QueryPlanner
        return QueryPlanner().required_tools(intent)


def execute_legacy(plan: AnalysisPlan, inputs: dict):
    from ai.router.orchestrator import SatQueryOrchestrator
    return SatQueryOrchestrator(planner=ApprovedPlanner(plan)).execute(
        query=plan.parsed.query, **inputs,
    )
