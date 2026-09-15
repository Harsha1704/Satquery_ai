from query_engine.capabilities import enforce_worldwide_capability
from query_engine.capability_validator import CapabilityValidator
from query_engine.parser import QueryParser
from query_engine.policy import QueryError
from query_engine.schemas import AnalysisPlan, AnalysisRequest, InputConfiguration, Intent, PlanStep
from query_engine.tool_registry import allowed_tools
from query_engine.validators import validate_request


class QueryPlanner:
    def __init__(self, parser=None, capability_validator=None):
        self.parser = parser or QueryParser()
        self.capability_validator = capability_validator or CapabilityValidator()

    def plan(self, request: AnalysisRequest, input_configuration: InputConfiguration | None = None) -> AnalysisPlan:
        if request.aoi is not None:
            enforce_worldwide_capability(request.query)
        parsed = self.parser.parse(request.query)
        warnings = ["Routing uses the legacy rule parser; its score is not calibrated model confidence."]
        if request.inputs.before_path and request.inputs.after_path:
            parsed = parsed.model_copy(update={"intent": Intent.CHANGE, "operation": "compare", "routing_confidence": None})
            warnings.append("The explicit before/after image pair selects change analysis.")
        source = "earth_engine" if request.aoi is not None else "local"
        if (request.before_year is None) != (request.after_year is None):
            raise QueryError("invalid_years", "Supply both before_year and after_year.")
        if request.before_year is not None:
            parsed = parsed.model_copy(update={"years": [request.before_year, request.after_year]})
        tools = allowed_tools(parsed.intent, source)
        validate_request(request, parsed.intent, parsed.years)
        capability = None
        if source == "earth_engine" or input_configuration is not None:
            capability = self.capability_validator.enforce(parsed, input_configuration, source)
        else:
            warnings.append("Input capability validation is deferred until server-resolved files are inspected.")
        steps = [
            PlanStep(step_id="validate_inputs", tool="input.validator", expected_output="validated_inputs"),
        ]
        if parsed.intent == Intent.CHANGE:
            steps.append(PlanStep(step_id="validate_alignment", tool="geospatial.alignment", dependencies=["validate_inputs"], expected_output="aligned_pair"))
        dependencies = [steps[-1].step_id]
        for index, tool in enumerate(tools, start=1):
            step_id = f"execute_{index}_{tool}"
            steps.append(PlanStep(step_id=step_id, tool=tool, dependencies=dependencies, expected_output="analysis_result"))
            dependencies = [step_id]
        steps.append(PlanStep(step_id="validate_result", tool="result.validator", dependencies=dependencies, expected_output="validated_result"))
        steps.append(PlanStep(step_id="publish_evidence", tool="evidence.publisher", dependencies=["validate_result"], expected_output="evidence_package"))
        return AnalysisPlan(
            parsed=parsed, tools=tools, source=source, warnings=warnings,
            input_configuration=input_configuration, capability=capability, steps=steps,
        )
