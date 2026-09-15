from datetime import datetime, timezone

from query_engine.policy import QueryError
from query_engine.schemas import AnalysisRequest, Intent


def validate_request(request: AnalysisRequest, intent: Intent, years: list[int]) -> None:
    inputs = request.inputs.model_dump(exclude_none=True)
    if request.aoi is not None:
        if inputs:
            raise QueryError("ambiguous_source", "Supply either an AOI or local images, not both.")
        if len(years) != 2 or not 1985 <= years[0] < years[1] <= datetime.now(timezone.utc).year:
            raise QueryError("invalid_years", "Temporal AOI analysis needs a clear time range. Include two years (for example 2020 and 2026), use 'since 2021', or say 'last 5 years'.")
        return
    if request.before_year is not None or request.after_year is not None:
        raise QueryError("unused_years", "Explicit year fields apply to AOI imagery retrieval only.")
    required = {
        Intent.CHANGE: {"before_path", "after_path"},
        Intent.FUSION: {"optical_path", "sar_path"},
        Intent.SAR: {"sar_path"} if "sar_path" in inputs else {"image_path"},
    }.get(intent, {"image_path"})
    if set(inputs) != required:
        raise QueryError("invalid_inputs", "This analysis requires exactly: " + ", ".join(sorted(required)))
