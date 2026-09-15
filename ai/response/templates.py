"""
Response templates for SatQuery-AI.

Keeps natural-language response wording separate
from analysis and orchestration logic.
"""

SEMANTIC_TEMPLATE = (
    "Semantic analysis completed. {description}"
)

MULTISPECTRAL_TEMPLATE = (
    "Multispectral analysis completed. {description}"
)

SAR_TEMPLATE = (
    "SAR analysis completed. {description}"
)

FUSION_TEMPLATE = (
    "Optical-SAR fusion completed. {description}"
)

CHANGE_TEMPLATE = (
    "Change detection completed. {description}"
)

ERROR_TEMPLATE = (
    "SatQuery-AI could not complete the requested analysis. "
    "{message}"
)

UNKNOWN_TEMPLATE = (
    "I could not determine which SatQuery-AI capability "
    "is required for this query."
)