from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskResult:
    """
    Standard result returned by a SatQuery-AI task executor.

    evidence:
        Structured evidence generated during analysis.

        Example:
        [
            {
                "type": "ndvi_map",
                "path": "data/evidence/ndvi_map.png"
            }
        ]
    """

    success: bool

    task: str

    message: str

    data: Dict[str, Any] = field(
        default_factory=dict
    )

    evidence: List[Any] = field(
        default_factory=list
    )

    error: Optional[str] = None