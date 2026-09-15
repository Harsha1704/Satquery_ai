from dataclasses import dataclass, field
from typing import Dict, Any, List

from ai.query_router import Intent


@dataclass
class Task:
    query: str
    intent: Intent
    required_tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)