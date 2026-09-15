"""Replaceable natural-language parser with conservative paraphrase normalization.

The original query is always preserved in the returned contract. Normalization
only helps the tested legacy router understand common natural-language variants;
it never invents a new executable capability.
"""
import re
from datetime import datetime, timezone

from query_engine.schemas import Intent, ParsedQuery


_ALIAS_PATTERNS = (
    (r"\bgreenery\b", "vegetation"),
    (r"\bgreen\s+cover\b", "vegetation"),
    (r"\bplant\s+health\b", "vegetation"),
    (r"\bcrop\s+health\b", "vegetation"),
    (r"\bconstruction\s+growth\b", "urban expansion"),
    (r"\bdevelopment\s+growth\b", "urban expansion"),
    (r"\bnew\s+development\b", "urban expansion"),
    (r"\bbuilt\s*up\b", "built-up"),
    (r"\bwater\s+extent\b", "water change"),
    (r"\bwater\s+body\s+shrink(?:age|ing)?\b", "water decrease"),
    (r"\bwater\s+body\s+expand(?:ed|ing)?\b", "water increase"),
    (r"\binundated\b", "flooded"),
    (r"\binundation\b", "flood"),
    (r"\bwhat\s+is\s+different\b", "what changed"),
    (r"\bhow\s+is\s+this\s+area\s+different\b", "what changed in this area"),
)


def _normalize_for_router(query: str) -> str:
    normalized = query
    for pattern, replacement in _ALIAS_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


class QueryParser:
    def parse(self, query: str) -> ParsedQuery:
        from ai.router.planner import QueryPlanner

        original_query = query.strip()
        router_query = _normalize_for_router(original_query)
        result = QueryPlanner().plan(router_query)
        years = list(result.years)

        # Resolve relative dates explicitly rather than silently using a hidden
        # default. The current year is used only when the user expresses a
        # relative period (e.g. "last 5 years") or an open-ended "since 2021".
        match = re.search(
            r"\b(?:last|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b",
            original_query.lower(),
        )
        current = datetime.now(timezone.utc).year
        if match:
            words = "one two three four five six seven eight nine ten".split()
            token = match.group(1)
            count = int(token) if token.isdigit() else words.index(token) + 1
            years = [current - count, current]
        elif len(years) == 1 and re.search(r"\b(?:since|from|after)\b", original_query.lower()):
            years = [years[0], current]

        return ParsedQuery(
            query=original_query,
            intent=Intent(result.intent.value),
            targets=result.targets,
            operation=result.operation,
            years=years,
            routing_confidence=result.confidence,
            change_direction=result.change_direction,
            transition_from=result.transition_from,
            transition_to=result.transition_to,
            parser="legacy_rules_v1+paraphrase_normalization",
        )
