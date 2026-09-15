from __future__ import annotations

from pathlib import Path
from pprint import pprint

from ai.router.orchestrator import SatQueryOrchestrator

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "outputs" / "temporal_high_change" / "noida_airport_jewar" / "noida_airport_jewar_2017.tif"
AFTER = ROOT / "outputs" / "temporal_high_change" / "noida_airport_jewar" / "noida_airport_jewar_2025.tif"


def main() -> None:
    if not BEFORE.exists():
        raise FileNotFoundError(BEFORE)
    if not AFTER.exists():
        raise FileNotFoundError(AFTER)

    o = SatQueryOrchestrator()
    result = o.execute(
        "What changed between 2017 and 2025?",
        before_path=str(BEFORE),
        after_path=str(AFTER),
    )
    execution = result.get("execution", result)

    print("=" * 80)
    print("SATQUERY BI-TEMPORAL SEMANTIC QUALITY-GATE TEST")
    print("=" * 80)
    print("Success:", execution.get("success"))
    print("Primary evidence:", execution.get("primary_evidence"))
    print()
    print("ANSWER")
    print("-" * 80)
    print(execution.get("answer"))
    print()
    print("SEMANTIC QUALITY GATE")
    print("-" * 80)
    semantic = execution.get("semantic_change_assessment", {})
    pprint(semantic.get("quality_gate", {}))
    print()
    print("SEMANTIC PERCENTAGES")
    print("-" * 80)
    print("Before:")
    pprint(semantic.get("before_percentages", {}))
    print("After:")
    pprint(semantic.get("after_percentages", {}))
    print()
    print("EXECUTION SUMMARY")
    print("-" * 80)
    pprint(execution.get("execution_summary", {}))

    assert execution.get("success") is True
    assert semantic.get("available") is True
    assert "unknown" in semantic.get("before_percentages", {})
    assert "unknown" in semantic.get("after_percentages", {})
    assert semantic.get("reliable") is False
    assert semantic.get("quality_gate", {}).get("passed") is False
    assert execution.get("primary_evidence") != "semantic"
    answer = str(execution.get("answer", ""))
    assert "quality gate failed" in answer
    assert "does not support a definitive semantic land-cover transition" in answer
    assert "bare land decreased by 98.7" not in answer.lower()

    print()
    print("TEST PASSED")


if __name__ == "__main__":
    main()
