from __future__ import annotations

from pathlib import Path

from ai.router.orchestrator import SatQueryOrchestrator


ROOT = Path(__file__).resolve().parents[1]

BEFORE = (
    ROOT
    / "outputs"
    / "temporal_high_change"
    / "navi_mumbai_airport"
    / "navi_mumbai_airport_2017.tif"
)

AFTER = (
    ROOT
    / "outputs"
    / "temporal_high_change"
    / "navi_mumbai_airport"
    / "navi_mumbai_airport_2025.tif"
)


def main():
    if not BEFORE.exists():
        raise FileNotFoundError(
            f"Missing test image: {BEFORE}"
        )

    if not AFTER.exists():
        raise FileNotFoundError(
            f"Missing test image: {AFTER}"
        )

    visual = (
        SatQueryOrchestrator
        ._assess_visual_pair(
            str(BEFORE),
            str(AFTER),
        )
    )

    print("=" * 78)
    print("SATQUERY BI-TEMPORAL VISUAL CHANGE TEST")
    print("=" * 78)
    print("Available:", visual.get("available"))
    print("Level:", visual.get("level"))
    print(
        "Mean absolute RGB difference:",
        visual.get("mean_absolute_rgb_difference"),
    )
    print(
        "Pixels > 10:",
        visual.get("pixels_difference_gt_10_pct"),
        "%",
    )
    print(
        "Pixels > 20:",
        visual.get("pixels_difference_gt_20_pct"),
        "%",
    )
    print(
        "Pixels > 30:",
        visual.get("pixels_difference_gt_30_pct"),
        "%",
    )

    answer = (
        SatQueryOrchestrator
        ._build_combined_change_answer(
            visual_assessment=visual,
            changed_percentage=0.0,
            before_year=2017,
            after_year=2025,
            fallback_answer=(
                "No significant change was detected."
            ),
        )
    )

    print()
    print("COMBINED ANSWER")
    print("-" * 78)
    print(answer)

    assert visual["available"] is True
    assert visual["level"] == "significant"
    assert (
        visual["pixels_difference_gt_20_pct"]
        > 25.0
    )
    assert "Substantial visual differences" in answer
    assert "0.00%" in answer

    print()
    print("TEST PASSED")


if __name__ == "__main__":
    main()
