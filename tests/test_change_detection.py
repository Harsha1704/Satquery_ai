import numpy as np

from ai.change_detection import (
    ChangeDetectionEngine,
    ChangeStatistics
)


def test_change_detection():

    print("\nSatQuery Change Detection Test")
    print("=" * 55)

    # -------------------------------------------------
    # Create synthetic BEFORE observation
    # -------------------------------------------------

    before = np.ones(
        (100, 100),
        dtype=np.float32
    ) * 0.20

    # -------------------------------------------------
    # Create synthetic AFTER observation
    # -------------------------------------------------

    after = before.copy()

    # Simulate a changed region.
    after[25:75, 25:75] = 0.80

    detector = ChangeDetectionEngine()
    statistics_engine = ChangeStatistics()

    # -------------------------------------------------
    # Detect change
    # -------------------------------------------------

    result = detector.detect(
        before,
        after,
        threshold=0.20
    )

    difference = result["difference"]
    change_mask = result["change_mask"]

    print("Before image created : PASS")
    print("After image created  : PASS")
    print("Difference map       : PASS")
    print("Change mask          : PASS")

    # -------------------------------------------------
    # Calculate statistics
    # -------------------------------------------------

    statistics = statistics_engine.calculate(
        difference,
        change_mask
    )

    classification = statistics_engine.classify(
        statistics["change_percentage"]
    )

    print("\nChange Statistics")
    print("-" * 55)

    print(
        "Total pixels     :",
        statistics["total_pixels"]
    )

    print(
        "Changed pixels   :",
        statistics["changed_pixels"]
    )

    print(
        "Change percentage:",
        f"{statistics['change_percentage']:.2f}%"
    )

    print(
        "Mean difference  :",
        f"{statistics['mean_difference']:.4f}"
    )

    print(
        "Max difference   :",
        f"{statistics['max_difference']:.4f}"
    )

    print(
        "Classification   :",
        classification
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    assert difference.shape == before.shape
    assert change_mask.shape == before.shape

    assert statistics["total_pixels"] == 10000

    assert (
        statistics["changed_pixels"] > 0
    )

    assert (
        statistics["change_percentage"] > 0
    )

    print("\nAll change detection tests passed.")


if __name__ == "__main__":
    test_change_detection()