from math import isclose

from training.validate_vehicle_predictions import _summarize_review


def test_review_metrics_report_false_positive_categories() -> None:
    metrics = _summarize_review(
        [
            {"review_status": "true_positive"},
            {"review_status": "true_positive"},
            {
                "review_status": "false_positive",
                "false_positive_category": "duplicate",
            },
            {
                "review_status": "false_positive",
                "false_positive_category": "parking_marking",
            },
        ],
        ground_truth=3,
    )

    assert metrics["predicted_cars"] == 4
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 2
    assert metrics["false_positive_by_category"] == {
        "duplicate": 1,
        "parking_marking": 1,
    }
    assert metrics["missed_cars"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 2 / 3
    assert isclose(metrics["counting_error_percent"], 100 / 3)
