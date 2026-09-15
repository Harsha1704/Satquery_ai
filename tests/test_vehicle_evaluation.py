from math import isclose

from training.evaluate_vehicle_detector import _average_precision_at_50, _match_predictions


def test_vehicle_evaluation_matches_one_prediction_per_ground_truth() -> None:
    ground_truth = [
        {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
        {"x1": 50.0, "y1": 50.0, "x2": 70.0, "y2": 70.0},
    ]
    predictions = [
        {"confidence": 0.95, "bbox": {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0}},
        {"confidence": 0.90, "bbox": {"x1": 11.0, "y1": 11.0, "x2": 29.0, "y2": 29.0}},
        {"confidence": 0.80, "bbox": {"x1": 50.0, "y1": 50.0, "x2": 70.0, "y2": 70.0}},
    ]

    matched, missed = _match_predictions(predictions, ground_truth, 0.5)

    assert [item["matched"] for item in matched] == [True, False, True]
    assert missed == []
    assert isclose(_average_precision_at_50(matched, len(ground_truth)), 5 / 6)
