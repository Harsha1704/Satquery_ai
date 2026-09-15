from email.mime import image

import numpy as np

from ai.semantic import (
    SemanticModel,
    SemanticPredictor
)


def test_semantic_model():

    print("\nSatQuery Semantic Model Test")
    print("=" * 55)

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    model = SemanticModel()

    result = model.predict(image)

    print(
        "Prediction shape :",
        result.shape
    )

    print(
        "Prediction dtype :",
        result.dtype
    )

    print(
        "Unique classes   :",
        np.unique(result)
    )

    assert result.shape == (100, 100)
    assert result.dtype == np.uint8

    print("Model prediction : PASS")

    predictor = SemanticPredictor(model)

    # Test raw prediction API
    prediction_mask = predictor.predict(image)

    assert isinstance(prediction_mask, np.ndarray)
    assert prediction_mask.shape == (100, 100)
    assert prediction_mask.dtype == np.uint8

    print("Raw prediction   : PASS")

    # Test metadata prediction API
    prediction = predictor.predict_with_metadata(image)

    assert "mask" in prediction
    assert "class_map" in prediction
    assert "shape" in prediction
    assert "dtype" in prediction
    assert "classes" in prediction

    assert prediction["mask"].shape == (100, 100)
    assert prediction["mask"].dtype == np.uint8

    print("Metadata prediction: PASS")

    print("Predictor         : PASS")
    print("Class map         :", prediction["class_map"])

    print("\nSemantic Model Test: PASS")


if __name__ == "__main__":
    test_semantic_model()