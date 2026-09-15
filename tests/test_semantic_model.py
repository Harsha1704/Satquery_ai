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

    prediction = predictor.predict(image)

    assert "prediction_mask" in prediction
    assert "class_map" in prediction

    print("Predictor         : PASS")
    print("Class map         :", prediction["class_map"])

    print("\nSemantic Model Test: PASS")


if __name__ == "__main__":
    test_semantic_model()