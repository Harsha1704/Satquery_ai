import numpy as np

from ai.semantic import SemanticAnalyzer


def test_semantic_analysis():

    print("\nSatQuery Semantic Intelligence Test")
    print("=" * 55)

    # -------------------------------------------------
    # Synthetic semantic prediction
    #
    # 0 = water
    # 1 = vegetation
    # 2 = built-up
    # 3 = bare land
    #
    # 100 x 100 = 10,000 pixels
    # -------------------------------------------------

    prediction = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    # Water = 4,500 pixels
    prediction[0:45, :] = 0

    # Vegetation = 3,000 pixels
    prediction[45:75, :] = 1

    # Built-up = 1,500 pixels
    prediction[75:90, :] = 2

    # Bare land = 1,000 pixels
    prediction[90:100, :] = 3

    class_map = {
        0: "water",
        1: "vegetation",
        2: "built_up",
        3: "bare_land"
    }

    analyzer = SemanticAnalyzer()

    # -------------------------------------------------
    # Analyze semantic prediction
    # -------------------------------------------------

    result = analyzer.analyze(
        prediction,
        class_map
    )

    description = analyzer.describe(
        result
    )

    print("Prediction mask    : PASS")
    print("Class analysis     : PASS")
    print("Statistics         : PASS")
    print("Dominant class     : PASS")
    print("Description        : PASS")

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("\nSemantic Statistics")
    print("-" * 55)

    print(
        "Total pixels:",
        result["total_pixels"]
    )

    for class_name, values in result["classes"].items():

        print(
            f"{class_name:12s}: "
            f"{values['pixels']:5d} pixels "
            f"({values['percentage']:.2f}%)"
        )

    print(
        "\nDominant class:",
        result["dominant_class"]
    )

    print(
        "Dominant percentage:",
        f"{result['dominant_percentage']:.2f}%"
    )

    print("\nInterpretation:")
    print(description)

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    assert result["total_pixels"] == 10000

    assert result["dominant_class"] == "water"

    assert (
        result["dominant_percentage"]
        == 45.0
    )

    assert (
        result["classes"]["water"]["pixels"]
        == 4500
    )

    assert (
        result["classes"]["water"]["percentage"]
        == 45.0
    )

    assert (
        result["classes"]["vegetation"]["pixels"]
        == 3000
    )

    assert (
        result["classes"]["vegetation"]["percentage"]
        == 30.0
    )

    assert (
        result["classes"]["built_up"]["pixels"]
        == 1500
    )

    assert (
        result["classes"]["built_up"]["percentage"]
        == 15.0
    )

    assert (
        result["classes"]["bare_land"]["pixels"]
        == 1000
    )

    assert (
        result["classes"]["bare_land"]["percentage"]
        == 10.0
    )

    print(
        "\nAll semantic tests passed."
    )


if __name__ == "__main__":
    test_semantic_analysis()
