import numpy as np

from ai.orchestrator import TaskPlanner, TaskExecutor


def test_semantic_orchestrator():

    print("\nSatQuery Semantic Orchestrator Test")
    print("=" * 60)

    query = "Identify water bodies"

    # -------------------------------------------------
    # Create planner
    # -------------------------------------------------

    planner = TaskPlanner()

    task = planner.create_task(query)

    print("\nQuery:")
    print(query)

    print("\nPlanning")
    print("-" * 60)

    print("Intent         :", task.intent.value)
    print("Required tools :", task.required_tools)

    assert task.intent.value == "semantic_analysis"

    assert (
        task.required_tools
        == ["semantic_analyzer"]
    )

    print("Planner        : PASS")

    # -------------------------------------------------
    # Create synthetic prediction mask
    # -------------------------------------------------

    prediction_mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    # Water
    prediction_mask[0:45, :] = 0

    # Vegetation
    prediction_mask[45:75, :] = 1

    # Built-up
    prediction_mask[75:90, :] = 2

    # Bare land
    prediction_mask[90:100, :] = 3

    task.parameters[
        "prediction_mask"
    ] = prediction_mask

    task.parameters[
        "semantic_class"
    ] = "water"

    # -------------------------------------------------
    # Execute
    # -------------------------------------------------

    executor = TaskExecutor()

    result = executor.execute(
        task,
        image_path="datasets/raw/test_image.png"
    )

    print("\nExecution")
    print("-" * 60)

    print("Success :", result.success)
    print("Task    :", result.task)
    print("Message :", result.message)

    if result.error:
        print("Error   :", result.error)

    assert result.success is True

    assert (
        result.task
        == "semantic_analysis"
    )

    assert (
        result.data["requested_class"]
        == "water"
    )

    statistics = result.data[
        "statistics"
    ]

    assert (
        statistics["classes"]["water"]["pixels"]
        == 4500
    )

    assert (
        statistics["classes"]["water"]["percentage"]
        == 45.0
    )

    print("\nEvidence")
    print("-" * 60)

    for item in result.evidence:
        print(" -", item)

    print(
        "\nSemantic Orchestrator: PASS"
    )


if __name__ == "__main__":
    test_semantic_orchestrator()