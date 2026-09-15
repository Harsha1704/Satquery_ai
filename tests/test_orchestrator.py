import os

from ai.orchestrator import SatQueryOrchestrator


IMAGE_PATH = "datasets/raw/test_multispectral.tif"


def test_orchestrator():

    print("\nSatQuery End-to-End Intelligence Test")
    print("=" * 60)

    query = "How much vegetation is present?"

    print("\nUser Query:")
    print(query)

    orchestrator = SatQueryOrchestrator()

    result = orchestrator.run(
        query=query,
        image_path=IMAGE_PATH
    )

    print("\nExecution Result")
    print("-" * 60)

    print("Success :", result.success)
    print("Task    :", result.task)
    print("Message :", result.message)

    print("\nEvidence")
    for item in result.evidence:
        print(" -", item)

    print("\nData")
    print(result.data)

    evidence_image = result.data.get(
        "evidence_image"
    )

    print("\nVisual Evidence")
    print(
        "File exists:",
        "PASS"
        if evidence_image and os.path.exists(evidence_image)
        else "FAIL"
    )

    assert result.success is True
    assert result.task == "spectral_analysis"
    assert evidence_image is not None
    assert os.path.exists(evidence_image)


if __name__ == "__main__":
    test_orchestrator()