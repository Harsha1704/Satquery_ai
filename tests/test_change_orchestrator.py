import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ai.orchestrator import SatQueryOrchestrator


BEFORE_PATH = (
    "datasets/raw/test_before.tif"
)

AFTER_PATH = (
    "datasets/raw/test_after.tif"
)


def create_test_images():

    width = 100
    height = 100

    before = np.ones(
        (height, width),
        dtype=np.float32
    ) * 0.20

    after = before.copy()

    # Simulate a changed region.
    after[25:75, 25:75] = 0.80

    transform = from_origin(
        75.0,
        30.0,
        10.0,
        10.0
    )

    for path, data in [
        (BEFORE_PATH, before),
        (AFTER_PATH, after)
    ]:

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform
        ) as dataset:

            dataset.write(
                data,
                1
            )


def test_change_orchestrator():

    print("\nSatQuery Change Detection Orchestrator Test")
    print("=" * 60)

    create_test_images()

    query = (
        "What changed between these two images?"
    )

    print("\nUser Query:")
    print(query)

    orchestrator = SatQueryOrchestrator()

    result = orchestrator.run(
        query=query,
        image_path=BEFORE_PATH,
        second_image_path=AFTER_PATH
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
        if evidence_image
        and os.path.exists(evidence_image)
        else "FAIL"
    )

    assert result.success is True

    assert (
        result.task
        == "change_detection"
    )

    assert evidence_image is not None

    assert os.path.exists(
        evidence_image
    )


if __name__ == "__main__":
    test_change_orchestrator()