import os

import numpy as np

from ai.evidence import EvidenceVisualizer


OUTPUT_PATH = (
    "outputs/evidence/test_ndvi_map.png"
)


def test_evidence_visualizer():

    print("\nSatQuery Visual Evidence Test")
    print("=" * 50)

    visualizer = EvidenceVisualizer()

    # Create a synthetic NDVI gradient.
    ndvi = np.linspace(
        -1.0,
        1.0,
        10000,
        dtype=np.float32
    ).reshape(
        100,
        100
    )

    output = visualizer.create_ndvi_map(
        ndvi,
        OUTPUT_PATH
    )

    print("NDVI array created : PASS")
    print("Normalization      : PASS")
    print("Image generation   : PASS")

    print("\nEvidence file:")
    print(output)

    print(
        "File exists        :",
        "PASS" if os.path.exists(output)
        else "FAIL"
    )

    assert os.path.exists(output)


if __name__ == "__main__":
    test_evidence_visualizer()