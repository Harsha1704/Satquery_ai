# tests/test_execution_summary.py

from pprint import pprint
import unittest

if __name__ != "__main__":
    raise unittest.SkipTest("Manual execution-summary report script; run directly for diagnostic output.")

from ai.execution import (
    ConfidenceManager,
    ExecutionSummaryBuilder,
    ResultAdapter,
)


print("\n" + "=" * 70)
print("1. CALIBRATED CONFIDENCE")
print("=" * 70)

manager = ConfidenceManager()

pprint(
    manager.normalize(
        0.91,
        "calibrated_probability",
    ).to_dict()
)


print("\n" + "=" * 70)
print("2. REMOTECLIP RELATIVE CONFIDENCE")
print("=" * 70)

pprint(
    manager.normalize(
        1.0,
        "relative_tile_relevance",
    ).to_dict()
)


print("\n" + "=" * 70)
print("3. ROUTER CONFIDENCE")
print("=" * 70)

pprint(
    manager.normalize(
        0.86,
        "routing_confidence",
    ).to_dict()
)


print("\n" + "=" * 70)
print("4. EXECUTION SUMMARY")
print("=" * 70)

builder = ExecutionSummaryBuilder()

summary = builder.build(
    query="Locate buildings in this image",
    intent="text_guided_grounding",
    success=True,
    tools=[
        "remoteclip",
        "text_guided_region_grounder",
    ],
    confidence=1.0,
    confidence_type="relative_tile_relevance",
    inputs={
        "image_path": "test_1.png",
        "target": "building",
    },
    model="RemoteCLIP-ViT-B-32",
    device="cpu",
    evidence=(
        "outputs/evidence/"
        "remoteclip_building.png"
    ),
    limitations=[
        (
            "Grounding identifies high-relevance image "
            "regions rather than exact object boundaries."
        ),
        (
            "Confidence is relative tile relevance and "
            "not a calibrated probability."
        ),
    ],
)

pprint(summary)


print("\n" + "=" * 70)
print("5. RESULT ADAPTER")
print("=" * 70)

adapter = ResultAdapter()

existing_result = {
    "success": True,
    "answer": (
        "RemoteCLIP identified 6 "
        "high-relevance regions."
    ),
    "count": 6,
    "confidence": 1.0,
    "confidence_type": (
        "relative_tile_relevance"
    ),
    "model": "RemoteCLIP-ViT-B-32",
    "device": "cpu",
    "evidence": (
        "outputs/evidence/"
        "remoteclip_building.png"
    ),
}

standardized = adapter.standardize(
    existing_result,
    query="Locate buildings in this image",
    intent="text_guided_grounding",
    tools=[
        "remoteclip",
        "text_guided_region_grounder",
    ],
)

pprint(standardized)


print("\n" + "=" * 70)
print("PRIORITY 11 CORE TEST COMPLETE")
print("=" * 70)
