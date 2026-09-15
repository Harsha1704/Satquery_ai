# tests/test_report_generator.py

from pathlib import Path

from ai.execution import ResultAdapter
from ai.report import SatQueryReportGenerator


def main():

    adapter = ResultAdapter()

    result = {
        "success": True,

        "answer": (
            "Approximately 1.18% of the analyzed "
            "area changed between 2020 and 2025."
        ),

        "changed_percentage": 1.1801,

        "changed_pixels": 12374,

        "total_pixels": 1048576,

        "confidence": 0.9227,

        "confidence_type": (
            "model_score"
        ),

        "model": "ChangeFormerV6",

        "device": "cpu",

        "evidence": [
            (
                "outputs/evidence/"
                "changeformer_change_mask.png"
            ),
            (
                "outputs/evidence/"
                "changeformer_before_overlay.png"
            ),
            (
                "outputs/evidence/"
                "changeformer_after_overlay.png"
            ),
            (
                "outputs/evidence/"
                "changeformer_comparison.png"
            ),
        ],

        "limitations": [
            (
                "ChangeFormer is used for binary "
                "building/structural change detection."
            ),
            (
                "The model does not independently "
                "identify semantic land-cover transitions."
            ),
        ],
    }

    standardized = adapter.standardize(
        result,

        query=(
            "What percentage changed "
            "between 2020 and 2025?"
        ),

        intent="change_detection",

        tools=[
            "changeformer",
            "change_vqa",
            "evidence_generator",
        ],

        inputs={
            "before_image": (
                "data/LEVIR-CD/test/A/test_1.png"
            ),
            "after_image": (
                "data/LEVIR-CD/test/B/test_1.png"
            ),
            "year_before": 2020,
            "year_after": 2025,
        },
    )

    generator = SatQueryReportGenerator()

    report = generator.generate(
        standardized,
        query=(
            "What percentage changed "
            "between 2020 and 2025?"
        ),
        report_name=(
            "satquery_change_analysis_demo"
        ),
    )

    print("\n" + "=" * 70)
    print("PRIORITY 13 REPORT GENERATION")
    print("=" * 70)

    print(
        "SUCCESS:",
        report["success"],
    )

    print(
        "HTML:",
        report["html_report"],
    )

    print(
        "JSON:",
        report["json_report"],
    )

    html_path = Path(
        report["html_report"]
    )

    json_path = Path(
        report["json_report"]
    )

    assert html_path.exists()
    assert json_path.exists()

    assert html_path.stat().st_size > 0
    assert json_path.stat().st_size > 0

    print(
        "HTML SIZE:",
        html_path.stat().st_size,
        "bytes",
    )

    print(
        "JSON SIZE:",
        json_path.stat().st_size,
        "bytes",
    )

    print("\n" + "=" * 70)
    print("PRIORITY 13 DOWNLOADABLE REPORT: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()