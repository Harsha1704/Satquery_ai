"""Render SatQuery vehicle predictions and prepare a manual-review manifest.

Each numbered box in the PNG corresponds to one row in the CSV. Mark every
row as `true_positive` or `false_positive`. Categorize each false positive,
then run this command again with `--review-csv` and `--ground-truth` to
calculate precision, recall, counting error, and the dominant error classes.
Do not tune thresholds before this review is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.detection import ObjectDetector


DEFAULT_IMAGE = PROJECT_ROOT / "data" / "raw" / "image_06.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "object_detection"
REVIEW_STATUSES = {"true_positive", "false_positive"}
FALSE_POSITIVE_CATEGORIES = {
    "parking_marking",
    "road_marking",
    "building_texture",
    "vegetation",
    "shadow",
    "duplicate",
    "other",
}
FALSE_POSITIVE_CATEGORY_ALIASES = {
    # Preserve compatibility with early review sheets while reporting the
    # current category vocabulary consistently.
    "building_edge": "building_texture",
    "vegetation_texture": "vegetation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--predictions-json",
        type=Path,
        help="Existing prediction JSON. Recreates review assets without inference.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Existing reviewed CSV. Skips inference and writes metrics.",
    )
    parser.add_argument(
        "--ground-truth",
        type=int,
        help="Manually counted number of real cars in the image.",
    )
    return parser.parse_args()


def _output_paths(image_path: Path, output_dir: Path) -> Dict[str, Path]:
    stem = image_path.stem
    return {
        "image": output_dir / f"{stem}_car_predictions.png",
        "json": output_dir / f"{stem}_car_predictions.json",
        "csv": output_dir / f"{stem}_car_review.csv",
        "metrics": output_dir / f"{stem}_car_metrics.json",
    }


def _draw_predictions(
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    destination: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index, detection in enumerate(detections, start=1):
        bbox = detection["bbox"]
        x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
        label = f"{index} {float(detection['confidence']):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        label_height = text_box[3] - text_box[1] + 3
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, text_box[2] + 3, y1), fill="red")
        draw.text((x1 + 1, label_y + 1), label, fill="white", font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _write_review_csv(
    detections: Iterable[Dict[str, Any]],
    destination: Path,
) -> None:
    fields = [
        "id",
        "prediction",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "review_status",
        "false_positive_category",
        "review_notes",
    ]
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, detection in enumerate(detections, start=1):
            writer.writerow(
                {
                    "id": index,
                    "prediction": detection["class"],
                    "confidence": detection["confidence"],
                    **detection["bbox"],
                    "review_status": "unreviewed",
                    "false_positive_category": "",
                    "review_notes": "",
                }
            )


def _read_review(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Review CSV has no predictions: {path}")
    invalid_statuses = {
        row.get("review_status", "").strip()
        for row in rows
        if row.get("review_status", "").strip() not in REVIEW_STATUSES
    }
    if invalid_statuses:
        raise ValueError(
            "Every row must be reviewed as one of "
            f"{sorted(REVIEW_STATUSES)}. Invalid statuses: "
            f"{sorted(invalid_statuses)}"
        )

    invalid_categories = {
        row.get("false_positive_category", "").strip()
        for row in rows
        if row["review_status"].strip() == "false_positive"
        and FALSE_POSITIVE_CATEGORY_ALIASES.get(
            row.get("false_positive_category", "").strip(),
            row.get("false_positive_category", "").strip(),
        ) not in FALSE_POSITIVE_CATEGORIES
    }
    if invalid_categories:
        raise ValueError(
            "Every false positive needs one category from "
            f"{sorted(FALSE_POSITIVE_CATEGORIES)}. Invalid categories: "
            f"{sorted(invalid_categories)}"
        )
    return rows


def _summarize_review(
    rows: list[Dict[str, str]],
    ground_truth: Optional[int],
) -> Dict[str, Any]:
    counts = Counter(row["review_status"].strip() for row in rows)
    error_categories = Counter(
        FALSE_POSITIVE_CATEGORY_ALIASES.get(
            row["false_positive_category"].strip(),
            row["false_positive_category"].strip(),
        )
        for row in rows
        if row["review_status"].strip() == "false_positive"
    )
    predicted = len(rows)
    true_positives = counts["true_positive"]
    false_positives = counts["false_positive"]
    metrics: Dict[str, Any] = {
        "predicted_cars": predicted,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_positive_by_category": dict(sorted(error_categories.items())),
        "precision": true_positives / predicted if predicted else None,
        "ground_truth_cars": ground_truth,
        "missed_cars": None,
        "recall": None,
        "counting_error_percent": None,
    }
    if ground_truth is not None:
        if ground_truth < true_positives:
            raise ValueError("Ground-truth count cannot be less than true positives.")
        metrics["missed_cars"] = ground_truth - true_positives
        metrics["recall"] = true_positives / ground_truth if ground_truth else None
        metrics["counting_error_percent"] = (
            abs(predicted - ground_truth) / ground_truth * 100 if ground_truth else None
        )
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.review_csv:
        rows = _read_review(args.review_csv.resolve())
        metrics = _summarize_review(rows, args.ground_truth)
        destination = _output_paths(args.image, output_dir)["metrics"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print(f"Metrics written to: {destination}")
        return

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if args.predictions_json:
        result = json.loads(
            args.predictions_json.resolve().read_text(encoding="utf-8")
        )
    else:
        detector = ObjectDetector(confidence_threshold=args.confidence)
        result = detector.detect(str(image_path), requested_class="car")
    detections = result["detections"]
    paths = _output_paths(image_path, output_dir)

    _draw_predictions(image_path, detections, paths["image"])
    _write_review_csv(detections, paths["csv"])
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Predicted cars: {len(detections)}")
    print(f"Annotated image: {paths['image']}")
    print(f"Review CSV: {paths['csv']}")
    print(f"Prediction JSON: {paths['json']}")


if __name__ == "__main__":
    main()
