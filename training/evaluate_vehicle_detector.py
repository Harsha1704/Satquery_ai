"""Evaluate the unchanged SatQuery vehicle baseline on clean YOLO-labeled images."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.detection import ObjectDetector


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".jfif", ".png", ".tif", ".tiff"}
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "vehicles"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "vehicles" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def _iou(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
    left = max(float(box_a["x1"]), float(box_b["x1"]))
    top = max(float(box_a["y1"]), float(box_b["y1"]))
    right = min(float(box_a["x2"]), float(box_b["x2"]))
    bottom = min(float(box_a["y2"]), float(box_b["y2"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a["x2"]) - float(box_a["x1"])) * max(
        0.0, float(box_a["y2"]) - float(box_a["y1"])
    )
    area_b = max(0.0, float(box_b["x2"]) - float(box_b["x1"])) * max(
        0.0, float(box_b["y2"]) - float(box_b["y1"])
    )
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _load_yolo_labels(label_path: Path, width: int, height: int) -> List[Dict[str, float]]:
    boxes: List[Dict[str, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 values.")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_number} has class {class_id}; only class 0 (car) is supported.")
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number} has non-normalized coordinates.")
        box_width *= width
        box_height *= height
        center_x *= width
        center_y *= height
        boxes.append(
            {
                "x1": center_x - box_width / 2,
                "y1": center_y - box_height / 2,
                "x2": center_x + box_width / 2,
                "y2": center_y + box_height / 2,
            }
        )
    return boxes


def _match_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, float]],
    iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
    """Greedily match high-confidence predictions to one ground-truth box."""
    matched_ground_truth: set[int] = set()
    matched_predictions: List[Dict[str, Any]] = []

    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, truth in enumerate(ground_truth):
            if index in matched_ground_truth:
                continue
            overlap = _iou(prediction["bbox"], truth)
            if overlap > best_iou:
                best_index, best_iou = index, overlap

        item = {**prediction, "matched": best_iou >= iou_threshold, "matched_iou": round(best_iou, 6)}
        if item["matched"]:
            matched_ground_truth.add(best_index)
        matched_predictions.append(item)

    missed = [truth for index, truth in enumerate(ground_truth) if index not in matched_ground_truth]
    return matched_predictions, missed


def _average_precision_at_50(records: Iterable[Dict[str, Any]], ground_truth_count: int) -> float:
    if not ground_truth_count:
        return 0.0
    ordered = sorted(records, key=lambda item: float(item["confidence"]), reverse=True)
    true_positives, false_positives = 0, 0
    precisions, recalls = [], []
    for record in ordered:
        if record["matched"]:
            true_positives += 1
        else:
            false_positives += 1
        precisions.append(true_positives / (true_positives + false_positives))
        recalls.append(true_positives / ground_truth_count)

    recall_points = [0.0, *recalls, 1.0]
    precision_points = [0.0, *precisions, 0.0]
    for index in range(len(precision_points) - 1, 0, -1):
        precision_points[index - 1] = max(precision_points[index - 1], precision_points[index])
    return sum(
        (recall_points[index + 1] - recall_points[index]) * precision_points[index + 1]
        for index in range(len(recall_points) - 1)
    )


def _draw_overlay(
    image_path: Path,
    predictions: Iterable[Dict[str, Any]],
    missed: Iterable[Dict[str, float]],
    destination: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction in predictions:
        box = prediction["bbox"]
        color = "lime" if prediction["matched"] else "red"
        label = f"TP {prediction['confidence']:.2f}" if prediction["matched"] else f"FP {prediction['confidence']:.2f}"
        draw.rectangle(tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2")), outline=color, width=3)
        draw.text((float(box["x1"]), max(0, float(box["y1"]) - 12)), label, fill=color, font=font)
    for truth in missed:
        draw.rectangle(tuple(float(truth[key]) for key in ("x1", "y1", "x2", "y2")), outline="yellow", width=3)
        draw.text((float(truth["x1"]), max(0, float(truth["y1"]) - 12)), "FN", fill="yellow", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1].")
    dataset = args.dataset.resolve()
    images_dir, labels_dir = dataset / "images", dataset / "labels"
    image_paths = sorted(path for path in images_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(
            f"No clean evaluation images found in {images_dir}. Add verified image/label pairs first."
        )
    missing_labels = [path for path in image_paths if not (labels_dir / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise FileNotFoundError("Missing ground-truth labels for: " + ", ".join(path.name for path in missing_labels))

    detector = ObjectDetector()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    per_image = []
    total_ground_truth = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        truth = _load_yolo_labels(labels_dir / f"{image_path.stem}.txt", width, height)
        result = detector.detect(str(image_path), requested_class="car")
        predictions, missed = _match_predictions(result["detections"], truth, args.iou)
        _draw_overlay(image_path, predictions, missed, output_dir / "overlays" / f"{image_path.stem}.png")
        records.extend({**prediction, "image": image_path.name} for prediction in predictions)
        total_ground_truth += len(truth)
        per_image.append(
            {
                "image": image_path.name,
                "predicted_cars": len(predictions),
                "ground_truth_cars": len(truth),
                "true_positives": sum(prediction["matched"] for prediction in predictions),
                "false_positives": sum(not prediction["matched"] for prediction in predictions),
                "false_negatives": len(missed),
            }
        )

    true_positives = sum(record["matched"] for record in records)
    false_positives = len(records) - true_positives
    false_negatives = total_ground_truth - true_positives
    precision = true_positives / len(records) if records else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth else 0.0
    metrics = {
        "baseline": "YOLOv8x VisDrone; unchanged inference configuration",
        "iou_threshold": args.iou,
        "evaluation_images": len(image_paths),
        "predicted_count": len(records),
        "ground_truth_count": total_ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": _average_precision_at_50(records, total_ground_truth),
        "counting_error_percent": abs(len(records) - total_ground_truth) / total_ground_truth * 100
        if total_ground_truth
        else None,
        "per_image": per_image,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
