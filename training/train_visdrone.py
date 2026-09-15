"""Train and validate SatQuery's aerial vehicle detector on VisDrone.

The first run downloads and converts VisDrone through the `download` block in
configs/visdrone_satquery.yaml. Subsequent runs reuse the converted dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "visdrone_satquery.yaml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--name", default="visdrone_satquery")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required. Activate the ML environment and run "
            "`pip install ultralytics`."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        pretrained=True,
        resume=args.resume,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None or not getattr(trainer, "best", None):
        raise RuntimeError("Training finished without exposing a best checkpoint.")

    best_weights = Path(trainer.best)
    if not best_weights.is_file():
        raise RuntimeError(f"Training completed without best weights: {best_weights}")

    # Validate the exported best checkpoint rather than the in-memory model.
    YOLO(str(best_weights)).val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )

    print("\nTraining and validation completed.")
    print(f"Best weights: {best_weights}")
    print(
        "Use them in SatQuery with:\n"
        f"  $env:SATQUERY_VEHICLE_WEIGHTS = '{best_weights}'"
    )
    return best_weights


if __name__ == "__main__":
    main()
