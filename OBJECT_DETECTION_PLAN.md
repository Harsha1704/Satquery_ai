# Object detection plan

## Current vehicle detector

Vehicle detection uses YOLO tiled inference with 512-pixel tiles, 25% overlap,
1.5x tile upscaling, 640-pixel inference, and four tiles per model call. The
post-processing stage performs class-aware global duplicate removal.

Before training, SatQuery downloads the public VisDrone checkpoint. After
training, point `SATQUERY_VEHICLE_WEIGHTS` at the resulting `best.pt`; this
uses local weights and skips the Hugging Face download.

## Training workflow

Activate the ML environment and run:

```powershell
.\.venv-ml\Scripts\python.exe training\train_visdrone.py --epochs 100 --batch 8
```

The data configuration downloads and converts the official VisDrone 2019 DET
train (6,471), validation (548), and test-dev (1,610) splits on the first run.
The output checkpoint is placed under `runs/detect/visdrone_satquery/weights/`.

## Validation gate

Evaluate the trained `best.pt` on the validation split and inspect precision,
recall, mAP50, and mAP50-95 before changing runtime thresholds. Then run the
detector on held-out aerial images with manually checked counts. `image_06.png`
is an annotated demonstration image and must not be used as training data.

### Per-image error analysis

For a diagnostic image, create an overlay and review sheet without changing
the detector:

```powershell
.\.venv-ml\Scripts\python.exe training\validate_vehicle_predictions.py
```

Mark every row in `image_06_car_review.csv` as either `true_positive` or
`false_positive`. Every false positive must additionally be categorized as
`parking_marking`, `building_texture`, `road_marking`, `vegetation`, `shadow`,
`duplicate`, or `other`. This makes the next intervention traceable to a
measured failure class, rather than a threshold change selected for one image.

When the review is complete, calculate the per-image metrics with the verified
ground-truth count:

```powershell
.\.venv-ml\Scripts\python.exe training\validate_vehicle_predictions.py `
  --review-csv outputs\object_detection\image_06_car_review.csv `
  --ground-truth <actual-car-count>
```

The resulting JSON reports actual cars, predicted cars, true positives, false
positives by category, missed cars, precision, recall, and counting error.
Only repeat this process across a held-out set before deciding whether domain
fine-tuning is necessary. Do not create coordinate rules for a demonstration
image.

## Buildings

The current LoveDA-based component produces semantic building regions. Connected
components therefore provide an estimate, not a validated count of individual
building instances. Building counting stays separate until an instance or
footprint detector is trained and evaluated.
