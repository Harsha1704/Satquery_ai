from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class ObjectDetector:
    """
    SatQuery-AI aerial/satellite object detection engine.

    VEHICLE DETECTION
    -----------------
    YOLOv8x fine-tuned on VisDrone aerial imagery.

    BUILDING DETECTION
    ------------------
    Existing SemanticModel / LoveDA segmentation pipeline.

    Notes
    -----
    Building detection remains an estimate because semantic
    segmentation does not provide true object instances.

    Vehicle detection uses overlapping tiled inference followed by
    confidence filtering, geometric filtering and global class-aware
    duplicate suppression.
    """

    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================

    MODEL_NAME = (
        "YOLOv8x VisDrone aerial detector "
        "+ tiled inference "
        "+ confidence filtering "
        "+ cross-tile NMS "
        "+ LoveDA building segmentation"
    )

    VEHICLE_MODEL_NAME = "YOLOv8x VisDrone"

    VISDRONE_REPO = "mshamrai/yolov8x-visdrone"
    VISDRONE_FILENAME = "best.pt"
    VEHICLE_WEIGHTS_ENV = "SATQUERY_VEHICLE_WEIGHTS"

    # ============================================================
    # SUPPORTED CLASSES
    # ============================================================

    SUPPORTED_OBJECTS = {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "van",
        "motor",
        "tricycle",
        "awning-tricycle",
        "building",
    }

    VISDRONE_CLASS_NAMES = {
        0: "person",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
    }

    CLASS_ALIASES = {
        "car": "car",
        "cars": "car",

        "vehicle": "car",
        "vehicles": "car",

        "van": "van",
        "vans": "van",

        "truck": "truck",
        "trucks": "truck",

        "bus": "bus",
        "buses": "bus",

        "person": "person",
        "people": "person",
        "persons": "person",

        "bicycle": "bicycle",
        "bicycles": "bicycle",
        "bike": "bicycle",
        "bikes": "bicycle",

        "motorcycle": "motor",
        "motorcycles": "motor",
        "motorbike": "motor",
        "motorbikes": "motor",
        "motor": "motor",

        "tricycle": "tricycle",
        "tricycles": "tricycle",

        "awning-tricycle": "awning-tricycle",
        "awning tricycle": "awning-tricycle",

        "building": "building",
        "buildings": "building",
        "house": "building",
        "houses": "building",
        "built up": "building",
        "built-up": "building",
    }

    # ============================================================
    # CONFIDENCE THRESHOLDS
    # ============================================================

    # The previous implementation accepted cars at ~0.10.
    #
    # For satellite counting this is too permissive and can create
    # many false positives.
    #
    # These are deliberately conservative starting values.
    CLASS_THRESHOLDS = {
        "car": 0.25,
        "van": 0.25,
        "truck": 0.25,
        "bus": 0.25,
        "motor": 0.25,
        "bicycle": 0.25,
        "person": 0.25,
        "tricycle": 0.25,
        "awning-tricycle": 0.25,
    }

    # ============================================================
    # BOX FILTERS
    # ============================================================

    # Very tiny boxes are usually unreliable in satellite imagery.
    MIN_BOX_WIDTH = {
        "car": 3.0,
        "van": 3.0,
        "truck": 4.0,
        "bus": 5.0,
        "motor": 2.0,
        "bicycle": 2.0,
        "person": 3.0,
        "tricycle": 3.0,
        "awning-tricycle": 3.0,
    }

    MIN_BOX_HEIGHT = {
        "car": 3.0,
        "van": 3.0,
        "truck": 4.0,
        "bus": 5.0,
        "motor": 2.0,
        "bicycle": 2.0,
        "person": 4.0,
        "tricycle": 3.0,
        "awning-tricycle": 3.0,
    }

    # Extremely large boxes are generally not useful for vehicle
    # counting and can be false detections.
    MAX_BOX_IMAGE_RATIO = 0.20

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
        nms_iou_threshold: float = 0.45,
        tile_batch_size: int = 4,
        tile_upscale: float = 1.5,
        inference_size: int = 640,
        building_tile_size: int = 512,
        building_tile_overlap: float = 0.25,
        min_building_area: int = 80,
        max_building_area_ratio: float = 0.20,
        max_building_bbox_ratio: float = 0.35,
    ):
        self.confidence_threshold = float(confidence_threshold)

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.device = self._select_device(device)

        self.tile_size = max(256, int(tile_size))

        self.tile_overlap = min(
            max(float(tile_overlap), 0.0),
            0.70,
        )

        self.nms_iou_threshold = min(
            max(float(nms_iou_threshold), 0.10),
            0.90,
        )

        self.tile_batch_size = max(
            1,
            int(tile_batch_size),
        )

        self.tile_upscale = min(
            max(float(tile_upscale), 1.0),
            3.0,
        )

        self.inference_size = max(32, int(inference_size))

        self.building_tile_size = max(
            256,
            int(building_tile_size),
        )

        self.building_tile_overlap = min(
            max(float(building_tile_overlap), 0.0),
            0.70,
        )

        self.min_building_area = max(
            20,
            int(min_building_area),
        )

        self.max_building_area_ratio = min(
            max(float(max_building_area_ratio), 0.001),
            0.95,
        )

        self.max_building_bbox_ratio = min(
            max(float(max_building_bbox_ratio), 0.001),
            0.95,
        )

        self.model = None
        self.categories: List[str] = []

        self._semantic_model = None

        self._load_model()

    # ============================================================
    # DEVICE
    # ============================================================

    @staticmethod
    def _select_device(
        device: Optional[str],
    ) -> str:
        try:
            import torch

            if device:
                requested = str(device).lower().strip()

                if requested == "cuda":
                    if torch.cuda.is_available():
                        return "cuda"

                    print(
                        "CUDA requested but unavailable. "
                        "Falling back to CPU."
                    )

                    return "cpu"

                if requested == "cpu":
                    return "cpu"

            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        except Exception:
            return "cpu"

    # ============================================================
    # LOAD VEHICLE MODEL
    # ============================================================

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "\n"
                "Ultralytics is required for the aerial "
                "vehicle detector.\n\n"
                "Run:\n"
                "pip install ultralytics huggingface_hub\n"
            ) from exc

        try:
            configured_weights = os.getenv(
                self.VEHICLE_WEIGHTS_ENV,
                "",
            ).strip()

            if configured_weights:
                weights_path = Path(
                    configured_weights
                ).expanduser()

                if not weights_path.is_file():
                    raise FileNotFoundError(
                        "Configured vehicle weights were not found: "
                        f"{weights_path}"
                    )

                print(
                    "\nLoading aerial vehicle model:\n"
                    f"  {self.VEHICLE_MODEL_NAME}\n"
                    f"  Local weights: {weights_path}\n"
                )
            else:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise ImportError(
                        "huggingface_hub is required when "
                        f"{self.VEHICLE_WEIGHTS_ENV} is not set.\n\n"
                        "Run:\n"
                        "pip install huggingface_hub\n"
                    ) from exc

                print(
                    "\nLoading aerial vehicle model:\n"
                    f"  {self.VEHICLE_MODEL_NAME}\n"
                    f"  Hugging Face: {self.VISDRONE_REPO}\n"
                )

                weights_path = hf_hub_download(
                    repo_id=self.VISDRONE_REPO,
                    filename=self.VISDRONE_FILENAME,
                )

            self.model = YOLO(str(weights_path))

            self.categories = list(
                self.VISDRONE_CLASS_NAMES.values()
            )

            print(
                "Aerial vehicle model loaded successfully."
            )

        except Exception as exc:
            self.model = None

            raise RuntimeError(
                "Failed to load the VisDrone YOLO model.\n"
                f"Repository: {self.VISDRONE_REPO}\n"
                f"Error: {type(exc).__name__}: {exc}"
            ) from exc

    # ============================================================
    # PUBLIC DETECTION API
    # ============================================================

    def detect(
        self,
        image_path: str,
        requested_class: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

        if not self.is_loaded():
            raise RuntimeError(
                "Object detection model is not loaded."
            )

        self._check_file(image_path)

        path = Path(image_path)

        requested = self._normalize_class_name(
            requested_class
        )

        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        threshold = min(
            max(threshold, 0.0),
            1.0,
        )

        # --------------------------------------------------------
        # BUILDING
        # --------------------------------------------------------

        if requested == "building":
            return self._detect_buildings(
                str(path)
            )

        # --------------------------------------------------------
        # VEHICLES
        # --------------------------------------------------------

        with Image.open(path) as source:
            image = source.convert("RGB")

            image_width, image_height = image.size

            detections = self._detect_with_tiles(
                image=image,
                requested_class=requested,
                confidence_threshold=threshold,
            )

        # --------------------------------------------------------
        # GLOBAL DUPLICATE REMOVAL
        # --------------------------------------------------------

        detections = self._remove_duplicate_detections(
            detections
        )

        # --------------------------------------------------------
        # COUNTS
        # --------------------------------------------------------

        counts = dict(
            Counter(
                item["class"]
                for item in detections
            )
        )

        return {
            "success": True,
            "task": "object_detection",
            "image": str(path),
            "model": self.get_model_name(),
            "device": self.get_device(),
            "requested_class": requested,
            "detection_count": len(detections),
            "class_counts": counts,
            "detections": detections,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "confidence_threshold": threshold,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "tile_batch_size": self.tile_batch_size,
            "tile_upscale": self.tile_upscale,
            "inference_size": self.inference_size,
            "nms_iou_threshold": self.nms_iou_threshold,
            "description": self._build_description(
                detections,
                requested,
            ),
        }

    # ============================================================
    # TILE POSITIONS
    # ============================================================

    @staticmethod
    def _tile_positions(
        dimension: int,
        tile_size: int,
        stride: int,
    ) -> List[int]:

        if dimension <= tile_size:
            return [0]

        positions: List[int] = []

        current = 0

        while current + tile_size < dimension:
            positions.append(current)
            current += stride

        last = max(
            0,
            dimension - tile_size,
        )

        if (
            not positions
            or positions[-1] != last
        ):
            positions.append(last)

        return positions

    # ============================================================
    # TILE INFERENCE
    # ============================================================

    def _detect_with_tiles(
        self,
        image: Image.Image,
        requested_class: Optional[str],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:

        width, height = image.size

        tile_size = min(
            self.tile_size,
            width,
            height,
        )

        stride = max(
            1,
            int(
                tile_size
                * (1.0 - self.tile_overlap)
            ),
        )

        x_positions = self._tile_positions(
            width,
            tile_size,
            stride,
        )

        y_positions = self._tile_positions(
            height,
            tile_size,
            stride,
        )

        jobs: List[
            Tuple[Image.Image, int, int, int, int]
        ] = []

        for y in y_positions:
            for x in x_positions:

                x2 = min(
                    x + tile_size,
                    width,
                )

                y2 = min(
                    y + tile_size,
                    height,
                )

                crop = image.crop(
                    (
                        x,
                        y,
                        x2,
                        y2,
                    )
                )

                original_crop_width = crop.width
                original_crop_height = crop.height

                if self.tile_upscale > 1.0:
                    crop = crop.resize(
                        (
                            int(
                                crop.width
                                * self.tile_upscale
                            ),
                            int(
                                crop.height
                                * self.tile_upscale
                            ),
                        ),
                        Image.Resampling.LANCZOS,
                    )

                jobs.append(
                    (
                        crop,
                        x,
                        y,
                        original_crop_width,
                        original_crop_height,
                    )
                )

        print(
            "Vehicle detection: "
            f"{len(jobs)} overlapping tiles"
        )

        results: List[
            Dict[str, Any]
        ] = []

        for start in range(
            0,
            len(jobs),
            self.tile_batch_size,
        ):

            batch = jobs[
                start:
                start + self.tile_batch_size
            ]

            results.extend(
                self._detect_tile_batch(
                    batch=batch,
                    requested_class=requested_class,
                    confidence_threshold=confidence_threshold,
                )
            )

        return results

    # ============================================================
    # SINGLE TILE YOLO
    # ============================================================

    def _detect_single_tile(
        self,
        crop: Image.Image,
        offset_x: int,
        offset_y: int,
        requested_class: Optional[str],
        confidence_threshold: float,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        return self._detect_tile_batch(
            batch=[
                (
                    crop,
                    offset_x,
                    offset_y,
                    (
                        tile_width
                        if tile_width is not None
                        else int(crop.width / self.tile_upscale)
                    ),
                    (
                        tile_height
                        if tile_height is not None
                        else int(crop.height / self.tile_upscale)
                    ),
                )
            ],
            requested_class=requested_class,
            confidence_threshold=confidence_threshold,
        )

    def _detect_tile_batch(
        self,
        batch: List[Tuple[Image.Image, int, int, int, int]],
        requested_class: Optional[str],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Run one YOLO prediction call for a group of equally sized tiles."""
        if self.model is None or not batch:
            return []

        try:
            model_results = self.model.predict(
                source=[np.asarray(job[0]) for job in batch],
                imgsz=self.inference_size,
                conf=0.20,
                iou=0.45,
                max_det=500,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            print(
                "Tile batch inference failed:",
                type(exc).__name__,
                str(exc),
            )
            return []

        if len(model_results) != len(batch):
            print(
                "Tile batch inference returned an unexpected number "
                "of results."
            )
            return []

        detections: List[Dict[str, Any]] = []

        for result, (
            crop,
            offset_x,
            offset_y,
            tile_width,
            tile_height,
        ) in zip(model_results, batch):
            detections.extend(
                self._detections_from_result(
                    result=result,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    requested_class=requested_class,
                    confidence_threshold=confidence_threshold,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
            )

        return detections

    def _detections_from_result(
        self,
        result: Any,
        offset_x: int,
        offset_y: int,
        requested_class: Optional[str],
        confidence_threshold: float,
        tile_width: int,
        tile_height: int,
    ) -> List[Dict[str, Any]]:
        """Convert one Ultralytics result back to global image coordinates."""

        detections: List[
            Dict[str, Any]
        ] = []

        scale = (
            self.tile_upscale
            if self.tile_upscale > 1.0
            else 1.0
        )

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if boxes is None:
            return detections

        xyxy = getattr(
            boxes,
            "xyxy",
            None,
        )

        confs = getattr(
            boxes,
            "conf",
            None,
        )

        classes = getattr(
            boxes,
            "cls",
            None,
        )

        if (
            xyxy is None
            or confs is None
            or classes is None
        ):
            return detections

        xyxy = xyxy.cpu().numpy()
        confs = confs.cpu().numpy()
        classes = classes.cpu().numpy()

        for box, confidence, class_id in zip(
            xyxy,
            confs,
            classes,
        ):

                confidence = float(
                    confidence
                )

                class_id = int(
                    class_id
                )

                class_name = (
                    self.VISDRONE_CLASS_NAMES.get(
                        class_id
                    )
                )

                if not class_name:
                    continue

                class_name = (
                    self._normalize_class_name(
                        class_name
                    )
                )

                if class_name is None:
                    continue

                # ------------------------------------------------
                # REQUESTED CLASS
                # ------------------------------------------------

                if (
                    requested_class
                    and class_name != requested_class
                ):
                    continue

                # ------------------------------------------------
                # CLASS-SPECIFIC CONFIDENCE
                # ------------------------------------------------

                class_threshold = (
                    self.CLASS_THRESHOLDS.get(
                        class_name,
                        0.25,
                    )
                )

                effective_threshold = max(
                    float(confidence_threshold),
                    float(class_threshold),
                )

                if confidence < effective_threshold:
                    continue

                # ------------------------------------------------
                # BOX
                # ------------------------------------------------

                if len(box) != 4:
                    continue

                x1, y1, x2, y2 = map(
                    float,
                    box,
                )

                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale

                x1 += offset_x
                x2 += offset_x

                y1 += offset_y
                y2 += offset_y

                if x2 <= x1 or y2 <= y1:
                    continue

                box_width = x2 - x1
                box_height = y2 - y1

                # ------------------------------------------------
                # MINIMUM BOX SIZE
                # ------------------------------------------------

                min_width = self.MIN_BOX_WIDTH.get(
                    class_name,
                    2.0,
                )

                min_height = self.MIN_BOX_HEIGHT.get(
                    class_name,
                    2.0,
                )

                if box_width < min_width:
                    continue

                if box_height < min_height:
                    continue

                # ------------------------------------------------
                # MAXIMUM BOX SIZE
                # ------------------------------------------------

                if tile_width > 0 and tile_height > 0:

                    width_ratio = (
                        box_width
                        / tile_width
                    )

                    height_ratio = (
                        box_height
                        / tile_height
                    )

                    if (
                        width_ratio
                        > self.MAX_BOX_IMAGE_RATIO
                    ):
                        continue

                    if (
                        height_ratio
                        > self.MAX_BOX_IMAGE_RATIO
                    ):
                        continue

                # ------------------------------------------------
                # TILE BOUNDARY INFORMATION
                # ------------------------------------------------

                local_x1 = x1 - offset_x
                local_y1 = y1 - offset_y
                local_x2 = x2 - offset_x
                local_y2 = y2 - offset_y

                edge_margin = max(
                    8.0,
                    min(
                        20.0,
                        min(
                            tile_width,
                            tile_height,
                        )
                        * 0.04,
                    ),
                )

                touches_left = (
                    local_x1 <= edge_margin
                )

                touches_top = (
                    local_y1 <= edge_margin
                )

                touches_right = (
                    local_x2
                    >= tile_width - edge_margin
                )

                touches_bottom = (
                    local_y2
                    >= tile_height - edge_margin
                )

                touches_edge = any(
                    (
                        touches_left,
                        touches_top,
                        touches_right,
                        touches_bottom,
                    )
                )

                detections.append(
                    {
                        "class": class_name,

                        "confidence": round(
                            confidence,
                            4,
                        ),

                        "bbox": {
                            "x1": round(
                                x1,
                                2,
                            ),
                            "y1": round(
                                y1,
                                2,
                            ),
                            "x2": round(
                                x2,
                                2,
                            ),
                            "y2": round(
                                y2,
                                2,
                            ),
                        },

                        "source":
                            "yolov8x_visdrone",

                        "_tile_x":
                            int(offset_x),

                        "_tile_y":
                            int(offset_y),

                        "_touches_tile_edge":
                            touches_edge,
                    }
                )

        return detections

    # ============================================================
    # CROSS-TILE DUPLICATE REMOVAL
    # ============================================================

    def _remove_duplicate_detections(
        self,
        detections: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        if not detections:
            return []

        grouped: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        for item in detections:
            grouped.setdefault(
                item["class"],
                [],
            ).append(item)

        final: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------
        # CLASS-AWARE GLOBAL NMS
        # --------------------------------------------------------

        for class_name, items in grouped.items():

            # Strong detections first.
            items.sort(
                key=lambda x: float(
                    x.get(
                        "confidence",
                        0.0,
                    )
                ),
                reverse=True,
            )

            kept: List[
                Dict[str, Any]
            ] = []

            for candidate in items:

                duplicate = False

                candidate_tile = (
                    candidate.get("_tile_x"),
                    candidate.get("_tile_y"),
                )

                candidate_edge = bool(
                    candidate.get(
                        "_touches_tile_edge",
                        False,
                    )
                )

                for existing in kept:

                    existing_tile = (
                        existing.get("_tile_x"),
                        existing.get("_tile_y"),
                    )

                    existing_edge = bool(
                        existing.get(
                            "_touches_tile_edge",
                            False,
                        )
                    )

                    iou = self._calculate_iou(
                        candidate["bbox"],
                        existing["bbox"],
                    )

                    containment = (
                        self._containment_ratio(
                            candidate["bbox"],
                            existing["bbox"],
                        )
                    )

                    # ------------------------------------------------
                    # SAME TILE
                    # ------------------------------------------------
                    #
                    # YOLO already performs local NMS, but this
                    # protects us against any duplicate returned
                    # by the model.
                    #
                    if candidate_tile == existing_tile:
                        if (
                            iou >= 0.50
                            or containment >= 0.75
                        ):
                            duplicate = True
                            break

                        continue

                    # ------------------------------------------------
                    # DIFFERENT TILES
                    # ------------------------------------------------

                    if (
                        iou
                        >= self.nms_iou_threshold
                    ):
                        duplicate = True
                        break

                    # Strong containment is a duplicate even if
                    # IoU isn't especially high.
                    if containment >= 0.80:
                        duplicate = True
                        break

                    # ------------------------------------------------
                    # EDGE-AWARE DUPLICATE RULE
                    # ------------------------------------------------
                    #
                    # A detection touching a tile edge is especially
                    # likely to have a better duplicate in an
                    # overlapping neighboring tile.
                    #
                    # Only suppress it when there is meaningful
                    # overlap with the neighboring detection.
                    #
                    if candidate_edge or existing_edge:

                        if iou >= 0.30:
                            duplicate = True
                            break

                        if containment >= 0.60:
                            duplicate = True
                            break

                if not duplicate:
                    kept.append(candidate)

            final.extend(kept)

        # --------------------------------------------------------
        # FINAL SORT
        # --------------------------------------------------------

        final.sort(
            key=lambda x: float(
                x.get(
                    "confidence",
                    0.0,
                )
            ),
            reverse=True,
        )

        # --------------------------------------------------------
        # REMOVE INTERNAL METADATA
        # --------------------------------------------------------

        for item in final:

            item.pop(
                "_tile_x",
                None,
            )

            item.pop(
                "_tile_y",
                None,
            )

            item.pop(
                "_touches_tile_edge",
                None,
            )

        return final

    # ============================================================
    # IOU
    # ============================================================

    @staticmethod
    def _calculate_iou(
        box_a: Dict[str, float],
        box_b: Dict[str, float],
    ) -> float:

        ax1 = float(box_a["x1"])
        ay1 = float(box_a["y1"])
        ax2 = float(box_a["x2"])
        ay2 = float(box_a["y2"])

        bx1 = float(box_b["x1"])
        by1 = float(box_b["y1"])
        bx2 = float(box_b["x2"])
        by2 = float(box_b["y2"])

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = (
            max(
                0.0,
                ax2 - ax1,
            )
            *
            max(
                0.0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0.0,
                bx2 - bx1,
            )
            *
            max(
                0.0,
                by2 - by1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ============================================================
    # CONTAINMENT
    # ============================================================

    @staticmethod
    def _containment_ratio(
        box_a: Dict[str, float],
        box_b: Dict[str, float],
    ) -> float:

        ax1 = float(box_a["x1"])
        ay1 = float(box_a["y1"])
        ax2 = float(box_a["x2"])
        ay2 = float(box_a["y2"])

        bx1 = float(box_b["x1"])
        by1 = float(box_b["y1"])
        bx2 = float(box_b["x2"])
        by2 = float(box_b["y2"])

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = (
            max(
                0.0,
                ax2 - ax1,
            )
            *
            max(
                0.0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0.0,
                bx2 - bx1,
            )
            *
            max(
                0.0,
                by2 - by1,
            )
        )

        smaller = min(
            area_a,
            area_b,
        )

        if smaller <= 0:
            return 0.0

        return intersection / smaller

    # ============================================================
    # BUILDING DETECTION
    # ============================================================

    def _detect_buildings(
        self,
        image_path: str,
    ) -> Dict[str, Any]:

        self._check_file(image_path)

        try:
            from ai.models import SemanticModel

            if self._semantic_model is None:
                self._semantic_model = SemanticModel(
                    device=self.get_device()
                )

            semantic_model = self._semantic_model

            if not semantic_model.is_loaded():
                return {
                    "success": False,
                    "task": "object_detection",
                    "requested_class": "building",
                    "message":
                        "Semantic model failed to load.",
                }

            mask = (
                self._predict_building_mask_tiled(
                    semantic_model,
                    image_path,
                )
            )

            binary = (
                mask == 2
            ).astype(np.uint8)

            detections = (
                self._extract_building_regions(
                    binary
                )
            )

            return {
                "success": True,
                "task": "object_detection",
                "image": str(image_path),
                "model": self.get_model_name(),
                "device": self.get_device(),
                "requested_class": "building",
                "detection_count": len(detections),
                "class_counts": {
                    "building": len(detections)
                },
                "detections": detections,
                "image_width": int(mask.shape[1]),
                "image_height": int(mask.shape[0]),
                "confidence_threshold": None,
                "building_tile_size":
                    self.building_tile_size,
                "building_tile_overlap":
                    self.building_tile_overlap,
                "source":
                    "semantic_segmentation",
                "description":
                    (
                        f"Detected approximately "
                        f"{len(detections)} "
                        "building-like regions."
                    ),
                "note":
                    (
                        "Building detection uses semantic "
                        "segmentation. Adjacent buildings "
                        "touching in the mask can appear as "
                        "one connected region."
                    ),
            }

        except Exception as exc:

            return {
                "success": False,
                "task": "object_detection",
                "requested_class": "building",
                "error": type(exc).__name__,
                "message": str(exc),
            }

    # ============================================================
    # BUILDING TILE PREDICTION
    # ============================================================

    def _predict_building_mask_tiled(
        self,
        semantic_model: Any,
        image_path: str,
    ) -> np.ndarray:

        with Image.open(image_path) as source:

            image = source.convert("RGB")

            width, height = image.size

            tile_size = min(
                self.building_tile_size,
                width,
                height,
            )

            stride = max(
                1,
                int(
                    tile_size
                    * (
                        1.0
                        - self.building_tile_overlap
                    )
                ),
            )

            xs = self._tile_positions(
                width,
                tile_size,
                stride,
            )

            ys = self._tile_positions(
                height,
                tile_size,
                stride,
            )

            building_votes = np.zeros(
                (
                    height,
                    width,
                ),
                dtype=np.uint16,
            )

            coverage = np.zeros(
                (
                    height,
                    width,
                ),
                dtype=np.uint16,
            )

            for y in ys:

                for x in xs:

                    x2 = min(
                        x + tile_size,
                        width,
                    )

                    y2 = min(
                        y + tile_size,
                        height,
                    )

                    crop = image.crop(
                        (
                            x,
                            y,
                            x2,
                            y2,
                        )
                    )

                    tile_mask = np.asarray(
                        semantic_model.predict(
                            crop,
                            original=True,
                        )
                    )

                    if tile_mask.ndim != 2:
                        raise ValueError(
                            "Tiled semantic prediction "
                            "must be 2D. "
                            f"Received {tile_mask.shape}."
                        )

                    crop_w = x2 - x
                    crop_h = y2 - y

                    if tile_mask.shape != (
                        crop_h,
                        crop_w,
                    ):
                        raise ValueError(
                            "Semantic model returned "
                            "an unexpected tile mask "
                            f"shape {tile_mask.shape}; "
                            f"expected {(crop_h, crop_w)}."
                        )

                    coverage[
                        y:y2,
                        x:x2
                    ] += 1

                    building_votes[
                        y:y2,
                        x:x2
                    ] += (
                        tile_mask == 2
                    ).astype(np.uint16)

            building_mask = (
                building_votes * 2
                >= coverage
            ).astype(np.uint8)

            return np.where(
                building_mask > 0,
                2,
                1,
            ).astype(np.uint8)

    # ============================================================
    # BUILDING REGION EXTRACTION
    # ============================================================

    def _extract_building_regions(
        self,
        binary_mask: np.ndarray,
    ) -> List[Dict[str, Any]]:

        binary_mask = (
            np.asarray(binary_mask) > 0
        ).astype(np.uint8)

        height, width = binary_mask.shape

        image_area = max(
            1,
            height * width,
        )

        try:
            import cv2

            kernel = np.ones(
                (
                    3,
                    3,
                ),
                dtype=np.uint8,
            )

            cleaned = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )

            (
                num_labels,
                labels,
                stats,
                centroids,
            ) = cv2.connectedComponentsWithStats(
                cleaned,
                connectivity=8,
            )

            candidates: List[
                Dict[str, Any]
            ] = []

            for label_id in range(
                1,
                num_labels,
            ):

                x = int(
                    stats[
                        label_id,
                        cv2.CC_STAT_LEFT,
                    ]
                )

                y = int(
                    stats[
                        label_id,
                        cv2.CC_STAT_TOP,
                    ]
                )

                w = int(
                    stats[
                        label_id,
                        cv2.CC_STAT_WIDTH,
                    ]
                )

                h = int(
                    stats[
                        label_id,
                        cv2.CC_STAT_HEIGHT,
                    ]
                )

                area = int(
                    stats[
                        label_id,
                        cv2.CC_STAT_AREA,
                    ]
                )

                if area < self.min_building_area:
                    continue

                if w < 6 or h < 6:
                    continue

                bbox_area = max(
                    1,
                    w * h,
                )

                area_ratio = (
                    area / image_area
                )

                bbox_ratio = (
                    bbox_area / image_area
                )

                fill_ratio = (
                    area / bbox_area
                )

                if (
                    area_ratio
                    > self.max_building_area_ratio
                ):
                    continue

                if (
                    bbox_ratio
                    > self.max_building_bbox_ratio
                ):
                    continue

                aspect = max(
                    w / max(h, 1),
                    h / max(w, 1),
                )

                if aspect > 20.0:
                    continue

                if fill_ratio < 0.10:
                    continue

                cx = float(
                    centroids[label_id][0]
                )

                cy = float(
                    centroids[label_id][1]
                )

                candidates.append(
                    {
                        "class": "building",
                        "confidence": None,
                        "bbox": {
                            "x1": float(x),
                            "y1": float(y),
                            "x2": float(x + w),
                            "y2": float(y + h),
                        },
                        "area_pixels": area,
                        "fill_ratio": round(
                            float(fill_ratio),
                            4,
                        ),
                        "centroid": {
                            "x": round(cx, 2),
                            "y": round(cy, 2),
                        },
                        "source":
                            "semantic_segmentation",
                    }
                )

            candidates.sort(
                key=lambda x:
                    x["area_pixels"],
                reverse=True,
            )

            final: List[
                Dict[str, Any]
            ] = []

            for candidate in candidates:

                duplicate = False

                for existing in final:

                    iou = self._calculate_iou(
                        candidate["bbox"],
                        existing["bbox"],
                    )

                    containment = (
                        self._containment_ratio(
                            candidate["bbox"],
                            existing["bbox"],
                        )
                    )

                    if (
                        iou >= 0.75
                        or containment >= 0.90
                    ):
                        duplicate = True
                        break

                if not duplicate:
                    final.append(candidate)

            final.sort(
                key=lambda x: (
                    x["bbox"]["y1"],
                    x["bbox"]["x1"],
                )
            )

            return final

        except ImportError:

            return (
                self._fallback_connected_components(
                    binary_mask
                )
            )

    # ============================================================
    # FALLBACK CONNECTED COMPONENTS
    # ============================================================

    def _fallback_connected_components(
        self,
        binary_mask: np.ndarray,
    ) -> List[Dict[str, Any]]:

        height, width = binary_mask.shape

        visited = np.zeros_like(
            binary_mask,
            dtype=bool,
        )

        detections: List[
            Dict[str, Any]
        ] = []

        for y in range(height):

            for x in range(width):

                if (
                    binary_mask[y, x] == 0
                    or visited[y, x]
                ):
                    continue

                stack = [(y, x)]

                visited[y, x] = True

                pixels: List[
                    Tuple[int, int]
                ] = []

                while stack:

                    cy, cx = stack.pop()

                    pixels.append(
                        (cy, cx)
                    )

                    for dy, dx in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    ):

                        ny = cy + dy
                        nx = cx + dx

                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and binary_mask[
                                ny,
                                nx
                            ] != 0
                            and not visited[
                                ny,
                                nx
                            ]
                        ):

                            visited[
                                ny,
                                nx
                            ] = True

                            stack.append(
                                (ny, nx)
                            )

                area = len(pixels)

                if area < self.min_building_area:
                    continue

                ys = [
                    p[0]
                    for p in pixels
                ]

                xs = [
                    p[1]
                    for p in pixels
                ]

                x1 = min(xs)
                x2 = max(xs) + 1

                y1 = min(ys)
                y2 = max(ys) + 1

                bbox_area = (
                    x2 - x1
                ) * (
                    y2 - y1
                )

                if bbox_area <= 0:
                    continue

                detections.append(
                    {
                        "class": "building",
                        "confidence": None,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        },
                        "area_pixels": area,
                        "source":
                            "semantic_segmentation",
                    }
                )

        detections.sort(
            key=lambda x: (
                x["bbox"]["y1"],
                x["bbox"]["x1"],
            )
        )

        return detections

    # ============================================================
    # CLASS NORMALIZATION
    # ============================================================

    @classmethod
    def _normalize_class_name(
        cls,
        class_name: Optional[str],
    ) -> Optional[str]:

        if not class_name:
            return None

        value = (
            str(class_name)
            .lower()
            .strip()
        )

        return cls.CLASS_ALIASES.get(
            value,
            value,
        )

    # ============================================================
    # DESCRIPTION
    # ============================================================

    @staticmethod
    def _build_description(
        detections: List[
            Dict[str, Any]
        ],
        requested_class: Optional[str],
    ) -> str:

        if not detections:

            if requested_class:
                return (
                    f"No '{requested_class}' "
                    "objects were detected."
                )

            return (
                "No supported objects "
                "were detected."
            )

        counts = Counter(
            item["class"]
            for item in detections
        )

        preferred = [
            "car",
            "van",
            "truck",
            "bus",
            "motor",
            "bicycle",
            "tricycle",
            "awning-tricycle",
            "person",
            "building",
        ]

        ordered = [
            c
            for c in preferred
            if c in counts
        ]

        ordered += [
            c
            for c in counts
            if c not in ordered
        ]

        parts = []

        for name in ordered:

            count = counts[name]

            if count == 1:
                noun = name

            else:

                if name == "person":
                    noun = "people"

                elif name == "motor":
                    noun = "motors"

                elif name == "bicycle":
                    noun = "bicycles"

                elif name == "building":
                    noun = "buildings"

                elif name == "tricycle":
                    noun = "tricycles"

                elif name == "awning-tricycle":
                    noun = "awning-tricycles"

                elif name == "van":
                    noun = "vans"

                elif name == "bus":
                    noun = "buses"

                else:
                    noun = f"{name}s"

            parts.append(
                f"{count} {noun}"
            )

        if len(parts) == 1:
            return f"Detected {parts[0]}."

        if len(parts) == 2:
            return (
                f"Detected {parts[0]} "
                f"and {parts[1]}."
            )

        return (
            "Detected "
            + ", ".join(parts[:-1])
            + ", and "
            + parts[-1]
            + "."
        )

    # ============================================================
    # FILE VALIDATION
    # ============================================================

    @staticmethod
    def _check_file(
        path: str,
    ) -> None:

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Input path is not a file: {path}"
            )

    # ============================================================
    # STATUS
    # ============================================================

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_device(self) -> str:
        return str(self.device)

    def get_model_name(self) -> str:
        return self.MODEL_NAME


__all__ = [
    "ObjectDetector"
]
