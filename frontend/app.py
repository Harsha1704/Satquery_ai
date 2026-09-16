from __future__ import annotations

import hashlib
import uuid
import re
from threading import Lock
import os
import sys
import traceback
import threading
import time
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from PIL import Image

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    Response,
    send_file,
)

# ============================================================
# PROJECT PATHS
# ============================================================

FRONTEND_DIR = Path(__file__).resolve().parent
ROOT = FRONTEND_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# RASTERIO / PROJ
# ============================================================

try:
    import rasterio

    bundled_proj = (
        Path(rasterio.__file__).resolve().parent
        / "proj_data"
    )

    if (bundled_proj / "proj.db").is_file():
        os.environ["PROJ_LIB"] = str(bundled_proj)
        os.environ["PROJ_DATA"] = str(bundled_proj)

        from rasterio.env import (
            set_proj_data_search_path,
        )

        set_proj_data_search_path(
            str(bundled_proj)
        )

    from rasterio.transform import array_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import (
        Resampling,
        calculate_default_transform,
        reproject,
        transform_bounds,
    )

except Exception:
    rasterio = None
    transform_bounds = None
    calculate_default_transform = None
    reproject = None
    Resampling = None
    array_bounds = None
    Window = None
    from_bounds = None


# ============================================================
# SATQUERY BACKEND
# ============================================================

from ai.execution import ResultAdapter
from ai.report import SatQueryReportGenerator
from ai.router.orchestrator import SatQueryOrchestrator
from ai.router.planner import QueryPlanner
from ai.validation import InputValidator

import gee_temporal


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(
        FRONTEND_DIR / "templates"
    ),
    static_folder=str(
        FRONTEND_DIR / "static"
    ),
)

app.config["MAX_CONTENT_LENGTH"] = (
    220 * 1024 * 1024
)


UPLOAD_DIR = (
    FRONTEND_DIR
    / "uploads"
)

REPORT_DIR = (
    ROOT
    / "outputs"
    / "reports"
)

MAP_PREVIEW_DIR = (
    FRONTEND_DIR
    / "static"
    / "map_previews"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAP_PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROI_DIR = (
    ROOT
    / "outputs"
    / "roi"
)

ROI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Stores successful analyses while Flask is running.
# Map View uses analysis_id instead of trusting file paths
# sent back from the browser.

ANALYSIS_REGISTRY: Dict[
    str,
    Dict[str, Any],
] = {}


MAX_REGISTERED_ANALYSES = 50

ANALYSIS_REGISTRY_LOCK = Lock()

# ============================================================
# SATQUERY COMPONENTS
# ============================================================

orchestrator = SatQueryOrchestrator()

planner = QueryPlanner()

validator = InputValidator()

adapter = ResultAdapter()

report_generator = SatQueryReportGenerator(
    output_dir=str(REPORT_DIR)
)


MODEL_RUNTIME_STATE = {
    "vqa": {
        "status": "not_started",
        "device": None,
        "model": None,
        "load_ms": None,
        "error": None,
    }
}


def preload_competition_models():
    # Replace the snapshot atomically so the status route sees consistent fields.
    MODEL_RUNTIME_STATE["vqa"] = {
        "status": "loading", "device": None, "model": None,
        "load_ms": None, "error": None,
    }
    try:
        start = time.perf_counter()
        result = orchestrator.preload_vqa(
            warmup=os.getenv("SATQUERY_BLIP_WARMUP", "0") == "1",
        )
        warmup = result.get("warmup")
        if not result.get("success") or (
            isinstance(warmup, dict) and not warmup.get("success")
        ):
            raise RuntimeError(result.get("error") or "VQA preload or warmup failed.")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        MODEL_RUNTIME_STATE["vqa"] = {
            "status": "ready",
            "device": result.get("device"),
            "model": result.get("model"),
            "load_ms": round(elapsed_ms, 2),
            "error": None,
        }
        print(
            "[SatQuery][Startup] BLIP VQA ready | "
            f"device={result.get('device')} | time={elapsed_ms / 1000.0:.2f}s"
        )
    except Exception as error:
        MODEL_RUNTIME_STATE["vqa"] = {
            **MODEL_RUNTIME_STATE["vqa"], "status": "error", "error": str(error),
        }
        print(f"[SatQuery][Startup] BLIP preload failed: {error}")


if os.getenv("SATQUERY_PRELOAD_VQA", "1") == "1":
    threading.Thread(
        target=preload_competition_models,
        daemon=True,
        name="satquery-vqa-preloader",
    ).start()


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_intent(
    intent: Any,
) -> str:

    if intent is None:
        return "unknown"

    if hasattr(intent, "value"):
        return str(intent.value)

    value = str(intent)

    if value.startswith("Intent."):
        value = value.split(
            ".",
            1,
        )[1].lower()

    return value


def plan_query(
    query: str,
) -> Dict[str, Any]:

    try:
        plan = planner.plan(query)

    except AttributeError:

        try:
            plan = planner.create_plan(
                query
            )

        except AttributeError:
            plan = planner(query)

    if hasattr(plan, "to_dict"):
        return plan.to_dict()

    if isinstance(plan, dict):
        return plan

    data = {}

    for field in [
        "intent",
        "confidence",
        "required_tools",
        "tools",
        "target",
        "targets",
        "operation",
        "years",
        "change_direction",
    ]:
        if hasattr(plan, field):
            data[field] = getattr(
                plan,
                field,
            )

    return data


def plan_tools(
    plan: Dict[str, Any],
):

    tools = (
        plan.get("required_tools")
        or plan.get("tools")
        or []
    )

    if isinstance(tools, str):
        tools = [tools]

    return [
        str(tool)
        for tool in tools
    ]


def save_upload(
    field_name: str,
    prefix: str,
) -> Optional[str]:

    uploaded = request.files.get(
        field_name
    )

    if (
        not uploaded
        or not uploaded.filename
    ):
        return None

    suffix = (
        Path(uploaded.filename)
        .suffix
        .lower()
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    destination = (
        UPLOAD_DIR
        / f"{prefix}_{timestamp}{suffix}"
    )

    uploaded.save(destination)

    return str(destination)


def json_safe(
    value: Any,
):

    if (
        value is None
        or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    intent: str,
    image_path: Optional[str],
    before_path: Optional[str],
    after_path: Optional[str],
    optical_path: Optional[str],
    sar_path: Optional[str],
):

    if intent == "change_detection":

        if (
            not before_path
            or not after_path
        ):
            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "CHANGE_PAIR_REQUIRED",
                        "message":
                            "Upload both Before and After images.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_change_pair(
                before_path,
                after_path,
            )
        )

    if intent == "optical_sar_fusion":

        if (
            not optical_path
            or not sar_path
        ):
            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "OPTICAL_SAR_PAIR_REQUIRED",
                        "message":
                            "Upload both Optical and SAR imagery.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_optical_sar_pair(
                optical_path,
                sar_path,
            )
        )

    if intent == "sar_analysis":

        if not sar_path:
            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "SAR_REQUIRED",
                        "message":
                            "Upload a SAR image.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_single_image(
                sar_path,
                expected_modality="sar",
            )
        )

    if intent == "multispectral_analysis":

        path = (
            optical_path
            or image_path
        )

        if not path:
            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "MULTISPECTRAL_REQUIRED",
                        "message":
                            "Upload a multispectral GeoTIFF/TIFF or NPY file.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_multispectral(
                path
            )
        )

    if image_path:
        return (
            validator
            .validate_single_image(
                image_path
            )
        )

    return None


# ============================================================
# EXECUTION
# ============================================================

def execute_query(
    query: str,
    intent: str,
    image_path: Optional[str],
    before_path: Optional[str],
    after_path: Optional[str],
    optical_path: Optional[str],
    sar_path: Optional[str],
):

    kwargs = {}

    if intent == "change_detection":

        kwargs["before_path"] = (
            before_path
        )

        kwargs["after_path"] = (
            after_path
        )

    elif intent == "optical_sar_fusion":

        kwargs["optical_path"] = (
            optical_path
        )

        kwargs["sar_path"] = (
            sar_path
        )

    elif intent == "sar_analysis":

        kwargs["sar_path"] = (
            sar_path
        )

    elif intent == "multispectral_analysis":

        kwargs["image_path"] = (
            optical_path
            or image_path
        )

    else:

        kwargs["image_path"] = (
            image_path
        )

    return orchestrator.execute(
        query,
        **kwargs,
    )


def unwrap_execution_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(result, dict):
        return {
            "success": True,
            "answer": str(result),
        }

    execution = result.get(
        "execution"
    )

    if not isinstance(
        execution,
        dict,
    ):
        return dict(result)

    merged = dict(result)

    for key, value in execution.items():
        if value is not None:
            merged[key] = value

    merged["routing"] = {
        "query":
            result.get("query"),
        "intent":
            result.get("intent"),
        "routing_confidence":
            result.get("confidence"),
        "matched_keywords":
            result.get(
                "matched_keywords",
                [],
            ),
        "required_tools":
            result.get(
                "required_tools",
                [],
            ),
    }

    if execution.get("answer") is not None:
        merged["answer"] = (
            execution["answer"]
        )

    if execution.get("model"):
        merged["model"] = (
            execution["model"]
        )

    if execution.get("device"):
        merged["device"] = (
            execution["device"]
        )

    evidence = (
        execution.get(
            "visual_evidence"
        )
        or execution.get(
            "evidence"
        )
        or execution.get(
            "evidence_path"
        )
    )

    if evidence is not None:
        merged["evidence"] = evidence

    if execution.get("limitations"):
        merged["limitations"] = (
            execution["limitations"]
        )

    return merged


# ============================================================
# EVIDENCE
# ============================================================

def _raw_evidence_paths(
    result: Dict[str, Any],
):

    evidence = (
        result.get("evidence")
        or result.get("visual_evidence")
        or result.get("evidence_path")
    )

    if evidence is None:
        return []

    if not isinstance(
        evidence,
        (
            list,
            tuple,
        ),
    ):
        evidence = [evidence]

    paths = []

    for item in evidence:

        if isinstance(item, dict):

            path_value = (
                item.get("path")
                or item.get("file")
                or item.get("image")
            )

        else:
            path_value = item

        if not path_value:
            continue

        path = Path(
            str(path_value)
        )

        if not path.is_absolute():
            path = ROOT / path

        if path.exists():
            paths.append(
                path.resolve()
            )

    return paths


def evidence_items(
    result: Dict[str, Any],
):

    output = []

    for path in _raw_evidence_paths(
        result
    ):

        try:
            relative = (
                path
                .relative_to(
                    ROOT.resolve()
                )
            )

        except Exception:
            continue

        output.append(
            {
                "name": path.name,
                "url":
                    f"/project-file/{relative.as_posix()}",
            }
        )

    return output


# ============================================================
# MAP PREVIEW HELPERS
# ============================================================

def _preview_token(
    path: Path,
    suffix: str,
):

    try:
        signature = (
            f"{path.resolve()}::"
            f"{path.stat().st_mtime_ns}::"
            f"{suffix}"
        )

    except Exception:
        signature = (
            f"{path}::{suffix}"
        )

    digest = hashlib.sha1(
        signature.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:14]

    return digest


def _static_preview_url(
    path: Path,
):

    return (
        "/static/map_previews/"
        + path.name
    )


def _stretch_band(
    band: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:

    band = band.astype(
        np.float32
    )

    good = (
        valid_mask
        & np.isfinite(band)
    )

    values = band[good]

    if values.size == 0:
        return np.zeros(
            band.shape,
            dtype=np.uint8,
        )

    low, high = np.percentile(
        values,
        [
            2,
            98,
        ],
    )

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        low = float(
            np.nanmin(values)
        )

        high = float(
            np.nanmax(values)
        )

    if high <= low:
        result = np.zeros(
            band.shape,
            dtype=np.uint8,
        )

        result[good] = 128

        return result

    scaled = (
        (band - low)
        / (high - low)
    )

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    scaled[~good] = 0.0

    return (
        scaled * 255.0
    ).astype(
        np.uint8
    )


def _dataset_valid_mask(
    dataset,
):

    try:
        mask = (
            dataset.dataset_mask()
            > 0
        )

    except Exception:
        mask = np.ones(
            (
                dataset.height,
                dataset.width,
            ),
            dtype=bool,
        )

    return mask


def _band_descriptions(
    dataset,
):

    return [
        (
            description
            or ""
        )
        .strip()
        .upper()
        for description
        in dataset.descriptions
    ]


def _find_band(
    dataset,
    names,
):

    descriptions = (
        _band_descriptions(
            dataset
        )
    )

    names = {
        name.upper()
        for name in names
    }

    for index, description in enumerate(
        descriptions,
        start=1,
    ):
        if description in names:
            return index

    return None


def _source_rgba(
    dataset,
) -> np.ndarray:

    valid = _dataset_valid_mask(
        dataset
    )

    red_index = _find_band(
        dataset,
        {
            "B04",
            "B4",
            "RED",
        },
    )

    green_index = _find_band(
        dataset,
        {
            "B03",
            "B3",
            "GREEN",
        },
    )

    blue_index = _find_band(
        dataset,
        {
            "B02",
            "B2",
            "BLUE",
        },
    )

    if (
        red_index
        and green_index
        and blue_index
    ):

        indexes = [
            red_index,
            green_index,
            blue_index,
        ]

    elif dataset.count >= 3:

        indexes = [
            1,
            2,
            3,
        ]

    else:

        first = dataset.read(1)

        gray = _stretch_band(
            first,
            valid,
        )

        alpha = np.where(
            valid,
            255,
            0,
        ).astype(
            np.uint8
        )

        return np.dstack(
            [
                gray,
                gray,
                gray,
                alpha,
            ]
        )

    bands = dataset.read(
        indexes
    )

    rgb = []

    for band in bands:
        rgb.append(
            _stretch_band(
                band,
                valid,
            )
        )

    alpha = np.where(
        valid,
        255,
        0,
    ).astype(
        np.uint8
    )

    return np.dstack(
        [
            rgb[0],
            rgb[1],
            rgb[2],
            alpha,
        ]
    )


def _ndvi_rgba(
    dataset,
) -> Optional[np.ndarray]:

    red_index = _find_band(
        dataset,
        {
            "B04",
            "B4",
            "RED",
        },
    )

    nir_index = _find_band(
        dataset,
        {
            "B08",
            "B8",
            "NIR",
        },
    )

    if (
        red_index is None
        or nir_index is None
    ):
        return None

    red = dataset.read(
        red_index
    ).astype(
        np.float32
    )

    nir = dataset.read(
        nir_index
    ).astype(
        np.float32
    )

    valid = (
        _dataset_valid_mask(
            dataset
        )
        & np.isfinite(red)
        & np.isfinite(nir)
    )

    denominator = (
        nir + red
    )

    valid &= (
        np.abs(denominator)
        > 1e-6
    )

    ndvi = np.zeros(
        red.shape,
        dtype=np.float32,
    )

    ndvi[valid] = (
        (
            nir[valid]
            - red[valid]
        )
        /
        denominator[valid]
    )

    ndvi = np.clip(
        ndvi,
        -1.0,
        1.0,
    )

    normalized = (
        ndvi + 1.0
    ) / 2.0

    red_channel = (
        255.0
        * (
            1.0
            - normalized
        )
    ).astype(
        np.uint8
    )

    green_channel = (
        255.0
        * normalized
    ).astype(
        np.uint8
    )

    blue_channel = (
        70.0
        * (
            1.0
            - np.abs(ndvi)
        )
    ).astype(
        np.uint8
    )

    alpha = np.where(
        valid,
        220,
        0,
    ).astype(
        np.uint8
    )

    return np.dstack(
        [
            red_channel,
            green_channel,
            blue_channel,
            alpha,
        ]
    )


def _reproject_rgba_to_wgs84(
    dataset,
    rgba: np.ndarray,
    destination: Path,
):

    if (
        rasterio is None
        or calculate_default_transform
        is None
        or reproject is None
        or dataset.crs is None
    ):
        return None

    left = float(
        dataset.bounds.left
    )

    bottom = float(
        dataset.bounds.bottom
    )

    right = float(
        dataset.bounds.right
    )

    top = float(
        dataset.bounds.top
    )

    (
        destination_transform,
        destination_width,
        destination_height,
    ) = calculate_default_transform(
        dataset.crs,
        "EPSG:4326",
        dataset.width,
        dataset.height,
        left,
        bottom,
        right,
        top,
    )

    destination_width = max(
        1,
        int(destination_width),
    )

    destination_height = max(
        1,
        int(destination_height),
    )

    output = np.zeros(
        (
            4,
            destination_height,
            destination_width,
        ),
        dtype=np.uint8,
    )

    for channel in range(4):

        resampling = (
            Resampling.nearest
            if channel == 3
            else Resampling.bilinear
        )

        reproject(
            source=rgba[:, :, channel],
            destination=output[channel],
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            resampling=resampling,
        )

    rgba_output = np.moveaxis(
        output,
        0,
        -1,
    )

    Image.fromarray(
        rgba_output,
        mode="RGBA",
    ).save(
        destination
    )

    west, south, east, north = (
        array_bounds(
            destination_height,
            destination_width,
            destination_transform,
        )
    )

    return [
        float(west),
        float(south),
        float(east),
        float(north),
    ]


def _create_source_preview(
    raster_path: Path,
    layer_name: str,
):

    if rasterio is None:
        return None

    if raster_path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        return None

    try:

        with rasterio.open(
            raster_path
        ) as dataset:

            if dataset.crs is None:
                return None

            rgba = _source_rgba(
                dataset
            )

            token = _preview_token(
                raster_path,
                "source",
            )

            destination = (
                MAP_PREVIEW_DIR
                / f"{token}_source.png"
            )

            bounds = (
                _reproject_rgba_to_wgs84(
                    dataset,
                    rgba,
                    destination,
                )
            )

            if not bounds:
                return None

            return {
                "id":
                    f"{layer_name}_source",
                "name":
                    f"{layer_name.title()} raster",
                "role":
                    "source",
                "url":
                    _static_preview_url(
                        destination
                    ),
                "bounds_wgs84":
                    bounds,
                "opacity":
                    0.92,
                "visible":
                    True,
            }

    except Exception:
        traceback.print_exc()
        return None


def _create_ndvi_preview(
    raster_path: Path,
):

    if rasterio is None:
        return None

    if raster_path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        return None

    try:

        with rasterio.open(
            raster_path
        ) as dataset:

            if dataset.crs is None:
                return None

            rgba = _ndvi_rgba(
                dataset
            )

            if rgba is None:
                return None

            token = _preview_token(
                raster_path,
                "ndvi",
            )

            destination = (
                MAP_PREVIEW_DIR
                / f"{token}_ndvi.png"
            )

            bounds = (
                _reproject_rgba_to_wgs84(
                    dataset,
                    rgba,
                    destination,
                )
            )

            if not bounds:
                return None

            return {
                "id":
                    "ndvi_result",
                "name":
                    "NDVI result",
                "role":
                    "result",
                "url":
                    _static_preview_url(
                        destination
                    ),
                "bounds_wgs84":
                    bounds,
                "opacity":
                    0.72,
                "visible":
                    True,
            }

    except Exception:
        traceback.print_exc()
        return None


def _create_evidence_overlay(
    evidence_path: Path,
    raster_path: Path,
    index: int,
):

    if rasterio is None:
        return None

    if raster_path.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:
        return None

    try:

        image = Image.open(
            evidence_path
        ).convert(
            "RGBA"
        )

        with rasterio.open(
            raster_path
        ) as dataset:

            if dataset.crs is None:
                return None

            if (
                image.width
                != dataset.width
                or image.height
                != dataset.height
            ):
                return None

            rgba = np.asarray(
                image,
                dtype=np.uint8,
            )

            token = _preview_token(
                evidence_path,
                f"evidence_{index}",
            )

            destination = (
                MAP_PREVIEW_DIR
                / f"{token}_evidence.png"
            )

            bounds = (
                _reproject_rgba_to_wgs84(
                    dataset,
                    rgba,
                    destination,
                )
            )

            if not bounds:
                return None

            return {
                "id":
                    f"evidence_{index}",
                "name":
                    evidence_path.stem
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),
                "role":
                    "result",
                "url":
                    _static_preview_url(
                        destination
                    ),
                "bounds_wgs84":
                    bounds,
                "opacity":
                    0.72,
                "visible":
                    False,
            }

    except Exception:
        return None


# ============================================================
# GEOSPATIAL PAYLOAD
# ============================================================

def build_geospatial_payload(
    validation: Optional[
        Dict[str, Any]
    ],
    standardized: Dict[
        str,
        Any,
    ],
    intent: str,
):

    if (
        not validation
        or not isinstance(
            validation,
            dict,
        )
    ):
        return {
            "available": False,
            "reason":
                "No geospatial validation metadata available.",
            "layers": [],
            "visual_layers": [],
        }

    metadata = validation.get(
        "metadata"
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return {
            "available": False,
            "reason":
                "No geospatial metadata available.",
            "layers": [],
            "visual_layers": [],
        }

    candidates = []

    for key in [
        "optical",
        "sar",
        "before",
        "after",
    ]:

        value = metadata.get(key)

        if isinstance(value, dict):

            candidates.append(
                {
                    "name": key,
                    "metadata": value,
                }
            )

    if not candidates:

        candidates.append(
            {
                "name": "image",
                "metadata": metadata,
            }
        )

    layers = []

    visual_layers = []

    primary_raster_path = None

    for index, candidate in enumerate(
        candidates
    ):

        name = candidate["name"]

        info = candidate["metadata"]

        if not info.get(
            "georeferenced"
        ):
            continue

        crs = info.get("crs")

        bounds = info.get(
            "bounds"
        )

        if (
            not crs
            or not bounds
        ):
            continue

        try:

            bounds = [
                float(value)
                for value in bounds
            ]

            if (
                len(bounds) != 4
                or not all(
                    isfinite(value)
                    for value in bounds
                )
            ):
                continue

            if (
                str(crs).upper()
                not in {
                    "EPSG:4326",
                    "OGC:CRS84",
                }
            ):

                if transform_bounds is None:
                    continue

                west, south, east, north = (
                    transform_bounds(
                        crs,
                        "EPSG:4326",
                        bounds[0],
                        bounds[1],
                        bounds[2],
                        bounds[3],
                        densify_pts=21,
                    )
                )

            else:

                (
                    west,
                    south,
                    east,
                    north,
                ) = bounds

            if not (
                all(
                    isfinite(value)
                    for value
                    in (
                        west,
                        south,
                        east,
                        north,
                    )
                )
                and -180
                <= west
                < east
                <= 180
                and -90
                <= south
                < north
                <= 90
            ):
                continue

            layer = {
                "id":
                    f"{name}_{index}",
                "name":
                    name,
                "source_crs":
                    str(crs),
                "bounds_source":
                    bounds,
                "bounds_wgs84":
                    [
                        float(west),
                        float(south),
                        float(east),
                        float(north),
                    ],
                "width":
                    info.get("width"),
                "height":
                    info.get("height"),
                "bands":
                    info.get("bands"),
                "dtype":
                    info.get("dtype"),
                "resolution_x":
                    info.get(
                        "resolution_x"
                    ),
                "resolution_y":
                    info.get(
                        "resolution_y"
                    ),
                "nodata":
                    info.get("nodata"),
                "transform":
                    info.get(
                        "transform"
                    ),
            }

            layers.append(layer)

            path_value = info.get(
                "path"
            )

            if path_value:

                raster_path = Path(
                    str(path_value)
                )

                if raster_path.exists():

                    if (
                        primary_raster_path
                        is None
                    ):
                        primary_raster_path = (
                            raster_path
                        )

                    preview = (
                        _create_source_preview(
                            raster_path,
                            name,
                        )
                    )

                    if preview:

                        if index > 0:
                            preview[
                                "visible"
                            ] = False

                        visual_layers.append(
                            preview
                        )

        except Exception:
            traceback.print_exc()
            continue

    if not layers:

        return {
            "available": False,
            "reason":
                (
                    "The selected analysis does not "
                    "contain usable georeferenced imagery."
                ),
            "layers": [],
            "visual_layers": [],
        }

    # --------------------------------------------------------
    # NDVI MAP RESULT
    # --------------------------------------------------------

    if (
        intent
        == "multispectral_analysis"
        and primary_raster_path
    ):

        ndvi_preview = (
            _create_ndvi_preview(
                primary_raster_path
            )
        )

        if ndvi_preview:
            visual_layers.append(
                ndvi_preview
            )

    # --------------------------------------------------------
    # MODEL EVIDENCE OVERLAYS
    # --------------------------------------------------------

    if primary_raster_path:

        evidence_paths = (
            _raw_evidence_paths(
                standardized
            )
        )

        for evidence_index, evidence_path in enumerate(
            evidence_paths[:4]
        ):

            overlay = (
                _create_evidence_overlay(
                    evidence_path,
                    primary_raster_path,
                    evidence_index,
                )
            )

            if overlay:
                visual_layers.append(
                    overlay
                )

    return {
        "available": True,
        "layers": layers,
        "visual_layers":
            visual_layers,
        "has_source_raster":
            any(
                layer.get("role")
                == "source"
                for layer
                in visual_layers
            ),
        "has_result_overlay":
            any(
                layer.get("role")
                == "result"
                for layer
                in visual_layers
            ),
    }


def register_analysis(
    query: str,
    intent: str,
    inputs: Dict[str, Any],
    validation: Optional[Dict[str, Any]],
    standardized: Dict[str, Any],
    response: Dict[str, Any],
) -> str:
    analysis_id = uuid.uuid4().hex[:20]
    record = {
        "query": query,
        "intent": intent,
        "inputs": dict(inputs),
        "validation": validation,
        "standardized": standardized,
        "response": response,
        "created_at": datetime.utcnow().isoformat(),
    }
    with ANALYSIS_REGISTRY_LOCK:
        ANALYSIS_REGISTRY[analysis_id] = record
        while len(ANALYSIS_REGISTRY) > MAX_REGISTERED_ANALYSES:
            ANALYSIS_REGISTRY.pop(next(iter(ANALYSIS_REGISTRY)))
    return analysis_id


def _validate_wgs84_bounds(
    bounds: Any,
):

    if (
        not isinstance(
            bounds,
            (
                list,
                tuple,
            ),
        )
        or len(bounds) != 4
    ):

        raise ValueError(
            "ROI bounds must be [west, south, east, north]."
        )

    west, south, east, north = [
        float(value)
        for value
        in bounds
    ]

    if not all(
        isfinite(value)
        for value
        in (
            west,
            south,
            east,
            north,
        )
    ):

        raise ValueError(
            "ROI bounds contain invalid coordinates."
        )

    if not (
        -180.0
        <= west
        < east
        <= 180.0
    ):

        raise ValueError(
            "Invalid longitude bounds."
        )

    if not (
        -90.0
        <= south
        < north
        <= 90.0
    ):

        raise ValueError(
            "Invalid latitude bounds."
        )

    return (
        west,
        south,
        east,
        north,
    )


def crop_geotiff_to_roi(
    source_path: str,
    bounds_wgs84,
    destination_path: Path,
) -> str:

    if rasterio is None:

        raise RuntimeError(
            "Rasterio is required for geospatial ROI analysis."
        )

    source = Path(
        source_path
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Source raster not found: {source}"
        )

    if source.suffix.lower() not in {
        ".tif",
        ".tiff",
    }:

        raise ValueError(
            "Map ROI analysis requires a georeferenced TIFF/GeoTIFF source."
        )

    (
        west,
        south,
        east,
        north,
    ) = _validate_wgs84_bounds(
        bounds_wgs84
    )

    with rasterio.open(
        source
    ) as src:

        if src.crs is None:

            raise ValueError(
                "The source raster does not contain a CRS."
            )

        (
            source_left,
            source_bottom,
            source_right,
            source_top,
        ) = transform_bounds(
            "EPSG:4326",
            src.crs,
            west,
            south,
            east,
            north,
            densify_pts=21,
        )

        raster_left = float(
            src.bounds.left
        )

        raster_bottom = float(
            src.bounds.bottom
        )

        raster_right = float(
            src.bounds.right
        )

        raster_top = float(
            src.bounds.top
        )

        clipped_left = max(
            source_left,
            raster_left,
        )

        clipped_bottom = max(
            source_bottom,
            raster_bottom,
        )

        clipped_right = min(
            source_right,
            raster_right,
        )

        clipped_top = min(
            source_top,
            raster_top,
        )

        if (
            clipped_left
            >= clipped_right
            or clipped_bottom
            >= clipped_top
        ):

            raise ValueError(
                "The selected map area does not overlap the uploaded GeoTIFF."
            )

        window = from_bounds(
            clipped_left,
            clipped_bottom,
            clipped_right,
            clipped_top,
            transform=src.transform,
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        full_window = Window(
            col_off=0,
            row_off=0,
            width=src.width,
            height=src.height,
        )

        window = window.intersection(
            full_window
        )

        if (
            window.width < 1
            or window.height < 1
        ):

            raise ValueError(
                "The selected area is smaller than one raster pixel."
            )

        data = src.read(
            window=window
        )

        roi_transform = (
            src.window_transform(
                window
            )
        )

        profile = (
            src.profile.copy()
        )

        profile.update(
            {
                "height":
                    int(
                        window.height
                    ),

                "width":
                    int(
                        window.width
                    ),

                "transform":
                    roi_transform,
            }
        )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            destination_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                data
            )

            dst.write_mask(src.dataset_mask(window=window))
            dst.update_tags(**src.tags())
            for band in range(1, src.count + 1):
                dst.update_tags(band, **src.tags(band))

            for index, description in enumerate(
                src.descriptions,
                start=1,
            ):

                if description:

                    dst.set_band_description(
                        index,
                        description,
                    )

    return str(
        destination_path
    )


def geotiff_to_rgb_png(
    source_path: str,
    destination_path: Path,
) -> str:

    if rasterio is None:

        raise RuntimeError(
            "Rasterio is required."
        )

    with rasterio.open(
        source_path
    ) as dataset:

        rgba = _source_rgba(
            dataset
        )

    rgb = rgba[
        :,
        :,
        :3
    ]

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        rgb,
        mode="RGB",
    ).save(
        destination_path
    )

    return str(
        destination_path
    )


def make_roi_name(
    analysis_id: str,
    label: str,
    extension: str = ".tif",
):

    token = (
        uuid.uuid4()
        .hex[:10]
    )

    return (
        ROI_DIR
        / (
            f"{analysis_id}_"
            f"{label}_"
            f"{token}"
            f"{extension}"
        )
    )


def prepare_roi_execution(
    analysis_id: str,
    intent: str,
    original_inputs: Dict[
        str,
        Any,
    ],
    bounds_wgs84,
):

    cropped = {}

    def crop_input(
        input_key: str,
        label: str,
    ):

        path = (
            original_inputs.get(
                input_key
            )
        )

        if not path:

            raise ValueError(
                f"The original analysis does not contain {input_key}."
            )

        output = make_roi_name(
            analysis_id,
            label,
            ".tif",
        )

        cropped[input_key] = crop_geotiff_to_roi(
            path,
            bounds_wgs84,
            output,
        )
        return cropped[input_key]


    # --------------------------------------------------------
    # MULTISPECTRAL
    # --------------------------------------------------------

    if intent == "multispectral_analysis":

        source_key = (
            "optical_path"
            if original_inputs.get(
                "optical_path"
            )
            else "image_path"
        )

        optical_roi = crop_input(
            source_key,
            "multispectral",
        )

        validation = (
            validator
            .validate_multispectral(
                optical_roi
            )
        )

        return {
            "source_inputs": cropped,
            "image_path":
                optical_roi,

            "optical_path":
                optical_roi,

            "sar_path":
                None,

            "before_path":
                None,

            "after_path":
                None,

            "validation":
                validation,

            "map_source":
                optical_roi,
        }


    # --------------------------------------------------------
    # SAR
    # --------------------------------------------------------

    if intent == "sar_analysis":

        sar_roi = crop_input(
            "sar_path",
            "sar",
        )

        validation = (
            validator
            .validate_single_image(
                sar_roi,
                expected_modality="sar",
            )
        )

        return {
            "source_inputs": cropped,
            "image_path":
                None,

            "optical_path":
                None,

            "sar_path":
                sar_roi,

            "before_path":
                None,

            "after_path":
                None,

            "validation":
                validation,

            "map_source":
                sar_roi,
        }


    # --------------------------------------------------------
    # OPTICAL + SAR
    # --------------------------------------------------------

    if intent == "optical_sar_fusion":

        optical_roi = crop_input(
            "optical_path",
            "optical",
        )

        sar_roi = crop_input(
            "sar_path",
            "sar",
        )

        validation = (
            validator
            .validate_optical_sar_pair(
                optical_roi,
                sar_roi,
            )
        )

        return {
            "source_inputs": cropped,
            "image_path":
                None,

            "optical_path":
                optical_roi,

            "sar_path":
                sar_roi,

            "before_path":
                None,

            "after_path":
                None,

            "validation":
                validation,

            "map_source":
                optical_roi,
        }


    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    if intent == "change_detection":

        before_tif = crop_input(
            "before_path",
            "before",
        )

        after_tif = crop_input(
            "after_path",
            "after",
        )

        validation = (
            validator
            .validate_change_pair(
                before_tif,
                after_tif,
            )
        )

        before_png = (
            geotiff_to_rgb_png(
                before_tif,
                make_roi_name(
                    analysis_id,
                    "before_rgb",
                    ".png",
                ),
            )
        )

        after_png = (
            geotiff_to_rgb_png(
                after_tif,
                make_roi_name(
                    analysis_id,
                    "after_rgb",
                    ".png",
                ),
            )
        )

        return {
            "source_inputs": cropped,
            "image_path":
                None,

            "optical_path":
                None,

            "sar_path":
                None,

            "before_path":
                before_png,

            "after_path":
                after_png,

            "validation":
                validation,

            "map_source":
                before_tif,
        }


    # --------------------------------------------------------
    # VQA / GROUNDING / OBJECT DETECTION
    # --------------------------------------------------------

    source_key = None

    for key in [
        "image_path",
        "optical_path",
        "before_path",
    ]:

        if original_inputs.get(
            key
        ):

            source_key = key
            break

    if source_key is None:

        raise ValueError(
            "No georeferenced source image is available for the selected analysis."
        )

    roi_tif = crop_input(
        source_key,
        "visual",
    )

    validation = (
        validator
        .validate_single_image(
            roi_tif
        )
    )

    roi_png = (
        geotiff_to_rgb_png(
            roi_tif,
            make_roi_name(
                analysis_id,
                "visual_rgb",
                ".png",
            ),
        )
    )

    return {
        "source_inputs": cropped,
        "image_path":
            roi_png,

        "optical_path":
            None,

        "sar_path":
            None,

        "before_path":
            None,

        "after_path":
            None,

        "validation":
            validation,

        "map_source":
            roi_tif,
    }


def report_links(
    report: Dict[str, Any],
):

    output = {
        "html": None,
        "json": None,
    }

    for key in [
        "html",
        "json",
    ]:

        report_key = (
            f"{key}_report"
        )

        value = report.get(
            report_key
        )

        if not value:
            continue

        path = Path(
            value
        )

        if not path.exists():
            continue

        try:

            relative = (
                path
                .resolve()
                .relative_to(
                    ROOT.resolve()
                )
            )

        except Exception:
            continue

        output[key] = (
            "/project-file/"
            + relative.as_posix()
        )

    return output


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


@app.route("/map-view")
def map_view():
    return render_template(
        "map.html"
    )

@app.route("/why-satquery")
@app.route("/why_satquery")
def why_satquery():
    return render_template(
        "why_satquery.html"
    )


@app.route("/api/map-artifacts/<job_id>/<path:artifact_path>")
def map_artifact_proxy(job_id: str, artifact_path: str):
    """Serve API evidence through the Flask origin used by the map page.

    The browser therefore never needs a cross-origin request to the analysis
    API. The FastAPI endpoint remains the authority for published artifacts.
    """
    if not re.fullmatch(r"ana_[0-9a-f]{32}", job_id):
        abort(404)
    parts = artifact_path.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        abort(404)

    safe_path = "/".join(quote(part, safe="") for part in parts)
    endpoint = (
        "http://127.0.0.1:8000/api/v1/jobs/"
        f"{job_id}/artifacts/{safe_path}"
    )
    try:
        with urlopen(endpoint, timeout=30) as upstream:
            content_type = upstream.headers.get_content_type()
            if not content_type.startswith(("image/", "application/geo+json")):
                abort(415)
            return Response(
                upstream.read(),
                status=upstream.status,
                content_type=content_type,
                headers={"Cache-Control": "private, max-age=300"},
            )
    except HTTPError as exc:
        abort(exc.code if exc.code in {404, 415} else 502)
    except (URLError, TimeoutError):
        abort(503)


@app.route(
    "/project-file/<path:relative_path>"
)
def project_file(
    relative_path: str,
):

    target = (
        ROOT
        / relative_path
    ).resolve()

    try:

        target.relative_to(
            ROOT.resolve()
        )

    except ValueError:
        abort(403)

    if (
        not target.exists()
        or not target.is_file()
    ):
        abort(404)

    return send_file(target)


# ============================================================
# API
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"],
)
def analyze():

    try:

        query = (
            request.form.get(
                "query"
            )
            or ""
        ).strip()

        if not query:

            return jsonify(
                {
                    "success":
                        False,
                    "query":
                        "",
                    "error":
                        "Enter a natural-language query.",
                    "answer":
                        "Enter a natural-language query.",
                }
            ), 400

        plan = plan_query(
            query
        )

        intent = normalize_intent(
            plan.get("intent")
        )

        tools = plan_tools(
            plan
        )

        image_path = save_upload(
            "single_image",
            "single",
        )

        before_path = save_upload(
            "before_image",
            "before",
        )

        after_path = save_upload(
            "after_image",
            "after",
        )

        optical_path = save_upload(
            "optical_image",
            "optical",
        )

        sar_path = save_upload(
            "sar_image",
            "sar",
        )

        validation = validate_inputs(
            intent,
            image_path,
            before_path,
            after_path,
            optical_path,
            sar_path,
        )

        if (
            validation is not None
            and not validation.get(
                "valid",
                False,
            )
        ):

            errors = (
                validation.get(
                    "errors"
                )
                or []
            )

            messages = []

            for error in errors:

                if isinstance(
                    error,
                    dict,
                ):
                    message = (
                        error.get(
                            "message"
                        )
                        or error.get(
                            "code"
                        )
                    )

                else:
                    message = str(
                        error
                    )

                if (
                    message
                    and message
                    not in messages
                ):
                    messages.append(
                        message
                    )

            answer = (
                " ".join(messages)
                if messages
                else
                "Input validation failed."
            )

            return jsonify(
                {
                    "success":
                        False,
                    "query":
                        query,
                    "intent":
                        intent,
                    "answer":
                        answer,
                    "error":
                        "Input validation failed.",
                    "confidence":
                        0,
                    "validation":
                        json_safe(
                            validation
                        ),
                    "reports":
                        {},
                    "evidence":
                        [],
                    "geospatial":
                        {
                            "available":
                                False,
                            "reason":
                                answer,
                            "layers":
                                [],
                            "visual_layers":
                                [],
                        },
                }
            ), 400

        outer_result = execute_query(
            query=query,
            intent=intent,
            image_path=image_path,
            before_path=before_path,
            after_path=after_path,
            optical_path=optical_path,
            sar_path=sar_path,
        )

        result = (
            unwrap_execution_result(
                outer_result
            )
        )

        if tools:
            result[
                "tools_used"
            ] = tools

        if (
            result.get(
                "confidence"
            )
            is None
        ):

            result[
                "confidence"
            ] = plan.get(
                "confidence",
                0.0,
            )

            result[
                "confidence_type"
            ] = (
                "routing_confidence"
            )

        inputs = {
            key: value
            for key, value
            in {
                "image_path":
                    image_path,
                "before_path":
                    before_path,
                "after_path":
                    after_path,
                "optical_path":
                    optical_path,
                "sar_path":
                    sar_path,
            }.items()
            if value is not None
        }

        standardized = (
            adapter.standardize(
                result,
                query=query,
                intent=intent,
                tools=tools,
                inputs=inputs,
                validation=validation,
                default_confidence_type=(
                    "relative_tile_relevance"
                    if intent
                    == "text_guided_grounding"
                    else "model_score"
                ),
            )
        )

        for key in [
            "answer",
            "model",
            "device",
            "evidence",
            "limitations",
        ]:

            if (
                result.get(key)
                is not None
            ):
                standardized[
                    key
                ] = result[key]

        report = (
            report_generator.generate(
                standardized,
                query=query,
            )
        )

        confidence_details = (
            standardized.get(
                "confidence_details"
            )
            or {}
        )

        percentage = (
            confidence_details.get(
                "percentage"
            )
        )

        if percentage is None:

            raw = standardized.get(
                "confidence",
                0.0,
            )

            try:
                raw = float(raw)

            except Exception:
                raw = 0.0

            percentage = (
                raw * 100.0
                if raw <= 1.0
                else raw
            )

        html_report = Path(
            report["html_report"]
        )

        json_report = Path(
            report["json_report"]
        )

        reports = {
            "html": None,
            "json": None,
        }

        if html_report.exists():

            relative = (
                html_report
                .resolve()
                .relative_to(
                    ROOT.resolve()
                )
            )

            reports["html"] = (
                "/project-file/"
                + relative.as_posix()
            )

        if json_report.exists():

            relative = (
                json_report
                .resolve()
                .relative_to(
                    ROOT.resolve()
                )
            )

            reports["json"] = (
                "/project-file/"
                + relative.as_posix()
            )

        geospatial = (
            build_geospatial_payload(
                validation,
                standardized,
                intent,
            )
        )

        response = {
            "success":
                bool(
                    standardized.get(
                        "success",
                        True,
                    )
                ),

            "query":
                query,

            "intent":
                intent,

            "task":
                intent
                .replace(
                    "_",
                    " ",
                )
                .title(),

            "answer":
                (
                    standardized.get(
                        "answer"
                    )
                    or standardized.get(
                        "description"
                    )
                    or standardized.get(
                        "message"
                    )
                    or
                    "Analysis completed."
                ),

            "confidence":
                round(
                    float(
                        percentage
                        or 0.0
                    ),
                    2,
                ),

            "confidence_details":
                json_safe(
                    confidence_details
                ),

            "model":
                (
                    standardized.get(
                        "model"
                    )
                    or standardized.get(
                        "model_name"
                    )
                    or
                    "SatQuery AI Pipeline"
                ),

            "device":
                standardized.get(
                    "device"
                )
                or "cpu",

            "tools":
                tools,

            "validation":
                json_safe(
                    validation
                ),

            "evidence":
                evidence_items(
                    standardized
                ),

            "geospatial":
                geospatial,

            "reports":
                reports,

            "result":
                json_safe(
                    standardized
                ),
        }

        if response["success"]:
            response["analysis_id"] = register_analysis(
                query=query,
                intent=intent,
                inputs=inputs,
                validation=validation,
                standardized=standardized,
                response=response,
            )

        return jsonify(
            response
        )

    except Exception as exc:

        traceback.print_exc()

        return jsonify(
            {
                "success":
                    False,

                "error":
                    "analysis_execution_failed",

                "answer":
                    "The analysis could not be completed. Please verify the input imagery and try again.",

                "confidence":
                    0,

            }
        ), 500


@app.route(
    "/api/map-analyze",
    methods=["POST"],
)
def map_analyze():

    try:

        payload = (
            request.get_json(
                silent=True
            )
            or {}
        )

        if not isinstance(payload, dict):
            raise ValueError("Send a JSON object with analysis_id, query and bounds.")

        analysis_id = str(
            payload.get(
                "analysis_id"
            )
            or ""
        ).strip()

        query = str(
            payload.get(
                "query"
            )
            or ""
        ).strip()

        bounds = (
            payload.get(
                "bounds"
            )
        )


        if not analysis_id:

            return jsonify(
                {
                    "success":
                        False,

                    "error":
                        "Select a valid geospatial analysis first.",
                }
            ), 400


        if not query:

            return jsonify(
                {
                    "success":
                        False,

                    "error":
                        "Enter a question about the selected area.",
                }
            ), 400


        record = (
            ANALYSIS_REGISTRY.get(
                analysis_id
            )
        )


        if not record:

            return jsonify(
                {
                    "success":
                        False,

                    "error":
                        (
                            "This analysis is no longer available in the running server. "
                            "Run the GeoTIFF analysis again from the Analyze page."
                        ),
                }
            ), 404


        bounds = (
            _validate_wgs84_bounds(
                bounds
            )
        )


        plan = plan_query(
            query
        )


        intent = normalize_intent(
            plan.get(
                "intent"
            )
        )


        tools = plan_tools(
            plan
        )


        roi = prepare_roi_execution(
            analysis_id=analysis_id,
            intent=intent,
            original_inputs=record[
                "inputs"
            ],
            bounds_wgs84=bounds,
        )


        validation = (
            roi.get(
                "validation"
            )
        )


        if (
            validation is not None
            and not validation.get(
                "valid",
                False,
            )
        ):

            errors = (
                validation.get(
                    "errors"
                )
                or []
            )

            messages = []

            for error in errors:

                if isinstance(
                    error,
                    dict,
                ):

                    message = (
                        error.get(
                            "message"
                        )
                        or error.get(
                            "code"
                        )
                    )

                else:

                    message = str(
                        error
                    )

                if (
                    message
                    and message
                    not in messages
                ):

                    messages.append(
                        message
                    )


            return jsonify(
                {
                    "success":
                        False,

                    "intent":
                        intent,

                    "query":
                        query,

                    "error":
                        (
                            " ".join(
                                messages
                            )
                            or
                            "ROI validation failed."
                        ),

                    "validation":
                        json_safe(
                            validation
                        ),
                }
            ), 400


        outer_result = execute_query(
            query=query,
            intent=intent,
            image_path=roi.get(
                "image_path"
            ),
            before_path=roi.get(
                "before_path"
            ),
            after_path=roi.get(
                "after_path"
            ),
            optical_path=roi.get(
                "optical_path"
            ),
            sar_path=roi.get(
                "sar_path"
            ),
        )


        result = (
            unwrap_execution_result(
                outer_result
            )
        )


        if tools:

            result[
                "tools_used"
            ] = tools


        if (
            result.get(
                "confidence"
            )
            is None
        ):

            result[
                "confidence"
            ] = plan.get(
                "confidence",
                0.0,
            )

            result[
                "confidence_type"
            ] = (
                "routing_confidence"
            )


        roi_inputs = {
            key: value
            for key, value
            in {
                "image_path":
                    roi.get(
                        "image_path"
                    ),

                "before_path":
                    roi.get(
                        "before_path"
                    ),

                "after_path":
                    roi.get(
                        "after_path"
                    ),

                "optical_path":
                    roi.get(
                        "optical_path"
                    ),

                "sar_path":
                    roi.get(
                        "sar_path"
                    ),
            }.items()
            if value
        }


        standardized = (
            adapter.standardize(
                result,
                query=query,
                intent=intent,
                tools=tools,
                inputs=roi_inputs,
                validation=validation,
                default_confidence_type=(
                    "relative_tile_relevance"
                    if intent
                    == "text_guided_grounding"
                    else
                    "model_score"
                ),
            )
        )


        for key in [
            "answer",
            "model",
            "device",
            "evidence",
            "limitations",
        ]:

            if (
                result.get(
                    key
                )
                is not None
            ):

                standardized[
                    key
                ] = result[
                    key
                ]


        standardized[
            "selected_region"
        ] = {
            "west":
                bounds[0],

            "south":
                bounds[1],

            "east":
                bounds[2],

            "north":
                bounds[3],

            "source_analysis_id":
                analysis_id,
        }


        roi_report_generator = SatQueryReportGenerator(
            output_dir=str(REPORT_DIR / f"roi_{uuid.uuid4().hex}")
        )
        report = (
            roi_report_generator.generate(
                standardized,
                query=query,
            )
        )


        reports = report_links(
            report
        )


        confidence_details = (
            standardized.get(
                "confidence_details"
            )
            or {}
        )


        percentage = (
            confidence_details.get(
                "percentage"
            )
        )


        if percentage is None:

            raw = standardized.get(
                "confidence",
                0.0,
            )

            try:
                raw = float(
                    raw
                )

            except Exception:
                raw = 0.0

            percentage = (
                raw * 100.0
                if raw <= 1.0
                else raw
            )


        geospatial = (
            build_geospatial_payload(
                validation,
                standardized,
                intent,
            )
        )


        response = {

            "success":
                bool(
                    standardized.get(
                        "success",
                        True,
                    )
                ),

            "query":
                query,

            "intent":
                intent,

            "task":
                intent
                .replace(
                    "_",
                    " ",
                )
                .title(),

            "answer":
                (
                    standardized.get(
                        "answer"
                    )
                    or standardized.get(
                        "description"
                    )
                    or standardized.get(
                        "message"
                    )
                    or
                    "Analysis completed."
                ),

            "confidence":
                round(
                    float(
                        percentage
                        or 0.0
                    ),
                    2,
                ),

            "confidence_details":
                json_safe(
                    confidence_details
                ),

            "model":
                (
                    standardized.get(
                        "model"
                    )
                    or standardized.get(
                        "model_name"
                    )
                    or
                    "SatQuery AI Pipeline"
                ),

            "device":
                standardized.get(
                    "device"
                )
                or "cpu",

            "tools":
                tools,

            "validation":
                json_safe(
                    validation
                ),

            "evidence":
                evidence_items(
                    standardized
                ),

            "geospatial":
                geospatial,

            "selected_region":
                {
                    "west":
                        bounds[0],

                    "south":
                        bounds[1],

                    "east":
                        bounds[2],

                    "north":
                        bounds[3],
                },

            "reports":
                reports,

            "result":
                json_safe(
                    standardized
                ),
        }


        if response["success"]:
            response["analysis_id"] = register_analysis(
                query=query,
                intent=intent,
                inputs=roi["source_inputs"],
                validation=validation,
                standardized=standardized,
                response=response,
            )

        return jsonify(
            response
        )


    except (ValueError, TypeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception as exc:

        traceback.print_exc()

        return jsonify(
            {
                "success":
                    False,

                "error":
                    "analysis_execution_failed",

                "answer": "The analysis could not be completed. Please verify the input imagery and try again.",
            }
        ), 500


def _extract_change_percentage(
    result: Dict[str, Any],
    standardized: Dict[str, Any],
) -> Optional[float]:
    """Best-effort extraction of a ChangeFormer changed-area percentage
    from whatever shape the orchestrator/adapter happens to return.
    Different pipeline versions have used different key names, so this
    checks the common ones and gives up cleanly (returns None) rather
    than guessing -- the narrative simply omits the structural
    percentage line when this comes back empty."""

    candidate_keys = [
        "change_percentage",
        "changed_area_percentage",
        "change_area_pct",
        "changed_pixels_percentage",
        "change_ratio",
    ]

    def _search(source: Any) -> Optional[float]:

        if not isinstance(source, dict):
            return None

        for key in candidate_keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return float(value) * 100.0 if value <= 1.0 else float(value)

        for nested_key in ["statistics", "metrics", "change_stats", "summary"]:
            nested = source.get(nested_key)
            found = _search(nested)
            if found is not None:
                return found

        return None

    return _search(result) or _search(standardized)


@app.route(
    "/api/map-temporal-analyze",
    methods=["POST"],
)
def map_temporal_analyze():
    """Historical, no-upload-required change analysis.

    The user draws any area on the Map View base map (no GeoTIFF
    analysis needs to be loaded first) and asks a question like
    'what changed between 2010 and 2026 in this area?'. This route:

      1. Parses the two years out of the question (or uses explicit
         before_year/after_year from the request body).
      2. Pulls cloud-filtered Sentinel-2 / Landsat composites for that
         bounding box in each year straight from Google Earth Engine.
      3. Feeds those two images into the existing bi-temporal
         change-detection pipeline (ChangeFormer via execute_query),
         exactly like an uploaded Before/After pair would be.
      4. Layers a plain-language NDVI/NDBI/water narrative on top of
         the model's answer so the response reads like a real
         explanation of what happened in that spot, not just a score.
    """

    try:

        payload = (
            request.get_json(silent=True)
            or {}
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "Send a JSON object with bounds and query."
            )

        query = str(
            payload.get("query") or ""
        ).strip()

        bounds = payload.get("bounds")

        raw_before_year = payload.get("before_year")
        raw_after_year = payload.get("after_year")

        if not query:
            return jsonify(
                {
                    "success": False,
                    "error": "Enter a question about the selected area.",
                }
            ), 400

        bounds = _validate_wgs84_bounds(bounds)

        if not gee_temporal.is_configured():
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Historical imagery is not configured on this server yet. "
                        "Set GEE_SERVICE_ACCOUNT_EMAIL and GEE_SERVICE_ACCOUNT_KEY "
                        "(and install earthengine-api) to enable Google Earth "
                        "Engine-backed historical comparisons."
                    ),
                }
            ), 503

        current_year = datetime.utcnow().year

        try:
            before_year = int(raw_before_year)
            after_year = int(raw_after_year)
        except (TypeError, ValueError):
            before_year, after_year = gee_temporal.extract_years_from_query(
                query, current_year
            )

        if before_year > after_year:
            before_year, after_year = after_year, before_year

        if before_year == after_year:
            after_year = min(current_year, before_year + 1)

        gee_workdir = UPLOAD_DIR / "gee_temporal"

        try:
            temporal_plan = gee_temporal.plan_temporal_fetch(
                bounds, before_year, after_year
            )

            before_composite = gee_temporal.fetch_year_composite(
                bounds,
                before_year,
                gee_workdir,
                spec=temporal_plan.before_spec,
                dimensions=(temporal_plan.width_px, temporal_plan.height_px),
            )
            after_composite = gee_temporal.fetch_year_composite(
                bounds,
                after_year,
                gee_workdir,
                spec=temporal_plan.after_spec,
                dimensions=(temporal_plan.width_px, temporal_plan.height_px),
            )

        except gee_temporal.GeeNotConfiguredError as exc:
            return jsonify(
                {"success": False, "error": str(exc)}
            ), 503

        except gee_temporal.RoiTooLargeError as exc:
            return jsonify(
                {"success": False, "error": str(exc)}
            ), 413

        except gee_temporal.NoImageryError as exc:
            return jsonify(
                {"success": False, "error": str(exc)}
            ), 422

        before_path = str(before_composite.path)
        after_path = str(after_composite.path)

        validation = validator.validate_change_pair(
            before_path, after_path
        )

        if (
            validation is not None
            and not validation.get("valid", False)
        ):

            errors = validation.get("errors") or []
            messages = []

            for error in errors:

                if isinstance(error, dict):
                    message = error.get("message") or error.get("code")
                else:
                    message = str(error)

                if message and message not in messages:
                    messages.append(message)

            return jsonify(
                {
                    "success": False,
                    "intent": "change_detection",
                    "query": query,
                    "error": (
                        " ".join(messages)
                        or "The fetched historical imagery failed validation."
                    ),
                    "validation": json_safe(validation),
                }
            ), 400

        plan = plan_query(query)
        tools = plan_tools(plan)

        outer_result = execute_query(
            query=query,
            intent="change_detection",
            image_path=None,
            before_path=before_path,
            after_path=after_path,
            optical_path=None,
            sar_path=None,
        )

        result = unwrap_execution_result(outer_result)

        if tools:
            result["tools_used"] = tools

        standardized = adapter.standardize(
            result,
            query=query,
            intent="change_detection",
            tools=tools,
            inputs={"before_path": before_path, "after_path": after_path},
            validation=validation,
            default_confidence_type="model_score",
        )

        for key in ["answer", "model", "device", "evidence", "limitations"]:
            if result.get(key) is not None:
                standardized[key] = result[key]

        model_answer = (
            standardized.get("answer")
            or standardized.get("description")
            or standardized.get("message")
        )

        changeformer_change_pct = _extract_change_percentage(result, standardized)

        narrative = gee_temporal.build_change_narrative(
            before_composite,
            after_composite,
            model_answer,
            changeformer_change_pct=changeformer_change_pct,
            cross_sensor=temporal_plan.cross_sensor,
        )

        standardized["answer"] = narrative

        standardized["selected_region"] = {
            "west": bounds[0],
            "south": bounds[1],
            "east": bounds[2],
            "north": bounds[3],
            "before_year": before_year,
            "after_year": after_year,
        }

        roi_report_generator = SatQueryReportGenerator(
            output_dir=str(REPORT_DIR / f"gee_{uuid.uuid4().hex}")
        )

        report = roi_report_generator.generate(
            standardized, query=query
        )

        reports = report_links(report)

        confidence_details = standardized.get("confidence_details") or {}
        percentage = confidence_details.get("percentage")

        if percentage is None:

            raw = standardized.get("confidence", 0.0)

            try:
                raw = float(raw)
            except Exception:
                raw = 0.0

            percentage = raw * 100.0 if raw <= 1.0 else raw

        geospatial = build_geospatial_payload(
            validation, standardized, "change_detection"
        )

        response = {
            "success": bool(standardized.get("success", True)),
            "query": query,
            "intent": "change_detection",
            "task": f"Historical Change \u00b7 {before_year} vs {after_year}",
            "answer": narrative,
            "confidence": round(float(percentage or 0.0), 2),
            "confidence_details": json_safe(confidence_details),
            "model": (
                standardized.get("model")
                or standardized.get("model_name")
                or "SatQuery AI Pipeline + Google Earth Engine"
            ),
            "device": standardized.get("device") or "cpu",
            "tools": tools,
            "validation": json_safe(validation),
            "evidence": evidence_items(standardized),
            "geospatial": geospatial,
            "selected_region": {
                "west": bounds[0],
                "south": bounds[1],
                "east": bounds[2],
                "north": bounds[3],
                "before_year": before_year,
                "after_year": after_year,
            },
            "reports": reports,
            "temporal": {
                "before_year": before_year,
                "after_year": after_year,
                "before_source": before_composite.collection_id,
                "after_source": after_composite.collection_id,
                "before_stats": before_composite.stats,
                "after_stats": after_composite.stats,
                "cross_sensor": temporal_plan.cross_sensor,
                "common_grid": {
                    "target_scale_m": temporal_plan.target_scale_m,
                    "width_px": temporal_plan.width_px,
                    "height_px": temporal_plan.height_px,
                },
                "changeformer_change_percentage": changeformer_change_pct,
            },
            "result": json_safe(standardized),
        }

        if response["success"]:
            response["analysis_id"] = register_analysis(
                query=query,
                intent="change_detection",
                inputs={"before_path": before_path, "after_path": after_path},
                validation=validation,
                standardized=standardized,
                response=response,
            )

        return jsonify(response)

    except (ValueError, TypeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception as exc:

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": "analysis_execution_failed",
                "answer": "The analysis could not be completed. Please verify the input imagery and try again.",
            }
        ), 500


@app.route('/api/runtime-status')
def runtime_status():
    return jsonify({'success': True, 'models': MODEL_RUNTIME_STATE})


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
