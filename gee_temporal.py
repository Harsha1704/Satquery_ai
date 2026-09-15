"""
gee_temporal.py
================

Historical, two-date ("before" vs "after") satellite imagery for an
arbitrary user-drawn bounding box, sourced live from Google Earth Engine.

This module is intentionally decoupled from the rest of the SatQuery AI
pipeline: it only knows how to turn `(bounds, year)` into a small,
georeferenced GeoTIFF plus a handful of summary statistics (NDVI, NDBI,
water fraction). Everything else -- running ChangeFormer, building the
map overlays, writing the PDF/HTML report -- is handled by the existing
`execute_query(intent="change_detection", ...)` path in app.py. This
module just gives that path real "before_path" / "after_path" files
instead of requiring the user to upload them.

SETUP (one-time, on your server):

1.  pip install earthengine-api requests

2.  Create a Google Cloud project and enable the "Earth Engine API".
    https://console.cloud.google.com/apis/library/earthengine.googleapis.com

3.  Create a service account in that project, grant it the
    "Earth Engine Resource Viewer" role (or register it directly at
    https://code.earthengine.google.com/register -> "service account"),
    and download its JSON key.

4.  Set these environment variables before starting Flask:

        GEE_SERVICE_ACCOUNT_EMAIL   = xxxx@your-project.iam.gserviceaccount.com
        GEE_SERVICE_ACCOUNT_KEY     = /absolute/path/to/service-account-key.json
        GEE_PROJECT_ID              = your-gcp-project-id

    (GEE_PROJECT_ID is required by newer earthengine-api versions even
    when using a service account.)

That's it -- no per-request auth, no OAuth popups. The module lazily
calls ee.Initialize() once per process.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import ee
except ImportError as exc:  # pragma: no cover
    ee = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ============================================================
# ERRORS
# ============================================================

class GeeNotConfiguredError(RuntimeError):
    """Raised when the earthengine-api package or credentials are missing."""


class NoImageryError(RuntimeError):
    """Raised when Earth Engine has no usable, cloud-free imagery for the
    requested area/year combination."""


# ============================================================
# INITIALIZATION
# ============================================================

_INITIALIZED = False


def is_configured() -> bool:
    """True if the earthengine-api package is importable and the
    required environment variables are present. Does not perform any
    network calls."""

    if ee is None:
        return False

    return bool(
        os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")
        and os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    )


def init_earth_engine() -> None:
    """Initialize the Earth Engine session once per process. Cheap to
    call repeatedly; only does work the first time."""

    global _INITIALIZED

    if _INITIALIZED:
        return

    if ee is None:
        raise GeeNotConfiguredError(
            "The 'earthengine-api' package is not installed on the server. "
            "Run: pip install earthengine-api"
        ) from _IMPORT_ERROR

    email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    project = os.environ.get("GEE_PROJECT_ID")

    if not email or not key_path:
        raise GeeNotConfiguredError(
            "Set GEE_SERVICE_ACCOUNT_EMAIL and GEE_SERVICE_ACCOUNT_KEY "
            "environment variables to enable historical imagery."
        )

    if not Path(key_path).is_file():
        raise GeeNotConfiguredError(
            f"GEE_SERVICE_ACCOUNT_KEY points to a file that does not exist: {key_path}"
        )

    credentials = ee.ServiceAccountCredentials(email, key_path)
    ee.Initialize(credentials, project=project)

    _INITIALIZED = True


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class YearComposite:
    year: int
    path: Path
    collection_id: str
    image_count: int
    scale_m: float
    stats: Dict[str, float] = field(default_factory=dict)


# ============================================================
# CLOUD MASKING
# ============================================================

def _mask_sentinel2(image):
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )
    return image.updateMask(mask).divide(10000)


def _mask_landsat_sr(image):
    qa = image.select("QA_PIXEL")
    dilated_cloud = 1 << 1
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    mask = (
        qa.bitwiseAnd(dilated_cloud).eq(0)
        .And(qa.bitwiseAnd(cloud).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow).eq(0))
    )
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return optical.updateMask(mask).copyProperties(image, ["system:time_start"])


# ============================================================
# ROI SIZE LIMITS
#
# Earth Engine's getDownloadURL() caps a single request at ~32MB and
# 10,000 px per side. More importantly, for an interactive hackathon
# demo we want fast, reliable requests -- not someone drawing half a
# state. Sentinel-2 (10 m) is capped tighter than Landsat (30 m)
# because the same area produces ~9x more pixels at 10 m.
# ============================================================

MAX_KM_FINE_RESOLUTION = 5.0    # cap when either year uses 10 m Sentinel-2
MAX_KM_COARSE_RESOLUTION = 10.0  # cap when both years use 30 m Landsat


class RoiTooLargeError(RuntimeError):
    """Raised when the drawn rectangle exceeds the size Earth Engine can
    reliably serve for the requested resolution."""


def bbox_size_km(bounds_wgs84: List[float]) -> Tuple[float, float]:
    """Approximate (width_km, height_km) of a WGS84 bounding box using
    an equirectangular approximation -- accurate to well under 1% at
    the neighborhood/city scale this feature targets."""

    import math

    west, south, east, north = bounds_wgs84
    lat_mid = math.radians((south + north) / 2.0)

    height_km = (north - south) * 111.32
    width_km = (east - west) * 111.32 * math.cos(lat_mid)

    return abs(width_km), abs(height_km)


def enforce_roi_limits(bounds_wgs84: List[float], target_scale_m: float) -> None:
    """Raise RoiTooLargeError if the selected rectangle is too big for
    the resolution we're about to fetch at."""

    width_km, height_km = bbox_size_km(bounds_wgs84)

    max_km = (
        MAX_KM_FINE_RESOLUTION
        if target_scale_m <= 10
        else MAX_KM_COARSE_RESOLUTION
    )

    if width_km > max_km or height_km > max_km:
        raise RoiTooLargeError(
            f"Selected area is about {width_km:.1f} km x {height_km:.1f} km, "
            f"which is too large for a {target_scale_m:.0f} m comparison "
            f"(limit is {max_km:.0f} km x {max_km:.0f} km). Draw a smaller "
            "rectangle, such as a single neighborhood or site."
        )


# ============================================================
# COLLECTION SELECTION
#
# Sentinel-2 Surface Reflectance only exists from 2017 onward in Earth
# Engine (COPERNICUS/S2_SR_HARMONIZED). Anything older must come from
# Landsat:
#
#   2017 - present  -> Sentinel-2 SR (10 m)
#   2013 - 2016     -> Landsat 8 (30 m)
#   1999 - 2012     -> Landsat 5, falling back to Landsat 7 (30 m)
#   1984 - 1998     -> Landsat 5 (30 m)
#
# Each spec also lists a `fallback_id` used automatically when the
# primary collection has zero clear scenes for that year/area (e.g. a
# gap in the Landsat 5/7 overlap window).
# ============================================================

def _sentinel2_spec() -> Dict[str, Any]:
    return {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "fallback_id": None,
        "mask_fn": _mask_sentinel2,
        "red": "B4", "nir": "B8", "swir": "B11", "green": "B3", "blue": "B2",
        "rgb": ["B4", "B3", "B2"],
        "scale": 10,
        "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
        "label": "Sentinel-2 SR",
    }


def _landsat8_spec() -> Dict[str, Any]:
    return {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "fallback_id": "LANDSAT/LC09/C02/T1_L2",
        "mask_fn": _mask_landsat_sr,
        "red": "SR_B4", "nir": "SR_B5", "swir": "SR_B6", "green": "SR_B3", "blue": "SR_B2",
        "rgb": ["SR_B4", "SR_B3", "SR_B2"],
        "scale": 30,
        "cloud_property": "CLOUD_COVER",
        "label": "Landsat 8",
    }


def _landsat5_spec() -> Dict[str, Any]:
    return {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "fallback_id": "LANDSAT/LE07/C02/T1_L2",
        "mask_fn": _mask_landsat_sr,
        "red": "SR_B3", "nir": "SR_B4", "swir": "SR_B5", "green": "SR_B2", "blue": "SR_B1",
        "rgb": ["SR_B3", "SR_B2", "SR_B1"],
        "scale": 30,
        "cloud_property": "CLOUD_COVER",
        "label": "Landsat 5",
    }


def _landsat7_spec() -> Dict[str, Any]:
    return {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "fallback_id": "LANDSAT/LT05/C02/T1_L2",
        "mask_fn": _mask_landsat_sr,
        "red": "SR_B3", "nir": "SR_B4", "swir": "SR_B5", "green": "SR_B2", "blue": "SR_B1",
        "rgb": ["SR_B3", "SR_B2", "SR_B1"],
        "scale": 30,
        "cloud_property": "CLOUD_COVER",
        "label": "Landsat 7",
    }


def _collection_for_year(year: int) -> Dict[str, Any]:
    """Pick the best available public archive for a given calendar
    year. This is the *primary* choice only -- `_resolve_collection`
    below falls back automatically if it turns out to be empty for
    the requested area."""

    if year >= 2017:
        return _sentinel2_spec()

    if year >= 2013:
        return _landsat8_spec()

    if year >= 1999:
        # Landsat 5 and 7 both cover this window. Landsat 5 has no
        # scan-line-corrector gaps, so prefer it and fall back to
        # Landsat 7 only if Landsat 5 has no clear scenes here.
        return _landsat5_spec()

    return _landsat5_spec()


# ============================================================
# COLLECTION RESOLUTION (WITH FALLBACK)
# ============================================================

def _build_filtered_collection(spec: Dict[str, Any], collection_id: str, region, year: int, max_cloud_pct: float):
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )

    if spec.get("cloud_property"):
        collection = collection.filter(
            ee.Filter.lt(spec["cloud_property"], max_cloud_pct)
        )

    return collection


def _resolve_collection(
    region,
    year: int,
    max_cloud_pct: float = 60.0,
) -> Tuple[Dict[str, Any], Any, int]:
    """Return (spec, filtered_collection, image_count) for a year,
    trying the primary sensor first and its documented fallback second
    if the primary has zero clear scenes over this exact area."""

    spec = _collection_for_year(year)

    collection = _build_filtered_collection(spec, spec["id"], region, year, max_cloud_pct)
    count = collection.size().getInfo()

    if count == 0:
        # Retry without the cloud filter -- some areas only get one or
        # two passes a year and a strict cloud threshold can zero them
        # out even though a usable (if imperfect) composite exists.
        collection = (
            ee.ImageCollection(spec["id"])
            .filterBounds(region)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
        )
        count = collection.size().getInfo()

    if count == 0 and spec.get("fallback_id"):
        fallback_spec = dict(spec)
        fallback_spec["id"] = spec["fallback_id"]
        fallback_spec["label"] = spec["fallback_id"].split("/")[1]

        collection = _build_filtered_collection(
            fallback_spec, fallback_spec["id"], region, year, max_cloud_pct
        )
        count = collection.size().getInfo()

        if count == 0:
            collection = (
                ee.ImageCollection(fallback_spec["id"])
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
            )
            count = collection.size().getInfo()

        if count > 0:
            spec = fallback_spec

    if count == 0:
        raise NoImageryError(
            f"No usable {spec['label']} scenes were found over this area "
            f"for {year} (including the automatic sensor fallback). Try a "
            "different year or a different area."
        )

    return spec, collection, int(count)


# ============================================================
# COMMON-GRID PLANNING
#
# Comparing a 2010 Landsat pixel to a 2026 Sentinel-2 pixel only makes
# sense if both are resampled onto the exact same geographic grid:
# same CRS, same pixel dimensions, same extent. We plan this once for
# the pair, then fetch both years onto that shared grid.
# ============================================================

@dataclass
class TemporalPlan:
    bounds: List[float]
    before_year: int
    after_year: int
    before_spec: Dict[str, Any]
    after_spec: Dict[str, Any]
    before_count: int
    after_count: int
    target_scale_m: float
    width_px: int
    height_px: int
    cross_sensor: bool


def plan_temporal_fetch(
    bounds_wgs84: List[float],
    before_year: int,
    after_year: int,
    max_cloud_pct: float = 60.0,
) -> TemporalPlan:
    """Resolve sensors for both years, pick a common resolution (the
    coarser of the two, so nothing is upsampled beyond its native
    detail), compute a shared pixel grid, and enforce the ROI size
    limit for that resolution -- all before any imagery is downloaded.
    """

    init_earth_engine()

    west, south, east, north = bounds_wgs84
    region = ee.Geometry.Rectangle(
        [west, south, east, north], proj="EPSG:4326", geodesic=False
    )

    before_spec, _before_collection, before_count = _resolve_collection(
        region, before_year, max_cloud_pct
    )
    after_spec, _after_collection, after_count = _resolve_collection(
        region, after_year, max_cloud_pct
    )

    target_scale = max(before_spec["scale"], after_spec["scale"])

    enforce_roi_limits(bounds_wgs84, target_scale)

    width_km, height_km = bbox_size_km(bounds_wgs84)

    width_px = max(32, min(10000, round((width_km * 1000.0) / target_scale)))
    height_px = max(32, min(10000, round((height_km * 1000.0) / target_scale)))

    return TemporalPlan(
        bounds=list(bounds_wgs84),
        before_year=before_year,
        after_year=after_year,
        before_spec=before_spec,
        after_spec=after_spec,
        before_count=before_count,
        after_count=after_count,
        target_scale_m=float(target_scale),
        width_px=int(width_px),
        height_px=int(height_px),
        cross_sensor=(before_spec["id"] != after_spec["id"]),
    )


# ============================================================
# FETCH ONE YEAR (ONTO A GIVEN OR SELF-RESOLVED GRID)
# ============================================================

def fetch_year_composite(
    bounds_wgs84: List[float],
    year: int,
    workdir: Path,
    max_cloud_pct: float = 60.0,
    spec: Optional[Dict[str, Any]] = None,
    dimensions: Optional[Tuple[int, int]] = None,
) -> YearComposite:
    """Build a cloud-filtered median composite for `year` over `bounds`,
    download it as an RGB GeoTIFF, and return summary statistics
    (NDVI / NDBI / water fraction) for the same area.

    If `spec` and `dimensions` are supplied (normally from
    `plan_temporal_fetch`), the image is resampled onto that exact
    pixel grid so it lines up with its paired before/after year even
    when the two years come from different sensors. Without them, the
    function resolves its own sensor/resolution independently (used
    for single-year previews).

    Raises NoImageryError if nothing usable is found for that year.
    """

    init_earth_engine()

    west, south, east, north = bounds_wgs84
    region = ee.Geometry.Rectangle(
        [west, south, east, north], proj="EPSG:4326", geodesic=False
    )

    if spec is None:
        spec, collection, image_count = _resolve_collection(region, year, max_cloud_pct)
    else:
        collection = _build_filtered_collection(spec, spec["id"], region, year, max_cloud_pct)
        image_count = collection.size().getInfo()
        if image_count == 0:
            collection = (
                ee.ImageCollection(spec["id"])
                .filterBounds(region)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
            )
            image_count = collection.size().getInfo()
        if image_count == 0:
            raise NoImageryError(
                f"No usable {spec.get('label', spec['id'])} scenes were found "
                f"over this area for {year}."
            )

    masked = collection.map(spec["mask_fn"])
    composite = masked.median().clip(region)

    ndvi = composite.normalizedDifference([spec["nir"], spec["red"]]).rename("NDVI")
    ndbi = composite.normalizedDifference([spec["swir"], spec["nir"]]).rename("NDBI")
    ndwi = composite.normalizedDifference([spec["green"], spec["nir"]]).rename("NDWI")

    stats_image = ndvi.addBands(ndbi).addBands(ndwi)

    # Stats are always computed at the sensor's own native resolution
    # (sharper than the shared RGB grid) so NDVI/NDBI numbers reflect
    # the real sensor, not a resampling artifact.
    native_scale = spec["scale"]

    raw_stats = stats_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=native_scale,
        maxPixels=1_000_000_000,
        bestEffort=True,
    ).getInfo()

    water_fraction = (
        ndwi.gt(0.0)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=native_scale,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .getInfo()
        .get("NDWI")
    )

    rgb_vis = (
        composite.select(spec["rgb"])
        .clamp(0, 0.3)
        .divide(0.3)
        .multiply(255)
        .uint8()
        .rename(["R", "G", "B"])
    )

    download_params: Dict[str, Any] = {
        "region": region,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    }

    if dimensions:
        # Shared grid mode: force both years onto identical
        # width x height so ChangeFormer compares like-for-like pixels
        # regardless of which sensor produced them.
        width_px, height_px = dimensions
        download_params["dimensions"] = f"{int(width_px)}x{int(height_px)}"
    else:
        download_params["scale"] = native_scale

    url = rgb_vis.getDownloadURL(download_params)

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"gee_{year}_{uuid.uuid4().hex[:8]}.tif"
    out_path.write_bytes(response.content)

    return YearComposite(
        year=year,
        path=out_path,
        collection_id=spec.get("label", spec["id"]),
        image_count=int(image_count),
        scale_m=float(native_scale),
        stats={
            "ndvi": raw_stats.get("NDVI"),
            "ndbi": raw_stats.get("NDBI"),
            "water_fraction": water_fraction,
        },
    )


# ============================================================
# NARRATIVE
# ============================================================

def _direction(delta: float, threshold: float = 0.04) -> str:
    if delta is None:
        return "is unknown"
    if delta > threshold:
        return "increased"
    if delta < -threshold:
        return "decreased"
    return "stayed roughly stable"


def build_change_narrative(
    before: YearComposite,
    after: YearComposite,
    model_answer: Optional[str] = None,
    changeformer_change_pct: Optional[float] = None,
    cross_sensor: bool = False,
) -> str:
    """Build a three-part, judge-defensible explanation:

    1. Structural change (ChangeFormer) -- labeled explicitly as visual
       change evidence, with a disclaimer that the model was trained
       on LEVIR-CD building imagery, so applying it to Landsat/
       Sentinel composites carries some domain shift.
    2. Spectral trends (NDVI / NDBI / water) -- the numbers that
       actually justify calling something "deforestation" or "urban
       expansion".
    3. Combined interpretation -- what the two signals together most
       likely mean on the ground.

    This intentionally avoids implying ChangeFormer itself understands
    land-cover change; it is presented as one input among several.
    """

    sections: List[str] = []

    # --- 1. Structural change (ChangeFormer) ------------------------
    if changeformer_change_pct is not None:
        sections.append(
            f"Structural change (ChangeFormer): visual change evidence "
            f"was detected across approximately {changeformer_change_pct:.1f}% "
            f"of the selected region between {before.year} and {after.year}. "
            "Note: this model was trained on building-scale imagery "
            "(LEVIR-CD) rather than historical Landsat/Sentinel land "
            "cover, so treat this as supporting structural evidence "
            "rather than a definitive land-cover verdict."
        )
    elif model_answer:
        sections.append(
            f"Structural change (ChangeFormer): {model_answer.strip()} "
            "Note: this model was trained on building-scale imagery "
            "(LEVIR-CD), so results on historical satellite composites "
            "should be read as supporting evidence, not ground truth."
        )

    # --- 2. Spectral trends ------------------------------------------
    ndvi_before = before.stats.get("ndvi")
    ndvi_after = after.stats.get("ndvi")
    ndbi_before = before.stats.get("ndbi")
    ndbi_after = after.stats.get("ndbi")
    water_before = before.stats.get("water_fraction")
    water_after = after.stats.get("water_fraction")

    ndvi_delta = (
        ndvi_after - ndvi_before
        if ndvi_after is not None and ndvi_before is not None
        else None
    )
    ndbi_delta = (
        ndbi_after - ndbi_before
        if ndbi_after is not None and ndbi_before is not None
        else None
    )
    water_delta = (
        water_after - water_before
        if water_after is not None and water_before is not None
        else None
    )

    spectral_lines: List[str] = []

    if ndvi_delta is not None:
        spectral_lines.append(
            f"vegetation (NDVI) {_direction(ndvi_delta)} from "
            f"{ndvi_before:.2f} in {before.year} to {ndvi_after:.2f} in "
            f"{after.year} ({ndvi_delta:+.2f})"
        )

    if ndbi_delta is not None:
        spectral_lines.append(
            f"built-up signal (NDBI) {_direction(ndbi_delta)} from "
            f"{ndbi_before:.2f} to {ndbi_after:.2f} ({ndbi_delta:+.2f})"
        )

    if water_delta is not None and abs(water_delta) > 0.03:
        direction = "increased" if water_delta > 0 else "decreased"
        spectral_lines.append(
            f"open-water coverage {direction} from {water_before * 100:.0f}% "
            f"to {water_after * 100:.0f}%"
        )

    if spectral_lines:
        sensor_note = ""
        if cross_sensor:
            sensor_note = (
                f" (both years resampled to a common {before.scale_m:.0f}-{after.scale_m:.0f} m "
                "grid for a fair pixel-for-pixel comparison, since they come "
                "from different satellites)"
            )
        sections.append(
            "Spectral trends: " + "; ".join(spectral_lines) + sensor_note + "."
        )

    # --- 3. Combined interpretation -----------------------------------
    if ndvi_delta is not None and ndbi_delta is not None:
        if ndvi_delta < -0.05 and ndbi_delta > 0.05:
            interpretation = (
                "Interpretation: decreased vegetation together with an "
                "increased built-up signal is consistent with urban "
                "expansion, new construction, or land clearing for "
                "development in this area."
            )
        elif ndvi_delta < -0.05 and ndbi_delta <= 0.05:
            interpretation = (
                "Interpretation: vegetation loss without a matching rise "
                "in the built-up signal suggests deforestation, drought "
                "stress, or a change in agricultural land use rather than "
                "construction."
            )
        elif ndvi_delta > 0.05:
            interpretation = (
                "Interpretation: this pattern is consistent with "
                "vegetation regrowth, reforestation, or expanded/greener "
                "agriculture in this area."
            )
        else:
            interpretation = (
                "Interpretation: no strong land-cover conversion signal "
                "was detected between the two dates; the area looks "
                "largely unchanged at the spectral level."
            )

        if (
            changeformer_change_pct is not None
            and changeformer_change_pct > 15
            and "unchanged" not in interpretation
        ):
            interpretation += (
                " ChangeFormer's structural evidence supports this "
                "reading with a substantial visually-changed area."
            )

        sections.append(interpretation)

    if not sections:
        return "No significant change could be quantified for this area and time span."

    return " ".join(sections)


# ============================================================
# YEAR PARSING
# ============================================================

import re

_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def extract_years_from_query(query: str, current_year: Optional[int] = None) -> Tuple[int, int]:
    """Pull a (before_year, after_year) pair out of free text like
    'what changed between 2010 and 2026' or 'compare 2015 to now'.
    Falls back to a sensible 10-year default window if it can't find
    two explicit years."""

    current_year = current_year or time.gmtime().tm_year

    years = sorted({int(y) for y in _YEAR_RE.findall(query)})

    if len(years) >= 2:
        return years[0], years[-1]

    if len(years) == 1:
        only = years[0]
        if only >= current_year:
            return max(1985, only - 10), only
        return only, current_year

    return max(1985, current_year - 10), current_year

# === SATQUERY HISTORICAL CHANGE ASSESSMENT V2 ===

def _satquery_metric_value(stats, key):
    try:
        value = stats.get(key)
        if value is None:
            return None
        value = float(value)
        if value != value:
            return None
        return value
    except Exception:
        return None


def _satquery_metric_level(delta, moderate, strong):
    magnitude = abs(float(delta))
    if magnitude >= strong:
        return "strong"
    if magnitude >= moderate:
        return "moderate"
    return "small"


def assess_historical_change(
    before,
    after,
    model_answer=None,
    changeformer_change_pct=None,
    cross_sensor=False,
):
    before_stats = getattr(before, "stats", {}) or {}
    after_stats = getattr(after, "stats", {}) or {}

    before_year = getattr(before, "year", None)
    after_year = getattr(after, "year", None)

    b_ndvi = _satquery_metric_value(before_stats, "ndvi")
    a_ndvi = _satquery_metric_value(after_stats, "ndvi")
    b_ndbi = _satquery_metric_value(before_stats, "ndbi")
    a_ndbi = _satquery_metric_value(after_stats, "ndbi")
    b_water = _satquery_metric_value(before_stats, "water_fraction")
    a_water = _satquery_metric_value(after_stats, "water_fraction")

    d_ndvi = (
        a_ndvi - b_ndvi
        if b_ndvi is not None and a_ndvi is not None
        else None
    )
    d_ndbi = (
        a_ndbi - b_ndbi
        if b_ndbi is not None and a_ndbi is not None
        else None
    )
    d_water = (
        a_water - b_water
        if b_water is not None and a_water is not None
        else None
    )

    signals = []

    if d_ndvi is not None:
        signals.append(
            {
                "metric": "ndvi",
                "delta": d_ndvi,
                "level": _satquery_metric_level(
                    d_ndvi,
                    moderate=0.04,
                    strong=0.08,
                ),
            }
        )

    if d_ndbi is not None:
        signals.append(
            {
                "metric": "ndbi",
                "delta": d_ndbi,
                "level": _satquery_metric_level(
                    d_ndbi,
                    moderate=0.025,
                    strong=0.05,
                ),
            }
        )

    if d_water is not None:
        signals.append(
            {
                "metric": "water_fraction",
                "delta": d_water,
                "level": _satquery_metric_level(
                    d_water,
                    moderate=0.02,
                    strong=0.05,
                ),
            }
        )

    strong_count = sum(
        1 for item in signals
        if item["level"] == "strong"
    )
    moderate_count = sum(
        1 for item in signals
        if item["level"] == "moderate"
    )

    if strong_count >= 2:
        spectral_level = "significant"
    elif strong_count >= 1 and moderate_count >= 1:
        spectral_level = "significant"
    elif strong_count >= 1:
        spectral_level = "meaningful"
    elif moderate_count >= 2:
        spectral_level = "meaningful"
    elif moderate_count >= 1:
        spectral_level = "limited"
    else:
        spectral_level = "stable"

    cf_pct = None
    try:
        if changeformer_change_pct is not None:
            cf_pct = max(
                0.0,
                float(changeformer_change_pct),
            )
    except Exception:
        cf_pct = None

    if cf_pct is None:
        structural_level = "unavailable"
    elif cf_pct < 0.01:
        structural_level = "very_low"
    elif cf_pct < 1.0:
        structural_level = "minor"
    elif cf_pct < 10.0:
        structural_level = "localized"
    elif cf_pct < 30.0:
        structural_level = "significant"
    else:
        structural_level = "major"

    if spectral_level == "significant":
        opening = (
            f"Historical comparison indicates significant spectral "
            f"change between {before_year} and {after_year}."
        )
    elif spectral_level == "meaningful":
        opening = (
            f"Historical comparison indicates meaningful spectral "
            f"change between {before_year} and {after_year}."
        )
    elif spectral_level == "limited":
        opening = (
            f"Historical comparison indicates limited spectral "
            f"change between {before_year} and {after_year}."
        )
    else:
        opening = (
            f"No strong spectral land-cover change signal was detected "
            f"between {before_year} and {after_year}."
        )

    details = []

    if d_ndvi is not None:
        if d_ndvi <= -0.04:
            ndvi_trend = "decreased"
        elif d_ndvi >= 0.04:
            ndvi_trend = "increased"
        else:
            ndvi_trend = "remained broadly stable"

        details.append(
            "Vegetation signal (NDVI) "
            f"{ndvi_trend} from {b_ndvi:.3f} to {a_ndvi:.3f} "
            f"({d_ndvi:+.3f})."
        )

    if d_ndbi is not None:
        if d_ndbi >= 0.025:
            ndbi_trend = "increased"
        elif d_ndbi <= -0.025:
            ndbi_trend = "decreased"
        else:
            ndbi_trend = "remained broadly stable"

        details.append(
            "Built-up spectral signal (NDBI) "
            f"{ndbi_trend} from {b_ndbi:.3f} to {a_ndbi:.3f} "
            f"({d_ndbi:+.3f})."
        )

    if d_water is not None:
        before_water_pct = b_water * 100.0
        after_water_pct = a_water * 100.0
        delta_water_pp = d_water * 100.0

        if delta_water_pp >= 2.0:
            water_trend = "increased"
        elif delta_water_pp <= -2.0:
            water_trend = "decreased"
        else:
            water_trend = "remained broadly stable"

        details.append(
            "Estimated water-covered fraction "
            f"{water_trend} from {before_water_pct:.2f}% "
            f"to {after_water_pct:.2f}% "
            f"({delta_water_pp:+.2f} percentage points)."
        )

    interpretation = []

    if (
        d_ndvi is not None
        and d_ndvi <= -0.04
        and d_ndbi is not None
        and d_ndbi >= 0.025
    ):
        interpretation.append(
            "The combined NDVI decrease and NDBI increase are "
            "consistent with increased surface development or loss "
            "of vegetated cover, but they do not by themselves prove "
            "a specific land-cover conversion."
        )

    if d_water is not None and abs(d_water) >= 0.02:
        interpretation.append(
            "The water metric also changed materially; this can reflect "
            "real shoreline/water-cover change as well as seasonal or "
            "compositing effects, so it should be interpreted with the "
            "before/after imagery."
        )

    if cf_pct is not None:
        interpretation.append(
            "ChangeFormerV6 detected approximately "
            f"{cf_pct:.2f}% LEVIR-style structural change. "
            "Because this checkpoint was trained on high-resolution "
            "LEVIR-CD building imagery rather than Sentinel/Landsat "
            "historical composites, this value is supporting structural "
            "evidence and does not override the spectral assessment."
        )
    elif model_answer:
        lower_answer = str(model_answer).lower()
        if (
            "failed" in lower_answer
            or "validation" in lower_answer
            or "error" in lower_answer
        ):
            interpretation.append(
                "ChangeFormer structural evidence was unavailable for "
                "this pair, so the historical conclusion is based on "
                "the Earth Engine spectral trends."
            )

    if cross_sensor:
        interpretation.append(
            "The dates use different sensor families, so both images "
            "were harmonized to a common comparison grid. Cross-sensor "
            "radiometric differences remain a source of uncertainty."
        )

    answer_parts = [opening]
    answer_parts.extend(details)
    answer_parts.extend(interpretation)

    return {
        "level": spectral_level,
        "spectral_change_detected": spectral_level
        in {"limited", "meaningful", "significant"},
        "structural_level": structural_level,
        "changeformer_changed_percentage": cf_pct,
        "metrics": {
            "ndvi": {
                "before": b_ndvi,
                "after": a_ndvi,
                "delta": d_ndvi,
            },
            "ndbi": {
                "before": b_ndbi,
                "after": a_ndbi,
                "delta": d_ndbi,
            },
            "water_fraction": {
                "before": b_water,
                "after": a_water,
                "delta": d_water,
            },
        },
        "answer": " ".join(
            part.strip()
            for part in answer_parts
            if part and str(part).strip()
        ),
        "method_note": (
            "Historical conclusion prioritizes Earth Engine spectral "
            "trends for medium-resolution Sentinel/Landsat composites. "
            "ChangeFormerV6 is reported separately as supporting "
            "building/structural evidence."
        ),
    }


def build_change_narrative(
    before,
    after,
    model_answer=None,
    changeformer_change_pct=None,
    cross_sensor=False,
):
    assessment = assess_historical_change(
        before=before,
        after=after,
        model_answer=model_answer,
        changeformer_change_pct=changeformer_change_pct,
        cross_sensor=cross_sensor,
    )
    return assessment["answer"]

# === END SATQUERY HISTORICAL CHANGE ASSESSMENT V2 ===

