"""Geometric and identity provenance for map-linked analyses."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def _raster_metadata(path: Path) -> dict[str, Any]:
    try:
        from query_engine.runtime import configure_raster_runtime
        configure_raster_runtime()
        import rasterio
        with rasterio.open(path) as dataset:
            return {
                "crs": str(dataset.crs) if dataset.crs else None,
                "bounds": [float(x) for x in dataset.bounds], "transform": [float(x) for x in dataset.transform],
                "resolution": [float(x) for x in dataset.res], "width": dataset.width, "height": dataset.height,
                "band_count": dataset.count, "bands": [name or f"band_{i}" for i, name in enumerate(dataset.descriptions, 1)],
                "nodata": dataset.nodata, "dtype": dataset.dtypes[0] if dataset.dtypes else None,
            }
    except Exception:
        return {}


def _geometry_checks(aoi_payload: dict | None, source: dict) -> tuple[dict, dict | None]:
    """Intersect the WGS84 AOI with a source bound polygon in source CRS."""
    if not aoi_payload or not source.get("crs") or not source.get("bounds"):
        return {"status": "WARNING", "reason": "AOI or source georeferencing is unavailable."}, None
    try:
        from pyproj import Transformer
        from shapely.geometry import box, mapping, shape
        from shapely.ops import transform
        aoi_wgs84 = shape(aoi_payload)
        source_crs = source["crs"]
        changed_crs = source_crs != "EPSG:4326"
        aoi_source = transform(Transformer.from_crs("EPSG:4326", source_crs, always_xy=True).transform, aoi_wgs84) if changed_crs else aoi_wgs84
        source_geometry = box(*source["bounds"])
        intersection = aoi_source.intersection(source_geometry)
        requested_area = float(aoi_source.area)
        intersection_area = float(intersection.area)
        coverage = round(intersection_area * 100.0 / requested_area, 4) if requested_area > 0 else 0.0
        if intersection.is_empty or intersection_area <= 0:
            return {"status": "FAIL", "code": "AOI_OUTSIDE_SOURCE", "intersects": False, "requested_aoi_coverage_pct": 0.0,
                    "source_crs": source_crs, "transformation_performed": changed_crs}, None
        effective_wgs84 = transform(Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform, intersection) if changed_crs else intersection
        check = {
            "status": "PASS" if coverage >= 99.999 else "WARNING", "intersects": True,
            "requested_aoi_coverage_pct": coverage, "requested_area_source_units": requested_area,
            "intersection_area_source_units": intersection_area, "source_crs": source_crs,
            "original_aoi_crs": "EPSG:4326", "transformation_performed": changed_crs,
            "effective_aoi": mapping(effective_wgs84), "effective_bounds": [float(x) for x in intersection.bounds],
        }
        return check, {"geometry": mapping(intersection), "bounds": [float(x) for x in intersection.bounds], "crs": source_crs}
    except Exception as exc:
        return {"status": "FAIL", "code": "SPATIAL_TRANSFORM_FAILED", "reason": str(exc)}, None


def _roi_metadata(source: dict, spatial: dict | None, *, analysis_id: str, requested_aoi: dict | None) -> dict:
    roi = {"roi_id": f"roi_{analysis_id[4:]}", "analysis_id": analysis_id, "parent_source_id": source.get("source_id"),
           "requested_aoi": requested_aoi, "effective_aoi": spatial.get("geometry") if spatial else None,
           "crs": spatial.get("crs") if spatial else source.get("crs"), "bounds": spatial.get("bounds") if spatial else source.get("bounds"),
           "transform": source.get("transform"), "resolution": source.get("resolution"), "nodata": source.get("nodata")}
    if spatial and source.get("transform") and source.get("width") and source.get("height"):
        try:
            import rasterio.windows
            from affine import Affine
            window = rasterio.windows.from_bounds(*spatial["bounds"], transform=Affine(*source["transform"]))
            window = window.round_offsets().round_lengths()
            roi.update({"width": max(0, int(window.width)), "height": max(0, int(window.height)),
                        "transform": [float(x) for x in rasterio.windows.transform(window, Affine(*source["transform"]))]})
        except Exception:
            roi.update({"width": source.get("width"), "height": source.get("height")})
    else:
        roi.update({"width": source.get("width"), "height": source.get("height")})
    return roi


def build_analysis_context(*, analysis_id: str, query: str, aoi, inputs: Mapping[str, Path], imagery: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the sole analysis context without exposing server file paths."""
    aoi_payload = aoi.model_dump(mode="json") if aoi is not None else None
    aoi_hash = hashlib.sha256(repr(aoi_payload).encode("utf-8")).hexdigest()[:24] if aoi_payload else None
    sources = []
    for role, path in inputs.items():
        file_hash = _sha256(path)
        sources.append({"source_id": f"src_{file_hash[:24]}", "input_hash": file_hash, "role": role, "kind": "local_file", **_raster_metadata(path)})
    for item in imagery or []:
        identity = "|".join(str(item.get(key, "")) for key in ("collection_id", "year", "scale_m", "product_id"))
        sources.append({"source_id": f"src_{hashlib.sha256(identity.encode()).hexdigest()[:24]}", "kind": "earth_engine_composite",
                        "product_id": item.get("product_id"), "collection_id": item.get("collection_id"),
                        "acquisition_year": item.get("year"), "resolution_m": item.get("scale_m"), "modality": item.get("modality", "multispectral")})
    aoi_source_checks, roi = [], None
    for source in sources:
        check, spatial = _geometry_checks(aoi_payload, source)
        aoi_source_checks.append({"source_id": source["source_id"], **check})
        if roi is None and spatial is not None:
            roi = _roi_metadata(source, spatial, analysis_id=analysis_id, requested_aoi=aoi_payload)
            effective_wgs84 = check.get("effective_aoi")
            if effective_wgs84:
                try:
                    from shapely.geometry import shape
                    roi["effective_aoi_wgs84"] = effective_wgs84
                    roi["display_bounds_wgs84"] = [float(x) for x in shape(effective_wgs84).bounds]
                except Exception:
                    pass
    if roi is None:
        roi = {"roi_id": f"roi_{analysis_id[4:]}", "analysis_id": analysis_id, "requested_aoi": aoi_payload,
               "effective_aoi": aoi_payload, "effective_aoi_wgs84": aoi_payload,
               "display_bounds_wgs84": aoi.bounds if aoi is not None else None,
               "crs": "EPSG:4326" if aoi_payload else None}
    result = {"result_id": f"res_{analysis_id[4:] if analysis_id.startswith('ana_') else analysis_id}", "analysis_id": analysis_id,
              "roi_id": roi["roi_id"], "source_ids": [item["source_id"] for item in sources], "crs": roi.get("crs"),
              "bounds": roi.get("bounds"), "transform": roi.get("transform"), "width": roi.get("width"), "height": roi.get("height"),
              "resolution": roi.get("resolution")}
    return {"analysis_id": analysis_id, "query": query, "aoi_id": f"aoi_{aoi_hash}" if aoi_hash else None,
            "aoi": aoi_payload, "aoi_crs": "EPSG:4326" if aoi_payload else None, "sources": sources, "roi": roi,
            "result": result, "result_id": result["result_id"], "aoi_source_checks": aoi_source_checks,
            "processing_history": [], "created_at": datetime.now(timezone.utc).isoformat()}


def evidence_records(*, analysis_id: str, context: dict[str, Any], artifacts: list[str], aoi=None, job_dir: Path | None = None) -> list[dict[str, Any]]:
    records = []
    result = context["result"]
    for artifact in artifacts:
        path = job_dir / artifact if job_dir else None
        metadata = _raster_metadata(path) if path and path.suffix.lower() in {".tif", ".tiff"} else {}
        digest = hashlib.sha256(f"{analysis_id}|{artifact}".encode()).hexdigest()[:24]
        record = {"evidence_id": f"ev_{digest}", "analysis_id": analysis_id, "result_id": result["result_id"],
                  "source_ids": result["source_ids"], "roi_id": result["roi_id"], "type": Path(artifact).suffix.lstrip(".") + "_artifact",
                  "path": artifact, "created_at": datetime.now(timezone.utc).isoformat()}
        if metadata.get("crs") and metadata.get("bounds"):
            record.update({"georeferenced": True, **metadata})
        else:
            record.update({"georeferenced": False, "spatial_reference": result["result_id"]})
        records.append(record)
    return records


def validate_spatial_consistency(context: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Return structured geometry, ROI/result, and evidence consistency checks."""
    failures, warnings = [], []
    checks: dict[str, Any] = {"aoi_source": context.get("aoi_source_checks", [])}
    aoi_checks = checks["aoi_source"]
    for check in aoi_checks:
        if check["status"] == "FAIL": failures.append(check.get("code", "AOI_SOURCE_INVALID"))
        elif check["status"] == "WARNING": warnings.append("Only part of the requested AOI is covered by a source image.")
    roi, result = context.get("roi", {}), context.get("result", {})
    roi_result_ok = result.get("analysis_id") == context.get("analysis_id") and result.get("roi_id") == roi.get("roi_id") and bool(result.get("crs") or not context.get("aoi"))
    checks["roi_result"] = {"status": "PASS" if roi_result_ok else "FAIL", "crs_match": result.get("crs") == roi.get("crs"), "bounds_match": result.get("bounds") == roi.get("bounds")}
    if not roi_result_ok: failures.append("RESULT_ROI_MISMATCH")
    evidence_checks = []
    for item in evidence:
        linked = item.get("analysis_id") == context.get("analysis_id") and item.get("result_id") == result.get("result_id") and item.get("roi_id") == roi.get("roi_id")
        spatial_ok = (item.get("georeferenced") and item.get("crs") and item.get("bounds")) or (not item.get("georeferenced") and item.get("spatial_reference") == result.get("result_id"))
        status = "PASS" if linked and spatial_ok else "FAIL"
        evidence_checks.append({"evidence_id": item.get("evidence_id"), "status": status, "analysis_id_match": item.get("analysis_id") == context.get("analysis_id"), "result_id_match": item.get("result_id") == result.get("result_id"), "spatial_reference_valid": spatial_ok})
        if status == "FAIL": failures.append("EVIDENCE_RESULT_MISMATCH")
    checks["result_evidence"] = evidence_checks
    return {"status": "FAIL" if failures else ("WARNING" if warnings else "PASS"), "checks": checks, "failures": failures, "warnings": list(dict.fromkeys(warnings))}
