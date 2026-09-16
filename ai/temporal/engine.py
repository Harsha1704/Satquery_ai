"""Deterministic layered temporal change intelligence for small/ROI raster windows.

The engine is intentionally CPU-safe.  Learned change detection is an optional
specialist and is never represented as executed by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import numpy as np

@dataclass(frozen=True)
class TemporalPair:
    before_path: Path; after_path: Path; start_datetime: str|None=None; end_datetime: str|None=None
    def validate(self)->dict:
        import rasterio
        def parse(value):
            if value is None: return None
            text=str(value)
            if len(text)==4 and text.isdigit(): text += "-01-01"
            try:
                parsed=datetime.fromisoformat(text.replace("Z", "+00:00"))
                return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
            except ValueError as exc: raise ValueError("INVALID_TEMPORAL_PAIR: invalid acquisition datetime") from exc
        start,end=parse(self.start_datetime),parse(self.end_datetime)
        if start and end and start >= end: raise ValueError("INVALID_TEMPORAL_PAIR")
        with rasterio.open(self.before_path) as before, rasterio.open(self.after_path) as after:
            if not before.crs or not after.crs: raise ValueError("INVALID_TEMPORAL_PAIR: missing CRS")
            return {"before":{"crs":str(before.crs),"bounds":[float(x) for x in before.bounds],"resolution":list(before.res),"bands":before.count},"after":{"crs":str(after.crs),"bounds":[float(x) for x in after.bounds],"resolution":list(after.res),"bands":after.count}}

def _hash(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def _otsu(values:np.ndarray)->float:
 values=values[np.isfinite(values)]
 if not len(values): raise ValueError("INSUFFICIENT_VALID_DATA")
 hist,edges=np.histogram(values,bins=128); prob=hist/hist.sum(); omega=np.cumsum(prob); means=np.cumsum(prob*((edges[:-1]+edges[1:])/2)); total=means[-1]; denom=omega*(1-omega); score=np.divide((total*omega-means)**2,denom,out=np.zeros_like(denom),where=denom>0); return float((edges[np.argmax(score)]+edges[np.argmax(score)+1])/2)
def _area_m2(geometry, crs)->float:
 from pyproj import Transformer
 from shapely.geometry import shape
 from shapely.ops import transform
 polygon=shape(geometry)
 if str(crs).upper()!="EPSG:6933": polygon=transform(Transformer.from_crs(crs,"EPSG:6933",always_xy=True).transform,polygon)
 return float(polygon.area)
def _direction(geometry, roi_bounds, crs)->str:
 from shapely.geometry import shape
 x,y=shape(geometry).centroid.coords[0]; l,b,r,t=roi_bounds; hx=(l+r)/2;hy=(b+t)/2
 vertical="north" if y>hy else "south"; horizontal="east" if x>hx else "west"
 if abs(x-hx)<(r-l)*.16:return vertical
 if abs(y-hy)<(t-b)*.16:return horizontal
 return vertical+"_"+horizontal

class TemporalChangeEngine:
    VERSION="deterministic-temporal-v1"
    def analyze(self,pair:TemporalPair,*,before_semantic=None,after_semantic=None,minimum_component_pixels=1)->dict:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.features import shapes
        from rasterio.warp import reproject, transform_bounds, transform_geom
        from rasterio.windows import from_bounds
        from scipy import ndimage
        metadata=pair.validate()
        with rasterio.open(pair.before_path) as before, rasterio.open(pair.after_path) as after:
            # The before grid is deterministic reference. Restrict it to the
            # actual intersection before resampling T2: no out-of-coverage
            # pixels may enter change statistics or rendered evidence.
            after_bounds_in_before = transform_bounds(after.crs, before.crs, *after.bounds, densify_pts=21)
            left=max(before.bounds.left,after_bounds_in_before[0]); bottom=max(before.bounds.bottom,after_bounds_in_before[1])
            right=min(before.bounds.right,after_bounds_in_before[2]); top=min(before.bounds.top,after_bounds_in_before[3])
            if left>=right or bottom>=top: raise ValueError("NO_SPATIAL_OVERLAP")
            window=from_bounds(left,bottom,right,top,transform=before.transform).round_offsets().round_lengths()
            if before.count != after.count:
                raise ValueError("INCOMPATIBLE_BANDS: temporal rasters must have matching band counts")
            if any(a and b and a != b for a, b in zip(before.descriptions, after.descriptions)):
                raise ValueError("INCOMPATIBLE_BANDS: temporal band descriptions differ")
            before_data=before.read(window=window, masked=True).astype("float32").filled(np.nan); roi_transform=before.window_transform(window)
            after_data=np.full((after.count,before_data.shape[1],before_data.shape[2]),np.nan,dtype="float32")
            for index in range(min(before.count,after.count)):
                source = after.read(index+1, masked=True).astype("float32").filled(np.nan)
                reproject(source,after_data[index],src_transform=after.transform,src_crs=after.crs,dst_transform=roi_transform,dst_crs=before.crs,resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
            count=min(before.count,after.count); before_data=before_data[:count];after_data=after_data[:count]
            valid=np.all(np.isfinite(before_data)&np.isfinite(after_data),axis=0)
            if before.nodata is not None: valid &= np.all(before_data != before.nodata,axis=0)
            if after.nodata is not None: valid &= np.all(after_data != after.nodata,axis=0)
            if valid.sum()==0: raise ValueError("INSUFFICIENT_VALID_DATA")
            spectral=np.mean(np.abs(after_data-before_data),axis=0); normalized=spectral/(np.mean(np.abs(before_data),axis=0)+1e-6)
            # Robust fusion: both magnitude and normalized evidence contribute equally; no probability claim.
            spectral_scale=np.nanpercentile(spectral[valid],95); normalized_scale=np.nanpercentile(normalized[valid],95)
            score=(spectral/max(spectral_scale,1e-9)+normalized/max(normalized_scale,1e-9))/2
            score_range=float(np.ptp(score[valid]))
            threshold=_otsu(score[valid]) if score_range > 1e-9 else None
            mask=((score>=threshold)&valid) if threshold is not None else np.zeros_like(valid, dtype=bool)
            mask=ndimage.binary_opening(mask);mask=ndimage.binary_closing(mask) & valid
            labels,count_objects=ndimage.label(mask); sizes=np.bincount(labels.ravel()); mask &= sizes[labels]>=minimum_component_pixels
            labels,count_objects=ndimage.label(mask)
            changed=int(mask.sum()); total=int(valid.sum()); transform=roi_transform; crs=str(before.crs); bounds=[float(x) for x in rasterio.transform.array_bounds(before_data.shape[1],before_data.shape[2],transform)]
            warnings=[]
            if threshold is None and np.any(spectral[valid] > 1e-9):
                raise ValueError("UNRESOLVED_UNIFORM_CHANGE: a constant nonzero difference cannot be classified by adaptive thresholding")
            valid_coverage=valid.sum()*100/mask.size
            if valid_coverage < 80: warnings.append("Valid comparison coverage is below 80%; interpret areas outside the effective common grid as unavailable.")
            if before.count != after.count: warnings.append("The input band counts differ; only their shared leading bands were compared.")
            if before.crs == after.crs:
                resolution_ratio=max(abs(before.res[0])/max(abs(after.res[0]),1e-12),abs(after.res[0])/max(abs(before.res[0]),1e-12),abs(before.res[1])/max(abs(after.res[1]),1e-12),abs(after.res[1])/max(abs(before.res[1]),1e-12))
                if resolution_ratio > 2: warnings.append("Input resolution differs by more than 2×; common-grid resampling may suppress or amplify small changes.")
            transition={}; semantic_mask=None
            if before_semantic is not None and after_semantic is not None:
                if before_semantic.shape!=mask.shape or after_semantic.shape!=mask.shape: raise ValueError("INVALID_TEMPORAL_PAIR: semantic grids differ")
                semantic_mask=(before_semantic!=after_semantic)&valid
                for src,dst in zip(before_semantic[semantic_mask],after_semantic[semantic_mask]): transition[f"{int(src)}_to_{int(dst)}"]=transition.get(f"{int(src)}_to_{int(dst)}",0)+1
            features=[]
            for geom,val in shapes(labels.astype("int32"),mask=mask,transform=transform):
                component=labels == int(val)
                area=_area_m2(geom,crs); direction=_direction(geom,bounds,crs)
                local_transition=None
                if semantic_mask is not None and np.any(semantic_mask & component):
                    counts={}
                    for src,dst in zip(before_semantic[semantic_mask & component],after_semantic[semantic_mask & component]): counts[f"{int(src)}_to_{int(dst)}"]=counts.get(f"{int(src)}_to_{int(dst)}",0)+1
                    local_transition=max(counts,key=counts.get)
                # GeoJSON shown by the web map must be WGS84.  Preserve the
                # analysis CRS in properties so metric calculations remain
                # traceable to the exact common-grid geometry.
                display_geom = transform_geom(crs, "EPSG:4326", geom, precision=8) if crs.upper() != "EPSG:4326" else geom
                features.append({"type":"Feature","geometry":display_geom,"properties":{"change_id":f"chg_{len(features)+1}","pixel_count":int(component.sum()),"area_m2":area,"area_ha":area/10000,"direction":direction,"change_type":local_transition or "spectral_change","change_score":float(np.mean(score[component])),"source_ids":[f"src_{_hash(pair.before_path)[:16]}",f"src_{_hash(pair.after_path)[:16]}"],"source_geometry_crs":crs,"geometry_crs":"EPSG:4326","model_confidence":None}})
            features.sort(key=lambda item:item["properties"]["area_m2"],reverse=True)
            changed_area=sum(f["properties"]["area_m2"] for f in features); dominant=max(transition,key=transition.get) if transition else None
            result_id="res_"+hashlib.sha256((str(pair.before_path)+str(pair.after_path)).encode()).hexdigest()[:20]
            stats={"changed_area_m2":changed_area,"changed_area_ha":changed_area/10000,"changed_percentage":changed*100/total,"unchanged_percentage":(total-changed)*100/total,"valid_pixel_percentage":valid_coverage,"area_method":"EPSG:6933_equal_area_polygon_area"}
            description=f"Between {pair.start_datetime or 'the first observation'} and {pair.end_datetime or 'the second observation'}, {stats['changed_area_ha']:.4f} hectares ({stats['changed_percentage']:.2f}% of valid effective-ROI pixels) were identified as changed using fused deterministic spectral evidence."
            return {"task":"temporal_change_analysis","status":"EXPERIMENTAL","result_id":result_id,"pair":metadata,"effective_roi":{"crs":crs,"bounds":bounds,"transform":[float(x) for x in transform],"width":before_data.shape[2],"height":before_data.shape[1]},"statistics":stats,"threshold":{"method":"otsu_on_fused_spectral_score","value":threshold,"distribution":{"median":float(np.median(score[valid])),"p95":float(np.percentile(score[valid],95))}},"features":{"spectral_magnitude":True,"normalized_spectral_difference":True,"semantic_transition":semantic_mask is not None,"learned_detector":"NOT_EXECUTED"},"transition_matrix":transition,"dominant_transition":dominant,"change_polygons":{"type":"FeatureCollection","features":features},"largest_change_direction":features[0]["properties"]["direction"] if features else "none","description":description,"change_score":"deterministic_fused_feature_score","model_confidence":None,"registration":{"method":"metadata_common_grid_reprojection","status":"passed","estimated_shift_pixels":None,"quality_warnings":warnings},"provenance":{"before_hash":_hash(pair.before_path),"after_hash":_hash(pair.after_path),"algorithm":self.VERSION,"resampling":"bilinear_continuous_features","categorical_resampling":"nearest_required","time_range":[pair.start_datetime,pair.end_datetime]}}
