"""Validated pixel-box to geographic geometry conversion using the authoritative ROI transform."""
from __future__ import annotations
import math
from typing import Any
def validate_box(box, width:int, height:int)->list[float]:
 if not isinstance(box,(list,tuple)) or len(box)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in box): raise ValueError("INVALID_GROUNDING_RESULT")
 x1,y1,x2,y2=map(float,box);x1=max(0,min(width,x1));x2=max(0,min(width,x2));y1=max(0,min(height,y1));y2=max(0,min(height,y2))
 if x2<=x1 or y2<=y1: raise ValueError("INVALID_GROUNDING_RESULT")
 return [x1,y1,x2,y2]
def pixel_box_to_geo(box, roi:dict[str,Any])->dict[str,Any]:
 required=("analysis_id","roi_id","crs","transform","width","height")
 if any(roi.get(k) is None for k in required): raise ValueError("INVALID_GROUNDING_RESULT")
 x1,y1,x2,y2=validate_box(box,int(roi["width"]),int(roi["height"])); from affine import Affine; from pyproj import Transformer
 transform=Affine(*roi["transform"]); corners=[transform*(x,y) for x,y in ((x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1))]
 if str(roi["crs"]).upper()!="EPSG:4326":
  transformer=Transformer.from_crs(roi["crs"],"EPSG:4326",always_xy=True); corners=[transformer.transform(x,y) for x,y in corners]
 return {"bbox_pixel":[x1,y1,x2,y2],"geometry":{"type":"Polygon","coordinates":[corners]},"crs":"EPSG:4326","analysis_id":roi["analysis_id"],"roi_id":roi["roi_id"],"source_id":roi.get("parent_source_id")}
