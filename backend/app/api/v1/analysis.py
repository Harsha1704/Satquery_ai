from pathlib import Path
import os
import re

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.schemas.analysis import AnalysisRequest, JobResponse
from query_engine.schemas import AnalysisPlan

router = APIRouter()


@router.post("/analysis/plan", response_model=AnalysisPlan)
def plan(body: AnalysisRequest, request: Request):
    return request.app.state.analysis_service.plan(body)


@router.post("/analyses", response_model=JobResponse, include_in_schema=False)
@router.post("/analysis", response_model=JobResponse)
def analyze(body: AnalysisRequest, request: Request, response: Response):
    result, status_code = request.app.state.analysis_service.analyze(body)
    response.status_code = status_code
    return result


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request):
    return request.app.state.analysis_service.get_job(job_id)


def _artifact_cors_headers(request: Request) -> dict[str, str]:
    """Return explicit browser headers for map-renderable job artifacts.

    MapLibre loads image sources with ``fetch`` rather than a plain ``<img>``
    element, so the artifact response itself must be CORS-readable by the
    workspace origin.  CORSMiddleware remains the global policy; these headers
    are a defense-in-depth guarantee for FileResponse / byte-range responses.
    """
    configured = os.environ.get("SATQUERY_CORS_ORIGINS")
    allowed = {
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    }
    if configured:
        allowed.update(x.strip() for x in configured.split(",") if x.strip())
    origin = (request.headers.get("origin") or "").rstrip("/")
    headers = {
        "Cache-Control": "private, max-age=300",
        "Cross-Origin-Resource-Policy": "cross-origin",
    }
    if origin in allowed:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Range"
        headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range, Accept-Ranges"
        headers["Vary"] = "Origin"
    return headers


def _safe_job_dir(request: Request, job_id: str) -> Path:
    if not re.fullmatch(r"ana_[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    root = request.app.state.analysis_service.job_root.resolve()
    job_dir = (root / job_id).resolve()
    if root not in job_dir.parents or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_dir


@router.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_job_artifact(job_id: str, artifact_path: str, request: Request):
    """Serve only artifacts explicitly published by the completed job.

    Internal request payloads, worker outputs, logs, and report JSON are never
    downloadable through the artifact route even if a caller guesses a path.
    """
    job_dir = _safe_job_dir(request, job_id)
    normalized = str(Path(artifact_path).as_posix()).lstrip("/")
    job = request.app.state.analysis_service.get_job(job_id)
    published = set(job.result.evidence if job.result else [])
    if normalized not in published:
        raise HTTPException(status_code=404, detail="Published artifact not found.")

    candidate = (job_dir / normalized).resolve()
    if job_dir not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geojson"}:
        raise HTTPException(status_code=415, detail="Unsupported published artifact type.")
    return FileResponse(candidate, headers=_artifact_cors_headers(request))


@router.get("/jobs/{job_id}/report")
def get_job_report(job_id: str, request: Request):
    import html
    job = request.app.state.analysis_service.get_job(job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail="No completed analysis is available for this job.")
    d=job.model_dump(mode="json"); result=d.get("result") or {}; stats=result.get("statistics") or {}
    imagery=stats.get("imagery") or {}; before=imagery.get("before") or {}; after=imagery.get("after") or {}; layer=stats.get("task_layer") or {}; comparison=stats.get("comparison") or {}
    parsed=(d.get("plan") or {}).get("parsed") or {}; query=str(parsed.get("query") or "SatQuery analysis"); years=parsed.get("years") or []
    text=(query+" "+" ".join(map(str,parsed.get("targets") or []))).lower()
    if any(x in text for x in ("vegetation","ndvi","crop","forest")): title="Vegetation Change Intelligence"; metric="NDVI"; b=before.get("ndvi"); a=after.get("ndvi")
    elif any(x in text for x in ("urban","built_up","built-up","construction")): title="Urban Expansion Intelligence"; metric="NDBI"; b=before.get("ndbi"); a=after.get("ndbi")
    elif any(x in text for x in ("water","flood","river","lake","ndwi")): title="Water Change Intelligence"; metric="Water extent"; b=before.get("water_fraction"); a=after.get("water_fraction")
    else: title="Temporal Change Intelligence"; metric="Change"; b=None; a=None
    def num(v):
        try: return float(v)
        except (TypeError,ValueError): return None
    b=num(b); a=num(a); delta=(a-b) if b is not None and a is not None else None
    if metric=="Water extent":
        before_txt=f"{b*100:.2f}%" if b is not None else "Unavailable"; after_txt=f"{a*100:.2f}%" if a is not None else "Unavailable"; delta_txt=f"{delta*100:+.2f} pp" if delta is not None else "Unavailable"
    else:
        before_txt=f"{b:.3f}" if b is not None else "Unavailable"; after_txt=f"{a:.3f}" if a is not None else "Unavailable"; delta_txt=f"{delta:+.3f}" if delta is not None else "Unavailable"
    area=num(layer.get("affected_area_km2")); share=num(layer.get("changed_percentage")); coverage=num(layer.get("analysis_coverage_pct"))
    validation=stats.get("validation") or {}; validation_quality=str(validation.get("quality") or "Not reported")
    alignment=(validation.get("alignment") or {}).get("label") or "Not reported"
    sensitivity=(validation.get("threshold_sensitivity") or {}).get("status") or "Not reported"
    sensor_consistency=validation.get("sensor_consistency") or "Not reported"
    validation_flags=validation.get("flags") or []
    routing=num((result.get("confidence") or {}).get("routing")); routing_txt=f"{routing*100:.0f}%" if routing is not None else "Rule-routed"
    imgs=(result.get("provenance") or {}).get("imagery") or []; collections=[]
    for x in imgs:
        c=str(x.get("collection_id") or ""); name="Sentinel-2 MSI" if "S2" in c else ("Landsat multispectral" if "LANDSAT" in c else (c or "Earth observation imagery"))
        if name not in collections: collections.append(name)
    source=" / ".join(collections) or "Earth observation imagery"
    source_consistency=(result.get("provenance") or {}).get("source_consistency") or {}
    source_consistency_txt = "Matched" if source_consistency.get("matched") is True else ("Changed · review" if source_consistency.get("matched") is False else "Not applicable")
    evidence=[x for x in result.get("evidence") or [] if str(x).lower().endswith((".png",".jpg",".jpeg"))]
    preferred=[]
    for token in ("before_after_change_comparison","before_after_comparison","scene_","ndvi_change_layer","ndbi_change_layer","water_change_layer","rgb_difference","change_mask"):
        hit=next((x for x in evidence if token in x.lower() and x not in preferred),None)
        if hit: preferred.append(hit)
    ev="".join(f'<figure><img src="/api/v1/jobs/{job_id}/artifacts/{html.escape(x)}"><figcaption>{html.escape(Path(x).stem.replace("_"," ").title())}</figcaption></figure>' for x in preferred[:6])
    answer=html.escape(str(result.get("answer") or "Analysis completed.")); limitations=result.get("limitations") or []
    lim="".join(f'<li>{html.escape(str(x))}</li>' for x in limitations[:8]) or '<li>No additional limitations were returned.</li>'
    val_flags="".join(f'<li>{html.escape(str(x))}</li>' for x in validation_flags[:6]) or '<li>No validation flags were raised.</li>'
    if b is not None and a is not None:
        lo=min(b,a); hi=max(b,a); span=max(hi-lo,abs(hi)*.2,0.01); wb=25+55*(b-lo)/span; wa=25+55*(a-lo)/span
    else: wb=wa=35
    period=" → ".join(map(str,years)) if years else "Selected observation period"
    visual_heading = "Before · After · Change" if comparison.get("change_available") else "Before · After · Change evidence unavailable"
    report=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>{html.escape(title)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#06131d;color:#eaf7ff;font-family:Inter,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:38px}}.top{{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid #183747;padding-bottom:24px}}.brand{{color:#59caff;font-weight:800;letter-spacing:.12em}}h1{{font-size:38px;margin:8px 0}}.period{{color:#9ab4c2}}.badge{{background:#0d3b32;color:#78e6bd;border:1px solid #176650;padding:8px 12px;border-radius:999px;height:max-content}}.finding{{margin:24px 0;background:linear-gradient(135deg,#0a293a,#08202d);border:1px solid #16465c;border-radius:18px;padding:22px}}.finding span,.section-label{{color:#58caff;font-size:11px;font-weight:800;letter-spacing:.15em}}.finding p{{font-size:17px;line-height:1.6;margin-bottom:0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:#0a2230;border:1px solid #153b4d;border-radius:14px;padding:16px}}.metric span{{display:block;color:#7e9eae;font-size:11px}}.metric strong{{display:block;font-size:21px;margin-top:6px}}section{{margin-top:28px}}h2{{font-size:22px}}.chart{{background:#091f2c;border:1px solid #153747;border-radius:16px;padding:18px}}.barrow{{display:grid;grid-template-columns:90px 1fr 90px;gap:12px;align-items:center;margin:14px 0}}.track{{height:16px;background:#102e3e;border-radius:99px;overflow:hidden}}.fill{{height:100%;background:linear-gradient(90deg,#3caee8,#6de2c0);border-radius:99px}}.decision{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.decision div{{background:#0a2230;padding:15px;border-radius:12px}}.decision span{{display:block;color:#779aaa;font-size:10px}}.decision strong{{display:block;margin-top:5px}}.validation-note{{margin-top:12px;padding:14px;border-radius:12px;background:#071b27;border:1px solid #173849;color:#9db5c1;line-height:1.55}}.evidence{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0;background:#071b27;border:1px solid #173849;border-radius:14px;overflow:hidden}}figure img{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}}figcaption{{padding:10px;font-size:11px;color:#a9c1cc}}ul{{color:#9db5c1;line-height:1.6}}footer{{margin-top:34px;padding-top:18px;border-top:1px solid #173646;color:#668998;font-size:11px}}@media(max-width:800px){{.metrics,.decision,.evidence{{grid-template-columns:1fr 1fr}}main{{padding:20px}}}}@media print{{body{{background:white;color:#10202a}}.metric,.finding,.chart,.decision div,figure{{background:#f5f8fa;color:#10202a}}}}
</style></head><body><main><div class='top'><div><div class='brand'>SATQUERY AI · GEOSPATIAL INTELLIGENCE REPORT</div><h1>{html.escape(title)}</h1><div class='period'>{html.escape(period)} · Job {html.escape(job_id)}</div></div><div class='badge'>Completed</div></div>
<div class='finding'><span>EXECUTIVE FINDING</span><p>{answer}</p></div>
<div class='metrics'><div class='metric'><span>Detected-change area</span><strong>{f'~{area:.2f} km²' if area is not None else 'Unavailable'}</strong></div><div class='metric'><span>Changed share</span><strong>{f'{share:.1f}%' if share is not None else 'Unavailable'}</strong></div><div class='metric'><span>{html.escape(metric)} change</span><strong>{delta_txt}</strong></div><div class='metric'><span>Workflow confidence</span><strong>{routing_txt}</strong></div></div>
<section><div class='section-label'>QUANTITATIVE ANALYTICS</div><h2>Before vs after</h2><div class='chart'><div class='barrow'><b>Before</b><div class='track'><div class='fill' style='width:{wb:.0f}%'></div></div><strong>{before_txt}</strong></div><div class='barrow'><b>After</b><div class='track'><div class='fill' style='width:{wa:.0f}%'></div></div><strong>{after_txt}</strong></div></div></section>
<section><div class='section-label'>AI DECISION</div><h2>Why this workflow was selected</h2><div class='decision'><div><span>QUERY</span><strong>{html.escape(query)}</strong></div><div><span>SATELLITE</span><strong>{html.escape(source)}</strong></div><div><span>ANALYSIS</span><strong>{html.escape(metric)} temporal comparison</strong></div><div><span>PERIOD</span><strong>{html.escape(period)}</strong></div><div><span>VALID COVERAGE</span><strong>{f'{coverage:.0f}%' if coverage is not None else 'Not reported'}</strong></div><div><span>EVIDENCE TYPE</span><strong>Task-specific spectral change</strong></div><div><span>PLAN / EXECUTION SOURCE</span><strong>{html.escape(source_consistency_txt)}</strong></div></div></section>
<section><div class='section-label'>EVIDENCE VALIDATION</div><h2>Can this result be trusted?</h2><div class='decision'><div><span>EVIDENCE QUALITY</span><strong>{html.escape(validation_quality)}</strong></div><div><span>SPATIAL ALIGNMENT</span><strong>{html.escape(str(alignment))}</strong></div><div><span>VALID COVERAGE</span><strong>{f'{coverage:.1f}%' if coverage is not None else 'Not reported'}</strong></div><div><span>THRESHOLD SENSITIVITY</span><strong>{html.escape(str(sensitivity).title())}</strong></div><div><span>SENSOR CONSISTENCY</span><strong>{html.escape(str(sensor_consistency))}</strong></div><div><span>STATISTICS FOOTPRINT</span><strong>{'Exact AOI polygon' if validation.get('exact_aoi_statistics') else 'Comparison footprint'}</strong></div></div><div class='validation-note'><ul>{val_flags}</ul><b>Note:</b> Evidence quality is a rule-based validation grade, not calibrated model accuracy.</div></section>
<section><div class='section-label'>VISUAL EVIDENCE</div><h2>{html.escape(visual_heading)}</h2><div class='evidence'>{ev or '<p>No browser-preview evidence was produced for this run.</p>'}</div></section>
<section><div class='section-label'>TECHNICAL TRANSPARENCY</div><h2>Method & limitations</h2><ul>{lim}</ul></section><footer>Generated by SatQuery AI from saved analysis outputs. Workflow confidence describes query/workflow routing and must not be interpreted as calibrated result accuracy.</footer></main></body></html>"""
    return HTMLResponse(content=report,headers={"Content-Disposition":f'inline; filename="SatQuery_{job_id}_report.html"',"Cache-Control":"no-store"})
