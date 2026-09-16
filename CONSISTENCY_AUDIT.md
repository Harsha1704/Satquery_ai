# SatQuery analysis-context consistency audit

## Execution path audited

`frontend/templates/map.html` loads `frontend/static/js/map.js`. The map captures a WGS84 GeoJSON AOI and sends it to the FastAPI Map Context endpoint. `backend/app/api/v1/map_context.py` validates and converts it into the versioned `query_engine.schemas.AnalysisRequest`; the same planner produces the preview fingerprint and the execution plan. `backend/app/api/v1/map.py` submits that approved plan to `AnalysisJobManager`, which calls `AnalysisService`, the private query-engine worker, evidence generation, spatial validation, and the saved-job report endpoint.

## Findings and corrections

1. **Stale browser responses after AOI edits:** an AOI edit cleared visible layers but did not invalidate an in-flight request. A late response could therefore repopulate the map with a previous AOI's result. The map now snapshots the AOI per request, assigns a run token, invalidates that token whenever the AOI changes, and requests best-effort cancellation of the superseded job.
2. **Plan/execution imagery mismatch was publishable:** the backend recorded a disagreement between the planned imagery and worker provenance as a limitation, then could publish evidence and a report. It now returns `409 imagery_plan_mismatch` and publishes no result when provenance differs.
3. **Collection identity comparison was brittle:** planning can expose either a provider collection ID or its human label. The source-consistency check now accepts either authoritative identifier while requiring the same ordered before/after year and collection pair.
4. **Selected-scene limitation:** the current Map View has no imagery-catalog search-and-select workflow. It performs an explicitly labelled, server-resolved annual composite workflow. It must not be represented as a user-selected individual scene until a catalog-selection contract (item ID, asset ID, acquisition datetime and signed asset access) is implemented end-to-end.

## Canonical context

```
Map AOI + query
  -> canonical AnalysisRequest + approved plan fingerprint
  -> planner / orchestrator / worker
  -> AnalysisService spatial and source-consistency gate
  -> canonical AnalysisResult provenance
  -> evidence, map overlay, statistics, saved-job report
```

The completed result carries the AOI, imagery provenance, execution context, analysis context, evidence identities, spatial-consistency checks, and job/analysis ID. Evidence lives under each job directory, avoiding cross-analysis filename collisions.

## Remaining risks

The annual Earth Engine path creates cloud-filtered median composites, not a single selected scene; therefore an individual acquisition datetime and scene ID are unavailable by design. Full scene-selection support requires an imagery discovery API and a UI control before it can be claimed. The legacy Flask upload API remains a separate compatibility flow; Map View uses FastAPI on port 8000 as its authoritative context pipeline.
