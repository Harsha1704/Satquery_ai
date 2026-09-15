# SatQuery AI — Phase 1 API

This adds a typed query engine and synchronous FastAPI API around the existing
AI implementation. The Flask UI, specialist AI, model checkpoints, and
`gee_temporal.py` remain in place. React, Redis, PostGIS, authentication, and
deployment infrastructure are later phases.

## Run

From the repository root, using the existing Python 3.10 ML environment:

```powershell
.venv-ml/Scripts/python.exe -m pip install -r backend/requirements.txt
.venv-ml/Scripts/python.exe -m backend.main
```

Open **http://127.0.0.1:8000/docs** for the interactive API documentation.
`python backend/main.py` and `uvicorn backend.main:app` are also supported.
The compatibility launcher binds to localhost. This phase is for trusted local
use: project IDs are labels, `user_id` is `local`, and no access-control boundary
or multi-user authentication is implemented.

| Endpoint | Behavior |
| --- | --- |
| `GET /api/v1/health` | API liveness; does not load models or verify model readiness |
| `POST /api/v1/analysis/plan` | Parse, validate, and preview an approved plan without inference |
| `POST /api/v1/analysis` | Run analysis synchronously and return a completed or failed job |
| `POST /api/v1/analyses` | Alias of the singular analysis endpoint |
| `GET /api/v1/jobs/{job_id}` | Read a completed or failed job saved on disk |

Analysis returns HTTP 200 on success. Invalid requests return 422; missing GEE
configuration returns 503; oversized AOIs return 413; worker failures return 500;
timeouts return 504. Failures after job creation include a job ID and are saved.
This phase does **not** return 202 or provide an asynchronous polling queue.

## Local image request

Place the image in `frontend/uploads/`, or configure `SATQUERY_INPUT_ROOT`.
Input names are relative to that directory. Absolute paths, traversal, and
symlinks escaping the input root are rejected.

```json
{
  "query": "What is visible in this satellite image?",
  "project_id": "example",
  "inputs": {"image_path": "sample.png"}
}
```

Supported input roles:

| Analysis | Inputs |
| --- | --- |
| VQA, grounding, semantic, detection, multispectral | `image_path` |
| Change | `before_path` and `after_path` |
| SAR | `sar_path` or `image_path` |
| Optical + SAR | `optical_path` and `sar_path` |

An explicit before/after pair selects change analysis, matching the legacy UI.
The underlying engine still determines valid image content and band layouts;
multispectral analysis expects the legacy 12-band Sentinel-2 order.

## Historical AOI request

```json
{
  "query": "Has vegetation decreased in this region since 2020?",
  "aoi": {
    "type": "Polygon",
    "coordinates": [[[77,12],[77.01,12],[77.01,12.01],[77,12.01],[77,12]]]
  }
}
```

AOIs currently support one closed, axis-aligned WGS84 rectangle. Empty polygons,
holes, arbitrary polygons, and antimeridian-crossing areas are rejected rather
than silently analyzed as a different area. Automatic retrieval supports
historical change analysis only. Use explicit years in the question, “since
2020”, “last five years”, or both `before_year` and `after_year`. Ambiguous or
invalid date ranges are rejected. Bounds and imagery size limits are also
checked by the existing GEE provider.

Live retrieval requires the existing Earth Engine dependencies and these server
settings: `GEE_SERVICE_ACCOUNT_EMAIL`, `GEE_SERVICE_ACCOUNT_KEY`, and
`GEE_PROJECT_ID`. Use absolute paths for credential files and model-weight
environment settings because analysis workers have job-local working directories.
The API never imports the Flask application to retrieve imagery.

## Architecture and storage

```text
FastAPI → AnalysisService → QueryParser → QueryPlanner → tool allowlist
        → isolated worker → approved-plan adapter → SatQueryOrchestrator
```

The parser is injectable and currently delegates to the tested legacy rules.
It is **not** a general-purpose LLM parser. The existing `query_engine/parser/`
package is used instead of creating a conflicting `parser.py`. Tool selection
comes from a closed registry; clients cannot submit executable tool names or
code. The legacy orchestrator receives the approved intent through its planner
injection point. Registry entries represent whole legacy pipelines, including
their internal specialist calls.

Each request copies inputs into `outputs/jobs/ana_<uuid>/inputs/` and runs a
separate Python process with that job directory as its working directory.
Legacy evidence filenames therefore cannot overwrite another API job's files.
No process-wide directory change occurs in the API server.

```text
outputs/jobs/ana_<uuid>/
  inputs/                  # snapshots of submitted images
  imagery/                 # fetched historical composites
  outputs/evidence/        # unchanged legacy relative output convention
  artifacts/ layers/       # reserved for later standardized exports
  statistics/statistics.json
  reports/result.json      # persisted job/result contract and JSON report
  request.json
  worker-result.json       # legacy details, retained locally
  execution.log
```

Public evidence references are job-relative paths. Download routes, map-ready
layer conversion, HTML/PDF reports, and object-storage artifact IDs are not part
of this phase; `layers` remains empty. Raw legacy results stay in the job
directory for inspection. Only routing confidence is populated currently;
other confidence types remain null rather than reinterpreting generic legacy
scores as calibrated probabilities.

Workers load models afresh per request. RemoteCLIP's existing explicit relative
checkpoint cache is also job-local and may download weights again. A persistent
worker/model-cache adapter is a later optimization. Use the existing Flask UI
for workloads where warm model reuse is required today.

Configuration:

| Environment variable | Default |
| --- | --- |
| `SATQUERY_INPUT_ROOT` | repository `frontend/uploads` |
| `SATQUERY_JOB_ROOT` | repository `outputs/jobs` |
| `SATQUERY_ANALYSIS_TIMEOUT` | `900` seconds |

## Verification

```powershell
.venv-ml/Scripts/python.exe -m unittest tests.test_phase1_api -v
```

Tests cover schema and policy validation, planning, approved-intent adaptation,
concurrent job isolation, persistence, missing imagery configuration, and a real
multispectral worker using a synthetic raster. They do not download model weights
or call live Earth Engine. Raster workers apply the same bundled-PROJ setup as
the legacy Flask app to avoid conflicts with Windows PostGIS installations.
