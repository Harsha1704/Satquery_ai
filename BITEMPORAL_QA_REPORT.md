# Bi-temporal QA report

## Scope

Validated the saved Noida/Jewar pair:

- `outputs/temporal_high_change/noida_airport_jewar/noida_airport_jewar_2017.tif`
- `outputs/temporal_high_change/noida_airport_jewar/noida_airport_jewar_2025.tif`

Both files are 422 × 434, three-band `uint8` GeoTIFFs in WGS84. They have the
same affine grid (origin 77.59, 28.199; pixel size 0.000101895734597° ×
0.000089861751152°), bounds `[77.59, 28.16, 77.633, 28.199]`, and nodata value
zero. Their only TIFF-level tags are resolution and `AREA_OR_POINT`; the dates
are therefore evidenced by the filenames, not embedded acquisition metadata.

## Findings and corrections

- A pair with identical array dimensions but different affine transforms could
  previously reach visual, semantic, and structural inference. Bi-temporal
  validation now rejects it with `GRID_MISALIGNMENT` before those stages run.
- Reversed dated filenames now return `InvalidTemporalOrder`.
- RGB support metrics now exclude the combined nodata mask and report the valid
  pixel count. They remain explicitly supporting evidence, never a land-cover
  label.
- Semantic quality gating has an explicit boundary: unknown coverage of 35.0%
  passes; 35.1% fails. A failed gate prevents semantic evidence from becoming a
  definitive result.
- Visual evidence artifacts use a unique analysis directory, preventing a
  later analysis from overwriting another request's evidence.
- Unexpected Flask analysis failures return a stable error code and safe
  message; exception details and tracebacks are no longer exposed in the API
  response.

## Measured Noida RGB support

| Metric | Result |
| --- | ---: |
| Valid pixels | 183,148 (100.0%) |
| Mean absolute RGB difference | 31.0969 DN |
| Pixels above 10 DN | 78.2034% |
| Pixels above 20 DN | 48.5345% |
| Pixels above 30 DN | 32.8843% |

The model-backed bitemporal execution completed successfully for this pair.

## Validation run

- Contract and scientific-integrity checks: 23 passed.
- Model-backed Noida execution (`SATQUERY_RUN_INTEGRATION=1`): 1 passed.
- Complete default suite: 99 passed, 5 intentionally skipped, 33 subtests
  passed.
- Python compilation, JavaScript syntax validation, and `git diff --check`:
  passed.

## Remaining limitations

- The saved rasters do not contain acquisition-date metadata; filename years
  are the available temporal provenance for this fixture.
- Earth Engine and remote model downloads were not independently exercised
  against live upstream services because this environment's configured proxy
  rejects connections. The application now reports imagery connectivity
  failures as structured `503 imagery_service_unavailable` responses.
- The five skipped tests are manually invoked diagnostics or explicitly gated
  integrations, as listed by the test run.
