# SatQuery AI dashboard

Start the dashboard from the project root:

```powershell
.\.venv-ml\Scripts\python.exe frontend/app.py
```

Open http://127.0.0.1:5000. This is the Flask dashboard; the root `app.py` is the separate Streamlit interface.

BLIP VQA preloads in a background thread by default and reuses the same model and answer cache for requests. Set `SATQUERY_PRELOAD_VQA=0` to load on the first question instead, or `SATQUERY_BLIP_WARMUP=1` to run a short inference during preload. The device is selected automatically. `/api/runtime-status` reports preload progress (`not_started`, `loading`, `ready`, or `error`); the dashboard remains available while loading. Model weights must already be cached or downloadable on first use.

Run the real VQA performance check with `.\.venv-ml\Scripts\python.exe -m tests.test_vqa_performance`. It prints preload time, two inference times, and a repeated cached question, and fails if inference, timing telemetry, or caching fails. Run runtime regression checks with `.\.venv-ml\Scripts\python.exe -m unittest tests.test_vqa_runtime -v`.

The layout follows the provided 1672 × 941 dashboard reference. The page uses local SVG icons and local imagery, with responsive layouts for smaller screens. The initial result is explicitly marked **Example**. Generated decorative imagery is never submitted to the analysis service or substituted for actual model evidence.

- Six analysis modes, typed questions and keyboard submission (Ctrl/Cmd + Enter).
- File picker, drag and drop, separate before/after and optical/SAR fields, file removal and a 200 MB combined upload limit.
- Actual `/api/analyze` results, confidence, evidence gallery, downloads and session history.
- Reports lists completed reports. Datasets offers scenario presets. Model Hub describes the existing pipelines.
- Map View at `/map-view` lets you choose a completed georeferenced analysis, inspect its source raster and result layers, and run a question on a selected region. New Analysis preserves map history. Source previews and supported result rasters are reprojected to WGS84 for map placement.
- Leaflet and the OpenStreetMap basemap need an internet connection. The supplied CARTO endpoint returned API-key-required tiles, so the app uses the standard OpenStreetMap tile service with attribution. Follow its [tile usage policy](https://operations.osmfoundation.org/policies/tiles/) when deploying. Inputs without usable georeferencing show a clear empty state. Projected bounds are converted with Rasterio; when available, its matching bundled PROJ database is selected for this app process to avoid conflicts with other Windows GIS installations.
- A repository URL is not configured for the GitHub button.

Browser verification script: `node outputs/ui-review/check-ui.mjs` (requires the server above and installed Chrome). Screenshots are saved in `outputs/ui-review/`.

Map backend checks: `.\.venv-ml\Scripts\python.exe -m unittest tests.test_frontend_map tests.test_map_roi -v`.

## Analyze a selected area

1. In Analyze, choose **Multispectral Analysis**, upload `data/test/geotiff/sentinel2_test.tif`, and submit **Calculate NDVI**.
2. Open **Map View** and choose that completed analysis from **Dataset / Analysis**.
3. Click **Select Area**, then click two opposite corners within the raster footprint. Click **Fix Area**.
4. Enter **Calculate NDVI for this selected area** and run the analysis.
5. Inspect the cropped NDVI overlay and click **Generate Report**. The cropped analysis also appears in the dataset selector.

The server crops the original GeoTIFF pixels, preserving georeferencing and masks, before running the selected pipeline. Reports for selected regions are saved separately. Visual model inputs use PNG previews while the registry retains the cropped GeoTIFF for later map analysis.

The server keeps the latest 50 successful analysis records in memory. After restarting the server, or if an analysis expires from this registry, rerun its original upload before selecting another region. Browser history can still show older results, but it cannot restore their server records. This local development workflow requires the same server process for upload and selected-area analysis.

## Artwork

Both new assets were made with the built-in imagegen tool using the user-supplied dashboard as the visual reference, then copied into this project:

- `static/images/hero-satellite.png`: decorative satellite background.
- `static/images/example-roads.png`: illustrative initial preview, visibly labeled as an example.

The original input image is a visual reference only. All dashboard text, controls and layout are implemented in HTML/CSS/JavaScript, not baked into an image.

### Final generation prompts

**Hero**

Create a single clean bitmap background asset for a working website, using the attached dashboard screenshot only as a visual reference. Recreate ONLY the satellite photograph behind the central heading 'Understand Our Planet with the Power of AI', visible from x=320 to x=1170 and y=86 to y=370 in the reference. Output a 1536x640 wide landscape photograph: realistic straight-down satellite view of lush dark green forests, dispersed urban streets and small settlements, teal river curving down the far right edge, wispy white scattered clouds especially across the upper center and bottom right. Match the reference's natural terrain, composition, cloud shapes, rich green and teal colors and photographic texture closely. Full bleed photograph only. REMOVE ALL text, badges, panels, UI, buttons, borders and typography; reconstruct the land underneath. Do not reproduce the entire dashboard. No letters, no logos. Keep detail and lighter clouds at the top and upper right, dark forest across left. This is a decorative website background, not scientific evidence. Save the generated asset.

**Example roads**

Recreate ONLY the photographic satellite image in the right 'Visual Evidence' panel of this reference screenshot, bounded approximately x1205 y511 to x1632 y700. Output a clean high resolution wide 2.25:1 landscape raster photograph with no UI, no borders, no text, no overlays. Match this satellite view as closely as possible: near-vertical aerial imagery of a small US town, dense light grey suburban streets and scattered warehouse buildings in the left half, a long wide divided motorway running diagonally from bottom left toward upper center-right, a very dark green and brown wooded open area occupying the center, a nearly vertical muted green river along the right edge, pale sand and light-colored fields beyond the river at far right. Muted realistic satellite colors, natural texture and scale, no dramatic perspective. Preserve the reference geography and composition closely. This is decorative EXAMPLE imagery for a website dashboard, not actual scientific or analysis evidence. Only the photo fills the output.
