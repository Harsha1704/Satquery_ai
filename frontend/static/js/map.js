(() => {
"use strict";

const DEFAULT_CENTER = [78.9629, 20.5937];
const DEFAULT_ZOOM = 3.2;
const BASE_STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";
const API_BASE = window.SATQUERY_API_BASE || "http://127.0.0.1:8000";
const AOI_STORAGE_KEY = "satquery_selected_aoi";
const CONTEXT_STORAGE_KEY = "satquery_latest_context";
const SAT_SOURCE = "sq-satellite-source";
const SAT_LAYER = "sq-satellite-layer";
const AOI_SOURCE = "sq-aoi-source";
const AOI_FILL = "sq-aoi-fill";
const AOI_LINE = "sq-aoi-line";
const AOI_POINT = "sq-aoi-point";
const PREVIEW_SOURCE = "sq-preview-source";
const PREVIEW_FILL = "sq-preview-fill";
const PREVIEW_LINE = "sq-preview-line";
const ANALYSIS_SOURCE = "sq-analysis-evidence-source";
const ANALYSIS_LAYER = "sq-analysis-evidence-layer";
const TEMPORAL_CHANGE_SOURCE = "sq-temporal-change-polygons";
const TEMPORAL_CHANGE_FILL = "sq-temporal-change-fill";
const TEMPORAL_CHANGE_LINE = "sq-temporal-change-line";
const EXECUTION_STEPS = ["imagery", "preprocess", "analysis", "compare", "evidence", "statistics", "report"];

let map;
let currentAoi = null;
let drawMode = null;
let drawStart = null;
let polygonPoints = [];
let baseMode = "hybrid";
let currentJob = null;
let currentPlanPayload = null;
let executionTimer = null;
let executionIndex = 0;
let overlayArtifact = null;
let satelliteTileErrors = 0;
let satelliteErrorResetTimer = null;
let satelliteFallbackActive = false;
let activeJobId = null;
let currentAnalysisId = null;
let cancellationRequested = false;
// Every AOI/query submission owns a monotonically increasing token.  Network
// responses from an older token are never allowed to update the current map.
let activeRunToken = 0;

const $ = (id) => document.getElementById(id);
const els = {
  systemPill: $("systemPill"), systemLabel: $("systemLabel"),
  searchForm: $("locationSearchForm"), searchInput: $("locationSearchInput"), searchStatus: $("searchStatus"), searchResults: $("searchResults"),
  aoiTools: $("aoiTools"), clearAoiBtn: $("clearAoiBtn"), fitAoiBtn: $("fitAoiBtn"), aoiSummary: $("aoiSummary"), aoiStateText: $("aoiStateText"),
  query: $("mapNaturalQuery"), send: $("sendAoiQueryBtn"), connectionStatus: $("backendConnectionStatus"), badge: $("backendStateBadge"), backendPlan: $("backendPlan"),
  workflow: $("workflowSteps"), planSection: $("planSection"), togglePlanBtn: $("togglePlanBtn"), sourceSelection: $("sourceSelection"), selectedSourceName: $("selectedSourceName"), selectedSourceMeta: $("selectedSourceMeta"),
  resultSection: $("analysisResultSection"), emptyResultSection: $("emptyResultSection"), resultTitle: $("resultTitle"), resultPeriod: $("resultPeriod"), resultQualityBadge: $("resultQualityBadge"),
  metricArea: $("metricArea"), metricShare: $("metricShare"), metricPrimary: $("metricPrimary"), metricSatellite: $("metricSatellite"), metricConfidence: $("metricConfidence"), metricResultQuality: $("metricResultQuality"), mapAnswer: $("mapAnswer"),
  qualityRouting: $("qualityRouting"), qualityData: $("qualityData"), qualityEvidence: $("qualityEvidence"), evidenceGallery: $("evidenceGallery"), technicalEvidence: $("technicalEvidence"),
  validationCard: $("validationCard"), validationGrade: $("validationGrade"), validationAlignment: $("validationAlignment"), validationCoverage: $("validationCoverage"), validationSensitivity: $("validationSensitivity"), validationSensor: $("validationSensor"), validationSourcePlan: $("validationSourcePlan"), validationFlags: $("validationFlags"),
  comparisonViewer: $("comparisonViewer"), openComparisonBtn: $("openComparisonBtn"), compareScenesBtn: $("compareScenesBtn"), comparisonHeading: $("comparisonHeading"),
  compareBeforeImg: $("compareBeforeImg"), compareAfterImg: $("compareAfterImg"), compareChangeImg: $("compareChangeImg"), compareBeforeLabel: $("compareBeforeLabel"), compareAfterLabel: $("compareAfterLabel"), comparisonChangeCard: $("comparisonChangeCard"), comparisonChangeLabel: $("comparisonChangeLabel"),
  mapComparison: $("mapComparison"), closeComparisonBtn: $("closeComparisonBtn"), mapBeforeImg: $("mapBeforeImg"), mapAfterImg: $("mapAfterImg"), mapChangeImg: $("mapChangeImg"), mapBeforeLabel: $("mapBeforeLabel"), mapAfterLabel: $("mapAfterLabel"), mapComparisonTitle: $("mapComparisonTitle"), mapChangeFigure: $("mapChangeFigure"), mapChangeCaption: $("mapChangeCaption"),
  viewEvidenceBtn: $("viewEvidenceBtn"), generateReportBtn: $("generateReportBtn"), topReportsLink: $("topReportsLink"), reportsHub: $("reportsHub"), reportsList: $("reportsList"), reportsSummary: $("reportsSummary"), closeReportsHub: $("closeReportsHub"),
  vectorBtn: $("darkMapBtn"), satelliteBtn: $("satelliteMapBtn"), hybridBtn: $("hybridMapBtn"), resetBtn: $("resetGlobeBtn"), baseMapLabel: $("baseMapLabel"),
  aoiLayerToggle: $("aoiLayerToggle"), analysisLayerToggle: $("analysisLayerToggle"), activeLayerLabel: $("activeLayerLabel"), mapHint: $("mapHint"), analysisLegend: $("analysisLegend"),
  lat: $("cursorLat"), lng: $("cursorLng"), zoom: $("zoomLevel")
};

function safeText(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function setInsights(open) {
  $("insightsPanel").hidden = !open;
  document.body.classList.toggle("insights-open", open);
  $("toggleInsightsBtn").setAttribute("aria-expanded", String(open));
}

function setMapFocus(focused) {
  document.body.classList.toggle("map-focused", focused);
  $("toggleCommandBtn").setAttribute("aria-expanded", String(!focused));
  $("toggleCommandBtn").textContent = focused ? "New query" : "Focus map";
  if (focused) setInsights(false);
}

function mapPadding() {
  const mobile = window.innerWidth <= 760;
  const commandVisible = !document.body.classList.contains("map-focused");
  const sidebar = commandVisible ? document.querySelector(".command-panel").offsetWidth : 0;
  return mobile ? {top:110, bottom:commandVisible ? Math.round(map.getContainer().clientHeight * .59) : 45, left:30, right:45}
    : {top:120,bottom:90,left:sidebar + 40,right:document.body.classList.contains("insights-open") ? $("insightsPanel").offsetWidth + 30 : 70};
}

$("toggleCommandBtn").addEventListener("click", () => {
  const focused = !document.body.classList.contains("map-focused");
  setMapFocus(focused);
  if (!focused && window.innerWidth <= 760) setInsights(false);
});
$("toggleInsightsBtn").addEventListener("click", () => setInsights($("insightsPanel").hidden));
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (drawMode) { stopDrawing(); els.mapHint.textContent = "Drawing cancelled. Choose a tool to start again."; }
  else if (!els.mapComparison.classList.contains("hidden")) els.mapComparison.classList.add("hidden");
  else if (!els.reportsHub.classList.contains("hidden")) els.reportsHub.classList.add("hidden");
  else if (!$("insightsPanel").hidden) { setInsights(false); $("toggleInsightsBtn").focus(); }
});

function compactLabel(value) {
  return String(value || "--")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function pct(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const p = n <= 1 ? n * 100 : n;
  return `${p.toFixed(digits)}%`;
}

function humanTool(value) {
  const aliases = {
    satellite_retrieval: "Retrieve imagery",
    satellite_catalog_search: "Search satellite catalog",
    cloud_filtering: "Cloud filtering",
    scene_quality_filter: "Scene quality filtering",
    multispectral_preprocessing: "Multispectral preprocessing",
    sar_preprocessing: "SAR preprocessing",
    ndvi_analysis: "NDVI analysis",
    built_up_segmentation: "Built-up analysis",
    water_segmentation: "Water analysis",
    temporal_pair_selection: "Select temporal pair",
    temporal_change_detection: "Temporal change detection",
    change_detection: "Change detection",
    change_reasoning: "Evidence-grounded change reasoning",
    temporal_evidence: "Temporal evidence validation",
    change_statistics: "Change statistics",
    area_statistics: "Area statistics",
    affected_area_statistics: "Affected-area statistics",
    flood_change_detection: "Flood change detection",
    object_detection: "Object detection",
    geospatial_postprocessing: "Geospatial post-processing",
    visual_question_answering: "Vision-language analysis"
  };
  return aliases[String(value)] || compactLabel(value);
}

function bboxOf(feature) {
  if (!feature?.geometry) return null;
  const pairs = [];
  const walk = (value) => {
    if (Array.isArray(value) && value.length >= 2 && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]))) {
      pairs.push([Number(value[0]), Number(value[1])]);
      return;
    }
    if (Array.isArray(value)) value.forEach(walk);
  };
  walk(feature.geometry.coordinates);
  if (!pairs.length) return null;
  return [
    Math.min(...pairs.map((p) => p[0])), Math.min(...pairs.map((p) => p[1])),
    Math.max(...pairs.map((p) => p[0])), Math.max(...pairs.map((p) => p[1]))
  ];
}

function executionBounds(feature) {
  const bbox = bboxOf(feature);
  if (!bbox) return null;
  let [w, s, e, n] = bbox;
  if (Math.abs(e - w) < 1e-9) { w -= .005; e += .005; }
  if (Math.abs(n - s) < 1e-9) { s -= .005; n += .005; }
  return [w, s, e, n];
}

function polygonAreaKm2(feature) {
  const ring = feature?.geometry?.type === "Polygon" ? feature.geometry.coordinates?.[0] : null;
  if (!Array.isArray(ring) || ring.length < 4) return 0;
  const meanLat = ring.reduce((sum, p) => sum + Number(p[1] || 0), 0) / ring.length;
  const kx = 111.320 * Math.cos(meanLat * Math.PI / 180);
  const ky = 110.574;
  let sum = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const [x1, y1] = [ring[i][0] * kx, ring[i][1] * ky];
    const [x2, y2] = [ring[i + 1][0] * kx, ring[i + 1][1] * ky];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

function setSystemState(label, state = "") {
  els.systemLabel.textContent = label;
  els.systemPill.classList.remove("ready", "error");
  if (state) els.systemPill.classList.add(state);
}

function setBackendStatus(message, state = "standby") {
  els.connectionStatus.textContent = message;
  els.badge.classList.remove("ready", "error", "running", "completed");
  if (state === "ready") {
    els.badge.textContent = "Ready";
    els.badge.classList.add("ready");
  } else if (state === "completed") {
    els.badge.textContent = "Complete";
    els.badge.classList.add("completed");
  } else if (state === "error") {
    els.badge.textContent = "Needs attention";
    els.badge.classList.add("error");
  } else if (state === "running") {
    els.badge.textContent = "Processing";
    els.badge.classList.add("running");
  } else {
    els.badge.textContent = "Standby";
  }
}

function workflowReset() {
  if (executionTimer) clearInterval(executionTimer);
  executionTimer = null;
  executionIndex = 0;
  els.workflow.querySelectorAll(".workflow-step").forEach((step) => step.classList.remove("active", "done", "failed"));
}

function workflowMark(stepName, cls) {
  const step = els.workflow.querySelector(`[data-step="${stepName}"]`);
  if (!step) return;
  step.classList.remove("active", "done", "failed");
  if (cls) step.classList.add(cls);
}

function markPlanningComplete() {
  ["query", "aoi", "source", "workflow"].forEach((name) => workflowMark(name, "done"));
}

function beginExecutionProgress() {
  // The current backend call is synchronous, so we only animate phases we know
  // are entered before inference. Evidence/statistics/report are marked complete
  // only after the backend returns a successful result.
  executionIndex = 0;
  workflowMark("imagery", "active");
  const preAnalysis = ["imagery", "preprocess"];
  executionTimer = setInterval(() => {
    if (executionIndex >= preAnalysis.length) {
      clearInterval(executionTimer);
      executionTimer = null;
      workflowMark("analysis", "active");
      executionIndex = 2;
      return;
    }
    const current = preAnalysis[executionIndex];
    workflowMark(current, "done");
    executionIndex += 1;
    if (executionIndex < preAnalysis.length) workflowMark(preAnalysis[executionIndex], "active");
    else workflowMark("analysis", "active");
  }, 1800);
}

function finishExecutionProgress(success) {
  if (executionTimer) clearInterval(executionTimer);
  executionTimer = null;
  if (success) {
    EXECUTION_STEPS.forEach((name) => workflowMark(name, "done"));
  } else {
    const failedStep = EXECUTION_STEPS[Math.min(executionIndex, EXECUTION_STEPS.length - 1)] || "analysis";
    workflowMark(failedStep, "failed");
  }
}

function emptyFeatureCollection() { return { type: "FeatureCollection", features: [] }; }
function sourceData(id, data) {
  const src = map?.getSource(id);
  if (src?.setData) src.setData(data);
}

function rectangleFeature(a, b) {
  const west = Math.min(a.lng, b.lng), east = Math.max(a.lng, b.lng);
  const south = Math.min(a.lat, b.lat), north = Math.max(a.lat, b.lat);
  return {
    type: "Feature", properties: { satquery_role: "aoi", draw_mode: "rectangle" },
    geometry: { type: "Polygon", coordinates: [[[west, south],[east, south],[east, north],[west, north],[west, south]]] }
  };
}

function polygonFeature(points) {
  if (points.length < 3) return null;
  const ring = points.map((p) => [p.lng, p.lat]);
  ring.push([...ring[0]]);
  return { type: "Feature", properties: { satquery_role: "aoi", draw_mode: "polygon" }, geometry: { type: "Polygon", coordinates: [ring] } };
}

function pointFeature(p) {
  return { type: "Feature", properties: { satquery_role: "aoi", draw_mode: "point" }, geometry: { type: "Point", coordinates: [p.lng, p.lat] } };
}

function persistAoi() {
  if (currentAoi) localStorage.setItem(AOI_STORAGE_KEY, JSON.stringify(currentAoi));
  else localStorage.removeItem(AOI_STORAGE_KEY);
}

function renderAoi() {
  if (!map?.isStyleLoaded()) return;
  sourceData(AOI_SOURCE, currentAoi ? { type: "FeatureCollection", features: [currentAoi] } : emptyFeatureCollection());
  if (!currentAoi) {
    els.aoiStateText.textContent = "No AOI selected";
    els.aoiSummary.innerHTML = '<div class="empty-state">Draw an AOI to start a geospatial query.</div>';
    els.mapHint.textContent = "Search a location or draw an AOI to begin.";
    return;
  }
  const bbox = bboxOf(currentAoi);
  const area = polygonAreaKm2(currentAoi);
  els.aoiStateText.textContent = "AOI ready";
  els.aoiSummary.innerHTML = `
    <div class="aoi-grid">
      <div class="aoi-card"><span>Geometry</span><strong>${safeText(currentAoi.geometry.type)}</strong></div>
      <div class="aoi-card"><span>Approx. area</span><strong>${area ? area.toFixed(area > 100 ? 0 : 2) + " km²" : "Point AOI"}</strong></div>
      <div class="aoi-card" style="grid-column:1/-1"><span>Bounds</span><strong>${bbox ? bbox.map(v => v.toFixed(4)).join(" · ") : "--"}</strong></div>
    </div>`;
  els.mapHint.textContent = "AOI ready. Ask SatQuery what you want to know about this area.";
}

function saveAoi(feature) {
  if (feature && drawMode && window.innerWidth <= 760) setMapFocus(false);
  invalidateAnalysisForInputChange("The AOI changed. Any result for the previous area was invalidated.");
  currentAoi = feature || null;
  persistAoi();
  renderAoi();
}

function invalidateAnalysisForInputChange(message) {
  const staleJobId = activeJobId;
  activeRunToken += 1;
  activeJobId = null;
  currentAnalysisId = null;
  currentPlanPayload = null;
  cancellationRequested = false;
  clearAnalysisOverlay();
  resetResultPanel();

  // Cancellation is best effort.  The token above is the correctness guard:
  // even if a worker finishes after cancellation, its response is stale.
  if (staleJobId) {
    fetch(`${API_BASE}/api/v1/map/jobs/${encodeURIComponent(staleJobId)}`, {method:"DELETE"})
      .catch(() => {});
  }
  if (message) {
    setBackendStatus(message, "standby");
    els.mapHint.textContent = message;
  }
}

function setPreview(feature) {
  sourceData(PREVIEW_SOURCE, feature ? { type: "FeatureCollection", features: [feature] } : emptyFeatureCollection());
}

function stopDrawing() {
  drawMode = null; drawStart = null; polygonPoints = [];
  setPreview(null);
  if (!map) return;
  map.getCanvas().style.cursor = "";
  map.doubleClickZoom.enable();
  els.aoiTools.querySelectorAll("[data-mode]").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-pressed", "false"); });
}

function startDrawing(mode, button) {
  if (window.innerWidth <= 760) setMapFocus(true);
  stopDrawing();
  drawMode = mode;
  if (button) button.setAttribute("aria-pressed", "true");
  if (mode === "polygon") map.doubleClickZoom.disable();
  button?.classList.add("active");
  map.getCanvas().style.cursor = "crosshair";
  els.mapHint.textContent = mode === "polygon"
    ? "Polygon mode: click vertices, then double-click to finish."
    : mode === "rectangle"
      ? "Rectangle mode: click two opposite corners."
      : "Point mode: click the location to analyze.";
}

function fitAoi() {
  const bbox = executionBounds(currentAoi);
  if (!bbox) return;
  const [w,s,e,n] = bbox;
  map.fitBounds([[w,s],[e,n]], { padding:mapPadding(), maxZoom:14, duration:850 });
}

function addCustomLayers() {
  if (!map.getSource(SAT_SOURCE)) {
    map.addSource(SAT_SOURCE, {
      type:"raster",
      tiles:["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize:256,
      maxzoom:17,
      attribution:"Imagery © Esri, Maxar, Earthstar Geographics, GIS User Community"
    });
    const firstSymbol = (map.getStyle().layers || []).find((layer) => layer.type === "symbol")?.id;
    map.addLayer({ id:SAT_LAYER, type:"raster", source:SAT_SOURCE, paint:{"raster-opacity":.94} }, firstSymbol);
  }
  if (!map.getSource(AOI_SOURCE)) map.addSource(AOI_SOURCE, { type:"geojson", data:emptyFeatureCollection() });
  if (!map.getLayer(AOI_FILL)) map.addLayer({id:AOI_FILL,type:"fill",source:AOI_SOURCE,filter:["==",["geometry-type"],"Polygon"],paint:{"fill-color":"#55e6d5","fill-opacity":.11}});
  if (!map.getLayer(AOI_LINE)) map.addLayer({id:AOI_LINE,type:"line",source:AOI_SOURCE,paint:{"line-color":"#6af4e2","line-width":2.2,"line-opacity":.95}});
  if (!map.getLayer(AOI_POINT)) map.addLayer({id:AOI_POINT,type:"circle",source:AOI_SOURCE,filter:["==",["geometry-type"],"Point"],paint:{"circle-radius":8,"circle-color":"#65ead9","circle-stroke-color":"#eafffb","circle-stroke-width":2}});
  if (!map.getSource(PREVIEW_SOURCE)) map.addSource(PREVIEW_SOURCE, { type:"geojson", data:emptyFeatureCollection() });
  if (!map.getLayer(PREVIEW_FILL)) map.addLayer({id:PREVIEW_FILL,type:"fill",source:PREVIEW_SOURCE,filter:["==",["geometry-type"],"Polygon"],paint:{"fill-color":"#55aef6","fill-opacity":.08}});
  if (!map.getLayer(PREVIEW_LINE)) map.addLayer({id:PREVIEW_LINE,type:"line",source:PREVIEW_SOURCE,paint:{"line-color":"#83c9ff","line-width":1.8,"line-dasharray":[2,2]}});
}

function setBaseLayersVisible(visible) {
  (map.getStyle()?.layers || []).forEach((layer) => {
    if ([SAT_LAYER, AOI_FILL, AOI_LINE, AOI_POINT, PREVIEW_FILL, PREVIEW_LINE, ANALYSIS_LAYER].includes(layer.id)) return;
    try { map.setLayoutProperty(layer.id, "visibility", visible ? "visible" : "none"); } catch {}
  });
}

function activateBasemapFallback(reason = "Satellite reference tiles are unavailable here.") {
  if (satelliteFallbackActive) return;
  satelliteFallbackActive = true;
  try { if (map?.getLayer(SAT_LAYER)) map.setLayoutProperty(SAT_LAYER,"visibility","none"); } catch {}
  try { setBaseLayersVisible(true); } catch {}
  [els.vectorBtn, els.satelliteBtn, els.hybridBtn].forEach((b) => b?.classList.remove("active"));
  els.vectorBtn?.classList.add("active");
  if (els.baseMapLabel) els.baseMapLabel.textContent="Global vector · satellite unavailable";
  if (els.mapHint) els.mapHint.textContent=`${reason} The display basemap is independent of SatQuery analysis imagery; AOI analysis can continue.`;
}

function recordSatelliteTileError() {
  satelliteTileErrors += 1;
  if (satelliteErrorResetTimer) clearTimeout(satelliteErrorResetTimer);
  satelliteErrorResetTimer = setTimeout(()=>{ satelliteTileErrors=0; },12000);
  if (satelliteTileErrors >= 4) activateBasemapFallback();
}

function applyBaseMode(mode) {
  baseMode = mode;
  if (mode === "satellite" || mode === "hybrid") { satelliteFallbackActive=false; satelliteTileErrors=0; }
  [els.vectorBtn, els.satelliteBtn, els.hybridBtn].forEach((b) => b.classList.remove("active"));
  if (!map?.getLayer(SAT_LAYER)) return;
  if (mode === "satellite") {
    els.satelliteBtn.classList.add("active");
    setBaseLayersVisible(false);
    map.setLayoutProperty(SAT_LAYER,"visibility","visible");
    els.baseMapLabel.textContent="Satellite imagery";
  } else if (mode === "hybrid") {
    els.hybridBtn.classList.add("active");
    setBaseLayersVisible(true);
    map.setLayoutProperty(SAT_LAYER,"visibility","visible");
    els.baseMapLabel.textContent="Hybrid imagery";
  } else {
    els.vectorBtn.classList.add("active");
    setBaseLayersVisible(true);
    map.setLayoutProperty(SAT_LAYER,"visibility","none");
    els.baseMapLabel.textContent="Global vector";
  }
}

async function searchLocation(raw) {
  const query = raw.trim();
  if (!query) { els.searchStatus.textContent = "Enter a location first."; return; }
  els.searchStatus.textContent = "Searching global gazetteer…";
  els.searchResults.innerHTML = "";
  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.searchParams.set("q",query); url.searchParams.set("format","jsonv2"); url.searchParams.set("limit","5");
  try {
    const response = await fetch(url,{headers:{Accept:"application/json"}});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const results = await response.json();
    if (!results.length) { els.searchStatus.textContent="No matching locations found."; return; }
    els.searchStatus.textContent = `${results.length} location${results.length===1?"":"s"} found.`;
    results.forEach((result) => {
      const b=document.createElement("button"); b.type="button"; b.className="search-result"; b.textContent=result.display_name;
      b.addEventListener("click",()=>{
        if (Array.isArray(result.boundingbox)) {
          const [s,n,w,e]=result.boundingbox.map(Number);
          map.fitBounds([[w,s],[e,n]],{padding:mapPadding(),duration:900,maxZoom:13});
        } else map.flyTo({center:[Number(result.lon),Number(result.lat)],zoom:10,speed:1.15});
        els.searchResults.innerHTML="";
        els.searchStatus.textContent=result.display_name;
      });
      els.searchResults.appendChild(b);
    });
  } catch (error) {
    console.error(error);
    els.searchStatus.textContent="Location search unavailable. Navigate directly on the map.";
  }
}

function planField(plan, ...keys) {
  for (const key of keys) if (plan && plan[key] != null) return plan[key];
  return null;
}


function planProfile(plan) {
  const intent = String(planField(plan,"intent","analysis_type") || "").toLowerCase();
  const rawTargets = planField(plan,"targets","target") || [];
  const targets = (Array.isArray(rawTargets) ? rawTargets : [rawTargets]).map((x)=>String(x).toLowerCase());
  const combined = `${intent} ${targets.join(" ")}`;
  if (/vegetation|ndvi|crop|forest|green/.test(combined)) return "vegetation";
  if (/built.?up|urban|construction|expansion/.test(combined)) return "urban";
  if (/flood|inundation|waterlog/.test(combined)) return "flood";
  if (/water|lake|river|reservoir|ndwi/.test(combined)) return "water";
  if (/building|structural/.test(combined)) return "building";
  return "general";
}

function decisionForProfile(profile, source) {
  const decisions = {
    vegetation:{intent:"Vegetation change",analysis:"NDVI + temporal comparison",reason:"Multispectral red and near-infrared bands provide direct vegetation-index evidence.",output:"Vegetation loss/gain map + changed area + statistics"},
    urban:{intent:"Urban expansion",analysis:"NDBI + temporal comparison",reason:"Multispectral NIR and SWIR bands support built-up surface-change analysis.",output:"Built-up spectral increase map + area statistics"},
    flood:{intent:"Flood extent",analysis:"Executable water-transition temporal comparison",reason:"SatQuery uses the imagery source resolved by the executable plan. SAR is shown only when a validated SAR executor is actually enabled; otherwise optical water-index evidence is labeled honestly.",output:"Flood/water increase layer + affected-area statistics"},
    water:{intent:"Water change",analysis:"NDWI water-class transition + temporal comparison",reason:"Sentinel-2 green and near-infrared bands support NDWI-based surface-water extraction at 10 m resolution.",output:"Water gain/loss map + changed-area statistics"},
    building:{intent:"Building change",analysis:"Structural change detection",reason:"Structural evidence is prioritized for building-level temporal change.",output:"Structural change mask + supporting temporal evidence"},
    general:{intent:"Temporal change",analysis:"Multi-evidence change reasoning",reason:"SatQuery combines available spectral, semantic and visual evidence.",output:"Change evidence layer + statistics + explanation"}
  };
  const d = decisions[profile] || decisions.general;
  return {...d, source:source || "Auto-selected Earth observation source"};
}

function updateWorkflowLabels(plan) {
  const profile = planProfile(plan);
  const years = planField(plan,"years","requested_years") || [];
  const period = Array.isArray(years) && years.length >= 2 ? `${years[0]}–${years[1]}` : "selected dates";
  const analysisStep = els.workflow.querySelector('[data-step="analysis"]');
  const compareStep = els.workflow.querySelector('[data-step="compare"]');
  const reportStep = els.workflow.querySelector('[data-step="report"]');
  const labels = {vegetation:"Run NDVI analysis",urban:"Run NDBI analysis",flood:"Run flood/water analysis",water:"Run water-index analysis",building:"Run structural analysis",general:"Run GeoAI analysis"};
  if (analysisStep) analysisStep.querySelector("strong").textContent = labels[profile] || labels.general;
  if (compareStep) compareStep.querySelector("strong").textContent = `Compare ${period}`;
  if (reportStep) reportStep.querySelector("strong").textContent = "Generate answer & report";
}

function renderPlan(payload) {
  const plan=payload?.plan||{};
  const tools=planField(plan,"required_tools","tools")||[];
  const years=planField(plan,"years","requested_years")||[];
  const targets=planField(plan,"targets","target")||[];
  const confidence=Number(planField(plan,"routing_confidence","confidence"));
  const confidenceText=Number.isFinite(confidence)?pct(confidence):"Rule based";
  const source=planField(plan,"recommended_source","source","satellite_source")||"Auto-select";
  const sourceDetails=planField(plan,"source_details")||{};
  const intent=planField(plan,"intent","analysis_type")||"unknown";
  const yearsText=Array.isArray(years)&&years.length?years.join(" → "):"Auto from query";
  const targetsText=Array.isArray(targets)&&targets.length?targets.map(compactLabel).join(", "):"General scene";
  const toolsArray=Array.isArray(tools)?tools:[tools];
  const profile = planProfile(plan);
  const decision = decisionForProfile(profile, sourceDetails.platform || source);
  const inputConfiguration = plan.input_configuration || null;
  const inputText = inputConfiguration
    ? `${inputConfiguration.image_count || 0} input${inputConfiguration.image_count === 1 ? "" : "s"} · ${compactLabel(inputConfiguration.kind)}`
    : "AOI temporal imagery";
  const executionSteps = Array.isArray(plan.execution_steps) ? plan.execution_steps : [];
  updateWorkflowLabels(plan);

  els.selectedSourceName.textContent = sourceDetails.platform || source;
  const sourceMeta = [sourceDetails.sensor, sourceDetails.resolution].filter(Boolean).join(" · ");
  els.selectedSourceMeta.textContent = sourceMeta || "Selected automatically from the analysis requirements.";

  els.backendPlan.innerHTML=`
    <div class="plan-line"><span>Query detected</span><strong>${safeText(decision.intent)}</strong></div>
    <div class="plan-line"><span>Selected data</span><strong>${safeText(sourceDetails.platform || source)}</strong></div>
    <div class="plan-line"><span>Selected analysis</span><strong>${safeText(decision.analysis)}</strong></div>
    <div class="plan-line"><span>Input configuration</span><strong>${safeText(inputText)}</strong></div>
    <div class="plan-line"><span>Reason</span><strong>${safeText(decision.reason)}</strong></div>
    <div class="plan-line"><span>Time range</span><strong>${safeText(yearsText)}</strong></div>
    <div class="plan-line"><span>Expected output</span><strong>${safeText(decision.output)}</strong></div>
    <div class="plan-line"><span>Workflow confidence</span><strong>${safeText(confidenceText)}</strong></div>
    <div class="plan-tool-list">${toolsArray.map((tool)=>`<span class="plan-tool">${safeText(humanTool(tool))}</span>`).join("")}</div>
    ${executionSteps.length ? `<div class="plan-line"><span>Validated steps</span><strong>${executionSteps.length} dependency-ordered stages</strong></div>` : ""}`;
}

function resetResultPanel() {
  document.body.classList.remove("analysis-ready");
  currentJob = null;
  overlayArtifact = null;
  els.resultSection.classList.add("hidden");
  els.emptyResultSection.classList.remove("hidden");
  els.mapAnswer.textContent = "No completed analysis loaded yet.";
  els.evidenceGallery.innerHTML = "";
  els.evidenceGallery.classList.add("hidden");
  els.viewEvidenceBtn.disabled = true;
  els.compareScenesBtn.disabled = true;
  els.generateReportBtn.disabled = true;
  els.comparisonViewer.classList.add("hidden");
  els.validationCard?.classList.add("hidden");
  els.mapComparison.classList.add("hidden");
  els.activeLayerLabel.textContent = "No analysis layer";
  els.metricArea.textContent = "—";
  els.metricShare.textContent = "—";
  els.metricPrimary.textContent = "—";
  els.metricSatellite.textContent = "—";
  els.metricConfidence.textContent = "—";
  els.metricResultQuality.textContent = "—";
  els.qualityRouting.textContent = "—";
  els.qualityData.textContent = "—";
  els.qualityEvidence.textContent = "—";
  els.technicalEvidence.textContent = "No technical evidence loaded.";
  if (els.analysisLegend) {
    els.analysisLegend.innerHTML = "";
    els.analysisLegend.classList.add("hidden");
  }
}

function clearAnalysisOverlay() {
  if (!map) return;
  if (map.getLayer(ANALYSIS_LAYER)) map.removeLayer(ANALYSIS_LAYER);
  if (map.getSource(ANALYSIS_SOURCE)) map.removeSource(ANALYSIS_SOURCE);
  if (map.getLayer(TEMPORAL_CHANGE_LINE)) map.removeLayer(TEMPORAL_CHANGE_LINE);
  if (map.getLayer(TEMPORAL_CHANGE_FILL)) map.removeLayer(TEMPORAL_CHANGE_FILL);
  if (map.getSource(TEMPORAL_CHANGE_SOURCE)) map.removeSource(TEMPORAL_CHANGE_SOURCE);
  overlayArtifact = null;
}

function artifactUrl(jobId, artifact) {
  const encoded = String(artifact).split("/").map(encodeURIComponent).join("/");
  return `/api/map-artifacts/${encodeURIComponent(jobId)}/${encoded}`;
}

function collectionLabel(collectionId) {
  const value = String(collectionId || "").toUpperCase();
  if (value.includes("COPERNICUS/S2") || value.includes("SENTINEL-2")) return "Sentinel-2 MSI";
  if (value.includes("COPERNICUS/S1") || value.includes("SENTINEL-1")) return "Sentinel-1 SAR";
  if (value.includes("LANDSAT 5") || value.includes("LT05") || value.includes("LANDSAT/LT05")) return "Landsat 5 TM";
  if (value.includes("LANDSAT 7") || value.includes("LE07") || value.includes("LANDSAT/LE07")) return "Landsat 7 ETM+";
  if (value.includes("LANDSAT 8") || value.includes("LC08") || value.includes("LANDSAT/LC08")) return "Landsat 8 OLI";
  if (value.includes("LANDSAT 9") || value.includes("LC09") || value.includes("LANDSAT/LC09")) return "Landsat 9 OLI-2";
  if (value.includes("LANDSAT")) return "Landsat multispectral";
  return collectionId ? String(collectionId) : "Earth observation imagery";
}

function deriveActualSource(job) {
  const imagery = job?.result?.provenance?.imagery;
  if (Array.isArray(imagery) && imagery.length) {
    const labels = [...new Set(imagery.map((x)=>collectionLabel(x.collection_id)))];
    const scales = [...new Set(imagery.map((x)=>x.scale_m).filter((x)=>x != null))];
    return {
      label: labels.join(" / "),
      meta: scales.length ? `${scales.join("/")} m · ${imagery.reduce((s,x)=>s+(Number(x.image_count)||0),0)} source scenes` : "Earth Engine temporal composite"
    };
  }
  const planned = currentPlanPayload?.plan?.source_details;
  return { label: planned?.platform || currentPlanPayload?.plan?.recommended_source || "Earth observation imagery", meta: [planned?.sensor, planned?.resolution].filter(Boolean).join(" · ") };
}

function numberOrNull(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function signed(value, digits = 3) {
  const n = numberOrNull(value);
  if (n == null) return "—";
  const prefix = n > 0 ? "+" : "";
  return `${prefix}${n.toFixed(digits)}`;
}

function imageryStats(job) {
  const imagery = job?.result?.statistics?.imagery || {};
  return { before: imagery.before || {}, after: imagery.after || {} };
}

function analysisProfile(job) {
  const parsed = job?.plan?.parsed || {};
  const query = String(parsed.query || "").toLowerCase();
  const targets = Array.isArray(parsed.targets) ? parsed.targets.map((x)=>String(x).toLowerCase()) : [];
  const has = (...tokens) => tokens.some((token)=>targets.some((t)=>t.includes(token)) || query.includes(token));

  if (has("vegetation", "ndvi", "crop", "forest", "green cover")) {
    return { key:"vegetation", title:"Vegetation Change Analysis", primaryEvidence:"NDVI spectral evidence" };
  }
  if (has("built_up", "built-up", "built up", "urban", "construction", "expansion")) {
    return { key:"urban", title:"Urban Expansion Analysis", primaryEvidence:"NDBI built-up spectral evidence" };
  }
  if (has("flooded_area", "flood", "inundation", "waterlogging")) {
    return { key:"flood", title:"Flood Extent Analysis", primaryEvidence:"Executable water-transition evidence" };
  }
  if (has("water", "lake", "river", "reservoir", "ndwi")) {
    return { key:"water", title:"Water Change Analysis", primaryEvidence:"Water-index evidence" };
  }
  if (has("building", "buildings", "structural")) {
    return { key:"building", title:"Building Change Analysis", primaryEvidence:"Structural change evidence" };
  }
  return { key:"general", title:"Temporal Change Analysis", primaryEvidence:"Multi-source temporal evidence" };
}

function targetMetric(job, profile = analysisProfile(job)) {
  const {before, after} = imageryStats(job);
  const change = job?.result?.statistics?.change || {};

  if (profile.key === "vegetation") {
    const b = numberOrNull(before.ndvi), a = numberOrNull(after.ndvi);
    if (b != null && a != null) return { label:"NDVI change", value:signed(a-b), before:b, after:a, unit:"NDVI" };
  }
  if (profile.key === "urban") {
    const b = numberOrNull(before.ndbi), a = numberOrNull(after.ndbi);
    if (b != null && a != null) return { label:"NDBI change", value:signed(a-b), before:b, after:a, unit:"NDBI" };
  }
  if (profile.key === "water" || profile.key === "flood") {
    const b = numberOrNull(before.water_fraction), a = numberOrNull(after.water_fraction);
    if (b != null && a != null) return { label:"Water extent change", value:`${signed((a-b)*100,2)} pp`, before:b, after:a, unit:"fraction" };
  }
  const structural = numberOrNull(change.structural_change_percentage);
  if (structural != null) return { label:"Structural change", value:`${structural.toFixed(2)}%` };
  const rgbPct = numberOrNull(change.rgb_pixels_difference_gt_20_pct);
  if (rgbPct != null) return { label:"RGB pixel difference", value:`${rgbPct.toFixed(1)}%` };
  const evidenceCount = Array.isArray(job?.result?.evidence) ? job.result.evidence.length : 0;
  return { label:"Evidence artifacts", value:`${evidenceCount}` };
}

function taskLayerStats(job) {
  const layer = job?.result?.statistics?.task_layer;
  return layer && typeof layer === "object" ? layer : null;
}

function taskLayerSentence(job, profile) {
  const layer = taskLayerStats(job);
  if (!layer) return "";
  const changed = numberOrNull(layer.changed_percentage);
  const affected = numberOrNull(layer.affected_area_km2);
  const pos = numberOrNull(layer.positive_percentage);
  const neg = numberOrNull(layer.negative_percentage);
  if (changed == null) return "";
  const areaText = affected == null ? "" : ` (about ${affected.toFixed(2)} km² of the execution AOI)`;
  if (profile.key === "vegetation") {
    return ` Thresholded NDVI change marks ${changed.toFixed(1)}% of valid pixels as meaningful change${areaText}; loss ${neg?.toFixed(1) ?? "—"}%, gain ${pos?.toFixed(1) ?? "—"}%.`;
  }
  if (profile.key === "urban") {
    return ` Thresholded NDBI change marks ${changed.toFixed(1)}% of valid pixels as meaningful built-up spectral change${areaText}; increase ${pos?.toFixed(1) ?? "—"}%, decrease ${neg?.toFixed(1) ?? "—"}%.`;
  }
  if (profile.key === "water" || profile.key === "flood") {
    return ` Thresholded water-index change marks ${changed.toFixed(1)}% of valid pixels as meaningful change${areaText}; increase ${pos?.toFixed(1) ?? "—"}%, decrease ${neg?.toFixed(1) ?? "—"}%.`;
  }
  return "";
}

function focusedSummary(job, profile, metric) {
  const {before, after} = imageryStats(job);
  const years = job?.plan?.parsed?.years || [];
  const period = years.length >= 2 ? `${years[0]} and ${years[1]}` : "the selected dates";

  if (profile.key === "vegetation" && metric.before != null && metric.after != null) {
    const delta = metric.after - metric.before;
    const direction = delta < -0.04 ? "decreased" : delta > 0.04 ? "increased" : "remained broadly stable";
    return `Vegetation signal ${direction} between ${period}. Mean NDVI changed from ${metric.before.toFixed(3)} to ${metric.after.toFixed(3)} (${signed(delta)}).${taskLayerSentence(job, profile)} RGB/structural layers shown on the map are supporting visual evidence; the NDVI values and NDVI change layer are the primary vegetation evidence.`;
  }
  if (profile.key === "urban" && metric.before != null && metric.after != null) {
    const delta = metric.after - metric.before;
    const ndviBefore = numberOrNull(before.ndvi), ndviAfter = numberOrNull(after.ndvi);
    const context = ndviBefore != null && ndviAfter != null ? ` NDVI changed from ${ndviBefore.toFixed(3)} to ${ndviAfter.toFixed(3)} (${signed(ndviAfter-ndviBefore)}), which is supporting land-cover context.` : "";
    const interpretation = delta > 0.04 ? "is consistent with increased built-up spectral response" : delta < -0.04 ? "does not indicate an increase in built-up spectral response" : "shows only a small built-up spectral shift";
    return `Built-up spectral signal (NDBI) changed from ${metric.before.toFixed(3)} to ${metric.after.toFixed(3)} (${signed(delta)}) between ${period}; this ${interpretation}.${context}${taskLayerSentence(job, profile)} This is evidence of surface-development change, not a direct building-count measurement.`;
  }
  if ((profile.key === "water" || profile.key === "flood") && metric.before != null && metric.after != null) {
    const beforePct = metric.before * 100, afterPct = metric.after * 100;
    return `Estimated water-covered fraction changed from ${beforePct.toFixed(2)}% to ${afterPct.toFixed(2)}% between ${period} (${signed(afterPct-beforePct,2)} percentage points).${taskLayerSentence(job, profile)} The primary map layer uses NDWI water/non-water transitions when available; RGB difference remains supporting visual evidence.`;
  }
  return job?.result?.answer || "Analysis completed.";
}

function evidenceLabel(path) {
  const key = String(path || "").toLowerCase();
  if (key.includes("before_after_change_comparison")) return "Before | After | Change";
  if (key.includes("before_after_comparison")) return "Before | After";
  if (/scene_\d{4}_rgb/.test(key)) { const m=key.match(/scene_(\d{4})_rgb/); return `${m?.[1] || "Source"} Scene`; }
  if (key.includes("change_overlay_preview")) return "Detected Change Overlay";
  if (key.includes("ndvi_change_layer")) return "NDVI Vegetation Change";
  if (key.includes("ndbi_change_layer")) return "NDBI Built-up Change";
  if (key.includes("water_change_layer")) return "Water-index Change";
  if (key.includes("bitemporal_rgb_difference")) return "RGB Temporal Difference";
  if (key.includes("changeformer_after_overlay")) return "After-scene Structural Overlay";
  if (key.includes("changeformer_before_overlay")) return "Before-scene Structural Overlay";
  if (key.includes("changeformer_change_mask")) return "Structural Change Mask";
  if (key.includes("changeformer_comparison")) return "Structural Before/After Comparison";
  if (key.includes("ndvi") && key.includes("change")) return "NDVI Change Layer";
  if (key.includes("ndbi") && key.includes("change")) return "NDBI Built-up Change Layer";
  if (key.includes("ndwi") || key.includes("water")) return "Water Change Layer";
  const name = String(path).split("/").pop()?.replace(/\.[^.]+$/, "") || "Evidence";
  return compactLabel(name.replace(/^outputs[\\_-]?/, ""));
}

function evidenceRole(path) {
  const key = String(path || "").toLowerCase();
  if (key.includes("before_after_change_comparison")) return "Before / after / change";
  if (key.includes("before_after_comparison")) return "Before / after source imagery";
  if (/scene_\d{4}_rgb/.test(key)) return "Source imagery";
  if (key.includes("change_overlay_preview")) return "Detected-change preview";
  if (key.includes("ndvi")) return "Primary vegetation evidence";
  if (key.includes("ndbi")) return "Primary built-up evidence";
  if (key.includes("ndwi") || key.includes("water")) return "Primary water evidence";
  if (key.includes("changeformer")) return "Supporting structural evidence";
  if (key.includes("rgb_difference")) return "Supporting visual evidence";
  return "Analysis evidence";
}

function validationStats(job) {
  const v = job?.result?.statistics?.validation;
  return v && typeof v === "object" ? v : null;
}

function renderValidation(job) {
  const v = validationStats(job);
  if (!v || !els.validationCard) {
    els.validationCard?.classList.add("hidden");
    return;
  }
  const alignment = v.alignment || {};
  const sensitivity = v.threshold_sensitivity || {};
  const coverage = numberOrNull(v.valid_coverage_pct);
  els.validationGrade.textContent = v.quality || "Unavailable";
  els.validationAlignment.textContent = alignment.label || compactLabel(alignment.status || "Unavailable");
  els.validationCoverage.textContent = coverage == null ? "Unavailable" : `${coverage.toFixed(1)}%`;
  els.validationSensitivity.textContent = sensitivity.status ? compactLabel(sensitivity.status) : "Unavailable";
  els.validationSensor.textContent = v.sensor_consistency || "Unavailable";
  const sourceConsistency = job?.result?.provenance?.source_consistency || {};
  els.validationSourcePlan.textContent = sourceConsistency.matched === true ? "Matched" : (sourceConsistency.matched === false ? "Changed · review" : "Not applicable");
  const flags = Array.isArray(v.flags) ? [...v.flags] : [];
  if (sourceConsistency.matched === false && sourceConsistency.note) flags.unshift(sourceConsistency.note);
  els.validationFlags.innerHTML = flags.slice(0,4).map((x)=>`<div class="flag">• ${safeText(x)}</div>`).join("");
  els.validationCard.classList.remove("hidden");
}

function qualityStatus(job) {
  if (job?.status !== "completed") return { label:"Failed", cls:"error" };
  const validation = validationStats(job);
  if (validation) {
    if (validation.status === "review") return { label:"Needs review", cls:"warn" };
    if (validation.status === "moderate") return { label:"Moderate", cls:"warn" };
    if (validation.status === "good") return { label:"Validated", cls:"" };
  }
  const stats = job?.result?.statistics?.change || {};
  const answer = String(job?.result?.answer || "").toLowerCase();
  if (stats.semantic_reliable === false || answer.includes("quality gate") || answer.includes("supporting evidence only")) return { label:"Quality-gated", cls:"warn" };
  return { label:"Completed", cls:"" };
}

function dataQualityLabel(job) {
  const imagery = job?.result?.provenance?.imagery;
  if (Array.isArray(imagery) && imagery.length >= 2) {
    const sources = [...new Set(imagery.map((x)=>String(x.collection_id || "")))];
    return sources.length === 1 ? "Same sensor" : "Harmonized";
  }
  const score = numberOrNull(job?.result?.statistics?.change?.semantic_quality_score);
  return score == null ? "Available" : pct(score);
}

function orderedEvidence(job) {
  const evidence = (job?.result?.evidence || []).filter((p)=>/\.(png|jpe?g)$/i.test(p));
  const preferred = preferredEvidence(job);
  const rank=(p)=>{ const k=String(p).toLowerCase(); if(p===preferred)return 0; if(k.includes("before_after_change_comparison")||k.includes("before_after_comparison"))return 1; if(/scene_\d{4}_rgb/.test(k))return 2; if(evidenceRole(p).startsWith("Primary"))return 3; if(k.includes("rgb_difference"))return 4; return 5; };
  return [...evidence].sort((a,b)=>rank(a)-rank(b)||a.localeCompare(b));
}

function renderEvidenceGallery(job) {
  const evidence = orderedEvidence(job);
  if (!evidence.length) {
    els.evidenceGallery.innerHTML = "";
    els.evidenceGallery.classList.add("hidden");
    return;
  }
  els.evidenceGallery.innerHTML = evidence.slice(0,6).map((path)=>`
    <button type="button" class="evidence-item" data-artifact="${safeText(path)}" title="${safeText(evidenceRole(path))}">
      <img src="${safeText(artifactUrl(job.job_id,path))}" alt="${safeText(evidenceLabel(path))}">
      <em>${safeText(evidenceRole(path))}</em>
      <span>${safeText(evidenceLabel(path))}</span>
    </button>`).join("");
  els.evidenceGallery.classList.remove("hidden");
  els.evidenceGallery.querySelectorAll(".evidence-item").forEach((button)=>button.addEventListener("click",()=>showEvidenceOnMap(job,button.dataset.artifact)));
}

function preferredEvidence(job) {
  const evidence = (job?.result?.evidence || []).filter((p)=>/\.(png|jpe?g)$/i.test(p));
  const profile = analysisProfile(job);
  const priorityByTarget = {
    vegetation:["ndvi_change","ndvi","rgb_difference","comparison","after_overlay","overlay","change_mask"],
    urban:["ndbi_change","built_up","after_overlay","comparison","rgb_difference","change_mask","overlay"],
    flood:["flood","water","ndwi","rgb_difference","comparison","overlay","change_mask"],
    water:["water","ndwi","rgb_difference","comparison","overlay","change_mask"],
    building:["change_mask","after_overlay","comparison","rgb_difference","overlay"],
    general:["after_overlay","comparison","rgb_difference","change_mask","overlay"]
  };
  for (const token of priorityByTarget[profile.key] || priorityByTarget.general) {
    const found = evidence.find((p)=>p.toLowerCase().includes(token));
    if (found) return found;
  }
  return evidence[0] || null;
}

function renderAnalysisLegend(job, artifact) {
  if (!els.analysisLegend) return;
  const layer = taskLayerStats(job);
  const key = String(artifact || "").toLowerCase();
  const isTaskLayer = key.includes("ndvi_change_layer") || key.includes("ndbi_change_layer") || key.includes("water_change_layer");
  if (!layer || !isTaskLayer || !Array.isArray(layer.legend)) {
    els.analysisLegend.innerHTML = "";
    els.analysisLegend.classList.add("hidden");
    return;
  }

  const rows = layer.legend.map((entry) => {
    const rgba = Array.isArray(entry.rgba) ? entry.rgba : [148,163,184,80];
    const alpha = (Number(rgba[3] ?? 255) / 255).toFixed(2);
    const bg = `rgba(${Number(rgba[0])||0},${Number(rgba[1])||0},${Number(rgba[2])||0},${alpha})`;
    return `<div class="analysis-legend-row"><span class="analysis-legend-swatch" style="background:${bg}"></span><span>${safeText(entry.label || "Change")}</span></div>`;
  }).join("");

  const threshold = numberOrNull(layer.threshold);
  const affected = numberOrNull(layer.affected_area_km2);
  const coverage = numberOrNull(layer.analysis_coverage_pct);
  const noteParts = [];
  if (layer.method === "water_class_transition") noteParts.push("NDWI water/non-water transition");
  else if (threshold != null) noteParts.push(`Threshold ±${threshold.toFixed(3)} ${String(layer.metric || "").toUpperCase()}`);
  if (affected != null) noteParts.push(`Approx. changed area ${affected.toFixed(2)} km²`);
  if (coverage != null) noteParts.push(`Valid coverage ${coverage.toFixed(0)}%`);
  const sensitivity = layer.threshold_sensitivity?.status;
  if (sensitivity) noteParts.push(`Threshold sensitivity ${compactLabel(sensitivity)}`);
  els.analysisLegend.innerHTML = `
    <div class="analysis-legend-title">${safeText(evidenceLabel(artifact))}</div>
    ${rows}
    <div class="analysis-legend-note">${safeText(noteParts.join(" · ") || layer.note || "Task-specific spectral change evidence")}</div>`;
  els.analysisLegend.classList.remove("hidden");
}

function evidenceBounds(job, artifact) {
  const records = job?.result?.provenance?.evidence_identity;
  const record = Array.isArray(records) ? records.find((item) => item.analysis_id === job.job_id && item.path === artifact) : null;
  const validBounds = (item) => {
    const b = item?.bounds;
    return item?.georeferenced && String(item.crs || "").toUpperCase() === "EPSG:4326"
      && Array.isArray(b) && b.length === 4 && b.every((v) => typeof v === "number" && Number.isFinite(v))
      && b[0] >= -180 && b[2] <= 180 && b[1] >= -90 && b[3] <= 90 && b[0] < b[2] && b[1] < b[3] ? b : null;
  };
  if (!record) return null;
  const direct = validBounds(record);
  if (direct) return direct;
  // Compatibility for existing GEE jobs: the scene renderer preserves the
  // complete RGB GeoTIFF extent. Require a unique same-job, same-year source.
  const scene = /^outputs\/evidence\/scene_(\d{4})_rgb\.png$/.exec(artifact);
  if (scene) {
    const sources = records.filter((item) => item.analysis_id === job.job_id
      && new RegExp(`^imagery/gee_${scene[1]}_[^/]+\\.tif$`).test(item.path));
    const transform = sources[0]?.transform;
    if (sources.length === 1 && Array.isArray(transform)
        && transform[0] > 0 && transform[4] < 0 && transform[1] === 0 && transform[3] === 0) {
      return validBounds(sources[0]);
    }
  }
  return null;
}

function showEvidenceOnMap(job, artifact) {
  if (!job || job.job_id !== currentAnalysisId) return;
  const identities = job?.result?.provenance?.evidence_identity;
  if (Array.isArray(identities) && !identities.some((item) => item.analysis_id === currentAnalysisId && item.path === artifact)) return;
  if (!artifact) return;
  if (artifact === comparisonData(job)?.comparison_artifact) { openComparison(); return; }
  // The composited comparison preview may have resampled pixels; use its
  // authoritative change layer for geographic display instead.
  if (artifact === comparisonData(job)?.change_artifact) {
    const layerArtifact = taskLayerStats(job)?.artifact;
    if (layerArtifact && evidenceBounds(job, layerArtifact)) artifact = layerArtifact;
  }
  const bbox = evidenceBounds(job, artifact);
  if (!bbox) { openEvidencePreview(job, artifact); return; }
  if (!map?.isStyleLoaded()) {
    els.mapHint.textContent = "Map is loading. Please select the image again in a moment.";
    return;
  }
  const [w,s,e,n] = bbox;
  clearAnalysisOverlay();
  map.addSource(ANALYSIS_SOURCE, {
    type:"image",
    url:artifactUrl(job.job_id,artifact),
    coordinates:[[w,n],[e,n],[e,s],[w,s]]
  });
  const taskKey=String(artifact).toLowerCase();
  const primaryTask=taskKey.includes("ndvi_change_layer")||taskKey.includes("ndbi_change_layer")||taskKey.includes("water_change_layer");
  map.addLayer({
    id:ANALYSIS_LAYER,
    type:"raster",
    source:ANALYSIS_SOURCE,
    paint:{"raster-opacity":primaryTask ? .8 : 1,"raster-fade-duration":0}
  }, map.getLayer(AOI_LINE) ? AOI_LINE : undefined);
  overlayArtifact = artifact;
  els.analysisLayerToggle.checked = true;
  els.activeLayerLabel.textContent = evidenceLabel(artifact);
  els.mapHint.textContent = `${primaryTask ? "PRIMARY DETECTED-CHANGE LAYER" : evidenceRole(artifact)}: ${evidenceLabel(artifact)}`;
  renderAnalysisLegend(job, artifact);
  document.querySelectorAll(".evidence-item, .comparison-card").forEach((button) => {
    const selected = button.dataset.artifact === artifact
      || (primaryTask && button.dataset.artifact === comparisonData(job)?.change_artifact);
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function openEvidencePreview(job, artifact) {
  const dialog = $("evidencePreview");
  $("evidencePreviewTitle").textContent = evidenceLabel(artifact);
  $("evidencePreviewImg").src = artifactUrl(job.job_id, artifact);
  $("evidencePreviewImg").alt = evidenceLabel(artifact);
  dialog.showModal();
}

function showTemporalChangePolygons(job) {
  const temporal = job?.result?.statistics?.temporal;
  const collection = temporal?.change_polygons;
  if (!collection || collection.type !== "FeatureCollection" || !Array.isArray(collection.features) || !map?.isStyleLoaded()) return false;
  clearAnalysisOverlay();
  map.addSource(TEMPORAL_CHANGE_SOURCE, {type:"geojson", data:collection});
  map.addLayer({id:TEMPORAL_CHANGE_FILL,type:"fill",source:TEMPORAL_CHANGE_SOURCE,paint:{"fill-color":"#f97316","fill-opacity":.28}});
  map.addLayer({id:TEMPORAL_CHANGE_LINE,type:"line",source:TEMPORAL_CHANGE_SOURCE,paint:{"line-color":"#fed7aa","line-width":2.2,"line-opacity":.95}});
  overlayArtifact = "temporal_change_polygons";
  els.analysisLayerToggle.checked = true;
  els.activeLayerLabel.textContent = "Detected temporal change polygons";
  els.mapHint.textContent = "DETECTED TEMPORAL CHANGE: map coordinates are transformed from the common analysis grid to WGS84.";
  if (collection.features.length) {
    const bounds = collection.features.map(bboxOf).filter(Boolean).reduce((all, box) => all ? [Math.min(all[0],box[0]),Math.min(all[1],box[1]),Math.max(all[2],box[2]),Math.max(all[3],box[3])] : box, null);
    if (bounds) map.fitBounds([[bounds[0],bounds[1]],[bounds[2],bounds[3]]], {padding:mapPadding(), maxZoom:15, duration:650});
  }
  return true;
}


function headlineMetrics(job, profile) {
  const layer = taskLayerStats(job);
  if (!layer) return {areaLabel:"Detected-change area",area:"Unavailable",shareLabel:"Changed share",share:"Unavailable"};
  let area = numberOrNull(layer.affected_area_km2), share = numberOrNull(layer.changed_percentage);
  let areaLabel = "Detected-change area", shareLabel = "Changed share";
  if (profile.key === "urban") { area = numberOrNull(layer.positive_area_km2); share = numberOrNull(layer.positive_percentage); areaLabel="Built-up increase evidence"; shareLabel="Analyzed area showing increase"; }
  if (profile.key === "vegetation") { area = numberOrNull(layer.negative_area_km2); share = numberOrNull(layer.negative_percentage); areaLabel="Vegetation-decrease evidence"; shareLabel="Analyzed area showing decrease"; }
  if (profile.key === "water" || profile.key === "flood") { area = numberOrNull(layer.positive_area_km2); share = numberOrNull(layer.positive_percentage); areaLabel=profile.key === "flood" ? "New inundation area" : "Water gain area"; shareLabel="Analyzed area showing gain"; }
  return {areaLabel,area:area == null ? "Unavailable" : `~${area.toFixed(2)} km²`,shareLabel,share:share == null ? "Unavailable" : `${share.toFixed(1)}%`};
}

function resultQualityText(job) {
  const validation = validationStats(job);
  if (validation?.quality) return validation.quality;
  const q = qualityStatus(job);
  if (q.cls === "warn") return "Quality-gated";
  if (q.cls === "error") return "Unavailable";
  const layer = taskLayerStats(job);
  const data = dataQualityLabel(job);
  if (layer && data === "Same sensor") return "Good";
  if (layer && data === "Harmonized") return "Moderate";
  return layer ? "Usable" : "Supporting evidence only";
}

function comparisonData(job) {
  const c = job?.result?.statistics?.comparison;
  if (!c || !c.before_artifact || !c.after_artifact) return null;
  return c;
}

function renderComparison(job) {
  const c = comparisonData(job);
  if (!c) {
    els.comparisonViewer.classList.add("hidden");
    els.compareScenesBtn.disabled = true;
    return;
  }
  const beforeUrl = artifactUrl(job.job_id,c.before_artifact);
  const afterUrl = artifactUrl(job.job_id,c.after_artifact);
  const hasChange = Boolean(c.change_artifact);
  const changeUrl = hasChange ? artifactUrl(job.job_id,c.change_artifact) : "";

  els.compareBeforeImg.src=beforeUrl;
  els.compareAfterImg.src=afterUrl;
  els.compareBeforeLabel.textContent = `${c.before_year || "Before"}`;
  els.compareAfterLabel.textContent = `${c.after_year || "After"}`;
  els.comparisonViewer.querySelector('[data-comparison="before"]').dataset.artifact=c.before_artifact;
  els.comparisonViewer.querySelector('[data-comparison="after"]').dataset.artifact=c.after_artifact;

  els.mapBeforeImg.src=beforeUrl;
  els.mapAfterImg.src=afterUrl;
  els.mapBeforeLabel.textContent=`${c.before_year || "Before"}`;
  els.mapAfterLabel.textContent=`${c.after_year || "After"}`;

  if (hasChange) {
    els.compareChangeImg.src=changeUrl;
    els.mapChangeImg.src=changeUrl;
    els.comparisonChangeCard.dataset.artifact=c.change_artifact;
    els.comparisonChangeCard.classList.remove("unavailable");
    els.mapChangeFigure.classList.remove("unavailable");
    els.comparisonChangeLabel.textContent="Detected change";
    els.mapChangeCaption.textContent="Detected change";
    els.comparisonHeading.textContent="Before · After · Detected Change";
    els.mapComparisonTitle.textContent=`${c.before_year || "Before"} · ${c.after_year || "After"} · Detected Change`;
  } else {
    els.compareChangeImg.removeAttribute("src");
    els.mapChangeImg.removeAttribute("src");
    delete els.comparisonChangeCard.dataset.artifact;
    els.comparisonChangeCard.classList.add("unavailable");
    els.mapChangeFigure.classList.add("unavailable");
    els.comparisonChangeLabel.textContent="Change evidence unavailable";
    els.mapChangeCaption.textContent="Change evidence unavailable";
    els.comparisonHeading.textContent="Before · After · Change evidence unavailable";
    els.mapComparisonTitle.textContent=`${c.before_year || "Before"} · ${c.after_year || "After"} · Change evidence unavailable`;
  }

  els.comparisonViewer.classList.remove("hidden");
  els.compareScenesBtn.disabled=false;
  els.comparisonViewer.querySelectorAll(".comparison-card").forEach((button)=>{
    button.onclick=()=>{ const artifact=button.dataset.artifact; if(artifact) showEvidenceOnMap(job,artifact); };
  });
}

function openComparison() {
  if (!currentJob || !comparisonData(currentJob)) return;
  els.mapComparison.classList.remove("hidden");
}

function renderPredictionStatistics(job, profile) {
  const layer = taskLayerStats(job);
  const readPercent = (value) => value == null || value === "" ? null :
    (Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 100 ? Number(value) : null);
  const gain = readPercent(layer?.positive_percentage);
  const loss = readPercent(layer?.negative_percentage);
  const changed = readPercent(layer?.changed_percentage);
  const stable = changed == null ? null : 100 - changed;
  const format = (value) => value == null ? "Unavailable" : `${value.toFixed(1)}%`;
  $("predictionGain").textContent = format(gain);
  $("predictionLoss").textContent = format(loss);
  $("predictionStable").textContent = format(stable);
  const area = (value) => value == null || !Number.isFinite(Number(value)) || Number(value) < 0
    ? "" : `${Math.round(Number(value) * 1e6).toLocaleString()} m²`;
  $("predictionGainArea").textContent = area(layer?.positive_area_km2);
  $("predictionLossArea").textContent = area(layer?.negative_area_km2);
  $("predictionStableArea").textContent = area(layer?.valid_area_km2 != null && layer?.affected_area_km2 != null
    ? layer.valid_area_km2 - layer.affected_area_km2 : null);
  const label = {vegetation:"Vegetation", urban:"Built-up signal", water:"Water", flood:"Water"}[profile.key] || "Signal";
  $("predictionGainLabel").textContent = `${label} gain`;
  $("predictionLossLabel").textContent = `${label} loss`;
  const net = gain == null || loss == null ? null : gain - loss;
  $("predictionNet").textContent = net == null ? "Unavailable" : `${net > 0 ? "+" : ""}${net.toFixed(1)} pp`;
  $("predictionNet").classList.toggle("negative", net != null && net < 0);
  $("predictionNote").textContent = layer?.note || (layer
    ? "Percentages describe valid analyzed pixels. Net change is gain minus loss in percentage points."
    : "Directional change measurements were not returned for this analysis.");
}

document.querySelectorAll("[data-result-target]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-result-target]").forEach((tab) => tab.classList.toggle("active", tab === button));
    const target = $(button.dataset.resultTarget);
    if (target instanceof HTMLDetailsElement) target.open = true;
    target?.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block:"nearest"});
    if (button.dataset.resultTarget === "technicalDetails") els.generateReportBtn.focus({preventScroll:true});
  });
});

function renderAnalysisResult(payload) {
  const job = payload?.job;
  if (!job || job.status !== "completed" || !job.result) return;
  if (job.job_id !== currentAnalysisId) return;
  currentJob = job;
  const parsed = job.plan?.parsed || {};
  const years = parsed.years || [];
  const profile = analysisProfile(job);
  const actualSource = deriveActualSource(job);
  const routing = job.result.confidence?.routing;
  const confidence = job.result.confidence || {};
  const metric = targetMetric(job, profile);
  const changeStats = job.result.statistics?.change || {};
  const quality = qualityStatus(job);
  const headline = headlineMetrics(job, profile);

  document.body.classList.add("analysis-ready");
  $("resultQuery").textContent = parsed.query || els.query.value;
  $("resultContext").textContent = [years.join(" → "), actualSource.label, actualSource.meta].filter(Boolean).join(" · ");
  renderPredictionStatistics(job, profile);

  els.resultTitle.textContent = profile.title;
  els.resultPeriod.textContent = years.length ? years.join(" → ") : "Current scene";
  els.resultQualityBadge.textContent = quality.label;
  els.resultQualityBadge.className = `quality-badge ${quality.cls}`.trim();
  els.metricArea.parentElement.querySelector("span").textContent = headline.areaLabel;
  els.metricArea.textContent = headline.area;
  els.metricShare.parentElement.querySelector("span").textContent = headline.shareLabel;
  els.metricShare.textContent = headline.share;
  els.metricPrimary.parentElement.querySelector("span").textContent = metric.label;
  els.metricPrimary.textContent = metric.value;
  els.metricSatellite.textContent = actualSource.label;
  els.metricConfidence.textContent = confidence.confidence_available
    ? compactLabel(confidence.confidence_level || "Available")
    : "Unavailable";
  els.metricResultQuality.textContent = resultQualityText(job);
  const validation = validationStats(job);
  const baseSummary = focusedSummary(job, profile, metric);
  els.mapAnswer.textContent = validation?.status === "review"
    ? `${baseSummary} Evidence validation recommends review before treating the mapped area as a definitive land-cover conversion.`
    : baseSummary;
  els.qualityRouting.textContent = routing == null ? "Unavailable" : `${compactLabel(confidence.confidence_provenance?.routing_confidence?.source || "Router")} · ${pct(routing)}`;
  els.qualityData.textContent = confidence.data_quality_confidence == null
    ? dataQualityLabel(job)
    : `${compactLabel(confidence.confidence_provenance?.data_quality_confidence?.source || "Quality gate")} · ${pct(confidence.data_quality_confidence)}`;
  els.qualityEvidence.textContent = confidence.model_confidence == null
    ? "Model probability unavailable"
    : `${compactLabel(confidence.confidence_provenance?.model_confidence?.source || "Model native")} · ${pct(confidence.model_confidence)}`;
  renderValidation(job);
  els.technicalEvidence.textContent = [
    job.result.answer ? `• Engine interpretation: ${job.result.answer}` : null,
    ...(job.result.limitations || []).map((x)=>`• ${x}`),
    changeStats.structural_change_level ? `• Structural change level: ${compactLabel(changeStats.structural_change_level)}` : null,
    changeStats.rgb_mean_absolute_difference != null ? `• Mean RGB difference: ${Number(changeStats.rgb_mean_absolute_difference).toFixed(2)} / 255` : null,
    changeStats.rgb_pixels_difference_gt_20_pct != null ? `• Pixels with RGB difference >20: ${Number(changeStats.rgb_pixels_difference_gt_20_pct).toFixed(1)}%` : null,
    actualSource.meta ? `• Imagery: ${actualSource.meta}` : null,
    validationStats(job)?.threshold_sensitivity?.spread_percentage_points != null ? `• Threshold sensitivity span: ${Number(validationStats(job).threshold_sensitivity.spread_percentage_points).toFixed(1)} percentage points (${compactLabel(validationStats(job).threshold_sensitivity.status)})` : null,
    validationStats(job)?.urban_ndvi_support_pct != null ? `• Urban cross-check: ${Number(validationStats(job).urban_ndvi_support_pct).toFixed(1)}% of NDBI-increase area also shows NDVI decline` : null,
    confidence.confidence_method ? `• Reliability method: ${compactLabel(confidence.confidence_method)} (${compactLabel(confidence.overall_type)})` : null,
    ...(confidence.confidence_warnings || []).map((x)=>`• Confidence: ${x}`),
    taskLayerStats(job)?.note ? `• Task-specific layer: ${taskLayerStats(job).note}` : null
  ].filter(Boolean).join("\n") || "No additional limitations were returned.";

  renderComparison(job);
  renderEvidenceGallery(job);
  els.resultSection.classList.remove("hidden");
  els.emptyResultSection.classList.add("hidden");
  els.viewEvidenceBtn.disabled = !preferredEvidence(job);
  els.compareScenesBtn.disabled = !comparisonData(job);
  els.generateReportBtn.disabled = false;

  saveAnalysisHistory(job);
  const preferred = preferredEvidence(job);
  if (!showTemporalChangePolygons(job) && preferred) showEvidenceOnMap(job, preferred);
}

function renderFailure(payload, fallbackMessage) {
  document.body.classList.remove("analysis-ready");
  $("resultQuery").textContent = "";
  $("resultContext").textContent = "";
  renderPredictionStatistics(null, {key:"general"});
  const job = payload?.job;
  currentJob = job || null;
  els.resultSection.classList.remove("hidden");
  els.emptyResultSection.classList.add("hidden");
  els.resultTitle.textContent = "Analysis could not complete";
  els.resultPeriod.textContent = job?.plan?.parsed?.years?.join(" → ") || "—";
  els.resultQualityBadge.textContent = "Needs attention";
  els.resultQualityBadge.className = "quality-badge error";
  els.metricArea.parentElement.querySelector("span").textContent = "Detected-change area";
  els.metricArea.textContent = "Unavailable";
  els.metricShare.parentElement.querySelector("span").textContent = "Changed share";
  els.metricShare.textContent = "Unavailable";
  els.metricPrimary.parentElement.querySelector("span").textContent = "Status";
  els.metricPrimary.textContent = "Not completed";
  els.metricSatellite.textContent = currentPlanPayload?.plan?.source_details?.platform || "—";
  els.metricConfidence.textContent = "—";
  els.metricResultQuality.textContent = "Unavailable";
  els.mapAnswer.textContent = job?.error || fallbackMessage || "The analysis engine returned an error.";
  els.qualityRouting.textContent = job?.plan?.parsed?.routing_confidence != null ? pct(job.plan.parsed.routing_confidence) : "—";
  els.qualityData.textContent = "Unavailable";
  els.qualityEvidence.textContent = "Not generated";
  els.technicalEvidence.textContent = `Error code: ${job?.error_code || "analysis_failed"}\n${job?.error || fallbackMessage || "Inspect the FastAPI terminal for details."}`;
  els.viewEvidenceBtn.disabled = true;
  els.compareScenesBtn.disabled = true;
  els.comparisonViewer.classList.add("hidden");
  els.validationCard?.classList.add("hidden");
  els.mapComparison.classList.add("hidden");
  els.generateReportBtn.disabled = true;
}

async function planQuery(query, aoi) {
  workflowMark("query","active");
  const response = await fetch(`${API_BASE}/api/v1/map/context`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      query,
      aoi,
      client:{source:"satquery-industry-workspace",map_center:{longitude:map.getCenter().lng,latitude:map.getCenter().lat},zoom:map.getZoom()}
    })
  });
  const payload = await response.json().catch(()=>({}));
  if (!response.ok) throw new Error(payload?.detail || payload?.error || `Planning HTTP ${response.status}`);
  if (payload?.plan?.planner === "unavailable") throw new Error((payload.plan.warnings||[]).join(" | ") || "No compatible planner is available.");
  return payload;
}

async function submitAnalysisJob(query, aoi, planFingerprint = null) {
  const response = await fetch(`${API_BASE}/api/v1/map/jobs`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({query,aoi,plan_fingerprint:planFingerprint})
  });
  const payload = await response.json().catch(()=>({}));
  if (!response.ok) throw new Error(payload?.detail || payload?.error || `Job submission HTTP ${response.status}`);
  return payload;
}

function workflowFromJobState(state) {
  const stage = String(state?.stage || "queued");
  const all = ["imagery","preprocess","analysis","compare","evidence","statistics","report"];
  const doneThrough = (name) => {
    const idx = all.indexOf(name);
    all.slice(0, idx + 1).forEach((step)=>workflowMark(step,"done"));
  };
  if (stage === "queued" || stage === "preparing") {
    workflowMark("imagery","active");
  } else if (stage === "retrieving_imagery") {
    workflowMark("imagery","active");
  } else if (stage === "preprocessing") {
    doneThrough("imagery"); workflowMark("preprocess","active");
  } else if (stage === "analyzing") {
    doneThrough("preprocess"); workflowMark("analysis","active");
  } else if (stage === "generating_evidence") {
    doneThrough("compare"); workflowMark("evidence","active");
  } else if (stage === "validating") {
    doneThrough("evidence"); workflowMark("statistics","active");
  } else if (stage === "finalizing") {
    doneThrough("statistics"); workflowMark("report","active");
  } else if (stage === "completed") {
    all.forEach((step)=>workflowMark(step,"done"));
  } else if (stage === "failed" || stage === "cancelled") {
    const active = els.workflow.querySelector(".workflow-step.active");
    if (active) { active.classList.remove("active"); active.classList.add("failed"); }
  }
  const progress = Number(state?.progress);
  const prefix = Number.isFinite(progress) ? `${Math.max(0,Math.min(100,progress))}% · ` : "";
  if (state?.message) {
    els.mapHint.textContent = `${prefix}${state.message}`;
    setBackendStatus(`${prefix}${state.message}`, stage === "failed" || stage === "cancelled" ? "error" : "running");
  }
}

async function pollAnalysisJob(jobId) {
  while (true) {
    const response = await fetch(`${API_BASE}/api/v1/map/jobs/${encodeURIComponent(jobId)}`, {cache:"no-store"});
    const state = await response.json().catch(()=>({}));
    if (!response.ok) throw new Error(state?.detail || state?.error || `Job status HTTP ${response.status}`);
    workflowFromJobState(state);
    if (["completed","failed","cancelled"].includes(state.status)) return state;
    await new Promise((resolve)=>setTimeout(resolve,700));
  }
}

async function cancelActiveJob() {
  if (!activeJobId || cancellationRequested) return;
  cancellationRequested = true;
  try {
    const response = await fetch(`${API_BASE}/api/v1/map/jobs/${encodeURIComponent(activeJobId)}`, {method:"DELETE"});
    const state = await response.json().catch(()=>({}));
    if (!response.ok) throw new Error(state?.detail || state?.error || `Cancellation HTTP ${response.status}`);
    setBackendStatus("Cancellation requested. SatQuery is stopping at the next safe checkpoint.","running");
    els.mapHint.textContent = "Cancellation requested…";
  } catch (error) {
    cancellationRequested = false;
    setBackendStatus(`Could not request cancellation: ${error.message}`,"error");
  }
}


async function sendAoiAndQuery() {
  if (activeJobId) {
    await cancelActiveJob();
    return;
  }
  if (els.send.disabled) return;
  const query=(els.query.value||"").trim();
  if (!currentAoi) { setBackendStatus("Draw an AOI before running the analysis.","error"); return; }
  if (!query) { setBackendStatus("Enter a natural-language question.","error"); els.query.focus(); return; }
  // Snapshot the upstream inputs before any asynchronous request starts.  The
  // planner and job submission therefore receive the exact same AOI.
  const submittedAoi = JSON.parse(JSON.stringify(currentAoi));
  const runToken = ++activeRunToken;
  setInsights(true);

  clearAnalysisOverlay();
  resetResultPanel();
  currentAnalysisId = null;
  workflowReset();
  currentPlanPayload = null;
  cancellationRequested = false;
  els.backendPlan.innerHTML = '<div class="empty-state">SatQuery is building the analysis plan…</div>';
  els.selectedSourceName.textContent = "Selecting automatically…";
  els.selectedSourceMeta.textContent = "Query, target, dates and AOI are being evaluated.";
  els.send.disabled=true;
  els.send.querySelector("span:first-child").textContent="Planning…";
  setBackendStatus("Understanding the request and building an executable geospatial plan…","running");
  els.mapHint.textContent="SatQuery AI is understanding your question…";

  try {
    const planPayload = await planQuery(query, submittedAoi);
    if (runToken !== activeRunToken) return;
    currentPlanPayload = planPayload;
    workflowMark("query","done");
    workflowMark("aoi","done");
    workflowMark("source","done");
    workflowMark("workflow","done");
    renderPlan(planPayload);
    localStorage.setItem(CONTEXT_STORAGE_KEY,JSON.stringify(planPayload));

    let submitted;
    try {
      submitted = await submitAnalysisJob(query, submittedAoi, planPayload.plan_fingerprint || planPayload?.plan?.fingerprint || null);
    } catch (error) {
      if (String(error.message||"").toLowerCase().includes("plan changed")) {
        setBackendStatus("Imagery availability changed after preview. Refreshing the executable plan once…","running");
        const refreshedPlan = await planQuery(query, submittedAoi);
        if (runToken !== activeRunToken) return;
        currentPlanPayload = refreshedPlan;
        renderPlan(refreshedPlan);
        submitted = await submitAnalysisJob(query, submittedAoi, refreshedPlan.plan_fingerprint || refreshedPlan?.plan?.fingerprint || null);
      } else {
        throw error;
      }
    }

    activeJobId = submitted.job_id;
    currentAnalysisId = submitted.job_id;
    els.send.disabled=false;
    els.send.querySelector("span:first-child").textContent="Cancel analysis";
    setBackendStatus("Analysis job accepted. Progress now comes from backend execution events.","running");
    workflowFromJobState(submitted);

    const state = await pollAnalysisJob(activeJobId);
    if (runToken !== activeRunToken || (state.job_id && state.job_id !== currentAnalysisId)) return;
    if (state.status !== "completed" || !state.job?.result) {
      renderFailure({job:state.job}, state.message || (state.status === "cancelled" ? "Analysis cancelled." : "Analysis did not complete."));
      setBackendStatus(state.status === "cancelled" ? "Analysis cancelled." : "Analysis stopped because execution or evidence requirements were not met.","error");
      els.mapHint.textContent = state.message || "Analysis needs attention. Review the result panel.";
      return;
    }

    renderAnalysisResult({job:state.job, execution_plan:state.execution_plan});
    setBackendStatus("Analysis completed. Evidence, statistics and report are ready.","completed");
    els.mapHint.textContent="Analysis complete. Evidence is displayed on the map.";
  } catch (error) {
    if (runToken !== activeRunToken) return;
    console.error("SatQuery map analysis error", error);
    renderFailure(null,error.message);
    setBackendStatus(`Analysis failed: ${error.message}`,"error");
    els.mapHint.textContent="Analysis failed. Review the error details; unsupported tasks are not simulated.";
  } finally {
    if (runToken === activeRunToken) {
      activeJobId=null;
      cancellationRequested=false;
      els.send.disabled=false;
      els.send.querySelector("span:first-child").textContent="Analyze with SatQuery AI";
    }
  }
}


function restoreState() {
  try { currentAoi=JSON.parse(localStorage.getItem(AOI_STORAGE_KEY)||"null"); } catch { currentAoi=null; }
  // Deliberately do not restore a previous analysis result. A new page load must not
  // present stale evidence as if it belongs to the current AOI/query.
  resetResultPanel();
}

async function checkBackend() {
  try {
    const response=await fetch(`${API_BASE}/api/v1/map/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data=await response.json();
    if (data?.status === "ok") {
      setSystemState("Geospatial engine ready","ready");
      setBackendStatus("Engine ready. Select an AOI and ask a question.","ready");
    } else throw new Error("Health check failed");
  } catch {
    setSystemState("Backend offline","error");
    setBackendStatus("FastAPI is not reachable on port 8000.","error");
  }
}

const HISTORY_KEY = "satquery.analysis.history.v1";

function loadAnalysisHistory() {
  try { const v=JSON.parse(localStorage.getItem(HISTORY_KEY)||"[]"); return Array.isArray(v)?v:[]; } catch { return []; }
}
function saveAnalysisHistory(job) {
  if (!job?.job_id || job.status!=="completed" || !job.result) return;
  const parsed=job.plan?.parsed||{}, profile=analysisProfile(job), layer=taskLayerStats(job), metric=targetMetric(job,profile), source=deriveActualSource(job), headline=headlineMetrics(job,profile);
  const item={job_id:job.job_id,title:profile.title,query:parsed.query||"",years:parsed.years||[],source:source.label,metric_label:metric.label,metric_value:metric.value,area:headline.area,share:headline.share,quality:resultQualityText(job),saved_at:new Date().toISOString()};
  const all=loadAnalysisHistory().filter(x=>x.job_id!==item.job_id); all.unshift(item); localStorage.setItem(HISTORY_KEY,JSON.stringify(all.slice(0,30)));
}
async function loadPersistentHistory() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/history?limit=50`, {cache:"no-store"});
    if (!response.ok) throw new Error(`History HTTP ${response.status}`);
    const payload = await response.json();
    const items = Array.isArray(payload?.items) ? payload.items : [];
    return items.map((item)=>{
      const q=String(item.query||"").toLowerCase();
      let title="Temporal Change Analysis";
      let metricLabel="Primary metric";
      if (/vegetation|green|ndvi|crop|forest/.test(q)) { title="Vegetation Change Analysis"; metricLabel="Changed area"; }
      else if (/urban|built|construction|development|ndbi/.test(q)) { title="Urban Expansion Analysis"; metricLabel="Changed area"; }
      else if (/water|flood|river|lake|ndwi/.test(q)) { title="Water Change Analysis"; metricLabel="Changed area"; }
      return {
        job_id:item.job_id,
        title,
        query:item.query||"",
        years:[item.before_year,item.after_year].filter((x)=>x!=null),
        source:item.sensor||"Earth observation imagery",
        metric_label:metricLabel,
        metric_value:item.changed_area_km2!=null?`${Number(item.changed_area_km2).toFixed(2)} km²`:"—",
        area:item.changed_area_km2!=null?`${Number(item.changed_area_km2).toFixed(2)} km²`:"—",
        share:item.changed_share_pct!=null?`${Number(item.changed_share_pct).toFixed(1)}%`:"—",
        quality:item.evidence_quality||"Completed",
        saved_at:item.completed_at||item.started_at,
        report_url:item.report_url
      };
    });
  } catch (error) {
    console.warn("Persistent analysis history unavailable; using local fallback.", error);
    return loadAnalysisHistory();
  }
}

async function renderReportsHub() {
  els.reportsList.innerHTML='<div class="history-empty"><strong>Loading analysis history…</strong></div>';
  const all=await loadPersistentHistory();
  els.reportsSummary.innerHTML=`<div><strong>${all.length}</strong><span>Saved analyses</span></div><div><strong>${new Set(all.map(x=>x.title)).size}</strong><span>Analysis types</span></div><div><strong>${all.filter(x=>String(x.quality).toLowerCase()==="good").length}</strong><span>Good-quality results</span></div>`;
  els.reportsList.innerHTML=all.length?all.map(x=>`<article class="history-card"><div class="history-card-top"><div><span>${safeText((x.years||[]).join(" → ")||"Current")}</span><h3>${safeText(x.title)}</h3></div><em>${safeText(x.quality||"Completed")}</em></div><p>${safeText(x.query||"")}</p><div class="history-metrics"><div><span>${safeText(x.metric_label||"Metric")}</span><strong>${safeText(x.metric_value||"—")}</strong></div><div><span>Changed area</span><strong>${safeText(x.area||"—")}</strong></div><div><span>Changed share</span><strong>${safeText(x.share||"—")}</strong></div><div><span>Satellite</span><strong>${safeText(x.source||"—")}</strong></div></div><div class="history-actions"><small>${safeText(x.saved_at?new Date(x.saved_at).toLocaleString():"")}</small><a href="${safeText(x.report_url?`${API_BASE}${x.report_url}`:`${API_BASE}/api/v1/jobs/${encodeURIComponent(x.job_id)}/report`)}" target="_blank" rel="noopener">Open intelligence report →</a></div></article>`).join(""):`<div class="history-empty"><strong>No reports yet</strong><p>Complete an analysis and it will appear here automatically.</p></div>`;
}

function openReportsHub(event){ if(event)event.preventDefault(); renderReportsHub(); els.reportsHub.classList.remove("hidden"); }
function closeReportsHub(){ els.reportsHub.classList.add("hidden"); }

function openCurrentReport(event) {
  if (event) event.preventDefault();
  if (!currentJob?.job_id) {
    setBackendStatus("Run an analysis before generating a report.","error");
    return;
  }
  window.open(`${API_BASE}/api/v1/jobs/${encodeURIComponent(currentJob.job_id)}/report`,"_blank","noopener");
}

function bindControls() {
  els.searchForm.addEventListener("submit",(e)=>{e.preventDefault();searchLocation(els.searchInput.value);});
  els.aoiTools.querySelectorAll("[data-mode]").forEach((button)=>button.addEventListener("click",()=>startDrawing(button.dataset.mode,button)));
  els.clearAoiBtn.addEventListener("click",()=>{stopDrawing();saveAoi(null);});
  els.fitAoiBtn.addEventListener("click",fitAoi);
  document.querySelectorAll(".suggestion-chip").forEach((chip)=>chip.addEventListener("click",()=>{els.query.value=chip.dataset.query||"";els.query.focus();}));
  els.send.addEventListener("click",sendAoiAndQuery);
  els.query.addEventListener("keydown",(e)=>{if((e.ctrlKey||e.metaKey)&&e.key==="Enter")sendAoiAndQuery();});
  els.vectorBtn.addEventListener("click",()=>applyBaseMode("vector"));
  els.satelliteBtn.addEventListener("click",()=>applyBaseMode("satellite"));
  els.hybridBtn.addEventListener("click",()=>applyBaseMode("hybrid"));
  els.resetBtn.addEventListener("click",()=>map.flyTo({center:DEFAULT_CENTER,zoom:DEFAULT_ZOOM,pitch:0,bearing:0,speed:1.1}));
  els.togglePlanBtn.addEventListener("click",()=>{
    els.backendPlan.classList.toggle("collapsed");
    els.togglePlanBtn.textContent = els.backendPlan.classList.contains("collapsed") ? "Show plan" : "Hide plan";
  });
  els.aoiLayerToggle.addEventListener("change",()=>{
    [AOI_FILL,AOI_LINE,AOI_POINT].forEach((id)=>{if(map.getLayer(id))map.setLayoutProperty(id,"visibility",els.aoiLayerToggle.checked?"visible":"none");});
  });
  els.analysisLayerToggle.addEventListener("change",()=>{
    if (map.getLayer(ANALYSIS_LAYER)) map.setLayoutProperty(ANALYSIS_LAYER,"visibility",els.analysisLayerToggle.checked?"visible":"none");
  });
  els.viewEvidenceBtn.addEventListener("click",()=>{
    const preferred = currentJob ? preferredEvidence(currentJob) : null;
    if (preferred) showEvidenceOnMap(currentJob,preferred);
  });
  els.generateReportBtn.addEventListener("click",openCurrentReport);
  els.compareScenesBtn.addEventListener("click", openComparison);
  els.openComparisonBtn.addEventListener("click", openComparison);
  els.closeComparisonBtn.addEventListener("click",()=>els.mapComparison.classList.add("hidden"));

  els.topReportsLink.addEventListener("click",openReportsHub);
  els.closeReportsHub.addEventListener("click",closeReportsHub);
}

function bindMapDrawing() {
  map.on("click",(e)=>{
    if (!drawMode) return;
    if (drawMode === "point") { saveAoi(pointFeature(e.lngLat)); stopDrawing(); return; }
    if (drawMode === "rectangle") {
      if (!drawStart) { drawStart=e.lngLat; els.mapHint.textContent="Rectangle mode: choose the opposite corner."; return; }
      const feature=rectangleFeature(drawStart,e.lngLat); saveAoi(feature); stopDrawing(); fitAoi(); return;
    }
    if (drawMode === "polygon") {
      polygonPoints.push(e.lngLat);
      const preview=polygonPoints.length>=3?polygonFeature(polygonPoints):{type:"Feature",properties:{},geometry:{type:"LineString",coordinates:polygonPoints.map(p=>[p.lng,p.lat])}};
      setPreview(preview); els.mapHint.textContent=`Polygon mode: ${polygonPoints.length} vertices · double-click to finish.`;
    }
  });
  map.on("mousemove",(e)=>{
    els.lat.textContent=e.lngLat.lat.toFixed(5); els.lng.textContent=e.lngLat.lng.toFixed(5);
    if (drawMode==="rectangle"&&drawStart)setPreview(rectangleFeature(drawStart,e.lngLat));
  });
  map.on("dblclick",(e)=>{
    if(drawMode!=="polygon")return;
    e.preventDefault();
    if(polygonPoints.length<3){els.mapHint.textContent="Polygon needs at least 3 vertices.";return;}
    saveAoi(polygonFeature(polygonPoints)); stopDrawing(); fitAoi();
  });
  map.on("zoom",()=>{els.zoom.textContent=map.getZoom().toFixed(1);});
}

function initializeMap() {
  if (window.innerWidth <= 760) setInsights(false);
  if (typeof maplibregl === "undefined") { setSystemState("MapLibre failed to load","error"); return; }
  restoreState();
  map=new maplibregl.Map({container:"map",style:BASE_STYLE_URL,center:DEFAULT_CENTER,zoom:DEFAULT_ZOOM,pitch:0,bearing:0,antialias:true,maxZoom:19,attributionControl:true});
  map.addControl(new maplibregl.NavigationControl({showCompass:true,showZoom:true,visualizePitch:true}),"top-right");
  map.addControl(new maplibregl.FullscreenControl(),"top-right");
  map.on("error",(event)=>{
    const sourceId = event?.sourceId || event?.source?.id || "";
    const message = String(event?.error?.message || "");
    if (sourceId === SAT_SOURCE || message.includes("World_Imagery") || message.includes("server.arcgisonline.com")) {
      recordSatelliteTileError();
    }
  });
  map.on("load",()=>{
    try{map.setProjection({type:"globe"});}catch{}
    addCustomLayers(); renderAoi(); applyBaseMode(baseMode); bindControls(); bindMapDrawing();
    els.zoom.textContent=map.getZoom().toFixed(1); checkBackend();
  });
}

window.SatQueryMap={
  getAOI:()=>currentAoi,
  clearAOI:()=>saveAoi(null),
  getMap:()=>map,
  analyze:sendAoiAndQuery
};
initializeMap();
})();
