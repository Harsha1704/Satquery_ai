(() => {
"use strict";

const DEFAULT_CENTER = [78.9629, 20.5937];
const DEFAULT_ZOOM = 3.3;
const BASE_STYLE_URL = "https://demotiles.maplibre.org/style.json";

const SATELLITE_SOURCE_ID = "satquery-satellite";
const SATELLITE_LAYER_ID = "satquery-satellite-layer";
const AOI_STORAGE_KEY = "satquery_selected_aoi";
const LAST_ANALYSIS_KEY = "satquery_last_analysis";
const ANALYSIS_PREFIX = "satquery-analysis-";

let map = null;
let draw = null;
let currentAoi = null;
let latestAnalysisBounds = null;
let analysisOverlayVisible = true;
let analysis = null;

const $ = (id) => document.getElementById(id);

const els = {
    mapQuery: $("mapQuery"),
    mapAnswer: $("mapAnswer"),
    mapModel: $("mapModel"),
    mapConfidence: $("mapConfidence"),
    datasetMetadata: $("datasetMetadata"),
    spatialMetadata: $("spatialMetadata"),
    sourceCRS: $("sourceCRS"),
    layerList: $("layerList"),
    cursorLat: $("cursorLat"),
    cursorLng: $("cursorLng"),
    zoomLevel: $("zoomLevel"),
    searchForm: $("locationSearchForm"),
    searchInput: $("locationSearchInput"),
    searchStatus: $("searchStatus"),
    searchResults: $("searchResults"),
    aoiSummary: $("aoiSummary"),
    selectedAoiDetails: $("selectedAoiDetails"),
    aoiStateBadge: $("aoiStateBadge"),
    mapHint: $("mapHint"),
    baseLayerToggle: $("baseLayerToggle"),
    satelliteLayerToggle: $("satelliteLayerToggle"),
    analysisLayerToggle: $("analysisLayerToggle"),
    clearAoiBtn: $("clearAoiBtn"),
    fitAoiBtn: $("fitAoiBtn"),
    fitBoundsBtn: $("fitBoundsBtn"),
    toggleOverlayBtn: $("toggleOverlayBtn"),
    resetGlobeBtn: $("resetGlobeBtn"),
    aoiTools: $("aoiTools"),
};

function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
}

function normalizePercent(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return n <= 1 ? n * 100 : n;
}

function setSearchStatus(message) {
    els.searchStatus.textContent = message;
}

function loadState() {
    try {
        const saved = localStorage.getItem(LAST_ANALYSIS_KEY);
        analysis = saved ? JSON.parse(saved) : null;
    } catch {
        analysis = null;
    }

    try {
        const savedAoi = localStorage.getItem(AOI_STORAGE_KEY);
        currentAoi = savedAoi ? JSON.parse(savedAoi) : null;
    } catch {
        currentAoi = null;
    }
}

function coordinatePairs(feature) {
    if (!feature?.geometry) return [];

    const { type, coordinates } = feature.geometry;

    if (type === "Point") return [coordinates];
    if (type === "LineString") return Array.isArray(coordinates) ? coordinates : [];
    if (type === "Polygon") return Array.isArray(coordinates?.[0]) ? coordinates[0] : [];
    if (type === "MultiPolygon") return Array.isArray(coordinates?.[0]?.[0]) ? coordinates[0][0] : [];

    return [];
}

function featureBBox(feature) {
    const points = coordinatePairs(feature);
    if (!points.length) return null;

    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;

    points.forEach(([lng, lat]) => {
        if (!Number.isFinite(lng) || !Number.isFinite(lat)) return;
        west = Math.min(west, lng);
        south = Math.min(south, lat);
        east = Math.max(east, lng);
        north = Math.max(north, lat);
    });

    if (![west, south, east, north].every(Number.isFinite)) return null;
    return [west, south, east, north];
}

function saveAoi(feature) {
    currentAoi = feature || null;

    if (currentAoi) {
        localStorage.setItem(AOI_STORAGE_KEY, JSON.stringify(currentAoi));
    } else {
        localStorage.removeItem(AOI_STORAGE_KEY);
    }

    window.dispatchEvent(new CustomEvent("satquery:aoi-change", {
        detail: currentAoi,
    }));

    renderAoiState();
}

function renderAoiState() {
    if (!currentAoi) {
        els.aoiSummary.innerHTML = `<div class="empty">No AOI selected.</div>`;
        els.selectedAoiDetails.innerHTML = `
            <div class="empty">
                Draw a polygon, rectangle, circle, freehand area, or point.
            </div>
        `;
        els.aoiStateBadge.textContent = "Waiting";
        els.aoiStateBadge.classList.remove("ready");
        els.mapHint.textContent = "Draw an AOI to prepare a live geospatial query.";
        return;
    }

    const bbox = featureBBox(currentAoi);
    const geometryType = currentAoi.geometry?.type || "Unknown";
    const vertexCount = coordinatePairs(currentAoi).length;

    els.aoiSummary.innerHTML = `
        <div class="aoi-summary-grid">
            <div class="aoi-stat">
                <span>Geometry</span>
                <strong>${escapeHTML(geometryType)}</strong>
            </div>
            <div class="aoi-stat">
                <span>Vertices</span>
                <strong>${vertexCount}</strong>
            </div>
        </div>
    `;

    els.selectedAoiDetails.innerHTML = `
        <div class="meta-row">
            <span>Geometry</span>
            <strong>${escapeHTML(geometryType)}</strong>
        </div>
        <div class="meta-row">
            <span>Bounding box</span>
            <strong>${bbox ? bbox.map(v => Number(v).toFixed(5)).join(", ") : "--"}</strong>
        </div>
        <div class="meta-row">
            <span>State</span>
            <strong>Ready for Phase 3 API connection</strong>
        </div>
    `;

    els.aoiStateBadge.textContent = "AOI Ready";
    els.aoiStateBadge.classList.add("ready");
    els.mapHint.textContent =
        "AOI selected. Phase 3 will send this GeoJSON + the user query to FastAPI.";
}

function fitFeature(feature) {
    const bbox = featureBBox(feature);
    if (!bbox || !map) return;

    const [west, south, east, north] = bbox;

    if (Math.abs(east - west) < 1e-8 && Math.abs(north - south) < 1e-8) {
        map.flyTo({ center: [west, south], zoom: 13, speed: 1.2 });
        return;
    }

    map.fitBounds([[west, south], [east, north]], {
        padding: 70,
        duration: 850,
        maxZoom: 15,
    });
}

function addSatelliteLayer() {
    if (!map.getSource(SATELLITE_SOURCE_ID)) {
        map.addSource(SATELLITE_SOURCE_ID, {
            type: "raster",
            tiles: [
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            attribution:
                "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and GIS User Community",
        });
    }

    if (!map.getLayer(SATELLITE_LAYER_ID)) {
        map.addLayer({
            id: SATELLITE_LAYER_ID,
            type: "raster",
            source: SATELLITE_SOURCE_ID,
            layout: { visibility: "none" },
            paint: {
                "raster-opacity": 0.92,
                "raster-fade-duration": 180,
            },
        });
    }
}

function setBaseMapVisible(visible) {
    const layers = map.getStyle()?.layers || [];

    layers.forEach((layer) => {
        if (
            layer.id === SATELLITE_LAYER_ID ||
            layer.id.startsWith(ANALYSIS_PREFIX) ||
            layer.id.startsWith("td-") ||
            layer.id.includes("terra-draw")
        ) {
            return;
        }

        try {
            map.setLayoutProperty(layer.id, "visibility", visible ? "visible" : "none");
        } catch {
            // Some layers may not expose visibility.
        }
    });
}

function setSatelliteVisible(visible) {
    if (!map.getLayer(SATELLITE_LAYER_ID)) return;

    map.setLayoutProperty(
        SATELLITE_LAYER_ID,
        "visibility",
        visible ? "visible" : "none"
    );
}

function analysisLayerIds() {
    return (map.getStyle()?.layers || [])
        .map((layer) => layer.id)
        .filter((id) => id.startsWith(ANALYSIS_PREFIX));
}

function setAnalysisVisible(visible) {
    analysisLayerIds().forEach((id) => {
        try {
            map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
        } catch {
            // Ignore transient style reloads.
        }
    });
}

function removeAnalysisLayers() {
    const style = map.getStyle();
    if (!style) return;

    [...(style.layers || [])]
        .map((layer) => layer.id)
        .filter((id) => id.startsWith(ANALYSIS_PREFIX))
        .forEach((id) => {
            if (map.getLayer(id)) map.removeLayer(id);
        });

    Object.keys(style.sources || {})
        .filter((id) => id.startsWith(ANALYSIS_PREFIX))
        .forEach((id) => {
            if (map.getSource(id)) map.removeSource(id);
        });

    latestAnalysisBounds = null;
}

function boundsPolygon(bounds) {
    const [west, south, east, north] = bounds;

    return {
        type: "Feature",
        properties: {},
        geometry: {
            type: "Polygon",
            coordinates: [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]],
        },
    };
}

function renderAnalysis() {
    if (!analysis) return;

    els.mapQuery.textContent = analysis.query || "No query available.";
    els.mapAnswer.textContent = analysis.answer || "No answer available.";
    els.mapModel.textContent =
        analysis.model ||
        analysis.execution?.model ||
        "--";

    const routingConfidence =
        analysis.routing?.routing_confidence ??
        analysis.routing_confidence ??
        analysis.confidence;

    const confidencePct = normalizePercent(routingConfidence);

    els.mapConfidence.textContent =
        confidencePct == null ? "--" : `${confidencePct.toFixed(1)}%`;

    const geo = analysis.geospatial;

    if (!geo?.available || !Array.isArray(geo.layers) || !geo.layers.length) {
        const reason =
            geo?.reason ||
            "This analysis does not contain georeferenced imagery.";

        els.datasetMetadata.innerHTML =
            `<div class="empty">${escapeHTML(reason)}</div>`;

        els.spatialMetadata.innerHTML = els.datasetMetadata.innerHTML;
        els.layerList.innerHTML =
            `<div class="empty">No analysis layers available.</div>`;
        return;
    }

    removeAnalysisLayers();

    els.datasetMetadata.innerHTML = "";
    els.spatialMetadata.innerHTML = "";
    els.layerList.innerHTML = "";

    let globalWest = Infinity;
    let globalSouth = Infinity;
    let globalEast = -Infinity;
    let globalNorth = -Infinity;

    geo.layers.forEach((layer, index) => {
        const rawBounds = layer.bounds_wgs84;

        if (
            !Array.isArray(rawBounds) ||
            rawBounds.length !== 4 ||
            !rawBounds.every((v) => Number.isFinite(Number(v)))
        ) {
            return;
        }

        const bounds = rawBounds.map(Number);
        const [west, south, east, north] = bounds;

        globalWest = Math.min(globalWest, west);
        globalSouth = Math.min(globalSouth, south);
        globalEast = Math.max(globalEast, east);
        globalNorth = Math.max(globalNorth, north);

        const sourceId = `${ANALYSIS_PREFIX}footprint-source-${index}`;
        const fillId = `${ANALYSIS_PREFIX}footprint-fill-${index}`;
        const lineId = `${ANALYSIS_PREFIX}footprint-line-${index}`;
        const color = index === 0 ? "#2f9cff" : "#2bd9d2";

        map.addSource(sourceId, {
            type: "geojson",
            data: boundsPolygon(bounds),
        });

        map.addLayer({
            id: fillId,
            type: "fill",
            source: sourceId,
            paint: {
                "fill-color": color,
                "fill-opacity": 0.08,
            },
        });

        map.addLayer({
            id: lineId,
            type: "line",
            source: sourceId,
            paint: {
                "line-color": color,
                "line-width": 2,
            },
        });

        const layerItem = document.createElement("div");
        layerItem.className = "layer-item";
        layerItem.innerHTML = `
            <span class="layer-color" style="background:${color}"></span>
            <span>${escapeHTML(layer.name || `Layer ${index + 1}`)}</span>
        `;
        els.layerList.appendChild(layerItem);

        map.on("click", fillId, (event) => {
            new maplibregl.Popup()
                .setLngLat(event.lngLat)
                .setHTML(`
                    <strong>${escapeHTML(layer.name || "Analysis layer")}</strong>
                    <br>CRS: ${escapeHTML(layer.source_crs || "--")}
                    <br>Bands: ${escapeHTML(layer.bands ?? "--")}
                `)
                .addTo(map);
        });

        map.on("mouseenter", fillId, () => {
            map.getCanvas().style.cursor = "pointer";
        });

        map.on("mouseleave", fillId, () => {
            map.getCanvas().style.cursor = "";
        });

        if (index === 0) {
            els.sourceCRS.textContent = layer.source_crs || "--";

            els.datasetMetadata.innerHTML = `
                <div class="meta-row"><span>Width</span><strong>${escapeHTML(layer.width ?? "--")}</strong></div>
                <div class="meta-row"><span>Height</span><strong>${escapeHTML(layer.height ?? "--")}</strong></div>
                <div class="meta-row"><span>Bands</span><strong>${escapeHTML(layer.bands ?? "--")}</strong></div>
                <div class="meta-row">
                    <span>Resolution</span>
                    <strong>${escapeHTML(layer.resolution_x ?? "--")} × ${escapeHTML(layer.resolution_y ?? "--")}</strong>
                </div>
            `;

            els.spatialMetadata.innerHTML = `
                <div class="meta-row"><span>West</span><strong>${west.toFixed(6)}</strong></div>
                <div class="meta-row"><span>South</span><strong>${south.toFixed(6)}</strong></div>
                <div class="meta-row"><span>East</span><strong>${east.toFixed(6)}</strong></div>
                <div class="meta-row"><span>North</span><strong>${north.toFixed(6)}</strong></div>
                <div class="meta-row"><span>NoData</span><strong>${escapeHTML(layer.nodata ?? "--")}</strong></div>
            `;
        }
    });

    if ([globalWest, globalSouth, globalEast, globalNorth].every(Number.isFinite)) {
        latestAnalysisBounds = [
            [globalWest, globalSouth],
            [globalEast, globalNorth],
        ];

        map.fitBounds(latestAnalysisBounds, {
            padding: 70,
            duration: 800,
            maxZoom: 15,
        });
    }

    if (geo.overlay_url && geo.layers[0]?.bounds_wgs84) {
        const [west, south, east, north] =
            geo.layers[0].bounds_wgs84.map(Number);

        const overlaySourceId = `${ANALYSIS_PREFIX}evidence-source`;
        const overlayLayerId = `${ANALYSIS_PREFIX}evidence-layer`;

        map.addSource(overlaySourceId, {
            type: "image",
            url: geo.overlay_url,
            coordinates: [
                [west, north],
                [east, north],
                [east, south],
                [west, south],
            ],
        });

        map.addLayer({
            id: overlayLayerId,
            type: "raster",
            source: overlaySourceId,
            paint: { "raster-opacity": 0.72 },
        });
    }

    setAnalysisVisible(els.analysisLayerToggle.checked);
}

async function searchLocation(query) {
    const trimmed = query.trim();

    if (!trimmed) {
        setSearchStatus("Enter a location first.");
        return;
    }

    setSearchStatus("Searching...");
    els.searchResults.innerHTML = "";

    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("q", trimmed);
    url.searchParams.set("format", "jsonv2");
    url.searchParams.set("limit", "5");
    url.searchParams.set("addressdetails", "1");

    try {
        const response = await fetch(url, {
            headers: { Accept: "application/json" },
        });

        if (!response.ok) {
            throw new Error(`Search failed with HTTP ${response.status}`);
        }

        const results = await response.json();

        if (!Array.isArray(results) || !results.length) {
            setSearchStatus("No matching locations found.");
            return;
        }

        setSearchStatus(`${results.length} result${results.length === 1 ? "" : "s"} found.`);

        results.forEach((result) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "search-result";
            button.textContent = result.display_name;

            button.addEventListener("click", () => {
                if (Array.isArray(result.boundingbox)) {
                    const south = Number(result.boundingbox[0]);
                    const north = Number(result.boundingbox[1]);
                    const west = Number(result.boundingbox[2]);
                    const east = Number(result.boundingbox[3]);

                    if ([west, south, east, north].every(Number.isFinite)) {
                        map.fitBounds([[west, south], [east, north]], {
                            padding: 80,
                            duration: 900,
                            maxZoom: 14,
                        });
                    }
                } else {
                    const lon = Number(result.lon);
                    const lat = Number(result.lat);

                    if (Number.isFinite(lon) && Number.isFinite(lat)) {
                        map.flyTo({
                            center: [lon, lat],
                            zoom: 11,
                            speed: 1.2,
                        });
                    }
                }

                els.searchResults.innerHTML = "";
                setSearchStatus(result.display_name);
            });

            els.searchResults.appendChild(button);
        });
    } catch (error) {
        console.error(error);
        setSearchStatus(
            "Location search is unavailable. You can still navigate and draw directly on the globe."
        );
    }
}

function buildDrawModes() {
    const td = window.terraDraw;
    if (!td) return [];

    const modes = [];

    [
        "TerraDrawRectangleMode",
        "TerraDrawPolygonMode",
        "TerraDrawCircleMode",
        "TerraDrawFreehandMode",
        "TerraDrawPointMode",
    ].forEach((name) => {
        const Ctor = td[name];

        if (typeof Ctor === "function") {
            modes.push(new Ctor({
                styles: {
                    fillColor: "#2f9cff",
                    fillOpacity: 0.18,
                    outlineColor: "#64c8ff",
                    outlineWidth: 2,
                },
            }));
        }
    });

    return modes;
}

function initializeDrawing() {
    if (!window.terraDraw || !window.terraDrawMapLibreGLAdapter) {
        console.warn("Terra Draw failed to load.");
        els.aoiTools.querySelectorAll("button").forEach((button) => {
            button.disabled = true;
        });
        return;
    }

    const TerraDraw = window.terraDraw.TerraDraw;
    const Adapter =
        window.terraDrawMapLibreGLAdapter.TerraDrawMapLibreGLAdapter;

    draw = new TerraDraw({
        adapter: new Adapter({
            map,
            lib: maplibregl,
        }),
        modes: buildDrawModes(),
    });

    draw.start();

    const syncLatestFeature = () => {
        const snapshot = draw.getSnapshot();

        if (!snapshot.length) {
            saveAoi(null);
            return;
        }

        saveAoi(snapshot[snapshot.length - 1]);
    };

    draw.on("finish", syncLatestFeature);
    draw.on("change", syncLatestFeature);

    els.aoiTools.querySelectorAll("[data-mode]").forEach((button) => {
        button.addEventListener("click", () => {
            const mode = button.dataset.mode;

            try {
                draw.setMode(mode);
            } catch (error) {
                console.error(`Unable to enable ${mode} mode`, error);
                return;
            }

            els.aoiTools.querySelectorAll("[data-mode]").forEach((item) => {
                item.classList.toggle("active", item === button);
            });

            els.mapHint.textContent =
                `Drawing mode: ${mode}. Complete the geometry on the map.`;
        });
    });
}

function restoreStoredAoiOnMap() {
    if (!currentAoi || !draw) {
        renderAoiState();
        return;
    }

    try {
        draw.addFeatures([currentAoi]);
    } catch (error) {
        console.warn("Stored AOI could not be restored:", error);
    }

    renderAoiState();
}

function initializeControls() {
    els.searchForm.addEventListener("submit", (event) => {
        event.preventDefault();
        searchLocation(els.searchInput.value);
    });

    els.baseLayerToggle.addEventListener("change", () => {
        setBaseMapVisible(els.baseLayerToggle.checked);
        setSatelliteVisible(els.satelliteLayerToggle.checked);
        setAnalysisVisible(els.analysisLayerToggle.checked);
    });

    els.satelliteLayerToggle.addEventListener("change", () => {
        setSatelliteVisible(els.satelliteLayerToggle.checked);
    });

    els.analysisLayerToggle.addEventListener("change", () => {
        setAnalysisVisible(els.analysisLayerToggle.checked);
    });

    els.clearAoiBtn.addEventListener("click", () => {
        if (draw) {
            try {
                draw.clear();
            } catch (error) {
                console.warn(error);
            }
        }

        saveAoi(null);

        els.aoiTools.querySelectorAll("[data-mode]").forEach((button) => {
            button.classList.remove("active");
        });
    });

    els.fitAoiBtn.addEventListener("click", () => {
        if (currentAoi) fitFeature(currentAoi);
    });

    els.fitBoundsBtn.addEventListener("click", () => {
        if (!latestAnalysisBounds) return;

        map.fitBounds(latestAnalysisBounds, {
            padding: 70,
            duration: 700,
            maxZoom: 15,
        });
    });

    els.toggleOverlayBtn.addEventListener("click", () => {
        analysisOverlayVisible = !analysisOverlayVisible;

        const overlayId = `${ANALYSIS_PREFIX}evidence-layer`;

        if (map.getLayer(overlayId)) {
            map.setLayoutProperty(
                overlayId,
                "visibility",
                analysisOverlayVisible ? "visible" : "none"
            );
        }
    });

    els.resetGlobeBtn.addEventListener("click", () => {
        map.flyTo({
            center: DEFAULT_CENTER,
            zoom: DEFAULT_ZOOM,
            pitch: 0,
            bearing: 0,
            speed: 1.1,
        });
    });
}

function initializeMap() {
    if (typeof maplibregl === "undefined") {
        $("map").textContent =
            "MapLibre could not load. Check the network connection and reload.";
        return;
    }

    map = new maplibregl.Map({
        container: "map",
        style: BASE_STYLE_URL,
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        pitch: 0,
        bearing: 0,
        antialias: true,
        attributionControl: true,
        maxZoom: 20,
    });

    map.addControl(new maplibregl.NavigationControl({
        visualizePitch: true,
        showCompass: true,
        showZoom: true,
    }), "top-right");

    map.addControl(new maplibregl.FullscreenControl(), "top-right");

    map.addControl(new maplibregl.ScaleControl({
        maxWidth: 120,
        unit: "metric",
    }), "bottom-right");

    map.on("mousemove", (event) => {
        els.cursorLat.textContent = event.lngLat.lat.toFixed(6);
        els.cursorLng.textContent = event.lngLat.lng.toFixed(6);
    });

    map.on("zoom", () => {
        els.zoomLevel.textContent = map.getZoom().toFixed(2);
    });

    map.on("load", () => {
        try {
            map.setProjection({ type: "globe" });
        } catch (error) {
            console.warn("Globe projection unavailable; falling back to Mercator.", error);
        }

        addSatelliteLayer();
        initializeDrawing();
        initializeControls();
        restoreStoredAoiOnMap();
        renderAnalysis();

        els.zoomLevel.textContent = map.getZoom().toFixed(2);
    });
}

window.SatQueryMap = {
    getAOI() {
        if (!currentAoi) return null;

        return {
            type: "Feature",
            geometry: currentAoi.geometry,
            properties: {
                ...(currentAoi.properties || {}),
                satquery_role: "aoi",
            },
            bbox: featureBBox(currentAoi),
        };
    },

    clearAOI() {
        if (draw) {
            try {
                draw.clear();
            } catch {
                // no-op
            }
        }

        saveAoi(null);
    },

    getMap() {
        return map;
    },
};

loadState();
initializeMap();
})();
