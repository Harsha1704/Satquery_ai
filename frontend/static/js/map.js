function initializeMap() {
'use strict';


const $ = id =>
    document.getElementById(
        id
    );


/* ============================================================
   MAP
============================================================ */

const map =
    L.map(
        'map',
        {
            zoomControl:
                true
        }
    ).setView(
        [
            20.5937,
            78.9629
        ],
        5
    );


L.tileLayer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
        maxZoom:
            19,

        className:
            'satquery-basemap',

        attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }
).addTo(
    map
);


/* ============================================================
   STATE
============================================================ */

let records =
    [];

let currentRecord =
    null;

let currentAnalysis =
    null;

let currentAnalysisId =
    null;

let footprintLayers =
    [];

let rasterLayers =
    [];

let resultLayers =
    [];

let datasetBounds =
    null;

let selectionMode =
    false;

let selectionStart =
    null;

let selectionRectangle =
    null;

let selectionMarker =
    null;

let fixedBounds =
    null;

let reportUrl =
    null;

let busy =
    false;


/* ------------------------------------------------------------
   Historical comparison (Google Earth Engine) selection state.
   Kept separate from the dataset-ROI selection above so a user
   can draw a historical-comparison area on the bare map even
   when no GeoTIFF analysis has been loaded yet.
------------------------------------------------------------ */

let temporalSelectionMode =
    false;

let temporalSelectionStart =
    null;

let temporalSelectionRectangle =
    null;

let temporalSelectionMarker =
    null;

let temporalFixedBounds =
    null;

let temporalBusy =
    false;


/* ============================================================
   HELPERS
============================================================ */

function escapeHTML(
    value
) {

    const div =
        document.createElement(
            'div'
        );

    div.textContent =
        value == null
            ? ''
            : String(
                value
            );

    return div.innerHTML;
}


function leafletBounds(
    bounds
) {

    if (
        !Array.isArray(
            bounds
        )
        ||
        bounds.length !== 4
    ) {
        return null;
    }

    const [
        west,
        south,
        east,
        north
    ] =
        bounds.map(
            Number
        );

    if (
        ![
            west,
            south,
            east,
            north
        ].every(
            Number.isFinite
        )
    ) {
        return null;
    }

    return [
        [
            south,
            west
        ],
        [
            north,
            east
        ]
    ];
}


function boundsArray(
    bounds
) {

    return [
        bounds.getWest(),
        bounds.getSouth(),
        bounds.getEast(),
        bounds.getNorth()
    ];
}


function clearAnalysisLayers() {

    [
        ...footprintLayers,
        ...rasterLayers,
        ...resultLayers
    ].forEach(
        item => {

            if (
                map.hasLayer(
                    item.layer
                )
            ) {

                map.removeLayer(
                    item.layer
                );
            }
        }
    );

    footprintLayers =
        [];

    rasterLayers =
        [];

    resultLayers =
        [];

    datasetBounds =
        null;

    $('layerList')
        .innerHTML =
            '<div class="empty">No layers available.</div>';
}


function clearSelection() {

    selectionMode =
        false;

    selectionStart =
        null;

    fixedBounds =
        null;

    if (
        selectionRectangle
    ) {

        map.removeLayer(
            selectionRectangle
        );

        selectionRectangle =
            null;
    }

    if (
        selectionMarker
    ) {

        map.removeLayer(
            selectionMarker
        );

        selectionMarker =
            null;
    }

    map
        .getContainer()
        .classList
        .remove(
            'selecting-area'
        );

    $('fixAreaBtn')
        .disabled =
            true;

    $('clearAreaBtn')
        .disabled =
            true;

    $('mapQueryInput')
        .disabled =
            true;

    $('analyzeAreaBtn')
        .disabled =
            true;

    $('roiReadyLabel')
        .textContent =
            'Fix an area before analysis';

    $('selectionStatus')
        .textContent =
            'No area selected.';

    $('selectedBounds')
        .innerHTML =
            '<div class="empty">Fix an area to continue.</div>';
}


/* ============================================================
   LOAD ANALYSES
============================================================ */

function loadRecords() {

    const output =
        [];

    try {

        const stored =
            JSON.parse(
                localStorage.getItem(
                    'satquery-map-history'
                )
                || '[]'
            );

        if (
            Array.isArray(
                stored
            )
        ) {

            output.push(
                ...stored
            );
        }

    } catch {
        // ignore
    }


    try {

        const session =
            JSON.parse(
                sessionStorage.getItem(
                    'satquery-history'
                )
                || '[]'
            );

        if (
            Array.isArray(
                session
            )
        ) {

            output.push(
                ...session
            );
        }

    } catch {
        // ignore
    }


    const unique =
        [];

    const ids =
        new Set();


    output.forEach(
        record => {

            if (
                !record
                ||
                !record.result
                ||
                !record.result.success
                ||
                !record.result.analysis_id
            ) {
                return;
            }

            const geo =
                record.result.geospatial;

            if (
                !geo
                ||
                !geo.available
                ||
                !Array.isArray(
                    geo.layers
                )
                ||
                !geo.layers.length
            ) {
                return;
            }

            if (
                ids.has(
                    record.result.analysis_id
                )
            ) {
                return;
            }

            ids.add(
                record.result.analysis_id
            );

            unique.push(
                record
            );
        }
    );


    return unique.sort(
        (
            a,
            b
        ) =>
            Number(
                b.time
                || 0
            )
            -
            Number(
                a.time
                || 0
            )
    );
}


/* ============================================================
   SELECTOR
============================================================ */

function populateAnalysisSelector() {

    records =
        loadRecords();

    const select =
        $('analysisSelect');

    select.replaceChildren();


    if (
        !records.length
    ) {

        const option =
            document.createElement(
                'option'
            );

        option.textContent =
            'No GeoTIFF analysis available';

        option.value =
            '';

        select.append(
            option
        );

        select.disabled =
            true;

        $('analysisHint')
            .textContent =
                'Run a successful GeoTIFF analysis from Analyze first.';

        return;
    }


    select.disabled =
        false;


    records.forEach(
        (
            record,
            index
        ) => {

            const option =
                document.createElement(
                    'option'
                );

            option.value =
                String(
                    index
                );

            option.textContent =
                `${record.query} — ${record.result.task || record.result.intent}`;

            select.append(
                option
            );
        }
    );


    $('analysisHint')
        .textContent =
            `${records.length} geospatial analysis${records.length === 1 ? '' : 'es'} available.`;


    select.value =
        '0';


    loadAnalysis(
        0
    );
}


/* ============================================================
   LAYER CONTROL
============================================================ */

function addLayerControl(
    name,
    object,
    color,
    checked
) {

    const label =
        document.createElement(
            'label'
        );

    label.className =
        'layer-toggle';


    const checkbox =
        document.createElement(
            'input'
        );

    checkbox.type =
        'checkbox';

    checkbox.checked =
        checked;


    const swatch =
        document.createElement(
            'span'
        );

    swatch.className =
        'layer-swatch';

    swatch.style.background =
        color;


    const text =
        document.createElement(
            'span'
        );

    text.textContent =
        name;


    checkbox.addEventListener(
        'change',
        () => {

            if (
                checkbox.checked
            ) {

                if (
                    !map.hasLayer(
                        object.layer
                    )
                ) {

                    object.layer.addTo(
                        map
                    );
                }

            } else {

                if (
                    map.hasLayer(
                        object.layer
                    )
                ) {

                    map.removeLayer(
                        object.layer
                    );
                }
            }
        }
    );


    object.checkbox =
        checkbox;


    label.append(
        checkbox,
        swatch,
        text
    );


    $('layerList')
        .append(
            label
        );
}


/* ============================================================
   LOAD ANALYSIS
============================================================ */

function loadAnalysis(
    index
) {

    clearAnalysisLayers();

    clearSelection();


    currentRecord =
        records[
            index
        ];

    if (
        !currentRecord
    ) {
        return;
    }


    currentAnalysis =
        currentRecord.result;

    currentAnalysisId =
        currentAnalysis.analysis_id;


    const geo =
        currentAnalysis.geospatial;


    $('mapQuestion')
        .textContent =
            currentAnalysis.query
            || '—';

    $('mapAnswer')
        .textContent =
            currentAnalysis.answer
            || '—';

    $('mapModel')
        .textContent =
            currentAnalysis.model
            || '—';

    $('mapTask')
        .textContent =
            currentAnalysis.task
            || currentAnalysis.intent
            || '—';


    const confidence =
        Number(
            currentAnalysis.confidence
            || 0
        );


    $('mapConfidence')
        .textContent =
            Number.isFinite(
                confidence
            )
                ? `${confidence.toFixed(1)}%`
                : '—';


    $('layerList')
        .replaceChildren();


    const boundsCollection =
        [];


    geo.layers.forEach(
        (
            layer,
            index
        ) => {

            const bounds =
                leafletBounds(
                    layer.bounds_wgs84
                );

            if (!bounds) {
                return;
            }


            boundsCollection.push(
                bounds
            );


            const rectangle =
                L.rectangle(
                    bounds,
                    {
                        color:
                            '#1eb7ff',

                        weight:
                            2,

                        dashArray:
                            '6 5',

                        fillOpacity:
                            0.01
                    }
                )
                .addTo(
                    map
                );


            const object = {
                layer:
                    rectangle,

                role:
                    'footprint'
            };


            footprintLayers.push(
                object
            );


            addLayerControl(
                `${layer.name} footprint`,
                object,
                '#1eb7ff',
                true
            );


            if (
                index === 0
            ) {

                renderMetadata(
                    layer
                );
            }
        }
    );


    (
        geo.visual_layers
        || []
    ).forEach(
        visual => {

            const bounds =
                leafletBounds(
                    visual.bounds_wgs84
                );

            if (
                !bounds
                ||
                !visual.url
            ) {
                return;
            }


            const overlay =
                L.imageOverlay(
                    visual.url,
                    bounds,
                    {
                        opacity:
                            Number(
                                visual.opacity
                            )
                            || 0.75
                    }
                );


            const visible =
                visual.visible
                !== false;


            if (visible) {

                overlay.addTo(
                    map
                );
            }


            const object = {

                layer:
                    overlay,

                role:
                    visual.role
                    || 'source',

                checkbox:
                    null
            };


            rasterLayers.push(
                object
            );


            if (
                object.role
                === 'result'
            ) {

                resultLayers.push(
                    object
                );
            }


            addLayerControl(
                visual.name
                || 'Raster',

                object,

                object.role
                === 'result'
                    ? '#22dda7'
                    : '#168dff',

                visible
            );
        }
    );


    if (
        boundsCollection.length
    ) {

        const combined =
            L.latLngBounds(
                boundsCollection[0]
            );


        boundsCollection
            .slice(1)
            .forEach(
                bounds =>
                    combined.extend(
                        bounds
                    )
            );


        datasetBounds =
            combined;


        map.fitBounds(
            combined,
            {
                padding:
                    [
                        30,
                        30
                    ],

                maxZoom:
                    17
            }
        );
    }


    $('fitBoundsBtn')
        .disabled =
            !datasetBounds;


    $('toggleOverlayBtn')
        .disabled =
            !resultLayers.length;


    $('selectAreaBtn')
        .disabled =
            !datasetBounds;


    $('mapEmptyState')
        .classList
        .add(
            'hidden'
        );


    reportUrl =
        currentAnalysis
            .reports
            ?.html
        || null;


    $('generateReportBtn')
        .disabled =
            !reportUrl;


    $('reportStatus')
        .textContent =
            reportUrl
                ? 'A report is available for the current analysis.'
                : 'Select an area and run a new analysis.';


    setTimeout(
        () =>
            map.invalidateSize(),
        120
    );
}


/* ============================================================
   METADATA
============================================================ */

function renderMetadata(
    layer
) {

    $('datasetMetadata')
        .innerHTML = `

        <div class="meta-row">
            <span>Width</span>
            <strong>${escapeHTML(layer.width)}</strong>
        </div>

        <div class="meta-row">
            <span>Height</span>
            <strong>${escapeHTML(layer.height)}</strong>
        </div>

        <div class="meta-row">
            <span>Bands</span>
            <strong>${escapeHTML(layer.bands)}</strong>
        </div>

        <div class="meta-row">
            <span>CRS</span>
            <strong>${escapeHTML(layer.source_crs)}</strong>
        </div>

        <div class="meta-row">
            <span>Resolution</span>
            <strong>
                ${escapeHTML(layer.resolution_x)}
                ×
                ${escapeHTML(layer.resolution_y)}
            </strong>
        </div>

    `;


    const bounds =
        layer.bounds_wgs84;


    $('spatialMetadata')
        .innerHTML = `

        <div class="meta-row">
            <span>West</span>
            <strong>${Number(bounds[0]).toFixed(6)}</strong>
        </div>

        <div class="meta-row">
            <span>South</span>
            <strong>${Number(bounds[1]).toFixed(6)}</strong>
        </div>

        <div class="meta-row">
            <span>East</span>
            <strong>${Number(bounds[2]).toFixed(6)}</strong>
        </div>

        <div class="meta-row">
            <span>North</span>
            <strong>${Number(bounds[3]).toFixed(6)}</strong>
        </div>

        <div class="meta-row">
            <span>NoData</span>
            <strong>${escapeHTML(layer.nodata ?? '—')}</strong>
        </div>

    `;
}


/* ============================================================
   AREA SELECTION
============================================================ */

$('selectAreaBtn')
    .addEventListener(
        'click',
        () => {

            clearSelection();

            if (typeof clearTemporalSelection === 'function') {
                clearTemporalSelection();
            }

            selectionMode =
                true;


            map
                .getContainer()
                .classList
                .add(
                    'selecting-area'
                );


            $('selectionStatus')
                .textContent =
                    'Click the first corner of the area.';


            $('clearAreaBtn')
                .disabled =
                    false;
        }
    );


map.on(
    'click',
    event => {

        if (
            temporalSelectionMode
        ) {

            handleTemporalMapClick(
                event
            );

            return;
        }


        if (
            !selectionMode
        ) {
            return;
        }


        if (
            !selectionStart
        ) {

            selectionStart =
                event.latlng;


            selectionMarker =
                L.circleMarker(
                    event.latlng,
                    {
                        radius:
                            5,

                        color:
                            '#21d8a4',

                        fillColor:
                            '#21d8a4',

                        fillOpacity:
                            1
                    }
                )
                .addTo(
                    map
                );


            $('selectionStatus')
                .textContent =
                    'Now click the opposite corner.';


            return;
        }


        const bounds =
            L.latLngBounds(
                selectionStart,
                event.latlng
            );


        if (
            selectionRectangle
        ) {

            map.removeLayer(
                selectionRectangle
            );
        }


        selectionRectangle =
            L.rectangle(
                bounds,
                {
                    color:
                        '#21d8a4',

                    weight:
                        3,

                    fillColor:
                        '#21d8a4',

                    fillOpacity:
                        0.10
                }
            )
            .addTo(
                map
            );


        selectionMode =
            false;


        map
            .getContainer()
            .classList
            .remove(
                'selecting-area'
            );


        $('fixAreaBtn')
            .disabled =
                false;


        $('selectionStatus')
            .textContent =
                'Area selected. Click Fix Area.';
    }
);


/* ============================================================
   FIX AREA
============================================================ */

$('fixAreaBtn')
    .addEventListener(
        'click',
        () => {

            if (
                !selectionRectangle
            ) {
                return;
            }


            fixedBounds =
                selectionRectangle
                    .getBounds();


            const west =
                fixedBounds
                    .getWest();

            const south =
                fixedBounds
                    .getSouth();

            const east =
                fixedBounds
                    .getEast();

            const north =
                fixedBounds
                    .getNorth();


            $('selectedBounds')
                .innerHTML = `

                <div class="meta-row">
                    <span>West</span>
                    <strong>${west.toFixed(6)}</strong>
                </div>

                <div class="meta-row">
                    <span>South</span>
                    <strong>${south.toFixed(6)}</strong>
                </div>

                <div class="meta-row">
                    <span>East</span>
                    <strong>${east.toFixed(6)}</strong>
                </div>

                <div class="meta-row">
                    <span>North</span>
                    <strong>${north.toFixed(6)}</strong>
                </div>

            `;


            $('selectionStatus')
                .textContent =
                    'Area fixed and ready for analysis.';


            $('roiReadyLabel')
                .textContent =
                    'Selected area ready';


            $('mapQueryInput')
                .disabled =
                    false;


            $('analyzeAreaBtn')
                .disabled =
                    false;


            $('mapQueryInput')
                .focus();
        }
    );


/* ============================================================
   CLEAR AREA
============================================================ */

$('clearAreaBtn')
    .addEventListener(
        'click',
        clearSelection
    );


/* ============================================================
   HISTORICAL COMPARISON (GOOGLE EARTH ENGINE)
============================================================ */

function populateYearSelects() {

    const currentYear =
        new Date().getFullYear();

    const beforeSelect =
        $('temporalBeforeYear');

    const afterSelect =
        $('temporalAfterYear');

    beforeSelect.replaceChildren();
    afterSelect.replaceChildren();

    for (
        let year = currentYear;
        year >= 1985;
        year--
    ) {

        const beforeOption =
            document.createElement('option');

        beforeOption.value =
            String(year);

        beforeOption.textContent =
            String(year);

        beforeSelect.append(
            beforeOption
        );


        const afterOption =
            document.createElement('option');

        afterOption.value =
            String(year);

        afterOption.textContent =
            String(year);

        afterSelect.append(
            afterOption
        );
    }

    beforeSelect.value =
        String(
            Math.max(
                1985,
                currentYear - 10
            )
        );

    afterSelect.value =
        String(currentYear);
}


populateYearSelects();


function clearTemporalSelection() {

    temporalSelectionMode =
        false;

    temporalSelectionStart =
        null;

    temporalFixedBounds =
        null;

    if (temporalSelectionRectangle) {
        map.removeLayer(temporalSelectionRectangle);
        temporalSelectionRectangle = null;
    }

    if (temporalSelectionMarker) {
        map.removeLayer(temporalSelectionMarker);
        temporalSelectionMarker = null;
    }

    map
        .getContainer()
        .classList
        .remove('selecting-area');

    $('temporalFixAreaBtn').disabled = true;
    $('temporalClearAreaBtn').disabled = true;
    $('temporalQueryInput').disabled = true;
    $('temporalAnalyzeBtn').disabled = true;

    $('temporalSelectionStatus').textContent =
        'No area selected.';

    $('temporalHint').textContent =
        'Draw and fix an area above, then compare two years.';
}


$('temporalSelectAreaBtn')
    .addEventListener(
        'click',
        () => {

            clearTemporalSelection();

            // A historical comparison and a dataset-ROI selection
            // can't run at the same time on one map.
            clearSelection();

            temporalSelectionMode = true;

            map
                .getContainer()
                .classList
                .add('selecting-area');

            $('temporalSelectionStatus').textContent =
                'Click the first corner of the area.';

            $('temporalClearAreaBtn').disabled = false;
        }
    );


function handleTemporalMapClick(event) {

    if (!temporalSelectionStart) {

        temporalSelectionStart =
            event.latlng;

        temporalSelectionMarker =
            L.circleMarker(
                event.latlng,
                {
                    radius: 5,
                    color: '#ffb020',
                    fillColor: '#ffb020',
                    fillOpacity: 1,
                }
            ).addTo(map);

        $('temporalSelectionStatus').textContent =
            'Now click the opposite corner.';

        return;
    }

    const bounds =
        L.latLngBounds(
            temporalSelectionStart,
            event.latlng
        );

    if (temporalSelectionRectangle) {
        map.removeLayer(temporalSelectionRectangle);
    }

    temporalSelectionRectangle =
        L.rectangle(
            bounds,
            {
                color: '#ffb020',
                weight: 3,
                fillColor: '#ffb020',
                fillOpacity: 0.10,
            }
        ).addTo(map);

    temporalSelectionMode = false;

    map
        .getContainer()
        .classList
        .remove('selecting-area');

    $('temporalFixAreaBtn').disabled = false;

    $('temporalSelectionStatus').textContent =
        'Area selected. Click Fix Area.';
}


/* ------------------------------------------------------------
   Mirrors gee_temporal.py's sensor thresholds and ROI size caps so
   an oversized rectangle is rejected instantly in the browser
   instead of round-tripping to Earth Engine first.
------------------------------------------------------------ */

function resolutionForYear(year) {
    // Sentinel-2 SR only exists in Earth Engine from 2017 onward.
    return year >= 2017 ? 10 : 30;
}


function roiSizeKm(bounds) {

    const nw =
        L.latLng(bounds.getNorth(), bounds.getWest());

    const ne =
        L.latLng(bounds.getNorth(), bounds.getEast());

    const sw =
        L.latLng(bounds.getSouth(), bounds.getWest());

    const widthKm =
        nw.distanceTo(ne) / 1000;

    const heightKm =
        nw.distanceTo(sw) / 1000;

    return { widthKm, heightKm };
}


function checkTemporalRoiSize(bounds, beforeYear, afterYear) {

    const targetScale =
        Math.max(
            resolutionForYear(beforeYear),
            resolutionForYear(afterYear)
        );

    const maxKm =
        targetScale <= 10 ? 5 : 10;

    const { widthKm, heightKm } =
        roiSizeKm(bounds);

    if (widthKm > maxKm || heightKm > maxKm) {
        return {
            ok: false,
            message:
                `Selected area is about ${widthKm.toFixed(1)} km x `
                + `${heightKm.toFixed(1)} km, which is too large for a `
                + `${targetScale} m comparison (limit ${maxKm} km x `
                + `${maxKm} km). Clear and draw a smaller area.`,
        };
    }

    return { ok: true };
}


$('temporalFixAreaBtn')
    .addEventListener(
        'click',
        () => {

            if (!temporalSelectionRectangle) {
                return;
            }

            const candidateBounds =
                temporalSelectionRectangle.getBounds();

            const beforeYear =
                Number($('temporalBeforeYear').value);

            const afterYear =
                Number($('temporalAfterYear').value);

            const sizeCheck =
                checkTemporalRoiSize(
                    candidateBounds,
                    beforeYear,
                    afterYear
                );

            if (!sizeCheck.ok) {
                $('temporalSelectionStatus').textContent =
                    sizeCheck.message;
                return;
            }

            temporalFixedBounds =
                candidateBounds;

            $('temporalSelectionStatus').textContent =
                'Area fixed. Pick two years and ask your question.';

            $('temporalHint').textContent =
                'Ready \u2014 imagery will be pulled for both years over this exact area.';

            $('temporalQueryInput').disabled = false;
            $('temporalAnalyzeBtn').disabled = false;

            $('temporalQueryInput').focus();
        }
    );


$('temporalClearAreaBtn')
    .addEventListener(
        'click',
        clearTemporalSelection
    );


$('temporalAnalyzeBtn')
    .addEventListener(
        'click',
        analyzeTemporalArea
    );


$('temporalQueryInput')
    .addEventListener(
        'keydown',
        event => {

            if (
                event.ctrlKey
                &&
                event.key === 'Enter'
            ) {
                analyzeTemporalArea();
            }
        }
    );


async function analyzeTemporalArea() {

    if (
        temporalBusy
        ||
        !temporalFixedBounds
    ) {
        return;
    }

    const query =
        $('temporalQueryInput')
            .value
            .trim()
        ||
        'What changed in this area?';

    const beforeYear =
        Number($('temporalBeforeYear').value);

    const afterYear =
        Number($('temporalAfterYear').value);

    if (beforeYear === afterYear) {
        $('temporalHint').textContent =
            'Pick two different years to compare.';
        return;
    }

    const sizeCheck =
        checkTemporalRoiSize(
            temporalFixedBounds,
            beforeYear,
            afterYear
        );

    if (!sizeCheck.ok) {
        $('temporalHint').textContent =
            sizeCheck.message;
        return;
    }

    temporalBusy = true;
    document.querySelector('.map-app').inert = true;

    reportUrl = null;
    $('generateReportBtn').disabled = true;
    $('reportStatus').textContent =
        `Pulling ${beforeYear} and ${afterYear} imagery from Earth Engine...`;

    $('temporalAnalyzeBtn').disabled = true;

    $('mapLoading').classList.remove('hidden');
    $('mapLoading').querySelector('strong').textContent =
        'Fetching historical imagery';
    $('mapLoading').querySelector('span').textContent =
        `Comparing ${beforeYear} \u2192 ${afterYear} over the selected area...`;

    try {

        const response =
            await fetch(
                '/api/map-temporal-analyze',
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query,
                        bounds: boundsArray(temporalFixedBounds),
                        before_year: beforeYear,
                        after_year: afterYear,
                    }),
                }
            );

        const data =
            await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.error
                || data.answer
                || 'Historical comparison failed.'
            );
        }

        $('mapQuestion').textContent =
            data.query || query;

        $('mapAnswer').textContent =
            data.answer || 'Analysis completed.';

        $('mapModel').textContent =
            data.model || '\u2014';

        $('mapTask').textContent =
            data.task
            || `Historical Change \u00b7 ${beforeYear} vs ${afterYear}`;

        const confidence =
            Number(data.confidence || 0);

        $('mapConfidence').textContent =
            `${confidence.toFixed(1)}%`;

        reportUrl =
            data.reports?.html
            || null;

        $('generateReportBtn').disabled =
            !reportUrl;

        $('reportStatus').textContent =
            reportUrl
                ? 'Historical comparison report is ready.'
                : 'Comparison completed, but no report was produced.';

        const newRecord = {
            query: data.query,
            mode: 'map_temporal',
            time: Date.now(),
            result: data,
        };

        let stored = [];

        try {
            stored =
                JSON.parse(
                    localStorage.getItem('satquery-map-history')
                    || '[]'
                );
        } catch {
            stored = [];
        }

        if (!Array.isArray(stored)) {
            stored = [];
        }

        stored.unshift(newRecord);

        try {
            localStorage.setItem(
                'satquery-map-history',
                JSON.stringify(stored.slice(0, 30))
            );
        } catch {
            $('reportStatus').textContent +=
                ' Browser history could not be saved.';
        }

        records.unshift(newRecord);

        const selector = $('analysisSelect');
        selector.replaceChildren();

        records.forEach((record, index) => {
            const option = document.createElement('option');
            option.value = String(index);
            option.textContent =
                `${record.query} - ${record.result.task || record.result.intent}`;
            selector.append(option);
        });

        selector.disabled = false;

        selector.value = '0';

        $('analysisHint').textContent =
            `${records.length} geospatial analysis${records.length === 1 ? '' : 'es'} available.`;

        loadAnalysis(0);

        clearTemporalSelection();

    } catch (error) {

        reportUrl = null;
        $('generateReportBtn').disabled = true;
        $('reportStatus').textContent =
            'Run a successful comparison to generate a report.';

        $('mapQuestion').textContent = query;

        $('mapAnswer').textContent =
            error.message || 'Historical comparison failed.';

        $('mapConfidence').textContent = '0.0%';
        $('mapModel').textContent = '\u2014';
        $('mapTask').textContent = 'Failed';

    } finally {

        document.querySelector('.map-app').inert = false;
        temporalBusy = false;

        $('mapLoading').classList.add('hidden');

        $('mapLoading').querySelector('strong').textContent =
            'Analyzing selected region';
        $('mapLoading').querySelector('span').textContent =
            'Cropping geospatial data and running SatQuery AI...';

        $('temporalAnalyzeBtn').disabled =
            !temporalFixedBounds;
    }
}


/* ============================================================
   QUERY EXAMPLES
============================================================ */

document
    .querySelectorAll(
        '[data-map-query]'
    )
    .forEach(
        button => {

            button.addEventListener(
                'click',
                () => {

                    if (
                        !fixedBounds
                    ) {
                        return;
                    }


                    $('mapQueryInput')
                        .value =
                            button
                                .dataset
                                .mapQuery;


                    $('mapQueryInput')
                        .focus();
                }
            );
        }
    );


/* ============================================================
   RUN SELECTED AREA ANALYSIS
============================================================ */

$('analyzeAreaBtn')
    .addEventListener(
        'click',
        analyzeSelectedArea
    );


$('mapQueryInput')
    .addEventListener(
        'keydown',
        event => {

            if (
                event.ctrlKey
                &&
                event.key
                === 'Enter'
            ) {

                analyzeSelectedArea();
            }
        }
    );


async function analyzeSelectedArea() {

    if (
        busy
        ||
        !fixedBounds
        ||
        !currentAnalysisId
    ) {
        return;
    }


    const query =
        $('mapQueryInput')
            .value
            .trim();


    if (!query) {

        $('mapQueryInput')
            .focus();

        return;
    }


    busy = true;
    document.querySelector('.map-app').inert = true;
    reportUrl = null;
    $('generateReportBtn').disabled = true;
    $('reportStatus').textContent = 'Analyzing the selected area...';


    $('analyzeAreaBtn')
        .disabled =
            true;


    $('mapLoading')
        .classList
        .remove(
            'hidden'
        );


    try {

        const response =
            await fetch(
                '/api/map-analyze',
                {
                    method:
                        'POST',

                    headers:
                        {
                            'Content-Type':
                                'application/json'
                        },

                    body:
                        JSON.stringify(
                            {
                                analysis_id:
                                    currentAnalysisId,

                                query:
                                    query,

                                bounds:
                                    boundsArray(
                                        fixedBounds
                                    )
                            }
                        )
                }
            );


        const data =
            await response.json();


        if (
            !response.ok
            ||
            !data.success
        ) {

            throw new Error(
                data.error
                ||
                data.answer
                ||
                'Selected-area analysis failed.'
            );
        }


        $('mapQuestion')
            .textContent =
                data.query
                || query;


        $('mapAnswer')
            .textContent =
                data.answer
                || 'Analysis completed.';


        $('mapModel')
            .textContent =
                data.model
                || '—';


        $('mapTask')
            .textContent =
                data.task
                || data.intent
                || '—';


        const confidence =
            Number(
                data.confidence
                || 0
            );


        $('mapConfidence')
            .textContent =
                `${confidence.toFixed(1)}%`;


        reportUrl =
            data.reports
                ?.html
            || null;


        $('generateReportBtn')
            .disabled =
                !reportUrl;


        $('reportStatus')
            .textContent =
                reportUrl
                    ? 'Selected-area report is ready.'
                    : 'Analysis completed, but no report was produced.';


        /*
            Add selected-area result as another
            map analysis so user can inspect it later.
        */

        const newRecord = {

            query:
                data.query,

            mode:
                'map_roi',

            time:
                Date.now(),

            result:
                data
        };


        let stored = [];

        try {

            stored =
                JSON.parse(
                    localStorage.getItem(
                        'satquery-map-history'
                    )
                    || '[]'
                );

        } catch {
            stored = [];
        }


        if (
            !Array.isArray(
                stored
            )
        ) {
            stored = [];
        }


        stored.unshift(
            newRecord
        );


        try {
            localStorage.setItem('satquery-map-history', JSON.stringify(stored.slice(0, 30)));
        } catch {
            $('reportStatus').textContent += ' Browser history could not be saved.';
        }
        records.unshift(newRecord);
        const selector = $('analysisSelect');
        selector.replaceChildren();
        records.forEach((record, index) => {
            const option = document.createElement('option');
            option.value = String(index);
            option.textContent = `${record.query} - ${record.result.task || record.result.intent}`;
            selector.append(option);
        });
        selector.value = String(records.indexOf(currentRecord));
        $('analysisHint').textContent = `${records.length} geospatial analyses available.`;

        /*
            Render new result overlays immediately.
        */

        renderResultLayers(
            data
        );


    } catch (error) {

        reportUrl = null;
        $('generateReportBtn').disabled = true;
        $('reportStatus').textContent = 'Run a successful selected-area analysis to generate a report.';
        renderResultLayers({});
        $('mapQuestion')
            .textContent =
                query;


        $('mapAnswer')
            .textContent =
                error.message
                ||
                'Analysis failed.';


        $('mapConfidence')
            .textContent =
                '0.0%';


        $('mapModel')
            .textContent =
                '—';


        $('mapTask')
            .textContent =
                'Failed';


    } finally {

        document.querySelector('.map-app').inert = false;
        busy =
            false;


        $('mapLoading')
            .classList
            .add(
                'hidden'
            );


        $('analyzeAreaBtn')
            .disabled =
                !fixedBounds;
    }
}


/* ============================================================
   RESULT OVERLAY
============================================================ */

function renderResultLayers(data) {
    const previous = new Set(resultLayers);
    resultLayers.forEach(object => {
        if (map.hasLayer(object.layer)) map.removeLayer(object.layer);
        object.checkbox?.closest('label')?.remove();
    });
    rasterLayers = rasterLayers.filter(object => !previous.has(object));
    resultLayers = [];
    const visuals = data.geospatial?.visual_layers || [];
    visuals.filter(visual => visual.role === 'result').forEach(visual => {
        const bounds = leafletBounds(visual.bounds_wgs84);
        if (!bounds || !visual.url) return;
        const overlay = L.imageOverlay(visual.url, bounds, {
            opacity: Number.isFinite(Number(visual.opacity)) ? Number(visual.opacity) : 0.72,
        }).addTo(map);
        const object = { layer: overlay, role: 'result', checkbox: null };
        resultLayers.push(object);
        rasterLayers.push(object);
        addLayerControl(visual.name || 'Selected-area result', object, '#22dda7', true);
    });
    $('toggleOverlayBtn').disabled = !resultLayers.length;
}

/* ============================================================
   FIT
============================================================ */

$('fitBoundsBtn')
    .addEventListener(
        'click',
        () => {

            if (
                datasetBounds
            ) {

                map.fitBounds(
                    datasetBounds,
                    {
                        padding:
                            [
                                30,
                                30
                            ],

                        maxZoom:
                            17
                    }
                );
            }
        }
    );


/* ============================================================
   TOGGLE RESULTS
============================================================ */

$('toggleOverlayBtn')
    .addEventListener(
        'click',
        () => {

            if (
                !resultLayers.length
            ) {
                return;
            }


            const visible =
                resultLayers.some(
                    item =>
                        map.hasLayer(
                            item.layer
                        )
                );


            resultLayers.forEach(
                item => {
                    if (item.checkbox) item.checkbox.checked = !visible;

                    if (visible) {

                        if (
                            map.hasLayer(
                                item.layer
                            )
                        ) {

                            map.removeLayer(
                                item.layer
                            );
                        }

                    } else {

                        item.layer.addTo(
                            map
                        );
                    }
                }
            );
        }
    );


/* ============================================================
   REPORT
============================================================ */

$('generateReportBtn')
    .addEventListener(
        'click',
        () => {

            if (
                reportUrl
            ) {

                window.open(
                    reportUrl,
                    '_blank',
                    'noopener'
                );
            }
        }
    );


/* ============================================================
   SELECT ANALYSIS
============================================================ */

$('analysisSelect')
    .addEventListener(
        'change',
        () => {

            const index =
                Number(
                    $('analysisSelect')
                        .value
                );


            if (
                Number.isInteger(
                    index
                )
                &&
                records[index]
            ) {

                loadAnalysis(
                    index
                );
            }
        }
    );


/* ============================================================
   CURSOR
============================================================ */

map.on(
    'mousemove',
    event => {

        $('cursorLat')
            .textContent =
                event.latlng.lat
                    .toFixed(
                        6
                    );


        $('cursorLng')
            .textContent =
                event.latlng.lng
                    .toFixed(
                        6
                    );
    }
);


map.on(
    'zoomend',
    () => {

        $('zoomLevel')
            .textContent =
                map.getZoom();
    }
);


$('zoomLevel')
    .textContent =
        map.getZoom();


/* ============================================================
   INITIALIZE
============================================================ */

$('selectAreaBtn').disabled = true;
populateAnalysisSelector();


setTimeout(
    () =>
        map.invalidateSize(),
    250
);

}

if (typeof L !== 'undefined') {
    initializeMap();
} else {
    document.getElementById('mapEmptyState').textContent = 'The map library could not load. Check your connection and reload.';
    document.querySelectorAll('.map-app button').forEach(button => { button.disabled = true; });
}