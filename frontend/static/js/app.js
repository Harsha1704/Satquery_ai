'use strict';

const $ = (id) => document.getElementById(id);

const form = $('analysisForm');
const queryInput = $('queryInput');

const inputs = Object.fromEntries(
    ['single', 'before', 'after', 'optical', 'sar']
        .map(
            key => [
                key,
                $(`${key}Image`)
            ]
        )
);

const modes = {
    single: {
        files: ['single'],
        title: 'Upload a satellite image',
        subtitle: 'PNG, JPG, JPEG, WebP, TIFF or GeoTIFF',
        formats: 'Single Image: PNG, JPG, JPEG, WebP, TIFF/GeoTIFF (Max 200MB)',
        image: 'example-roads.png',
    },

    change: {
        files: [
            'before',
            'after'
        ],
        title: 'Upload before & after satellite images',
        subtitle: 'Use two images of the same area',
        formats: 'Bi-temporal Change: PNG, JPG, JPEG, WebP, TIFF/GeoTIFF (2 images, Max 200MB total)',
        image: 'deforestation.jpg',
    },

    fusion: {
        files: [
            'optical',
            'sar'
        ],
        title: 'Upload optical/multispectral + SAR data',
        subtitle: 'Optical and SAR inputs are required',
        formats: 'Optical + SAR: GeoTIFF/TIFF or NPY for optical; GeoTIFF/TIFF or NPY for SAR',
        image: 'coastal.jpg',
    },

    multispectral: {
        files: ['optical'],
        title: 'Upload multispectral GeoTIFF/TIFF or NPY',
        subtitle: 'Requires spectral bands for NDVI/NDWI/NDBI',
        formats: 'Multispectral: multiband GeoTIFF/TIFF or NPY with at least 4 spectral bands',
        image: 'agriculture.webp',
    },

    sar: {
        files: ['sar'],
        title: 'Upload SAR GeoTIFF/TIFF or NPY',
        subtitle: 'Radar data such as Sentinel-1 VV/VH',
        formats: 'SAR: GeoTIFF/TIFF or NPY containing SAR/radar bands',
        image: 'sar.jpeg',
    },

    grounding: {
        files: ['single'],
        title: 'Upload a satellite image for text-guided detection',
        subtitle: 'Ask SatQuery to find regions using natural language',
        formats: 'Text-Grounded Detection: PNG, JPG, JPEG, WebP, TIFF/GeoTIFF',
        image: 'urban.webp',
    },
};


const sampleHistory = [
    {
        query: 'are there roads in image',
        mode: 'single',
        example: true
    },
    {
        query: 'change between 2020-2025',
        mode: 'change',
        example: true
    },
    {
        query: 'NDVI analysis',
        mode: 'multispectral',
        example: true
    },
];


let activeMode = 'single';

let evidenceItems = [];

let activeEvidence = 0;

let currentResult = null;

let busy = false;

let toastTimer;

let history = [];


try {

    const saved = JSON.parse(
        sessionStorage.getItem(
            'satquery-history'
        ) || '[]'
    );

    if (
        Array.isArray(
            saved
        )
    ) {

        history = saved
            .filter(
                item =>
                    item
                    && typeof item.query === 'string'
                    && modes[item.mode]
                    && item.result
            )
            .slice(
                0,
                20
            );
    }

} catch {

    // Storage is optional.

}


function icon(name) {

    const svg =
        document.createElementNS(
            'http://www.w3.org/2000/svg',
            'svg'
        );

    svg.setAttribute(
        'class',
        'icon'
    );

    svg.setAttribute(
        'aria-hidden',
        'true'
    );


    const use =
        document.createElementNS(
            svg.namespaceURI,
            'use'
        );

    use.setAttribute(
        'href',
        `/static/images/icons.svg#${name}`
    );

    svg.append(
        use
    );

    return svg;
}


function imageNode(
    url,
    description
) {

    const img =
        document.createElement(
            'img'
        );

    img.src = url;

    img.alt =
        description
        || 'Satellite imagery';

    return img;
}


function localUrl(value) {

    if (
        typeof value !== 'string'
        || !value
    ) {
        return null;
    }

    try {

        const url =
            new URL(
                value,
                location.origin
            );

        return (
            url.origin === location.origin
            && [
                'http:',
                'https:'
            ].includes(
                url.protocol
            )
        )
            ? url.href
            : null;

    } catch {

        return null;
    }
}


function toast(message) {

    clearTimeout(
        toastTimer
    );

    $('toast').textContent =
        message;

    $('toast')
        .classList
        .remove(
            'hidden'
        );

    toastTimer =
        setTimeout(
            () =>
                $('toast')
                    .classList
                    .add(
                        'hidden'
                    ),
            4500
        );
}


function formMessage(
    message = ''
) {

    $('formMessage').textContent =
        message;

    $('formMessage')
        .classList
        .toggle(
            'hidden',
            !message
        );
}


function updateCount() {

    $('charCount').textContent =
        `${queryInput.value.length}/500`;
}


function closeSidebar() {

    $('sidebar')
        .classList
        .remove(
            'open'
        );

    $('mobileMenuBtn')
        .setAttribute(
            'aria-expanded',
            'false'
        );
}


function setMode(mode) {

    if (
        !modes[mode]
        || busy
    ) {
        return;
    }


    activeMode =
        mode;


    document
        .querySelectorAll(
            '.mode-item'
        )
        .forEach(
            button => {

                const active =
                    button.dataset.mode
                    === mode;

                button.classList.toggle(
                    'active',
                    active
                );

                button.setAttribute(
                    'aria-pressed',
                    String(
                        active
                    )
                );
            }
        );


    Object
        .entries(
            inputs
        )
        .forEach(
            (
                [
                    key,
                    input
                ]
            ) => {

                input.disabled =
                    !modes[mode]
                        .files
                        .includes(
                            key
                        );
            }
        );


    refreshFiles();

    formMessage();

    closeSidebar();
}


function refreshFiles() {

    const required =
        modes[activeMode].files;


    Object
        .entries(
            inputs
        )
        .forEach(
            (
                [
                    key,
                    input
                ]
            ) => {

                const field =
                    document.querySelector(
                        `[data-file-wrapper="${key}"]`
                    );

                const file =
                    input.files[0];


                field.classList.toggle(
                    'hidden',
                    !required.includes(
                        key
                    )
                    || (
                        required.length === 1
                        && !file
                    )
                );


                field
                    .querySelector(
                        '.file-name'
                    )
                    .textContent =
                    file
                        ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`
                        : 'Choose a file';


                field
                    .querySelector(
                        '.remove-file'
                    )
                    .classList
                    .toggle(
                        'hidden',
                        !file
                    );
            }
        );


    const count =
        required.filter(
            key =>
                inputs[key]
                    .files
                    .length
        ).length;


    const config =
        modes[activeMode];


    $('uploadTitle').textContent =
        count === required.length
            ? `${
                count === 1
                    ? 'Input'
                    : 'Input pair'
            } ready for analysis`
            : config.title;


    $('uploadSubtitle').textContent =
        count
        && count < required.length
            ? `Choose the ${
                required.find(
                    key =>
                        !inputs[key]
                            .files
                            .length
                )
            } input to complete this analysis`
            : count
                ? 'Click to replace or drop new data'
                : config.subtitle;


    $('uploadFormats').textContent =
        `${config.formats} · Historical Map Comparison needs no upload`;
}


function validateFile(
    file,
    key
) {

    const extension =
        file.name
            .split('.')
            .pop()
            .toLowerCase();


    let allowed;

    let expected;


    if (
        activeMode === 'multispectral'
    ) {

        allowed = [
            'tif',
            'tiff',
            'npy'
        ];

        expected =
            'a multiband GeoTIFF/TIFF or NPY file';

    } else if (
        activeMode === 'sar'
        || key === 'sar'
    ) {

        allowed = [
            'tif',
            'tiff',
            'npy'
        ];

        expected =
            'a SAR GeoTIFF/TIFF or NPY file';

    } else if (
        activeMode === 'fusion'
        && key === 'optical'
    ) {

        allowed = [
            'tif',
            'tiff',
            'npy'
        ];

        expected =
            'an optical/multispectral GeoTIFF/TIFF or NPY file';

    } else {

        allowed = [
            'png',
            'jpg',
            'jpeg',
            'webp',
            'tif',
            'tiff'
        ];

        expected =
            'a PNG, JPG, JPEG, WebP, TIFF or GeoTIFF image';
    }


    if (
        !allowed.includes(
            extension
        )
    ) {

        return `Choose ${expected}.`;
    }


    if (
        !file.size
    ) {

        return (
            `${file.name} is empty. `
            + 'Choose another file.'
        );
    }


    if (
        file.size
        > 200 * 1048576
    ) {

        return (
            'The upload limit is 200 MB. '
            + 'Choose a smaller file.'
        );
    }


    return null;
}


function selectExample(
    button
) {

    setMode(
        button.dataset.mode
    );

    queryInput.value =
        button.dataset.query
        || '';

    updateCount();

    $('contentDialog')
        .close();

    queryInput.focus();
}


document
    .querySelectorAll(
        '.mode-item'
    )
    .forEach(
        button =>
            button.addEventListener(
                'click',
                () =>
                    setMode(
                        button.dataset.mode
                    )
            )
    );


document
    .querySelectorAll(
        '.example-chip,.use-card'
    )
    .forEach(
        button =>
            button.addEventListener(
                'click',
                () =>
                    selectExample(
                        button
                    )
            )
    );


queryInput.addEventListener(
    'input',
    () => {

        updateCount();

        formMessage();
    }
);


Object
    .entries(
        inputs
    )
    .forEach(
        (
            [
                key,
                input
            ]
        ) => {

            input.addEventListener(
                'change',
                () => {

                    const error =
                        input.files[0]
                        && validateFile(
                            input.files[0],
                            key
                        );

                    if (
                        error
                    ) {
                        input.value = '';
                    }

                    refreshFiles();

                    formMessage(
                        error
                        || ''
                    );
                }
            );
        }
    );


document
    .querySelectorAll(
        '[data-remove]'
    )
    .forEach(
        button =>
            button.addEventListener(
                'click',
                () => {

                    inputs[
                        button.dataset.remove
                    ].value = '';

                    refreshFiles();

                    formMessage();
                }
            )
    );


$('uploadZone')
    .addEventListener(
        'click',
        () => {

            const required =
                modes[activeMode]
                    .files;

            inputs[
                required.find(
                    key =>
                        !inputs[key]
                            .files
                            .length
                )
                || required[0]
            ].click();
        }
    );


let dragDepth = 0;


[
    'dragenter',
    'dragover',
    'dragleave',
    'drop'
].forEach(
    name =>
        $('uploadZone')
            .addEventListener(
                name,
                event => {

                    event.preventDefault();


                    if (
                        name === 'dragenter'
                    ) {
                        dragDepth++;
                    }


                    if (
                        name === 'dragleave'
                    ) {

                        dragDepth =
                            Math.max(
                                0,
                                dragDepth - 1
                            );
                    }


                    if (
                        name === 'drop'
                    ) {
                        dragDepth = 0;
                    }


                    $('uploadZone')
                        .classList
                        .toggle(
                            'dragging',
                            dragDepth > 0
                        );
                }
            )
);


$('uploadZone')
    .addEventListener(
        'drop',
        event => {

            const files =
                [
                    ...event
                        .dataTransfer
                        .files
                ];


            if (
                !files.length
                || busy
            ) {
                return;
            }


            const required =
                modes[activeMode]
                    .files;


            const available =
                files.length > 1
                    ? required
                    : [
                        required.find(
                            key =>
                                !inputs[key]
                                    .files
                                    .length
                        )
                        || required[0]
                    ];


            if (
                files.length
                > available.length
            ) {

                formMessage(
                    `This mode needs ${
                        required.length
                    } image${
                        required.length > 1
                            ? 's'
                            : ''
                    }.`
                );

                return;
            }


            for (
                let i = 0;
                i < files.length;
                i++
            ) {

                const error =
                    validateFile(
                        files[i],
                        available[i]
                    );

                if (
                    error
                ) {

                    formMessage(
                        error
                    );

                    return;
                }
            }


            files.forEach(
                (
                    file,
                    index
                ) => {

                    const transfer =
                        new DataTransfer();

                    transfer.items.add(
                        file
                    );

                    inputs[
                        available[index]
                    ].files =
                        transfer.files;
                }
            );


            refreshFiles();

            formMessage();
        }
    );


function setStatus(
    state,
    text
) {

    const badge =
        $('statusBadge');


    badge.className =
        `status-badge ${state}`;


    badge.replaceChildren(
        icon(
            state === 'success'
                ? 'check'
                : state === 'error'
                    ? 'close'
                    : 'radar'
        ),
        document.createTextNode(
            text
        )
    );
}


function setConfidence(
    value,
    empty = false
) {

    const number =
        Number(
            value
        );


    const percent =
        Number.isFinite(
            number
        )
            ? Math.max(
                0,
                Math.min(
                    100,
                    number
                )
            )
            : 0;


    $('confidenceRing')
        .style
        .setProperty(
            '--confidence',
            percent
        );


    $('confidenceValue')
        .textContent =
        empty
            ? '—'
            : `${percent.toFixed(1)}%`;
}


function showEvidence(
    index
) {

    const item =
        evidenceItems[index];


    if (
        !item
    ) {
        return;
    }


    activeEvidence =
        index;


    $('mainEvidence')
        .replaceChildren(
            imageNode(
                item.url,
                item.name
            )
        );


    $('evidenceThumbnails')
        .querySelectorAll(
            'button'
        )
        .forEach(
            (
                button,
                i
            ) => {

                button.classList.toggle(
                    'active',
                    i === index
                );

                button.setAttribute(
                    'aria-pressed',
                    String(
                        i === index
                    )
                );
            }
        );


    $('evidenceCaption')
        .textContent =
        item.name;
}


function renderEvidence(
    items,
    emptyMessage =
        'No visual evidence was generated for this analysis.'
) {

    evidenceItems =
        (
            Array.isArray(
                items
            )
                ? items
                : []
        )
            .filter(
                item =>
                    item
                    && localUrl(
                        item.url
                    )
            )
            .map(
                item => ({
                    name:
                        String(
                            item.name
                            || 'Visual evidence'
                        ),

                    url:
                        localUrl(
                            item.url
                        )
                })
            );


    $('evidenceThumbnails')
        .replaceChildren();


    $('downloadAllBtn')
        .disabled =
        !evidenceItems.length;


    if (
        !evidenceItems.length
    ) {

        const empty =
            document.createElement(
                'div'
            );

        empty.className =
            'evidence-empty';


        empty.append(
            icon(
                'image'
            ),
            document.createTextNode(
                emptyMessage
            )
        );


        $('mainEvidence')
            .replaceChildren(
                empty
            );

        return;
    }


    evidenceItems.forEach(
        (
            item,
            index
        ) => {

            const button =
                document.createElement(
                    'button'
                );

            button.type =
                'button';

            button.className =
                'evidence-thumb';

            button.setAttribute(
                'aria-label',
                `Show ${item.name}`
            );


            button.append(
                imageNode(
                    item.url,
                    ''
                )
            );


            button.addEventListener(
                'click',
                () =>
                    showEvidence(
                        index
                    )
            );


            $('evidenceThumbnails')
                .append(
                    button
                );
        }
    );


    showEvidence(
        0
    );
}


function updateResult(data) {

    currentResult =
        data;


    try {

        localStorage.setItem(
            'satquery_last_analysis',
            JSON.stringify(
                data
            )
        );

    } catch {

        toast(
            'Browser storage is unavailable. '
            + 'The Map View cannot retain this result.'
        );
    }


    $('exampleLabel')
        .classList
        .add(
            'hidden'
        );


    setStatus(
        data.success
            ? 'success'
            : 'error',

        data.success
            ? 'SUCCESS'
            : 'FAILED'
    );


    $('resultQuestion')
        .textContent =
        data.query
        || queryInput.value
        || 'Analysis request';


    $('resultAnswer')
        .textContent =
        data.answer
        || data.error
        || 'Analysis completed.';


    setConfidence(
        data.confidence
    );


    $('modelValue')
        .textContent =
        data.model
        || '—';


    $('deviceValue')
        .textContent =
        data.device
        || '—';


    $('taskValue')
        .textContent =
        data.task
        || data.intent
        || '—';


    renderEvidence(
        data.evidence
    );


    const report =
        data.success
        && localUrl(
            data.reports?.html
        );


    const reportButton =
        $('downloadReportBtn');


    if (
        report
    ) {

        reportButton.href =
            report;

        reportButton.setAttribute(
            'download',
            ''
        );

    } else {

        reportButton.removeAttribute(
            'href'
        );

        reportButton.removeAttribute(
            'download'
        );
    }


    reportButton.setAttribute(
        'aria-disabled',
        String(
            !report
        )
    );


    $('technicalBtn')
        .classList
        .remove(
            'hidden'
        );
}


function errorMessage(data) {

    const details =
        data.validation?.errors;


    return (
        Array.isArray(
            details
        )
        && details.length
    )
        ? details
            .map(
                item =>
                    typeof item === 'string'
                        ? item
                        : item.message
                            || item.code
            )
            .join(
                ' '
            )
        : data.error
            || data.answer
            || 'Analysis failed. Please try again.';
}


function renderHistory(
    container = $('recentList'),
    all = false
) {

    container.replaceChildren();


    const records =
        history.length
            ? history
            : sampleHistory;


    records
        .slice(
            0,
            all
                ? 20
                : 3
        )
        .forEach(
            record => {

                const button =
                    document.createElement(
                        'button'
                    );

                button.className =
                    'recent-item';

                button.type =
                    'button';


                const copy =
                    document.createElement(
                        'span'
                    );


                const title =
                    document.createElement(
                        'strong'
                    );

                title.textContent =
                    record.query;


                const time =
                    document.createElement(
                        'small'
                    );


                const minutes =
                    Math.max(
                        0,
                        Math.floor(
                            (
                                Date.now()
                                - record.time
                            )
                            / 60000
                        )
                    );


                time.textContent =
                    record.example
                        ? 'Example analysis'
                        : minutes < 1
                            ? 'Just now'
                            : minutes < 60
                                ? `${minutes} minutes ago`
                                : new Date(
                                    record.time
                                ).toLocaleString(
                                    [],
                                    {
                                        month:
                                            'short',

                                        day:
                                            'numeric',

                                        hour:
                                            '2-digit',

                                        minute:
                                            '2-digit'
                                    }
                                );


                copy.append(
                    title,
                    time
                );


                button.append(
                    imageNode(
                        `/static/images/${
                            modes[
                                record.mode
                            ].image
                        }`,
                        ''
                    ),
                    copy
                );


                button.addEventListener(
                    'click',
                    () => {

                        setMode(
                            record.mode
                        );

                        queryInput.value =
                            record.query;

                        updateCount();


                        if (
                            record.result
                        ) {

                            updateResult(
                                record.result
                            );

                        } else {

                            toast(
                                'Example question selected. '
                                + 'Upload your imagery to run it.'
                            );
                        }


                        $('contentDialog')
                            .close();

                        queryInput.focus();
                    }
                );


                container.append(
                    button
                );
            }
        );
}


function remember(data) {

    history.unshift(
        {
            query:
                data.query,

            mode:
                activeMode,

            time:
                Date.now(),

            result:
                data
        }
    );


    history =
        history.slice(
            0,
            20
        );


    try {

        sessionStorage.setItem(
            'satquery-history',
            JSON.stringify(
                history
            )
        );

    } catch {

        // Still usable without storage.

    }


    renderHistory();
}


form.addEventListener(
    'submit',
    async event => {

        event.preventDefault();


        if (
            busy
        ) {
            return;
        }


        const query =
            queryInput.value.trim();


        if (
            !query
        ) {

            formMessage(
                'Enter a question about your satellite imagery.'
            );

            queryInput.focus();

            return;
        }


        const required =
            modes[activeMode]
                .files;


        const missing =
            required.filter(
                key =>
                    !inputs[key]
                        .files
                        .length
            );


        if (
            missing.length
        ) {

            formMessage(
                `Upload ${
                    missing.join(
                        ' and '
                    )
                } imagery to continue.`
            );

            $('uploadZone')
                .focus();

            return;
        }


        const size =
            required.reduce(
                (
                    sum,
                    key
                ) =>
                    sum
                    + inputs[key]
                        .files[0]
                        .size,
                0
            );


        if (
            size > 200 * 1048576
        ) {

            formMessage(
                'Keep the total upload size below 200 MB.'
            );

            return;
        }


        formMessage();


        const data =
            new FormData(
                form
            );


        data.set(
            'query',
            query
        );


        busy =
            true;


        $('sendBtn')
            .disabled =
            true;


        $('loadingOverlay')
            .classList
            .remove(
                'hidden'
            );


        document
            .querySelector(
                '.page-shell'
            )
            .inert =
            true;


        document
            .querySelector(
                '.top-header'
            )
            .inert =
            true;


        try {

            const response =
                await fetch(
                    '/api/analyze',
                    {
                        method:
                            'POST',

                        body:
                            data
                    }
                );


            let payload;


            try {

                payload =
                    await response.json();

            } catch {

                throw new Error(
                    response.status === 413
                        ? (
                            'The upload is too large. '
                            + 'Use files totaling less than 200 MB.'
                        )
                        : (
                            `The server could not complete the request `
                            + `(${response.status}). Please try again.`
                        )
                );
            }


            if (
                !response.ok
                || !payload.success
            ) {

                updateResult(
                    {
                        ...payload,

                        query,

                        success:
                            false,

                        answer:
                            errorMessage(
                                payload
                            ),

                        confidence:
                            0,

                        reports:
                            {},

                        evidence:
                            []
                    }
                );

                return;
            }


            updateResult(
                payload
            );


            remember(
                payload
            );

        } catch (
            error
        ) {

            updateResult(
                {
                    query,

                    success:
                        false,

                    error:
                        error.message
                        || 'Could not reach the analysis server.',

                    confidence:
                        0
                }
            );

        } finally {

            busy =
                false;


            $('sendBtn')
                .disabled =
                false;


            $('loadingOverlay')
                .classList
                .add(
                    'hidden'
                );


            document
                .querySelector(
                    '.page-shell'
                )
                .inert =
                false;


            document
                .querySelector(
                    '.top-header'
                )
                .inert =
                false;


            $('sendBtn')
                .focus();
        }
    }
);


$('newAnalysisBtn')
    .addEventListener(
        'click',
        () => {

            if (
                busy
            ) {
                return;
            }


            form.reset();

            currentResult =
                null;


            try {

                localStorage.removeItem(
                    'satquery_last_analysis'
                );

            } catch {

                // Storage is optional.

            }


            setMode(
                'single'
            );

            updateCount();


            setStatus(
                '',
                'READY'
            );


            $('exampleLabel')
                .classList
                .add(
                    'hidden'
                );


            $('resultQuestion')
                .textContent =
                'Ask SatQuery AI a question.';


            $('resultAnswer')
                .textContent =
                'Your analysis result will appear here.';


            setConfidence(
                0,
                true
            );


            [
                'modelValue',
                'deviceValue',
                'taskValue'
            ].forEach(
                id => {

                    $(id).textContent =
                        '—';
                }
            );


            renderEvidence(
                [],
                'Visual evidence will appear here.'
            );


            $('downloadReportBtn')
                .removeAttribute(
                    'href'
                );


            $('downloadReportBtn')
                .setAttribute(
                    'aria-disabled',
                    'true'
                );


            $('technicalBtn')
                .classList
                .add(
                    'hidden'
                );


            queryInput.focus();
        }
    );


function download(
    url,
    name
) {

    const safeUrl =
        localUrl(
            url
        );


    if (
        !safeUrl
    ) {
        return;
    }


    const anchor =
        document.createElement(
            'a'
        );

    anchor.href =
        safeUrl;

    anchor.download =
        name;

    document.body.append(
        anchor
    );

    anchor.click();

    anchor.remove();
}


$('downloadAllBtn')
    .addEventListener(
        'click',
        () => {

            if (
                !currentResult
            ) {

                toast(
                    'Downloading example imagery.'
                );
            }


            evidenceItems.forEach(
                (
                    item,
                    index
                ) =>
                    setTimeout(
                        () =>
                            download(
                                item.url,
                                item.name
                            ),
                        index * 200
                    )
            );
        }
    );


$('downloadReportBtn')
    .addEventListener(
        'click',
        event => {

            if (
                $('downloadReportBtn')
                    .getAttribute(
                        'aria-disabled'
                    )
                === 'true'
            ) {

                event.preventDefault();

                toast(
                    'Run an analysis to generate a downloadable report.'
                );
            }
        }
    );


function openDialog(
    title
) {

    $('dialogTitle')
        .textContent =
        title;


    $('dialogBody')
        .replaceChildren();


    if (
        !$('contentDialog')
            .open
    ) {

        $('contentDialog')
            .showModal();
    }


    return $('dialogBody');
}


function paragraph(text) {

    const p =
        document.createElement(
            'p'
        );

    p.textContent =
        text;

    return p;
}


function showMap() {

    if (
        currentResult?.success
        && currentResult
            ?.geospatial
            ?.available === false
    ) {

        try {

            sessionStorage.setItem(
                'satquery_map_notice',
                currentResult
                    .geospatial
                    .reason
                || (
                    'The last analysis succeeded, '
                    + 'but its input has no CRS/bounds '
                    + 'and cannot be positioned on the map.'
                )
            );

        } catch {

            // Storage is optional.

        }
    }


    window.location.assign(
        '/map-view'
    );
}


function showReports() {

    const body =
        openDialog(
            'Reports'
        );


    const records =
        history.filter(
            item =>
                item.result.success
                && localUrl(
                    item.result
                        .reports
                        ?.html
                )
        );


    if (
        !records.length
    ) {

        body.append(
            paragraph(
                'Your downloadable reports will appear here '
                + 'after a successful analysis.'
            )
        );

        return;
    }


    records.forEach(
        record => {

            const row =
                document.createElement(
                    'div'
                );

            row.className =
                'report-row';


            const title =
                document.createElement(
                    'span'
                );

            title.textContent =
                record.query;


            const link =
                document.createElement(
                    'a'
                );

            link.textContent =
                'Download report';

            link.href =
                localUrl(
                    record.result
                        .reports
                        .html
                );

            link.download =
                '';


            row.append(
                title,
                link
            );

            body.append(
                row
            );
        }
    );
}


document
    .querySelectorAll(
        '[data-view]'
    )
    .forEach(
        button =>
            button.addEventListener(
                'click',
                () => {

                    const view =
                        button.dataset.view;


                    if (
                        view === 'analyze'
                    ) {

                        $('contentDialog')
                            .close();

                        queryInput.focus();
                    }


                    if (
                        view === 'map'
                    ) {

                        showMap();
                    }


                    if (
                        view === 'reports'
                    ) {

                        showReports();
                    }


                    if (
                        view === 'datasets'
                    ) {

                        const body =
                            openDialog(
                                'Datasets & analysis scenarios'
                            );

                        body.append(
                            paragraph(
                                'Choose a scenario to prepare your analysis, '
                                + 'then upload your own satellite data. '
                                + 'The images below are illustrative previews.'
                            )
                        );


                        const cards =
                            $('useCases')
                                .cloneNode(
                                    true
                                );

                        cards.removeAttribute(
                            'id'
                        );


                        cards
                            .querySelectorAll(
                                'button'
                            )
                            .forEach(
                                card =>
                                    card.addEventListener(
                                        'click',
                                        () =>
                                            selectExample(
                                                card
                                            )
                                    )
                            );


                        body.append(
                            cards
                        );
                    }


                    if (
                        view === 'models'
                    ) {

                        const body =
                            openDialog(
                                'Model Hub'
                            );


                        body.append(
                            paragraph(
                                'SatQuery routes your question '
                                + 'to the appropriate analysis pipeline. '
                                + 'Models load when you run an analysis.'
                            )
                        );


                        const list =
                            document.createElement(
                                'div'
                            );

                        list.className =
                            'dialog-list';


                        [
                            [
                                'BLIP VQA',
                                'Visual questions about a single satellite image.'
                            ],

                            [
                                'RemoteCLIP',
                                'Text and satellite image understanding for semantic analysis.'
                            ],

                            [
                                'ChangeFormer',
                                'Before and after structural change detection.'
                            ],

                            [
                                'Spectral, SAR & fusion tools',
                                'Spectral indices, radar backscatter and combined optical/SAR analysis.'
                            ]
                        ].forEach(
                            (
                                [
                                    title,
                                    description
                                ]
                            ) => {

                                const article =
                                    document.createElement(
                                        'article'
                                    );


                                const heading =
                                    document.createElement(
                                        'h3'
                                    );

                                heading.textContent =
                                    title;


                                article.append(
                                    heading,
                                    paragraph(
                                        description
                                    )
                                );


                                list.append(
                                    article
                                );
                            }
                        );


                        body.append(
                            list
                        );
                    }
                }
            )
    );


$('viewMapBtn')
    .addEventListener(
        'click',
        showMap
    );


$('technicalBtn')
    .addEventListener(
        'click',
        () => {

            const body =
                openDialog(
                    'Analysis details'
                );


            const pre =
                document.createElement(
                    'pre'
                );


            pre.textContent =
                JSON.stringify(
                    currentResult,
                    null,
                    2
                );


            body.append(
                pre
            );
        }
    );


$('viewAllBtn')
    .addEventListener(
        'click',
        () => {

            const body =
                openDialog(
                    'Recent Analyses'
                );


            body.append(
                paragraph(
                    history.length
                        ? (
                            'Analyses from this browser session. '
                            + 'Select one to revisit its result.'
                        )
                        : (
                            'No analyses yet. '
                            + 'Try one of these example questions '
                            + 'with your own imagery.'
                        )
                )
            );


            const list =
                document.createElement(
                    'div'
                );


            renderHistory(
                list,
                true
            );


            body.append(
                list
            );
        }
    );


$('closeDialogBtn')
    .addEventListener(
        'click',
        () =>
            $('contentDialog')
                .close()
    );


$('contentDialog')
    .addEventListener(
        'click',
        event => {

            const rect =
                $('contentDialog')
                    .getBoundingClientRect();


            if (
                event.target
                    === $('contentDialog')
                && (
                    event.clientX
                        < rect.left
                    || event.clientX
                        > rect.right
                    || event.clientY
                        < rect.top
                    || event.clientY
                        > rect.bottom
                )
            ) {

                $('contentDialog')
                    .close();
            }
        }
    );


$('mobileMenuBtn')
    .addEventListener(
        'click',
        () => {

            const open =
                $('sidebar')
                    .classList
                    .toggle(
                        'open'
                    );


            $('mobileMenuBtn')
                .setAttribute(
                    'aria-expanded',
                    String(
                        open
                    )
                );
        }
    );


document.addEventListener(
    'click',
    event => {

        if (
            !$('sidebar')
                .contains(
                    event.target
                )
            && !$('mobileMenuBtn')
                .contains(
                    event.target
                )
        ) {

            closeSidebar();
        }
    }
);


$('themeBtn')
    .addEventListener(
        'click',
        () => {

            const light =
                document.body
                    .classList
                    .toggle(
                        'light-theme'
                    );


            $('themeBtn')
                .setAttribute(
                    'aria-label',
                    `Switch to ${
                        light
                            ? 'dark'
                            : 'light'
                    } theme`
                );
        }
    );


$('profileBtn')
    .addEventListener(
        'click',
        () => {

            const body =
                openDialog(
                    'Harsha · Local workspace'
                );


            body.append(
                paragraph(
                    'Your recent analyses stay in this browser session. '
                    + 'New Analysis clears the current question and uploads. '
                    + 'Use the sun icon to switch the dashboard theme.'
                )
            );
        }
    );


const information = {

    about: [
        'About SatQuery AI',

        (
            'See the Earth. Ask Anything. '
            + 'SatQuery AI brings natural-language questions, '
            + 'satellite imagery and remote sensing analysis '
            + 'into one workspace.'
        )
    ],

    docs: [
        'Getting started',

        (
            '1. Choose an analysis mode. '
            + '2. Upload only the data type shown for that mode. '
            + 'Single-image VQA and text grounding accept normal images; '
            + 'multispectral and SAR analysis require remote-sensing data '
            + 'such as GeoTIFF/TIFF or NPY. '
            + '3. Ask a question and run the analysis. '
            + '4. For historical comparison, open Map View, '
            + 'draw an area, select two years and let Google Earth Engine '
            + 'fetch the imagery automatically. '
            + 'NPY files do not contain CRS/bounds, so a successful '
            + 'NPY analysis cannot be positioned on the map unless '
            + 'geospatial metadata is supplied separately.'
        )
    ],

    github: [
        'Project source',

        (
            'The dashboard is part of your local SatQuery-AI project. '
            + 'No GitHub repository URL is configured in this workspace.'
        )
    ],
};


document
    .querySelectorAll(
        '[data-info]'
    )
    .forEach(
        button =>
            button.addEventListener(
                'click',
                () => {

                    const [
                        title,
                        text
                    ] =
                        information[
                            button.dataset.info
                        ];


                    openDialog(
                        title
                    ).append(
                        paragraph(
                            text
                        )
                    );
                }
            )
    );


document.addEventListener(
    'keydown',
    event => {

        if (
            event.key === 'Escape'
        ) {

            closeSidebar();
        }


        if (
            (
                event.ctrlKey
                || event.metaKey
            )
            && event.key === 'Enter'
            && !$('contentDialog').open
            && !busy
        ) {

            event.preventDefault();

            form.requestSubmit();
        }
    }
);


setMode(
    'single'
);

updateCount();

renderHistory();


renderEvidence(
    [
        {
            name:
                'Illustrative example: roads and buildings',

            url:
                '/static/images/example-roads.png'
        },

        {
            name:
                'Example SAR imagery',

            url:
                '/static/images/sar.jpeg'
        },

        {
            name:
                'Example deforestation imagery',

            url:
                '/static/images/deforestation.jpg'
        },

        {
            name:
                'Example multispectral imagery',

            url:
                '/static/images/agriculture.webp'
        },
    ]
);