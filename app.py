# app.py

from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.router.orchestrator import SatQueryOrchestrator
from ai.router.planner import QueryPlanner
from ai.validation import InputValidator
from ai.execution import ResultAdapter
from ai.report import SatQueryReportGenerator


UPLOAD_DIR = ROOT / "data" / "uploads"
EVIDENCE_DIR = ROOT / "outputs" / "evidence"
REPORT_DIR = ROOT / "outputs" / "reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_query" not in st.session_state:
    st.session_state.analysis_query = ""

if "analysis_intent" not in st.session_state:
    st.session_state.analysis_intent = ""

if "analysis_validation" not in st.session_state:
    st.session_state.analysis_validation = None

if "analysis_tools" not in st.session_state:
    st.session_state.analysis_tools = []

if "recent_analyses" not in st.session_state:
    st.session_state.recent_analyses = []


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

:root {
    --sq-bg: #03111e;
    --sq-bg-2: #051827;
    --sq-panel: #081c2e;
    --sq-panel-2: #0d2438;
    --sq-panel-3: #102a42;

    --sq-border: rgba(112, 190, 255, 0.16);
    --sq-border-strong: rgba(67, 168, 255, 0.35);

    --sq-blue: #1688ff;
    --sq-blue-2: #2aa9ff;
    --sq-cyan: #19d4da;
    --sq-green: #21d995;
    --sq-orange: #ff9f43;
    --sq-purple: #9b6cff;

    --sq-text: #f6fbff;
    --sq-text-soft: #b8cbe0;
    --sq-muted: #6f8ca8;

    --sq-radius-xl: 24px;
    --sq-radius-lg: 18px;
    --sq-radius-md: 14px;
}

html,
body,
[class*="css"] {
    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;
}

body {
    background: var(--sq-bg);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 82% -12%,
            rgba(13, 97, 180, 0.24),
            transparent 32%
        ),
        radial-gradient(
            circle at 44% 20%,
            rgba(13, 64, 110, 0.16),
            transparent 34%
        ),
        linear-gradient(
            180deg,
            #03111e 0%,
            #041421 100%
        );
}

[data-testid="stHeader"] {
    background: rgba(3, 17, 30, 0.84);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.block-container {
    max-width: 1680px;
    padding-top: 1rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ---------------------------------------------------------
   SCROLLBAR
--------------------------------------------------------- */

::-webkit-scrollbar {
    width: 9px;
    height: 9px;
}

::-webkit-scrollbar-track {
    background: #04111d;
}

::-webkit-scrollbar-thumb {
    background: #153c5f;
    border-radius: 20px;
}

::-webkit-scrollbar-thumb:hover {
    background: #1b507d;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #031421 0%,
            #041827 100%
        );
    border-right: 1px solid rgba(89, 170, 230, 0.12);
}

[data-testid="stSidebar"] > div {
    padding-top: 0.8rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}


/* ---------------------------------------------------------
   BRAND
--------------------------------------------------------- */

.sq-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 8px 3px 20px 3px;
}

.sq-logo {
    width: 45px;
    height: 45px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background:
        linear-gradient(
            135deg,
            #0f78ff,
            #18c7ff
        );
    box-shadow:
        0 0 28px rgba(28, 145, 255, 0.28);
    font-size: 24px;
}

.sq-brand-title {
    font-size: 27px;
    line-height: 1;
    font-weight: 850;
    color: white;
}

.sq-brand-title span {
    color: #42a9ff;
}

.sq-brand-subtitle {
    margin-top: 5px;
    color: #87a3bd;
    font-size: 12px;
}


/* ---------------------------------------------------------
   TOP NAV
--------------------------------------------------------- */

.sq-topnav {
    height: 66px;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 14px 0 18px;
    margin-bottom: 14px;

    border: 1px solid rgba(88, 173, 240, 0.10);
    border-radius: 18px;

    background:
        linear-gradient(
            90deg,
            rgba(5, 28, 47, 0.95),
            rgba(5, 22, 38, 0.80)
        );

    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.15);
}

.sq-nav-left {
    display: flex;
    gap: 8px;
    align-items: center;
}

.sq-nav-item {
    padding: 10px 15px;
    color: #a9bfd4;
    font-size: 13px;
    font-weight: 700;

    border-radius: 11px;
    border: 1px solid transparent;
}

.sq-nav-item.active {
    color: #dff3ff;
    background:
        linear-gradient(
            135deg,
            rgba(17, 121, 232, 0.32),
            rgba(7, 83, 151, 0.30)
        );

    border-color:
        rgba(58, 166, 255, 0.28);
}

.sq-user {
    display: flex;
    align-items: center;
    gap: 10px;
}

.sq-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;

    display: grid;
    place-items: center;

    background:
        linear-gradient(
            135deg,
            #1d65a5,
            #113854
        );

    border: 1px solid rgba(255, 255, 255, 0.11);

    color: white;
    font-weight: 800;
}

.sq-username {
    color: #eaf6ff;
    font-size: 13px;
    font-weight: 700;
}


/* ---------------------------------------------------------
   SIDEBAR ANALYSIS MENU
--------------------------------------------------------- */

.sq-new-analysis {
    width: 100%;
    padding: 14px;

    border-radius: 14px;

    color: white;
    text-align: center;

    font-size: 14px;
    font-weight: 800;

    background:
        linear-gradient(
            135deg,
            #1684ff,
            #1e9cff
        );

    box-shadow:
        0 12px 30px rgba(20, 135, 255, 0.24);

    margin-bottom: 15px;
}

.sq-menu-card {
    display: flex;
    align-items: center;
    gap: 12px;

    padding: 12px 13px;
    margin-bottom: 7px;

    border-radius: 13px;

    background:
        rgba(8, 29, 48, 0.52);

    border:
        1px solid
        rgba(80, 159, 218, 0.06);
}

.sq-menu-card.active {
    background:
        linear-gradient(
            135deg,
            rgba(22, 64, 99, 0.96),
            rgba(10, 41, 66, 0.96)
        );

    border-color:
        rgba(67, 162, 232, 0.18);
}

.sq-menu-icon {
    width: 36px;
    height: 36px;

    border-radius: 10px;

    display: grid;
    place-items: center;

    font-size: 19px;

    background:
        rgba(30, 140, 255, 0.14);

    border:
        1px solid
        rgba(40, 155, 255, 0.18);
}

.sq-menu-title {
    color: #edf8ff;
    font-size: 13px;
    font-weight: 750;
}

.sq-menu-sub {
    color: #7995af;
    font-size: 11px;
    margin-top: 3px;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.sq-hero {
    min-height: 315px;
    padding: 50px 38px 35px 38px;

    border-radius: var(--sq-radius-xl);

    border:
        1px solid
        rgba(75, 165, 230, 0.20);

    background:
        radial-gradient(
            circle at 78% 17%,
            rgba(44, 158, 236, 0.25),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            rgba(5, 37, 60, 0.98),
            rgba(5, 23, 39, 0.95)
        );

    box-shadow:
        0 20px 55px rgba(0, 0, 0, 0.22);

    position: relative;
    overflow: hidden;
}

.sq-hero::after {
    content: "";
    position: absolute;

    width: 520px;
    height: 520px;

    right: -160px;
    top: -310px;

    border-radius: 50%;

    border:
        2px solid
        rgba(64, 164, 255, 0.10);

    box-shadow:
        0 0 100px rgba(28, 122, 206, 0.13);
}

.sq-kicker {
    display: inline-block;

    padding: 7px 11px;

    border-radius: 8px;

    background:
        rgba(13, 106, 186, 0.22);

    border:
        1px solid
        rgba(50, 165, 250, 0.32);

    color: #5ec0ff;

    font-size: 11px;
    letter-spacing: 0.8px;
    font-weight: 800;
}

.sq-hero-title {
    max-width: 760px;

    color: white;

    font-size: 42px;
    line-height: 1.15;
    font-weight: 850;

    margin-top: 17px;
}

.sq-hero-sub {
    color: #a7bfd5;

    max-width: 720px;

    font-size: 16px;
    line-height: 1.6;

    margin-top: 12px;
}

.sq-examples {
    margin-top: 27px;

    display: flex;
    gap: 9px;
    flex-wrap: wrap;
}

.sq-example {
    padding: 8px 12px;

    color: #ccecff;

    border:
        1px solid
        rgba(54, 166, 245, 0.32);

    background:
        rgba(5, 35, 58, 0.68);

    border-radius: 999px;

    font-size: 11px;
}


/* ---------------------------------------------------------
   MAIN INPUT
--------------------------------------------------------- */

[data-testid="stTextArea"] textarea {
    min-height: 116px !important;

    background:
        linear-gradient(
            135deg,
            rgba(10, 35, 57, 0.98),
            rgba(8, 28, 47, 0.98)
        ) !important;

    border:
        1px solid
        rgba(92, 172, 232, 0.22) !important;

    border-radius: 17px !important;

    color: #f4fbff !important;

    padding: 18px !important;

    font-size: 15px !important;

    box-shadow:
        inset 0 0 0 1px
        rgba(255, 255, 255, 0.01),
        0 15px 35px rgba(0, 0, 0, 0.16);
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #617d96 !important;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

div.stButton > button {
    min-height: 47px;

    border-radius: 13px;

    border:
        1px solid
        rgba(69, 168, 255, 0.35);

    color: white;

    font-weight: 800;

    background:
        linear-gradient(
            135deg,
            #157aff,
            #159fff
        );

    box-shadow:
        0 10px 30px
        rgba(15, 125, 255, 0.20);

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 13px 36px
        rgba(25, 145, 255, 0.28);

    border-color:
        rgba(69, 168, 255, 0.70);
}


/* ---------------------------------------------------------
   UPLOADER
--------------------------------------------------------- */

[data-testid="stFileUploader"] {
    border-radius: 16px;

    background:
        rgba(7, 28, 46, 0.68);

    border:
        1px dashed
        rgba(79, 168, 237, 0.26);

    padding: 7px;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent;
}

[data-testid="stFileUploaderDropzone"] button {
    border-radius: 10px !important;
}


/* ---------------------------------------------------------
   DEMO CARDS
--------------------------------------------------------- */

.sq-card-strip {
    margin-top: 15px;

    display: grid;

    grid-template-columns:
        repeat(5, 1fr);

    gap: 10px;
}

.sq-demo-card {
    min-height: 104px;

    padding: 14px;

    border-radius: 14px;

    background:
        linear-gradient(
            145deg,
            rgba(13, 39, 62, 0.98),
            rgba(7, 28, 46, 0.98)
        );

    border:
        1px solid
        rgba(78, 157, 218, 0.13);

    box-shadow:
        0 12px 28px rgba(0, 0, 0, 0.13);
}

.sq-demo-icon {
    font-size: 21px;
}

.sq-demo-title {
    margin-top: 8px;

    color: #f0f8ff;

    font-size: 12px;
    font-weight: 800;
}

.sq-demo-sub {
    color: #708ca5;

    font-size: 10px;

    margin-top: 4px;
}


/* ---------------------------------------------------------
   RESULT PANEL
--------------------------------------------------------- */

.sq-result-shell {
    border-radius: 20px;

    background:
        linear-gradient(
            160deg,
            rgba(8, 30, 49, 0.98),
            rgba(5, 22, 38, 0.98)
        );

    border:
        1px solid
        rgba(88, 172, 236, 0.16);

    padding: 17px;

    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.22);
}

.sq-result-header {
    display: flex;

    justify-content: space-between;
    align-items: center;

    margin-bottom: 12px;
}

.sq-result-title {
    color: #edf9ff;

    font-size: 16px;
    font-weight: 850;
}

.sq-success {
    padding: 6px 10px;

    border-radius: 9px;

    color: #43eeb2;

    background:
        rgba(25, 218, 148, 0.10);

    border:
        1px solid
        rgba(37, 225, 160, 0.35);

    font-size: 10px;
    font-weight: 850;
}

.sq-result-card {
    padding: 15px;

    border-radius: 14px;

    background:
        rgba(12, 37, 59, 0.88);

    border:
        1px solid
        rgba(82, 159, 216, 0.10);

    margin-bottom: 11px;
}

.sq-result-label {
    color: #74a7cf;

    font-size: 11px;
    font-weight: 700;

    margin-bottom: 6px;
}

.sq-result-value {
    color: #f2f9ff;

    font-size: 14px;
    line-height: 1.55;
}

.sq-answer {
    color: #f5fbff;

    font-size: 17px;
    font-weight: 650;

    line-height: 1.55;
}

.sq-model-grid {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 10px;

    margin-top: 12px;
}

.sq-model-label {
    color: #6e9bc0;

    font-size: 11px;
}

.sq-model-value {
    color: #edf8ff;

    font-size: 12px;
    font-weight: 700;
}


/* ---------------------------------------------------------
   CONFIDENCE RING
--------------------------------------------------------- */

.sq-confidence-row {
    display: grid;

    grid-template-columns:
        110px 1fr;

    gap: 13px;

    align-items: center;

    margin-top: 14px;
}

.sq-ring {
    --p: 85;

    width: 88px;
    height: 88px;

    border-radius: 50%;

    display: grid;
    place-items: center;

    background:
        conic-gradient(
            #1fcfe0 calc(var(--p) * 1%),
            #178dff calc(var(--p) * 1%),
            rgba(40, 91, 125, 0.35) 0
        );

    position: relative;
}

.sq-ring::before {
    content: "";

    width: 67px;
    height: 67px;

    border-radius: 50%;

    position: absolute;

    background: #0b2134;
}

.sq-ring span {
    z-index: 2;

    color: white;

    font-size: 15px;
    font-weight: 850;
}


/* ---------------------------------------------------------
   EVIDENCE
--------------------------------------------------------- */

.sq-evidence-title {
    display: flex;

    align-items: center;
    justify-content: space-between;

    margin-bottom: 10px;
}

.sq-evidence-placeholder {
    min-height: 255px;

    border-radius: 14px;

    border:
        1px dashed
        rgba(73, 167, 234, 0.24);

    display: grid;

    place-items: center;

    color: #65839c;

    background:
        rgba(4, 18, 30, 0.52);
}


/* ---------------------------------------------------------
   RECENT
--------------------------------------------------------- */

.sq-section-title {
    color: #9fb7cb;

    font-size: 12px;
    font-weight: 800;

    margin: 18px 0 10px 2px;

    text-transform: uppercase;

    letter-spacing: 0.75px;
}

.sq-recent {
    padding: 9px 4px;

    display: flex;

    gap: 10px;

    align-items: center;

    border-bottom:
        1px solid
        rgba(255,255,255,0.03);
}

.sq-recent-icon {
    width: 37px;
    height: 37px;

    border-radius: 10px;

    display: grid;
    place-items: center;

    background:
        linear-gradient(
            135deg,
            #184873,
            #0e2e48
        );
}

.sq-recent-title {
    color: #d7e8f5;

    font-size: 11px;
    font-weight: 700;
}

.sq-recent-time {
    color: #607e98;

    font-size: 9px;

    margin-top: 3px;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.sq-footer {
    display: flex;

    justify-content: space-between;

    margin-top: 25px;

    padding: 16px 4px 0;

    border-top:
        1px solid
        rgba(92, 170, 227, 0.10);

    color: #607b93;

    font-size: 10px;
}


/* ---------------------------------------------------------
   STREAMLIT TABS / EXPANDERS / STATUS
--------------------------------------------------------- */

[data-testid="stExpander"] {
    background:
        rgba(7, 27, 44, 0.72);

    border:
        1px solid
        rgba(78, 160, 221, 0.12);

    border-radius: 13px;
}

[data-testid="stStatusWidget"] {
    background:
        rgba(7, 27, 44, 0.82);

    border:
        1px solid
        rgba(76, 160, 223, 0.12);

    border-radius: 13px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
}

.stTabs [data-baseweb="tab"] {
    background:
        rgba(8, 30, 49, 0.75);

    border-radius: 10px;

    border:
        1px solid
        rgba(79, 159, 218, 0.10);

    padding:
        9px 14px;
}

.stTabs [aria-selected="true"] {
    background:
        rgba(21, 123, 210, 0.18) !important;
}


/* ---------------------------------------------------------
   RESPONSIVE
--------------------------------------------------------- */

@media (max-width: 1250px) {

    .sq-card-strip {
        grid-template-columns:
            repeat(3, 1fr);
    }

}

@media (max-width: 800px) {

    .sq-hero-title {
        font-size: 31px;
    }

    .sq-card-strip {
        grid-template-columns:
            1fr 1fr;
    }

    .sq-nav-item {
        display: none;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CACHED SERVICES
# ============================================================

@st.cache_resource
def get_orchestrator():
    return SatQueryOrchestrator()


@st.cache_resource
def get_planner():
    return QueryPlanner()


@st.cache_resource
def get_validator():
    return InputValidator()


@st.cache_resource
def get_result_adapter():
    return ResultAdapter()


@st.cache_resource
def get_report_generator():
    return SatQueryReportGenerator(
        output_dir=str(REPORT_DIR)
    )


# ============================================================
# HELPERS
# ============================================================

def save_uploaded_file(
    uploaded_file,
    prefix: str,
) -> Optional[str]:

    if uploaded_file is None:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    destination = (
        UPLOAD_DIR
        / f"{prefix}_{timestamp}{suffix}"
    )

    with open(
        destination,
        "wb",
    ) as output:
        shutil.copyfileobj(
            uploaded_file,
            output,
        )

    return str(destination)


def normalize_intent(
    intent: Any,
) -> str:

    if intent is None:
        return "unknown"

    if hasattr(intent, "value"):
        return str(intent.value)

    text = str(intent)

    if text.startswith("Intent."):
        text = (
            text
            .split(".", 1)[-1]
            .lower()
        )

    return text


def plan_query(
    query: str,
) -> Dict[str, Any]:

    planner = get_planner()

    try:
        plan = planner.plan(query)

    except AttributeError:

        try:
            plan = planner.create_plan(
                query
            )

        except AttributeError:
            plan = planner(query)

    if hasattr(
        plan,
        "to_dict",
    ):
        return plan.to_dict()

    if isinstance(
        plan,
        dict,
    ):
        return plan

    data = {}

    for field in [
        "intent",
        "confidence",
        "required_tools",
        "tools",
        "target",
        "targets",
        "operation",
        "years",
        "change_direction",
    ]:

        if hasattr(
            plan,
            field,
        ):
            data[field] = getattr(
                plan,
                field,
            )

    return data


def get_plan_tools(
    plan: Dict[str, Any],
) -> List[str]:

    tools = (
        plan.get("required_tools")
        or plan.get("tools")
        or []
    )

    if isinstance(
        tools,
        str,
    ):
        tools = [tools]

    return [
        str(tool)
        for tool in tools
    ]


def validate_inputs(
    intent: str,
    image_path: Optional[str],
    before_path: Optional[str],
    after_path: Optional[str],
    optical_path: Optional[str],
    sar_path: Optional[str],
) -> Optional[Dict[str, Any]]:

    validator = get_validator()

    if intent == "change_detection":

        if not before_path or not after_path:

            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "CHANGE_PAIR_REQUIRED",
                        "message":
                            "Upload both Before and After images.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_change_pair(
                before_path,
                after_path,
            )
        )

    if intent == "optical_sar_fusion":

        if not optical_path or not sar_path:

            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "OPTICAL_SAR_PAIR_REQUIRED",
                        "message":
                            "Upload both optical and SAR imagery.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_optical_sar_pair(
                optical_path,
                sar_path,
            )
        )

    if intent == "sar_analysis":

        if not sar_path:

            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "SAR_REQUIRED",
                        "message":
                            "Upload a SAR image.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_single_image(
                sar_path,
                expected_modality="sar",
            )
        )

    if intent == "multispectral_analysis":

        path = (
            optical_path
            or image_path
        )

        if not path:

            return {
                "valid": False,
                "errors": [
                    {
                        "code":
                            "MULTISPECTRAL_REQUIRED",
                        "message":
                            "Upload a multispectral image.",
                    }
                ],
                "warnings": [],
            }

        return (
            validator
            .validate_multispectral(
                path
            )
        )

    if image_path:

        return (
            validator
            .validate_single_image(
                image_path
            )
        )

    return None


def execute_query(
    query: str,
    intent: str,
    image_path: Optional[str],
    before_path: Optional[str],
    after_path: Optional[str],
    optical_path: Optional[str],
    sar_path: Optional[str],
) -> Dict[str, Any]:

    orchestrator = get_orchestrator()

    kwargs = {}

    if intent == "change_detection":

        kwargs["before_path"] = (
            before_path
        )

        kwargs["after_path"] = (
            after_path
        )

    elif intent == "optical_sar_fusion":

        kwargs["optical_path"] = (
            optical_path
        )

        kwargs["sar_path"] = (
            sar_path
        )

    elif intent == "sar_analysis":

        kwargs["sar_path"] = (
            sar_path
        )

    elif intent == "multispectral_analysis":

        kwargs["image_path"] = (
            optical_path
            or image_path
        )

    else:

        kwargs["image_path"] = (
            image_path
        )

    return orchestrator.execute(
        query,
        **kwargs,
    )


# ============================================================
# IMPORTANT FIX:
# UNWRAP ORCHESTRATOR EXECUTION RESULT
# ============================================================

def unwrap_execution_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(
        result,
        dict,
    ):
        return {
            "success": True,
            "answer": str(result),
        }

    execution = result.get(
        "execution"
    )

    if not isinstance(
        execution,
        dict,
    ):
        return result

    merged = dict(result)

    # Prefer actual execution values.
    for key, value in execution.items():

        if value is not None:
            merged[key] = value

    # Preserve outer query-routing information.
    merged["routing"] = {
        "query":
            result.get("query"),

        "intent":
            result.get("intent"),

        "routing_confidence":
            result.get("confidence"),

        "matched_keywords":
            result.get(
                "matched_keywords",
                [],
            ),

        "required_tools":
            result.get(
                "required_tools",
                [],
            ),
    }

    # Actual execution is authoritative.
    merged["success"] = execution.get(
        "success",
        result.get(
            "success",
            True,
        ),
    )

    # Model-generated answer.
    if execution.get("answer") is not None:
        merged["answer"] = (
            execution["answer"]
        )

    # Model / device.
    if execution.get("model"):
        merged["model"] = (
            execution["model"]
        )

    if execution.get("device"):
        merged["device"] = (
            execution["device"]
        )

    # Evidence handling.
    evidence = (
        execution.get(
            "visual_evidence"
        )
        or execution.get(
            "evidence"
        )
        or execution.get(
            "evidence_path"
        )
    )

    if evidence is not None:
        merged["evidence"] = evidence

    # Limitations.
    if execution.get("limitations"):
        merged["limitations"] = (
            execution["limitations"]
        )

    return merged


def get_answer(
    result: Dict[str, Any],
) -> str:

    for key in [
        "answer",
        "message",
        "description",
        "interpretation",
    ]:

        value = result.get(key)

        if value is not None:
            return str(value)

    return "Analysis completed."


def confidence_data(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    details = result.get(
        "confidence_details"
    )

    if isinstance(
        details,
        dict,
    ):
        return details

    raw = result.get(
        "confidence",
        0.0,
    )

    try:
        raw = float(raw)

    except Exception:
        raw = 0.0

    if raw > 1:
        raw /= 100.0

    raw = max(
        0.0,
        min(
            1.0,
            raw,
        ),
    )

    return {
        "score": raw,
        "percentage":
            round(
                raw * 100,
                2,
            ),
        "level": "unknown",
        "confidence_type":
            result.get(
                "confidence_type",
                "model_score",
            ),
        "calibrated": False,
    }


def get_evidence_paths(
    result: Dict[str, Any],
) -> List[Path]:

    evidence = (
        result.get("evidence")
        or result.get("visual_evidence")
        or result.get("evidence_path")
    )

    if not evidence:
        return []

    if not isinstance(
        evidence,
        (list, tuple),
    ):
        evidence = [evidence]

    output = []

    for item in evidence:

        if isinstance(
            item,
            dict,
        ):

            candidate = (
                item.get("path")
                or item.get("file")
                or item.get("image")
            )

        else:
            candidate = item

        if not candidate:
            continue

        path = Path(
            str(candidate)
        )

        if not path.is_absolute():
            path = (
                ROOT
                / path
            )

        if (
            path.exists()
            and path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".tif",
                ".tiff",
            }
        ):
            output.append(
                path
            )

    return output


def add_recent_analysis(
    query: str,
    intent: str,
):

    recent = (
        st.session_state
        .recent_analyses
    )

    recent.insert(
        0,
        {
            "query": query,
            "intent": intent,
            "time":
                datetime.now()
                .strftime(
                    "%H:%M"
                ),
        },
    )

    st.session_state.recent_analyses = (
        recent[:4]
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sq-brand">

    <div class="sq-logo">
        🛰️
    </div>

    <div>

        <div class="sq-brand-title">
            SatQuery<span>AI</span>
        </div>

        <div class="sq-brand-subtitle">
            See the Earth. Ask Anything.
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sq-new-analysis">
    ＋ &nbsp; New Analysis
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sq-menu-card active">

    <div class="sq-menu-icon">🖼️</div>

    <div>
        <div class="sq-menu-title">
            Single Image
        </div>
        <div class="sq-menu-sub">
            VQA, object detection
        </div>
    </div>

</div>

<div class="sq-menu-card">

    <div class="sq-menu-icon"
         style="color:#ff8a50;">
        ▣
    </div>

    <div>
        <div class="sq-menu-title">
            Bi-temporal Change
        </div>
        <div class="sq-menu-sub">
            Before / After analysis
        </div>
    </div>

</div>

<div class="sq-menu-card">

    <div class="sq-menu-icon"
         style="color:#a174ff;">
        ◩
    </div>

    <div>
        <div class="sq-menu-title">
            Optical + SAR Fusion
        </div>
        <div class="sq-menu-sub">
            Multimodal analysis
        </div>
    </div>

</div>

<div class="sq-menu-card">

    <div class="sq-menu-icon"
         style="color:#19d899;">
        ◉
    </div>

    <div>
        <div class="sq-menu-title">
            Multispectral Analysis
        </div>
        <div class="sq-menu-sub">
            NDVI, vegetation, indices
        </div>
    </div>

</div>

<div class="sq-menu-card">

    <div class="sq-menu-icon"
         style="color:#28a9ff;">
        ◌
    </div>

    <div>
        <div class="sq-menu-title">
            SAR Analysis
        </div>
        <div class="sq-menu-sub">
            Radar feature analysis
        </div>
    </div>

</div>

<div class="sq-menu-card">

    <div class="sq-menu-icon"
         style="color:#ffae34;">
        ◎
    </div>

    <div>
        <div class="sq-menu-title">
            Text-Grounded Detection
        </div>
        <div class="sq-menu-sub">
            Find anything with text
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sq-section-title">
    Upload imagery
</div>
""",
        unsafe_allow_html=True,
    )

    single_upload = st.file_uploader(
        "Single Image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "npy",
        ],
        key="single_upload",
    )

    with st.expander(
        "Bi-temporal inputs"
    ):

        before_upload = (
            st.file_uploader(
                "Before Image",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "tif",
                    "tiff",
                    "npy",
                ],
                key="before_upload",
            )
        )

        after_upload = (
            st.file_uploader(
                "After Image",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "tif",
                    "tiff",
                    "npy",
                ],
                key="after_upload",
            )
        )

    with st.expander(
        "Optical + SAR inputs"
    ):

        optical_upload = (
            st.file_uploader(
                "Optical / Multispectral",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "tif",
                    "tiff",
                    "npy",
                ],
                key="optical_upload",
            )
        )

        sar_upload = (
            st.file_uploader(
                "SAR Image",
                type=[
                    "tif",
                    "tiff",
                    "npy",
                ],
                key="sar_upload",
            )
        )

    st.markdown(
        """
<div class="sq-section-title">
    Recent Analyses
</div>
""",
        unsafe_allow_html=True,
    )

    if (
        st.session_state
        .recent_analyses
    ):

        for recent in (
            st.session_state
            .recent_analyses[:3]
        ):

            st.markdown(
                f"""
<div class="sq-recent">

    <div class="sq-recent-icon">
        🛰️
    </div>

    <div>

        <div class="sq-recent-title">
            {recent["query"][:32]}
        </div>

        <div class="sq-recent-time">
            {recent["intent"]} • {recent["time"]}
        </div>

    </div>

</div>
""",
                unsafe_allow_html=True,
            )

    else:

        st.markdown(
            """
<div style="
    color:#607c94;
    font-size:11px;
    padding:5px 2px 10px 2px;
">
    Your analyses will appear here.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div style="
    margin-top:20px;
    padding:13px;
    border-radius:13px;
    background:
        linear-gradient(
            135deg,
            rgba(11,69,110,.9),
            rgba(7,41,69,.9)
        );
    border:
        1px solid
        rgba(54,157,229,.18);
    color:#b9d5ea;
    font-size:10px;
">
    ✦ Powered by<br>
    <b style="color:white;">
        RemoteCLIP + BLIP + ChangeFormer
    </b>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown(
    """
<div class="sq-topnav">

    <div class="sq-nav-left">

        <div class="sq-nav-item active">
            ◉ &nbsp; Analyze
        </div>

        <div class="sq-nav-item">
            ◈ &nbsp; Map View
        </div>

        <div class="sq-nav-item">
            ◫ &nbsp; Datasets
        </div>

        <div class="sq-nav-item">
            ◇ &nbsp; Model Hub
        </div>

        <div class="sq-nav-item">
            ▣ &nbsp; Reports
        </div>

    </div>

    <div class="sq-user">

        <div style="
            color:#85a4bf;
            font-size:17px;
        ">
            ☼
        </div>

        <div class="sq-avatar">
            H
        </div>

        <div class="sq-username">
            Harsha⌄
        </div>

    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN 2-COLUMN LAYOUT
# ============================================================

main_col, result_col = st.columns(
    [2.25, 1],
    gap="medium",
)


# ============================================================
# LEFT MAIN WORKSPACE
# ============================================================

with main_col:

    st.markdown(
        """
<div class="sq-hero">

    <div class="sq-kicker">
        NATURAL LANGUAGE · MULTIMODAL · REMOTE SENSING
    </div>

    <div class="sq-hero-title">
        Understand Our Planet<br>
        with the Power of AI
    </div>

    <div class="sq-hero-sub">
        Upload satellite imagery and ask questions in
        natural language. SatQuery AI automatically
        selects the appropriate remote-sensing model
        and analysis pipeline.
    </div>

    <div class="sq-examples">

        <div class="sq-example">
            Are there roads in this image? →
        </div>

        <div class="sq-example">
            What changed between 2020 and 2025? →
        </div>

        <div class="sq-example">
            Calculate NDVI →
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height:14px'></div>",
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Ask SatQuery AI",
        value=(
            st.session_state
            .analysis_query
            if st.session_state
            .analysis_query
            else ""
        ),
        placeholder=(
            "Ask a question about your satellite imagery..."
        ),
        label_visibility="collapsed",
        height=116,
        max_chars=500,
    )

    analyze_button = st.button(
        "✦  Analyze Satellite Data",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        """
<div style="
    text-align:center;
    color:#5f7e99;
    font-size:10px;
    margin-top:4px;
">
    Supports PNG · JPG · TIFF · GeoTIFF · NPY
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sq-card-strip">

    <div class="sq-demo-card">
        <div class="sq-demo-icon">🏙️</div>
        <div class="sq-demo-title">
            Urban Area
        </div>
        <div class="sq-demo-sub">
            Single image VQA
        </div>
    </div>

    <div class="sq-demo-card">
        <div class="sq-demo-icon">🌲</div>
        <div class="sq-demo-title">
            Deforestation
        </div>
        <div class="sq-demo-sub">
            Change detection
        </div>
    </div>

    <div class="sq-demo-card">
        <div class="sq-demo-icon">🌾</div>
        <div class="sq-demo-title">
            Agriculture
        </div>
        <div class="sq-demo-sub">
            Multispectral NDVI
        </div>
    </div>

    <div class="sq-demo-card">
        <div class="sq-demo-icon">🌊</div>
        <div class="sq-demo-title">
            Flood Assessment
        </div>
        <div class="sq-demo-sub">
            SAR analysis
        </div>
    </div>

    <div class="sq-demo-card">
        <div class="sq-demo-icon">🗺️</div>
        <div class="sq-demo-title">
            Coastal Monitoring
        </div>
        <div class="sq-demo-sub">
            Optical + SAR
        </div>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# EXECUTION
# ============================================================

if analyze_button:

    if not query.strip():

        st.error(
            "Enter a natural-language query."
        )

        st.stop()

    try:

        with st.status(
            "SatQuery AI is analyzing satellite imagery...",
            expanded=True,
        ) as status:

            st.write(
                "Understanding your query..."
            )

            plan = plan_query(
                query.strip()
            )

            intent = normalize_intent(
                plan.get("intent")
            )

            tools = get_plan_tools(
                plan
            )

            st.write(
                f"Detected task: `{intent}`"
            )

            if tools:

                st.write(
                    "Selected AI tools: "
                    + ", ".join(tools)
                )

            st.write(
                "Preparing imagery..."
            )

            image_path = save_uploaded_file(
                single_upload,
                "single",
            )

            before_path = save_uploaded_file(
                before_upload,
                "before",
            )

            after_path = save_uploaded_file(
                after_upload,
                "after",
            )

            optical_path = save_uploaded_file(
                optical_upload,
                "optical",
            )

            sar_path = save_uploaded_file(
                sar_upload,
                "sar",
            )

            st.write(
                "Validating modality, format and metadata..."
            )

            validation = validate_inputs(
                intent=intent,
                image_path=image_path,
                before_path=before_path,
                after_path=after_path,
                optical_path=optical_path,
                sar_path=sar_path,
            )

            if (
                validation is not None
                and not validation.get(
                    "valid",
                    False,
                )
            ):

                status.update(
                    label="Input validation failed",
                    state="error",
                )

                errors = validation.get(
                    "errors",
                    [],
                )

                for error in errors:

                    if isinstance(
                        error,
                        dict,
                    ):
                        st.error(
                            error.get(
                                "message",
                                str(error),
                            )
                        )

                    else:
                        st.error(
                            str(error)
                        )

                st.stop()

            st.write(
                "Executing remote-sensing model..."
            )

            outer_result = execute_query(
                query=query.strip(),
                intent=intent,
                image_path=image_path,
                before_path=before_path,
                after_path=after_path,
                optical_path=optical_path,
                sar_path=sar_path,
            )

            # ------------------------------------------------
            # FIX THE PREVIOUS REPORT / GUI PROBLEM
            # ------------------------------------------------

            result = unwrap_execution_result(
                outer_result
            )

            input_metadata = {
                key: value
                for key, value
                in {
                    "image_path":
                        image_path,

                    "before_path":
                        before_path,

                    "after_path":
                        after_path,

                    "optical_path":
                        optical_path,

                    "sar_path":
                        sar_path,
                }.items()
                if value is not None
            }

            # Ensure model-level result includes tools.
            if tools:
                result["tools_used"] = (
                    tools
                )

            # Preserve routing confidence only when the
            # execution did not provide a model score.
            if (
                "confidence"
                not in result
                or result.get(
                    "confidence"
                ) is None
            ):

                result[
                    "confidence"
                ] = plan.get(
                    "confidence",
                    0.0,
                )

                result[
                    "confidence_type"
                ] = (
                    "routing_confidence"
                )

            adapter = get_result_adapter()

            standardized = (
                adapter.standardize(
                    result,
                    query=query.strip(),
                    intent=intent,
                    tools=tools,
                    inputs=input_metadata,
                    validation=validation,
                    default_confidence_type=(
                        "relative_tile_relevance"
                        if intent
                        == "text_guided_grounding"
                        else "model_score"
                    ),
                )
            )

            # Ensure correct answer/model survives adapter.
            if result.get("answer"):
                standardized[
                    "answer"
                ] = result[
                    "answer"
                ]

            if result.get("model"):
                standardized[
                    "model"
                ] = result[
                    "model"
                ]

            if result.get("device"):
                standardized[
                    "device"
                ] = result[
                    "device"
                ]

            if result.get("evidence"):
                standardized[
                    "evidence"
                ] = result[
                    "evidence"
                ]

            st.session_state.analysis_result = (
                standardized
            )

            st.session_state.analysis_query = (
                query.strip()
            )

            st.session_state.analysis_intent = (
                intent
            )

            st.session_state.analysis_validation = (
                validation
            )

            st.session_state.analysis_tools = (
                tools
            )

            add_recent_analysis(
                query=query.strip(),
                intent=intent,
            )

            status.update(
                label=(
                    "Analysis completed successfully"
                ),
                state="complete",
                expanded=False,
            )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                traceback.format_exc()
            )


# ============================================================
# RIGHT RESULT PANEL
# ============================================================

with result_col:

    result = (
        st.session_state
        .analysis_result
    )

    if result:

        answer = get_answer(
            result
        )

        confidence = confidence_data(
            result
        )

        confidence_pct = round(
            float(
                confidence.get(
                    "percentage",
                    0,
                )
            ),
            1,
        )

        model_name = (
            result.get("model")
            or result.get("model_name")
            or "SatQuery Pipeline"
        )

        device = (
            result.get("device")
            or "cpu"
        )

        intent = (
            st.session_state
            .analysis_intent
            or result.get(
                "intent",
                "unknown",
            )
        )

        evidence_paths = (
            get_evidence_paths(
                result
            )
        )

        st.markdown(
            f"""
<div class="sq-result-shell">

    <div class="sq-result-header">

        <div class="sq-result-title">
            ✥ &nbsp; Analysis Result
        </div>

        <div class="sq-success">
            ✦ SUCCESS
        </div>

    </div>

    <div class="sq-result-card">

        <div class="sq-result-label">
            💬 &nbsp; Your Question
        </div>

        <div class="sq-result-value">
            {st.session_state.analysis_query}
        </div>

    </div>

    <div class="sq-result-card">

        <div class="sq-result-label">
            ◉ &nbsp; Answer
        </div>

        <div class="sq-answer">
            {answer}
        </div>

        <div class="sq-confidence-row">

            <div
                class="sq-ring"
                style="--p:{confidence_pct};"
            >
                <span>
                    {confidence_pct}%
                </span>
            </div>

            <div>

                <div class="sq-model-grid">

                    <div class="sq-model-label">
                        Model
                    </div>

                    <div class="sq-model-value">
                        {model_name}
                    </div>

                    <div class="sq-model-label">
                        Device
                    </div>

                    <div class="sq-model-value">
                        {device}
                    </div>

                    <div class="sq-model-label">
                        Task
                    </div>

                    <div class="sq-model-value">
                        {intent.replace("_", " ").title()}
                    </div>

                </div>

            </div>

        </div>

    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="sq-result-shell">

    <div class="sq-evidence-title">

        <div class="sq-result-title">
            🖼️ &nbsp; Visual Evidence
        </div>

    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        if evidence_paths:

            for index, path in enumerate(
                evidence_paths[:4]
            ):

                try:

                    st.image(
                        str(path),
                        caption=(
                            "Satellite Evidence"
                            if index == 0
                            else path.name
                        ),
                        use_container_width=True,
                    )

                except Exception:

                    st.caption(
                        str(path)
                    )

        else:

            st.markdown(
                """
<div class="sq-evidence-placeholder">
    No visual evidence generated
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True,
        )

        report_generator = (
            get_report_generator()
        )

        report = (
            report_generator
            .generate(
                result,
                query=(
                    st.session_state
                    .analysis_query
                ),
            )
        )

        html_path = Path(
            report[
                "html_report"
            ]
        )

        json_path = Path(
            report[
                "json_report"
            ]
        )

        report_col1, report_col2 = (
            st.columns(2)
        )

        if html_path.exists():

            with open(
                html_path,
                "rb",
            ) as file:

                report_col1.download_button(
                    "⬇ Download Report",
                    data=file.read(),
                    file_name=html_path.name,
                    mime="text/html",
                    use_container_width=True,
                )

        if json_path.exists():

            with open(
                json_path,
                "rb",
            ) as file:

                report_col2.download_button(
                    "◫ Export JSON",
                    data=file.read(),
                    file_name=json_path.name,
                    mime="application/json",
                    use_container_width=True,
                )

        with st.expander(
            "Technical Analysis"
        ):

            tab1, tab2, tab3 = (
                st.tabs(
                    [
                        "Analysis",
                        "Validation",
                        "Execution",
                    ]
                )
            )

            with tab1:

                st.json(
                    {
                        key: value
                        for key, value
                        in result.items()
                        if key
                        not in {
                            "execution_summary",
                        }
                    }
                )

            with tab2:

                validation = (
                    st.session_state
                    .analysis_validation
                )

                if validation:
                    st.json(
                        validation
                    )

                else:
                    st.info(
                        "No validation metadata."
                    )

            with tab3:

                summary = (
                    result.get(
                        "execution_summary",
                        {},
                    )
                )

                st.json(
                    summary
                )

    else:

        st.markdown(
            """
<div class="sq-result-shell">

    <div class="sq-result-header">

        <div class="sq-result-title">
            ✥ &nbsp; Analysis Result
        </div>

    </div>

    <div class="sq-result-card">

        <div style="
            min-height:165px;
            display:grid;
            place-items:center;
            text-align:center;
        ">

            <div>

                <div style="
                    font-size:38px;
                    margin-bottom:12px;
                ">
                    🛰️
                </div>

                <div style="
                    color:#d7e9f7;
                    font-size:14px;
                    font-weight:750;
                ">
                    Ready for analysis
                </div>

                <div style="
                    color:#67859d;
                    font-size:11px;
                    margin-top:7px;
                    line-height:1.5;
                ">
                    Upload satellite imagery and
                    ask SatQuery AI a question.
                </div>

            </div>

        </div>

    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='height:10px'></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="sq-result-shell">

    <div class="sq-evidence-title">

        <div class="sq-result-title">
            🖼️ &nbsp; Visual Evidence
        </div>

    </div>

    <div class="sq-evidence-placeholder">
        Evidence will appear here after analysis
    </div>

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="sq-footer">

    <div>
        © 2026 SatQuery AI
        &nbsp; | &nbsp;
        Interactive Vision-Language Assistant
        for Multimodal Remote Sensing Analysis
    </div>

    <div>
        RemoteCLIP · BLIP · ChangeFormer · EuroSAT
    </div>

</div>
""",
    unsafe_allow_html=True,
)