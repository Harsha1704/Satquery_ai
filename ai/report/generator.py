# ai/report/generator.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import html
import base64
import mimetypes


class SatQueryReportGenerator:
    """
    Generates downloadable SatQuery AI analysis reports.

    Supported outputs:
        - HTML
        - JSON

    The report includes:
        - user query
        - detected intent
        - analysis result
        - confidence
        - model/tool information
        - validation results
        - execution summary
        - visual evidence
        - limitations
        - input metadata
    """

    def __init__(
        self,
        output_dir: str = "outputs/reports",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate(
        self,
        result: Dict[str, Any],
        *,
        query: Optional[str] = None,
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        timestamp = datetime.now()

        if report_name is None:
            report_name = (
                f"satquery_report_"
                f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
            )

        report_name = self._safe_filename(
            report_name
        )

        html_path = (
            self.output_dir
            / f"{report_name}.html"
        )

        json_path = (
            self.output_dir
            / f"{report_name}.json"
        )

        normalized = self._normalize_result(
            result=result,
            query=query,
            generated_at=timestamp,
        )

        self._write_json(
            normalized,
            json_path,
        )

        self._write_html(
            normalized,
            html_path,
        )

        return {
            "success": True,
            "html_report": str(html_path),
            "json_report": str(json_path),
            "generated_at": timestamp.isoformat(),
        }

    def _normalize_result(
        self,
        result: Dict[str, Any],
        query: Optional[str],
        generated_at: datetime,
    ) -> Dict[str, Any]:

        execution_summary = result.get(
            "execution_summary",
            {},
        )

        if not isinstance(
            execution_summary,
            dict,
        ):
            execution_summary = {}

        final_query = (
            query
            or execution_summary.get("query")
            or result.get("query")
            or "Not provided"
        )

        intent = (
            execution_summary.get("intent")
            or result.get("intent")
            or result.get("task")
            or "unknown"
        )

        confidence = (
            result.get("confidence_details")
            or execution_summary.get("confidence")
            or self._fallback_confidence(result)
        )

        evidence = (
            execution_summary.get("evidence")
            or result.get("evidence")
            or result.get("evidence_path")
            or result.get("visual_evidence")
        )

        limitations = (
            execution_summary.get("limitations")
            or result.get("limitations")
            or []
        )

        if isinstance(limitations, str):
            limitations = [limitations]

        validation = (
            execution_summary.get("validation")
            or result.get("validation")
        )

        model = (
            execution_summary.get("model")
            or result.get("model")
            or result.get("model_name")
        )

        device = (
            execution_summary.get("device")
            or result.get("device")
        )

        tools = (
            execution_summary.get("tools_used")
            or result.get("tools_used")
            or result.get("tools")
            or []
        )

        answer = (
            result.get("answer")
            or result.get("message")
            or execution_summary.get("message")
            or "Analysis completed."
        )

        return {
            "report": {
                "title": (
                    "SatQuery AI Remote Sensing "
                    "Analysis Report"
                ),
                "generated_at": (
                    generated_at.isoformat()
                ),
                "system": "SatQuery AI",
            },
            "query": final_query,
            "intent": str(intent),
            "success": bool(
                result.get("success", True)
            ),
            "answer": answer,
            "confidence": confidence,
            "model": model,
            "device": device,
            "tools_used": tools,
            "inputs": execution_summary.get(
                "inputs",
                result.get("inputs", {}),
            ),
            "validation": validation,
            "evidence": evidence,
            "limitations": limitations,
            "execution_steps": (
                execution_summary.get(
                    "execution_steps",
                    [],
                )
            ),
            "analysis_result": (
                self._clean_analysis_result(
                    result
                )
            ),
        }

    @staticmethod
    def _fallback_confidence(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        score = result.get(
            "confidence",
            result.get("score", 0.0),
        )

        try:
            score = float(score)
        except Exception:
            score = 0.0

        if score > 1.0 and score <= 100.0:
            score /= 100.0

        score = max(
            0.0,
            min(1.0, score),
        )

        return {
            "score": round(score, 4),
            "percentage": round(
                score * 100,
                2,
            ),
            "confidence_type": result.get(
                "confidence_type",
                "model_score",
            ),
            "calibrated": False,
        }

    @staticmethod
    def _clean_analysis_result(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        excluded = {
            "execution_summary",
            "confidence_details",
            "mask",
            "heatmap",
            "array",
            "image",
            "feature_map",
        }

        cleaned = {}

        for key, value in result.items():

            if key in excluded:
                continue

            cleaned[key] = (
                SatQueryReportGenerator
                ._make_serializable(value)
            )

        return cleaned

    @staticmethod
    def _make_serializable(
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(k):
                SatQueryReportGenerator
                ._make_serializable(v)
                for k, v in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                SatQueryReportGenerator
                ._make_serializable(v)
                for v in value
            ]

        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        return str(value)

    def _write_json(
        self,
        data: Dict[str, Any],
        path: Path,
    ):

        serializable = (
            self._make_serializable(data)
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                serializable,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _write_html(
        self,
        data: Dict[str, Any],
        path: Path,
    ):

        document = self._build_html(
            data
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(document)

    def _build_html(
        self,
        data: Dict[str, Any],
    ) -> str:

        query = self._escape(
            data.get("query")
        )

        intent = self._escape(
            data.get("intent")
        )

        answer = self._escape(
            data.get("answer")
        )

        model = self._escape(
            data.get("model")
            or "Not specified"
        )

        device = self._escape(
            data.get("device")
            or "Not specified"
        )

        confidence = (
            data.get("confidence")
            or {}
        )

        percentage = confidence.get(
            "percentage",
            0,
        )

        confidence_type = self._escape(
            confidence.get(
                "confidence_type",
                "unknown",
            )
        )

        calibrated = (
            "Yes"
            if confidence.get(
                "calibrated",
                False,
            )
            else "No"
        )

        tools_html = self._list_html(
            data.get(
                "tools_used",
                [],
            )
        )

        limitations_html = (
            self._list_html(
                data.get(
                    "limitations",
                    [],
                )
            )
        )

        steps_html = self._steps_html(
            data.get(
                "execution_steps",
                [],
            )
        )

        validation_html = (
            self._json_block(
                data.get("validation")
            )
        )

        inputs_html = self._json_block(
            data.get("inputs")
        )

        result_html = self._json_block(
            data.get(
                "analysis_result"
            )
        )

        evidence_html = (
            self._evidence_html(
                data.get("evidence"),
                path_parent=self.output_dir,
            )
        )

        generated_at = self._escape(
            data["report"]["generated_at"]
        )

        status = (
            "SUCCESS"
            if data.get("success")
            else "FAILED"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>SatQuery AI Analysis Report</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;
    background: #f4f7fb;
    color: #172033;
}}

.container {{
    width: min(1100px, 94%);
    margin: 30px auto;
}}

.header {{
    background:
        linear-gradient(
            135deg,
            #07152c,
            #153b72
        );
    color: white;
    padding: 34px;
    border-radius: 16px;
    margin-bottom: 22px;
}}

.header h1 {{
    margin: 0 0 8px 0;
    font-size: 30px;
}}

.header p {{
    margin: 5px 0;
    opacity: 0.88;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(210px, 1fr)
        );
    gap: 14px;
    margin-bottom: 20px;
}}

.card {{
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow:
        0 4px 18px
        rgba(20, 45, 80, 0.08);
    margin-bottom: 18px;
}}

.metric {{
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow:
        0 4px 18px
        rgba(20, 45, 80, 0.08);
}}

.metric-label {{
    font-size: 12px;
    text-transform: uppercase;
    color: #65738a;
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 19px;
    font-weight: 700;
}}

h2 {{
    font-size: 19px;
    margin-top: 0;
    color: #153b72;
}}

.answer {{
    font-size: 18px;
    line-height: 1.65;
}}

pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: #081323;
    color: #dbeafe;
    padding: 18px;
    border-radius: 10px;
    overflow-x: auto;
}}

ul {{
    line-height: 1.7;
}}

.step {{
    border-left: 4px solid #245da8;
    padding:
        8px
        14px;
    margin:
        10px
        0;
    background: #f6f9fe;
}}

.evidence {{
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid #d9e2ef;
    margin-top: 12px;
}}

.footer {{
    text-align: center;
    color: #77849a;
    font-size: 12px;
    margin:
        30px
        0;
}}

@media print {{

    body {{
        background: white;
    }}

    .container {{
        width: 100%;
        margin: 0;
    }}

    .card,
    .metric {{
        box-shadow: none;
        border: 1px solid #ddd;
    }}

}}

</style>
</head>

<body>

<div class="container">

<div class="header">

<h1>
SatQuery AI
</h1>

<p>
Interactive Vision-Language Assistant for
Multimodal Remote Sensing Analysis
</p>

<p>
Generated: {generated_at}
</p>

</div>


<div class="grid">

<div class="metric">
<div class="metric-label">
Status
</div>
<div class="metric-value">
{status}
</div>
</div>

<div class="metric">
<div class="metric-label">
Intent
</div>
<div class="metric-value">
{intent}
</div>
</div>

<div class="metric">
<div class="metric-label">
Routing Confidence
</div>
<div class="metric-value">
{percentage}%
</div>
</div>

<div class="metric">
<div class="metric-label">
Model
</div>
<div class="metric-value">
{model}
</div>
</div>

</div>


<div class="card">

<h2>User Query</h2>

<p class="answer">
{query}
</p>

</div>


<div class="card">

<h2>Analysis Answer</h2>

<p class="answer">
{answer}
</p>

</div>


<div class="card">

<h2>Routing Confidence</h2>

<p>
<strong>Score:</strong>
{percentage}%
</p>

<p>
<strong>Type:</strong>
{confidence_type}
</p>

<p>
<strong>Calibrated:</strong>
{calibrated}
</p>

</div>


<div class="card">

<h2>Model Information</h2>

<p>
<strong>Model:</strong>
{model}
</p>

<p>
<strong>Device:</strong>
{device}
</p>

<h3>Tools Used</h3>

{tools_html}

</div>


<div class="card">

<h2>Input Information</h2>

{inputs_html}

</div>


<div class="card">

<h2>Input Validation</h2>

{validation_html}

</div>


<div class="card">

<h2>Analysis Result</h2>

{result_html}

</div>


<div class="card">

<h2>Visual Evidence</h2>

{evidence_html}

</div>


<div class="card">

<h2>Execution Summary</h2>

{steps_html}

</div>


<div class="card">

<h2>Limitations</h2>

{limitations_html}

</div>


<div class="footer">

SatQuery AI • Remote Sensing Analysis Report

</div>

</div>

</body>
</html>
"""

    @staticmethod
    def _escape(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return html.escape(
            str(value)
        )

    def _json_block(
        self,
        value: Any,
    ) -> str:

        if value is None:
            return "<p>Not available.</p>"

        try:
            text = json.dumps(
                self._make_serializable(
                    value
                ),
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            text = str(value)

        return (
            "<pre>"
            + html.escape(text)
            + "</pre>"
        )

    def _list_html(
        self,
        values: Any,
    ) -> str:

        if not values:
            return "<p>None.</p>"

        if isinstance(values, str):
            values = [values]

        items = "".join(
            f"<li>{self._escape(value)}</li>"
            for value in values
        )

        return f"<ul>{items}</ul>"

    def _steps_html(
        self,
        steps: List[Dict[str, Any]],
    ) -> str:

        if not steps:
            return (
                "<p>No execution steps "
                "were recorded.</p>"
            )

        blocks = []

        for step in steps:

            number = self._escape(
                step.get("step", "")
            )

            name = self._escape(
                step.get(
                    "name",
                    "unknown",
                )
            )

            status = self._escape(
                step.get(
                    "status",
                    "unknown",
                )
            )

            blocks.append(
                f"""
<div class="step">
<strong>Step {number}</strong>
<br>
{name}
<br>
Status: {status}
</div>
"""
            )

        return "".join(blocks)

    def _evidence_html(
        self,
        evidence: Any,
        path_parent: Path,
    ) -> str:

        if not evidence:
            return (
                "<p>No visual evidence "
                "was generated.</p>"
            )

        if isinstance(
            evidence,
            (list, tuple),
        ):
            evidence_items = list(
                evidence
            )
        else:
            evidence_items = [
                evidence
            ]

        blocks = []

        for item in evidence_items:

            if isinstance(item, dict):

                evidence_path = (
                    item.get("path")
                    or item.get("file")
                    or item.get("image")
                )

            else:
                evidence_path = item

            if not evidence_path:
                continue

            evidence_path = str(
                evidence_path
            )

            escaped_path = self._escape(
                evidence_path
            )

            suffix = Path(
                evidence_path
            ).suffix.lower()

            if suffix in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }:

                image_src = self._evidence_image_src(
                    evidence_path,
                    path_parent,
                )

                blocks.append(
                    f"""
<p>
<strong>Evidence:</strong>
{escaped_path}
</p>

<img
    class="evidence"
    src="{self._escape(image_src)}"
    alt="SatQuery AI visual evidence">
"""
                )

            else:

                blocks.append(
                    f"""
<p>
<strong>Evidence:</strong>
{escaped_path}
</p>
"""
                )

        if not blocks:
            return (
                "<p>No visual evidence "
                "was generated.</p>"
            )

        return "".join(blocks)

    def _evidence_image_src(
        self,
        evidence_path: str,
        report_dir: Path,
    ) -> str:
        """Return a self-contained image source for portable HTML reports."""
        resolved = self._resolve_evidence_file(
            evidence_path,
            report_dir,
        )

        if resolved is not None:
            try:
                mime = (
                    mimetypes.guess_type(
                        resolved.name
                    )[0]
                    or "application/octet-stream"
                )
                payload = base64.b64encode(
                    resolved.read_bytes()
                ).decode("ascii")
                return f"data:{mime};base64,{payload}"
            except Exception:
                pass

        return self._relative_evidence_path(
            evidence_path,
            report_dir,
        )

    @staticmethod
    def _resolve_evidence_file(
        evidence_path: str,
        report_dir: Path,
    ) -> Optional[Path]:
        raw = Path(evidence_path)
        candidates = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(report_dir.parent / raw)
            candidates.append(Path.cwd() / raw)
            candidates.append(report_dir / raw)

        for candidate in candidates:
            try:
                candidate = candidate.resolve()
                if candidate.is_file():
                    return candidate
            except Exception:
                continue

        return None

    @staticmethod
    def _relative_evidence_path(
        evidence_path: str,
        report_dir: Path,
    ) -> str:

        try:
            evidence = Path(
                evidence_path
            ).resolve()

            report = report_dir.resolve()

            return str(
                Path(
                    "..",
                    "..",
                    evidence.relative_to(
                        Path.cwd()
                    ),
                )
            ).replace(
                "\\",
                "/",
            )

        except Exception:

            return evidence_path.replace(
                "\\",
                "/",
            )

    @staticmethod
    def _safe_filename(
        value: str,
    ) -> str:

        allowed = []

        for character in value:

            if (
                character.isalnum()
                or character in {
                    "-",
                    "_",
                }
            ):
                allowed.append(
                    character
                )

            else:
                allowed.append("_")

        return "".join(
            allowed
        ).strip("_")