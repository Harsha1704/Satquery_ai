from __future__ import annotations

from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]

ORCHESTRATOR = ROOT / "ai" / "router" / "orchestrator.py"
FRONTEND_APP = ROOT / "frontend" / "app.py"
FRONTEND_JS = ROOT / "frontend" / "static" / "js" / "app.js"

ORCH_MARKER = "# === SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ==="
APP_MARKER = "# === SATQUERY EVIDENCE FLATTEN V1 ==="
JS_MARKER = "// === SATQUERY CONFIDENCE LABEL V1 ==="


ORCH_HELPERS = r'''
    # === SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ===

    @staticmethod
    def _assess_visual_pair(
        before_path: str,
        after_path: str,
    ) -> Dict[str, Any]:
        # Scene-level RGB difference metrics. These complement ChangeFormer;
        # they are not semantic land-cover labels by themselves.
        from PIL import Image

        with Image.open(before_path) as before_image:
            before_rgb = np.asarray(
                before_image.convert("RGB"),
                dtype=np.float32,
            )

        with Image.open(after_path) as after_image:
            after_rgb = np.asarray(
                after_image.convert("RGB"),
                dtype=np.float32,
            )

        if before_rgb.shape != after_rgb.shape:
            return {
                "available": False,
                "reason": (
                    "Visual-difference metrics require aligned images "
                    "with identical dimensions."
                ),
            }

        absolute_difference = np.abs(
            after_rgb - before_rgb
        )

        per_pixel_difference = np.mean(
            absolute_difference,
            axis=2,
        )

        mean_absolute_difference = float(
            np.mean(
                absolute_difference
            )
        )

        changed_gt_10 = float(
            np.mean(
                per_pixel_difference > 10.0
            )
            * 100.0
        )

        changed_gt_20 = float(
            np.mean(
                per_pixel_difference > 20.0
            )
            * 100.0
        )

        changed_gt_30 = float(
            np.mean(
                per_pixel_difference > 30.0
            )
            * 100.0
        )

        if (
            changed_gt_20 >= 25.0
            or mean_absolute_difference >= 20.0
        ):
            level = "significant"
        elif (
            changed_gt_20 >= 10.0
            or mean_absolute_difference >= 10.0
        ):
            level = "meaningful"
        elif (
            changed_gt_20 >= 3.0
            or mean_absolute_difference >= 5.0
        ):
            level = "limited"
        else:
            level = "low"

        return {
            "available": True,
            "level": level,
            "mean_absolute_rgb_difference": round(
                mean_absolute_difference,
                4,
            ),
            "pixels_difference_gt_10_pct": round(
                changed_gt_10,
                4,
            ),
            "pixels_difference_gt_20_pct": round(
                changed_gt_20,
                4,
            ),
            "pixels_difference_gt_30_pct": round(
                changed_gt_30,
                4,
            ),
            "note": (
                "RGB scene differences can include real surface change, "
                "seasonality, illumination, atmosphere, compositing, and "
                "small registration effects. They are not a semantic "
                "land-cover label by themselves."
            ),
        }

    @staticmethod
    def _build_combined_change_answer(
        visual_assessment: Dict[str, Any],
        changed_percentage: float,
        before_year: Optional[int],
        after_year: Optional[int],
        fallback_answer: str,
    ) -> str:
        if before_year is not None and after_year is not None:
            period = (
                f"between {before_year} and {after_year}"
            )
        else:
            period = (
                "between the before and after images"
            )

        visual_available = bool(
            visual_assessment.get("available")
        )

        if not visual_available:
            return fallback_answer

        level = str(
            visual_assessment.get(
                "level",
                "low",
            )
        )

        pct20 = float(
            visual_assessment.get(
                "pixels_difference_gt_20_pct",
                0.0,
            )
        )

        mean_diff = float(
            visual_assessment.get(
                "mean_absolute_rgb_difference",
                0.0,
            )
        )

        if level == "significant":
            visual_text = (
                f"Substantial visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels, with "
                f"a mean absolute RGB difference of {mean_diff:.1f}/255."
            )
        elif level == "meaningful":
            visual_text = (
                f"Meaningful visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels, with "
                f"a mean absolute RGB difference of {mean_diff:.1f}/255."
            )
        elif level == "limited":
            visual_text = (
                f"Limited visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels."
            )
        else:
            visual_text = (
                f"The aligned scenes show only small RGB differences "
                f"{period}."
            )

        if changed_percentage < 0.01:
            structural_text = (
                "ChangeFormerV6 detected no confident LEVIR-style "
                "structural/building change (0.00%)."
            )
        else:
            structural_text = (
                "ChangeFormerV6 detected LEVIR-style structural/building "
                f"change across approximately {changed_percentage:.2f}% "
                "of the analyzed area."
            )

        limitation_text = (
            "Because ChangeFormerV6 was trained on high-resolution "
            "LEVIR-CD building imagery, its structural result should not "
            "override scene-level differences in medium-resolution "
            "Sentinel/Landsat composites. RGB differences can also be "
            "caused by seasonality, illumination, atmosphere, compositing, "
            "or small registration effects, so they should not be treated "
            "as a semantic land-cover transition by themselves."
        )

        return (
            f"{visual_text} "
            f"{structural_text} "
            f"{limitation_text}"
        )

    # === END SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ===
'''


def backup(path: Path) -> None:
    backup_path = path.with_name(
        path.name + ".before_bitemporal_visual_fix"
    )

    if not backup_path.exists():
        shutil.copy2(
            path,
            backup_path,
        )
        print(
            "Backup created:",
            backup_path,
        )


def patch_orchestrator() -> None:
    source = ORCHESTRATOR.read_text(
        encoding="utf-8"
    )

    if ORCH_MARKER not in source:
        anchor = (
            "    def _execute_change_detection(\n"
        )

        if anchor not in source:
            raise RuntimeError(
                "Could not find _execute_change_detection() "
                "in ai/router/orchestrator.py"
            )

        source = source.replace(
            anchor,
            ORCH_HELPERS + "\n" + anchor,
            1,
        )

    old_after_analysis = '''            changed_percentage = float(
                analysis.get(
                    "changed_percentage",
                    0.0,
                )
            )

            if changed_percentage < 0.01:
'''

    new_after_analysis = '''            changed_percentage = float(
                analysis.get(
                    "changed_percentage",
                    0.0,
                )
            )

            visual_assessment = (
                self._assess_visual_pair(
                    before_path=before_path,
                    after_path=after_path,
                )
            )

            combined_answer = (
                self._build_combined_change_answer(
                    visual_assessment=visual_assessment,
                    changed_percentage=changed_percentage,
                    before_year=before_year,
                    after_year=after_year,
                    fallback_answer=vqa_result.get(
                        "answer",
                        description,
                    ),
                )
            )

            if changed_percentage < 0.01:
'''

    if (
        "visual_assessment = (" not in source
        and old_after_analysis in source
    ):
        source = source.replace(
            old_after_analysis,
            new_after_analysis,
            1,
        )

    old_answer = '''                "answer": vqa_result.get(
                    "answer",
                    description,
                ),
                "description": description,
'''

    new_answer = '''                "answer": combined_answer,
                "description": combined_answer,
                "structural_change_percentage": (
                    changed_percentage
                ),
                "visual_change_assessment": (
                    self._make_json_safe(
                        visual_assessment
                    )
                ),
'''

    if (
        '"answer": combined_answer' not in source
        and old_answer in source
    ):
        source = source.replace(
            old_answer,
            new_answer,
            1,
        )

    old_summary = '''                    "changed_percentage": changed_percentage,
                    "change_level": confidence_label,
'''

    new_summary = '''                    "changed_percentage": changed_percentage,
                    "structural_change_percentage": changed_percentage,
                    "visual_change_level": visual_assessment.get(
                        "level"
                    ),
                    "visual_mean_absolute_rgb_difference": (
                        visual_assessment.get(
                            "mean_absolute_rgb_difference"
                        )
                    ),
                    "visual_pixels_difference_gt_20_pct": (
                        visual_assessment.get(
                            "pixels_difference_gt_20_pct"
                        )
                    ),
                    "change_level": confidence_label,
'''

    if (
        '"visual_change_level"' not in source
        and old_summary in source
    ):
        source = source.replace(
            old_summary,
            new_summary,
            1,
        )

    ORCHESTRATOR.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "Updated:",
        ORCHESTRATOR,
    )


NEW_EVIDENCE_FUNCTION = r'''# === SATQUERY EVIDENCE FLATTEN V1 ===
def evidence_items(result: Dict[str, Any]):
    evidence = (
        result.get("evidence")
        or result.get("visual_evidence")
        or result.get("evidence_path")
    )

    output = []
    seen_urls = set()

    def add_path(
        value: Any,
        label: Optional[str] = None,
    ) -> None:
        if value is None:
            return

        if isinstance(value, dict):
            direct = (
                value.get("path")
                or value.get("file")
                or value.get("image")
            )

            if direct:
                add_path(
                    direct,
                    value.get("name")
                    or label,
                )
                return

            for key, nested in value.items():
                add_path(
                    nested,
                    str(key)
                    .replace("_", " ")
                    .title(),
                )
            return

        if isinstance(value, (list, tuple, set)):
            for nested in value:
                add_path(
                    nested,
                    label,
                )
            return

        path = Path(
            str(value)
        )

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists() or not path.is_file():
            return

        try:
            relative = (
                path.resolve()
                .relative_to(
                    ROOT.resolve()
                )
            )
        except Exception:
            return

        url = (
            f"/project-file/"
            f"{relative.as_posix()}"
        )

        if url in seen_urls:
            return

        seen_urls.add(
            url
        )

        output.append(
            {
                "name": (
                    label
                    or path.stem
                    .replace("_", " ")
                    .title()
                ),
                "url": url,
            }
        )

    add_path(
        evidence
    )

    return output
# === END SATQUERY EVIDENCE FLATTEN V1 ===
'''


def patch_frontend_app() -> None:
    source = FRONTEND_APP.read_text(
        encoding="utf-8"
    )

    if APP_MARKER not in source:
        pattern = re.compile(
            r"def evidence_items\(result: Dict\[str, Any\]\):"
            r".*?"
            r"(?=\n\n@app\.route\(\"/\"\))",
            re.DOTALL,
        )

        source, count = pattern.subn(
            NEW_EVIDENCE_FUNCTION.rstrip(),
            source,
            count=1,
        )

        if count != 1:
            raise RuntimeError(
                "Could not replace evidence_items() "
                "in frontend/app.py"
            )

    if (
        '"confidence_type": confidence_type,' not in source
    ):
        anchor = '''        html_report = Path(report["html_report"])
'''

        insertion = '''        confidence_type = (
            result.get("confidence_type")
            or standardized.get("confidence_type")
            or confidence_details.get("type")
            or "model_score"
        )

        structural_change_percentage = (
            result.get("structural_change_percentage")
        )

        visual_change_assessment = (
            result.get("visual_change_assessment")
        )

'''

        if anchor not in source:
            raise RuntimeError(
                "Could not find report response section "
                "in frontend/app.py"
            )

        source = source.replace(
            anchor,
            insertion + anchor,
            1,
        )

        response_anchor = '''                "confidence_details": json_safe(confidence_details),
'''

        response_extra = '''                "confidence_details": json_safe(confidence_details),
                "confidence_type": confidence_type,
                "structural_change_percentage": (
                    structural_change_percentage
                ),
                "visual_change_assessment": json_safe(
                    visual_change_assessment
                ),
'''

        if response_anchor not in source:
            raise RuntimeError(
                "Could not find confidence response "
                "in frontend/app.py"
            )

        source = source.replace(
            response_anchor,
            response_extra,
            1,
        )

    FRONTEND_APP.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "Updated:",
        FRONTEND_APP,
    )


JS_HELPER = r'''
// === SATQUERY CONFIDENCE LABEL V1 ===
function updateConfidenceLabel(data) {
    const ring = $('confidenceRing');
    const block = ring
        ? ring.closest('.confidence-block')
        : null;
    const label = block
        ? block.querySelector('.field-label')
        : null;

    if (!label) return;

    const confidenceType = String(
        data?.confidence_type
        || data?.confidence_details?.type
        || ''
    ).toLowerCase();

    if (confidenceType === 'routing_confidence') {
        label.textContent = 'Routing Confidence';
    } else if (confidenceType === 'relative_tile_relevance') {
        label.textContent = 'Relative Relevance';
    } else if (confidenceType === 'model_score') {
        label.textContent = 'Model Score';
    } else {
        label.textContent = 'Confidence';
    }
}
// === END SATQUERY CONFIDENCE LABEL V1 ===
'''


def patch_frontend_js() -> None:
    source = FRONTEND_JS.read_text(
        encoding="utf-8"
    )

    if JS_MARKER not in source:
        anchor = '''function showEvidence(index) {
'''

        if anchor not in source:
            raise RuntimeError(
                "Could not find showEvidence() "
                "in frontend/static/js/app.js"
            )

        source = source.replace(
            anchor,
            JS_HELPER + "\n" + anchor,
            1,
        )

    old_update = '''    setConfidence(data.confidence);
    $('modelValue').textContent = data.model || '—';
    $('deviceValue').textContent = data.device || '—';
    $('taskValue').textContent = data.task || data.intent || '—';
    renderEvidence(data.evidence);
'''

    new_update = '''    setConfidence(data.confidence);
    updateConfidenceLabel(data);
    $('modelValue').textContent = data.model || '—';
    $('deviceValue').textContent = data.device || '—';

    let taskText = data.task || data.intent || '—';

    if (
        data.intent === 'change_detection'
        && data.structural_change_percentage !== null
        && data.structural_change_percentage !== undefined
        && Number.isFinite(
            Number(data.structural_change_percentage)
        )
    ) {
        taskText += (
            ` · Structural ${Number(
                data.structural_change_percentage
            ).toFixed(2)}%`
        );
    }

    $('taskValue').textContent = taskText;
    renderEvidence(data.evidence);
'''

    if (
        "Structural ${Number(" not in source
        and old_update in source
    ):
        source = source.replace(
            old_update,
            new_update,
            1,
        )

    FRONTEND_JS.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "Updated:",
        FRONTEND_JS,
    )


def main() -> None:
    for path in [
        ORCHESTRATOR,
        FRONTEND_APP,
        FRONTEND_JS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )
        backup(
            path
        )

    patch_orchestrator()
    patch_frontend_app()
    patch_frontend_js()

    print()
    print(
        "Bi-temporal visual-change fix installed."
    )


if __name__ == "__main__":
    main()
 