from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

ORCHESTRATOR = ROOT / "ai" / "router" / "orchestrator.py"
FRONTEND_APP = ROOT / "frontend" / "app.py"
FRONTEND_JS = ROOT / "frontend" / "static" / "js" / "app.js"

APP_MARKER = "# === SATQUERY EVIDENCE FLATTEN V1 ==="
JS_MARKER = "// === SATQUERY CONFIDENCE LABEL V1 ==="

NEW_EVIDENCE_FUNCTION = """# === SATQUERY EVIDENCE FLATTEN V1 ===
def evidence_items(result: Dict[str, Any]):
    evidence = (
        result.get("evidence")
        or result.get("visual_evidence")
        or result.get("evidence_path")
    )

    output = []
    seen_urls = set()

    def add_path(value, label=None):
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
                    value.get("name") or label,
                )
                return

            for key, nested in value.items():
                add_path(
                    nested,
                    str(key).replace("_", " ").title(),
                )
            return

        if isinstance(value, (list, tuple, set)):
            for nested in value:
                add_path(nested, label)
            return

        path = Path(str(value))

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists() or not path.is_file():
            return

        try:
            relative = path.resolve().relative_to(ROOT.resolve())
        except Exception:
            return

        url = f"/project-file/{relative.as_posix()}"

        if url in seen_urls:
            return

        seen_urls.add(url)

        output.append(
            {
                "name": (
                    label
                    or path.stem.replace("_", " ").title()
                ),
                "url": url,
            }
        )

    add_path(evidence)
    return output
# === END SATQUERY EVIDENCE FLATTEN V1 ===
"""

JS_HELPER = """
// === SATQUERY CONFIDENCE LABEL V1 ===
function updateConfidenceLabel(data) {
    const ring = $('confidenceRing');
    const block = ring ? ring.closest('.confidence-block') : null;
    const label = block ? block.querySelector('.field-label') : null;

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
"""

def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".before_bitemporal_frontend_repair")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("Backup created:", backup)

def replace_top_level_function(source: str, function_name: str, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = None
    end = None
    signature = f"def {function_name}("

    for i, line in enumerate(lines):
        if line.startswith(signature):
            start = i
            break

    if start is None:
        raise RuntimeError(
            f"Could not find top-level {function_name}() in {FRONTEND_APP}"
        )

    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith("def ") or line.startswith("@app.") or line.startswith("class "):
            end = i
            break

    if end is None:
        end = len(lines)

    return "".join(lines[:start] + [replacement.rstrip() + "\n\n"] + lines[end:])

def patch_frontend_app() -> None:
    source = FRONTEND_APP.read_text(encoding="utf-8")

    if APP_MARKER not in source:
        source = replace_top_level_function(
            source,
            "evidence_items",
            NEW_EVIDENCE_FUNCTION,
        )

    if '"confidence_type": confidence_type,' not in source:
        target = '        confidence_details = standardized.get("confidence_details") or {}\n'

        if target not in source:
            raise RuntimeError(
                "Could not find confidence_details assignment in frontend/app.py"
            )

        insertion = target + """
        confidence_type = (
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
"""

        source = source.replace(target, insertion, 1)

        response_target = (
            '                "confidence_details": json_safe(confidence_details),\n'
        )

        if response_target not in source:
            raise RuntimeError(
                "Could not find confidence_details inside JSON response."
            )

        response_replacement = response_target + """                "confidence_type": confidence_type,
                "structural_change_percentage": (
                    structural_change_percentage
                ),
                "visual_change_assessment": json_safe(
                    visual_change_assessment
                ),
"""

        source = source.replace(
            response_target,
            response_replacement,
            1,
        )

    FRONTEND_APP.write_text(source, encoding="utf-8")
    print("Updated:", FRONTEND_APP)

def patch_frontend_js() -> None:
    source = FRONTEND_JS.read_text(encoding="utf-8")

    if JS_MARKER not in source:
        source = source.rstrip() + "\n\n" + JS_HELPER.strip() + "\n"

    if "updateConfidenceLabel(data);" not in source:
        target = "    setConfidence(data.confidence);\n"

        if target not in source:
            raise RuntimeError(
                "Could not find setConfidence(data.confidence) in frontend/static/js/app.js"
            )

        source = source.replace(
            target,
            target + "    updateConfidenceLabel(data);\n",
            1,
        )

    old_task = (
        "    $('taskValue').textContent = data.task || data.intent || '—';\n"
    )

    if "structural_change_percentage" not in source and old_task in source:
        new_task = """    let taskText = data.task || data.intent || '—';

    if (
        data.intent === 'change_detection'
        && data.structural_change_percentage !== null
        && data.structural_change_percentage !== undefined
        && Number.isFinite(Number(data.structural_change_percentage))
    ) {
        taskText += ` · Structural ${Number(
            data.structural_change_percentage
        ).toFixed(2)}%`;
    }

    $('taskValue').textContent = taskText;
"""
        source = source.replace(old_task, new_task, 1)

    FRONTEND_JS.write_text(source, encoding="utf-8")
    print("Updated:", FRONTEND_JS)

def main():
    for path in [ORCHESTRATOR, FRONTEND_APP, FRONTEND_JS]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    backup_once(FRONTEND_APP)
    backup_once(FRONTEND_JS)

    orchestrator_source = ORCHESTRATOR.read_text(encoding="utf-8")

    if "# === SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ===" not in orchestrator_source:
        raise RuntimeError(
            "The orchestrator part of the previous patch was not installed."
        )

    print("Orchestrator visual-change patch already present.")

    patch_frontend_app()
    patch_frontend_js()

    print()
    print("Frontend repair completed successfully.")

if __name__ == "__main__":
    main()
