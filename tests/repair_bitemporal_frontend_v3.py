from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

ORCHESTRATOR = ROOT / "ai" / "router" / "orchestrator.py"
FRONTEND_APP = ROOT / "frontend" / "app.py"
FRONTEND_JS = ROOT / "frontend" / "static" / "js" / "app.js"

ORCH_MARKER = "# === SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ==="
APP_MARKER = "# === SATQUERY EVIDENCE FLATTEN V2 ==="
JS_MARKER = "// === SATQUERY CHANGE UI FIX V2 ==="


NEW_EVIDENCE_FUNCTION = '''# === SATQUERY EVIDENCE FLATTEN V2 ===
def evidence_items(result: Dict[str, Any]):
    evidence = (
        result.get("evidence")
        or result.get("visual_evidence")
        or result.get("evidence_path")
    )

    output = []
    seen = set()

    def add_value(value, label=None):
        if value is None:
            return

        if isinstance(value, dict):
            direct = (
                value.get("path")
                or value.get("file")
                or value.get("image")
            )

            if direct:
                add_value(
                    direct,
                    value.get("name") or label,
                )
                return

            for key, nested in value.items():
                add_value(
                    nested,
                    str(key)
                    .replace("_", " ")
                    .title(),
                )
            return

        if isinstance(value, (list, tuple, set)):
            for nested in value:
                add_value(
                    nested,
                    label,
                )
            return

        path = Path(str(value))

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists() or not path.is_file():
            return

        try:
            relative = (
                path.resolve()
                .relative_to(ROOT.resolve())
            )
        except Exception:
            return

        url = (
            f"/project-file/"
            f"{relative.as_posix()}"
        )

        if url in seen:
            return

        seen.add(url)

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

    add_value(evidence)

    return output
# === END SATQUERY EVIDENCE FLATTEN V2 ===
'''


JS_HELPER = r'''
// === SATQUERY CHANGE UI FIX V2 ===
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
        || data?.result?.confidence_type
        || data?.confidence_details?.type
        || ''
    ).toLowerCase();

    let text = 'Confidence';

    if (confidenceType === 'routing_confidence') {
        text = 'Routing Confidence';
    } else if (confidenceType === 'relative_tile_relevance') {
        text = 'Relative Relevance';
    } else if (confidenceType === 'model_score') {
        text = 'Model Score';
    }

    label.replaceChildren(
        icon('radar'),
        document.createTextNode(text)
    );
}

function getStructuralChangePercentage(data) {
    const value =
        data?.structural_change_percentage
        ?? data?.result?.structural_change_percentage
        ?? data?.result?.execution_summary?.structural_change_percentage
        ?? data?.result?.execution_summary?.changed_percentage;

    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : null;
}
// === END SATQUERY CHANGE UI FIX V2 ===
'''


def backup_once(path: Path) -> None:
    backup = path.with_name(
        path.name + ".before_change_ui_fix_v3"
    )

    if not backup.exists():
        shutil.copy2(path, backup)
        print("Backup created:", backup)


def replace_top_level_function(
    source: str,
    name: str,
    replacement: str,
) -> str:
    lines = source.splitlines(keepends=True)

    start = None
    end = None

    for index, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = index
            break

    if start is None:
        raise RuntimeError(
            f"Could not find top-level function {name}()"
        )

    for index in range(start + 1, len(lines)):
        line = lines[index]

        if (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("@app.")
        ):
            end = index
            break

    if end is None:
        end = len(lines)

    return "".join(
        lines[:start]
        + [replacement.rstrip() + "\n\n"]
        + lines[end:]
    )


def patch_frontend_app() -> None:
    source = FRONTEND_APP.read_text(
        encoding="utf-8"
    )

    if APP_MARKER not in source:
        source = replace_top_level_function(
            source,
            "evidence_items",
            NEW_EVIDENCE_FUNCTION,
        )

    if '"structural_change_percentage",' not in source:
        marker = '''            "limitations",
'''

        replacement = '''            "limitations",
            "confidence_type",
            "structural_change_percentage",
            "visual_change_assessment",
'''

        if marker in source:
            source = source.replace(
                marker,
                replacement,
                1,
            )

    if '"structural_change_percentage": json_safe(' not in source:
        marker = '''                "confidence": round(float(percentage or 0.0), 2),
'''

        if marker not in source:
            raise RuntimeError(
                "Could not find the confidence field "
                "inside the /api/analyze JSON response."
            )

        replacement = marker + '''                "confidence_type": (
                    result.get("confidence_type")
                    or standardized.get("confidence_type")
                    or "model_score"
                ),
                "structural_change_percentage": json_safe(
                    result.get(
                        "structural_change_percentage"
                    )
                ),
                "visual_change_assessment": json_safe(
                    result.get(
                        "visual_change_assessment"
                    )
                ),
'''

        source = source.replace(
            marker,
            replacement,
            1,
        )

    FRONTEND_APP.write_text(
        source,
        encoding="utf-8",
    )

    print("Updated:", FRONTEND_APP)


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
                "inside frontend/static/js/app.js"
            )

        source = source.replace(
            anchor,
            JS_HELPER.strip() + "\n\n" + anchor,
            1,
        )

    if "updateConfidenceLabel(data);" not in source:
        marker = '''    setConfidence(data.confidence);
'''

        if marker not in source:
            raise RuntimeError(
                "Could not find setConfidence(data.confidence)."
            )

        source = source.replace(
            marker,
            marker + '''    updateConfidenceLabel(data);
''',
            1,
        )

    old_task = '''    $('taskValue').textContent = data.task || data.intent || '—';
'''

    if old_task in source:
        new_task = '''    let taskText = data.task || data.intent || '—';

    const structuralChange =
        getStructuralChangePercentage(data);

    if (
        data.intent === 'change_detection'
        && structuralChange !== null
    ) {
        taskText += (
            ` | Structural ${structuralChange.toFixed(2)}%`
        );
    }

    $('taskValue').textContent = taskText;
'''

        source = source.replace(
            old_task,
            new_task,
            1,
        )

    FRONTEND_JS.write_text(
        source,
        encoding="utf-8",
    )

    print("Updated:", FRONTEND_JS)


def main():
    for path in [
        ORCHESTRATOR,
        FRONTEND_APP,
        FRONTEND_JS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    orchestrator_source = (
        ORCHESTRATOR.read_text(
            encoding="utf-8"
        )
    )

    if ORCH_MARKER not in orchestrator_source:
        raise RuntimeError(
            "The visual-change orchestrator patch "
            "is not present. Stop here."
        )

    backup_once(FRONTEND_APP)
    backup_once(FRONTEND_JS)

    print(
        "Orchestrator visual-change patch: PRESENT"
    )

    patch_frontend_app()
    patch_frontend_js()

    print()
    print(
        "Bi-temporal frontend/UI repair V3 completed."
    )


if __name__ == "__main__":
    main()
