"""SatQuery release-candidate preflight checks.

Run from the repository root:
    python scripts/preflight.py
    python scripts/preflight.py --strict
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, ok: bool, detail: str, required: bool = True):
    return {"name": name, "ok": bool(ok), "detail": detail, "required": required}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat recommended release checks as required.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = []
    required_paths = [
        "backend/app/main.py",
        "frontend/app.py",
        "frontend/templates/map.html",
        "frontend/static/js/map.js",
        "query_engine/planner.py",
        "query_engine/capabilities.py",
        "gee_temporal.py",
    ]
    for rel in required_paths:
        path = ROOT / rel
        checks.append(check(f"file:{rel}", path.is_file(), str(path)))

    ai_dir = ROOT / "ai"
    checks.append(check(
        "ai-package",
        (ai_dir / "__init__.py").is_file(),
        "The legacy specialist execution package must be included in a deployable release.",
        required=True,
    ))

    for module in ("fastapi", "flask", "pydantic", "numpy", "PIL", "requests", "rasterio", "ee"):
        ok = importlib.util.find_spec(module) is not None
        checks.append(check(f"import:{module}", ok, f"Python module {module}", required=args.strict or module not in {"ee"}))

    outputs = ROOT / "outputs"
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        probe = outputs / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except Exception as exc:
        writable = False
        write_detail = str(exc)
    else:
        write_detail = str(outputs)
    checks.append(check("outputs-writable", writable, write_detail))

    junk = list((ROOT / "frontend").glob("*.before_*"))
    junk += list((ROOT / "frontend").glob("phase*_backup*"))
    checks.append(check(
        "clean-source-tree",
        len(junk) == 0,
        f"{len(junk)} backup file/folder(s) remain in frontend; release builder excludes them.",
        required=args.strict,
    ))

    py = sys.version_info
    checks.append(check(
        "python-version",
        py >= (3, 11),
        f"Python {py.major}.{py.minor}.{py.micro}; Python 3.11+ is recommended for supported Google client releases.",
        required=args.strict,
    ))

    failed = [x for x in checks if x["required"] and not x["ok"]]
    payload = {"status": "pass" if not failed else "fail", "checks": checks, "failed_required": len(failed)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in checks:
            icon = "PASS" if item["ok"] else ("FAIL" if item["required"] else "WARN")
            print(f"[{icon}] {item['name']}: {item['detail']}")
        print(f"\nPreflight: {payload['status'].upper()} ({len(failed)} required failure(s))")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
