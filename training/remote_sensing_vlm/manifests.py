"""Strict JSONL manifests for domain adaptation, VQA, and grounding."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED = {"domain_adaptation": {"sample_id", "image", "text", "dataset", "split"},
            "vqa": {"sample_id", "image", "question", "answer", "dataset", "split"},
            "grounding": {"sample_id", "image", "text", "boxes", "dataset", "split"}}

def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records=[]
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: records.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
    return records

def validate_manifest(path: str | Path, role: str, *, check_images: bool = True) -> dict[str, Any]:
    if role not in REQUIRED: raise ValueError(f"Unsupported dataset role: {role}")
    records=read_manifest(path); errors=[]; seen_ids=set(); image_splits={}; counts=Counter()
    for item in records:
        missing=REQUIRED[role] - set(item)
        if missing: errors.append(f"{item.get('sample_id','<unknown>')}: missing {sorted(missing)}"); continue
        sid=str(item["sample_id"]); counts[str(item.get("split"))]+=1
        if sid in seen_ids: errors.append(f"duplicate sample_id: {sid}")
        seen_ids.add(sid)
        image=Path(str(item["image"])); fingerprint=str(image.resolve()) if image.exists() else str(image)
        image_splits.setdefault(fingerprint,set()).add(str(item["split"]))
        if check_images and not image.is_file(): errors.append(f"{sid}: image missing: {image}")
        if role == "vqa" and (not str(item["question"]).strip() or not str(item["answer"]).strip()): errors.append(f"{sid}: empty question or answer")
        if role == "grounding":
            for box in item["boxes"]:
                if not isinstance(box, list) or len(box) != 4 or any(not isinstance(x,(int,float)) for x in box) or box[2] <= box[0] or box[3] <= box[1]: errors.append(f"{sid}: invalid box")
    leakage=[image for image,splits in image_splits.items() if len(splits)>1]
    if leakage: errors.append(f"cross-split image leakage: {len(leakage)} image(s)")
    return {"valid": not errors, "role": role, "sample_count": len(records), "splits": dict(counts), "errors": errors,
            "manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
