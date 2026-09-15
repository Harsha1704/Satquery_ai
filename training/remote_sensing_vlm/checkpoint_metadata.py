from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

REQUIRED = {"model_name","base_model","base_revision","domain_adapted","adaptation_method","adaptation_dataset","training_status","evaluation_status"}
def sha256(path: str | Path) -> str:
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): d.update(block)
    return d.hexdigest()
def validate(metadata: dict[str,Any], checkpoint: str | Path | None = None) -> dict[str,Any]:
    missing=REQUIRED-set(metadata); errors=[f"missing {key}" for key in sorted(missing)]
    if metadata.get("domain_adapted") and (not metadata.get("checkpoint_sha256") or metadata.get("training_status") != "completed"): errors.append("adapted checkpoint lacks completed training provenance")
    if checkpoint:
        actual=sha256(checkpoint)
        if metadata.get("checkpoint_sha256") != actual: errors.append("checkpoint checksum mismatch")
    return {"valid":not errors,"errors":errors}
def write(path: str|Path, metadata: dict[str,Any]) -> None:
    check=validate(metadata)
    if not check["valid"]: raise ValueError("; ".join(check["errors"]))
    Path(path).write_text(json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8")
