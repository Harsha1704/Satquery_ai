"""Evaluation report schema for an actual adapted checkpoint; no metrics are fabricated."""
from __future__ import annotations
import json
from pathlib import Path
from .checkpoint_metadata import validate
def report_not_evaluated(metadata_path: str, output: str) -> dict:
 metadata=json.loads(Path(metadata_path).read_text()); check=validate(metadata); result={"status":"NOT_EVALUATED","checkpoint_metadata_valid":check["valid"],"validation":check,"metrics":None};Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2),encoding="utf-8");return result
