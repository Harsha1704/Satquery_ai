"""Baseline evaluator. It never reports metrics until a real held-out manifest runs."""
from __future__ import annotations
import argparse, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from .dataset import hardware
from .manifests import validate_manifest

def baseline_report(manifest: str, output: str, *, model="Salesforce/blip-vqa-base", revision="main", seed=42) -> dict:
    validation=validate_manifest(manifest,"vqa")
    report={"status":"NOT_EVALUATED","model":model,"revision":revision,"dataset_manifest":str(manifest),"sample_count":validation["sample_count"],"seed":seed,"hardware":hardware(),"timestamp":datetime.now(timezone.utc).isoformat(),"metrics":None,"validation":validation}
    try: report["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: report["git_commit"]=None
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",default="reports/vlm/vqa_baseline.json");a=p.parse_args();print(json.dumps(baseline_report(a.manifest,a.output),indent=2))
if __name__=="__main__": main()
