from __future__ import annotations
import argparse, json
from .manifests import validate_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--role",choices=["domain_adaptation","vqa","grounding"],required=True);p.add_argument("--skip-image-check",action="store_true");a=p.parse_args();r=validate_manifest(a.manifest,a.role,check_images=not a.skip_image_check);print(json.dumps(r,indent=2));raise SystemExit(0 if r["valid"] else 2)
if __name__=="__main__": main()
