"""Single metadata-governed entry point for future adapted SatQuery VLMs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from training.remote_sensing_vlm.checkpoint_metadata import validate
class RemoteSensingVLM:
 CAPABILITY_TRAINING_REQUIRED="TRAINING_REQUIRED"
 def __init__(self, metadata_path: str|Path|None=None): self.metadata_path=Path(metadata_path) if metadata_path else None; self.metadata=None; self.loaded=False
 def load(self) -> None:
  if not self.metadata_path or not self.metadata_path.is_file(): raise RuntimeError("CHECKPOINT_NOT_AVAILABLE")
  metadata=json.loads(self.metadata_path.read_text(encoding="utf-8")); check=validate(metadata)
  if not check["valid"] or not metadata.get("domain_adapted"): raise RuntimeError("Invalid remote-sensing adapter metadata: "+"; ".join(check["errors"]))
  checkpoint=self.metadata_path.parent / str(metadata.get("checkpoint_file",""))
  if not checkpoint.is_file(): raise RuntimeError("CHECKPOINT_NOT_AVAILABLE")
  if checkpoint.suffix != ".safetensors": raise RuntimeError("Unsafe checkpoint format; safetensors is required.")
  if not validate(metadata,checkpoint)["valid"]: raise RuntimeError("CHECKPOINT_CHECKSUM_FAILED")
  self.metadata=metadata;self.loaded=True
 def unload(self): self.loaded=False
 def is_loaded(self)->bool: return self.loaded
 def get_model_metadata(self)->dict[str,Any]: return self.metadata or {"model_name":"SatQuery Remote-Sensing VLM","domain_adapted":False,"status":self.CAPABILITY_TRAINING_REQUIRED}
 def get_capabilities(self)->dict[str,str]: return {"remote_sensing_vqa": "AVAILABLE" if self.loaded else self.CAPABILITY_TRAINING_REQUIRED,"text_guided_grounding":"EXPERIMENTAL"}
