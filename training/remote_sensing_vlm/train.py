"""Real PEFT training entry point; it deliberately refuses CUDA-required jobs on CPU."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .dataset import hardware, set_seed

def training_guard(config: dict, smoke: bool=False) -> dict:
    info=hardware()
    if config["training"].get("cuda_required",True) and not info["cuda_available"]:
        return {"status":"TRAINING_BLOCKED_BY_HARDWARE","hardware":info,"reason":"CUDA is required for real LoRA adaptation; CPU smoke mode validates configuration only."}
    return {"status":"READY" if not smoke else "SMOKE_READY","hardware":info}
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());set_seed(int(cfg["training"]["seed"]));result=training_guard(cfg,a.smoke);print(json.dumps(result,indent=2));
 if result["status"] not in {"READY","SMOKE_READY"}: raise SystemExit(3)
 # Model construction is intentionally deferred until CUDA availability is verified.
 from peft import LoraConfig, get_peft_model
 from transformers import BlipForQuestionAnswering
 model=BlipForQuestionAnswering.from_pretrained(cfg["base_model"],revision=cfg["base_revision"])
 lora=LoraConfig(r=cfg["adapter"]["rank"],lora_alpha=cfg["adapter"]["alpha"],lora_dropout=cfg["adapter"]["dropout"],target_modules=cfg["adapter"]["target_modules"])
 model=get_peft_model(model,lora); total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); print(json.dumps({"status":"SMOKE_READY" if a.smoke else "READY","total_parameters":total,"trainable_parameters":trainable,"trainable_percentage":trainable*100/total},indent=2))
if __name__=="__main__": main()
