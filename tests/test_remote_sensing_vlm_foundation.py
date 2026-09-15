import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai.models.remote_sensing_vlm import RemoteSensingVLM
from training.remote_sensing_vlm.checkpoint_metadata import sha256, validate
from training.remote_sensing_vlm.manifests import validate_manifest
from training.remote_sensing_vlm.train import training_guard

class RemoteSensingVLMFoundationTests(unittest.TestCase):
    def _manifest(self, directory, records):
        path=Path(directory)/"manifest.jsonl"; path.write_text("\n".join(json.dumps(x) for x in records),encoding="utf-8"); return path
    def test_vqa_manifest_rejects_empty_answers_and_split_leakage(self):
        with TemporaryDirectory() as directory:
            path=self._manifest(directory,[{"sample_id":"a","image":"same.png","question":"?","answer":"yes","dataset":"RSVQA","split":"train"},{"sample_id":"b","image":"same.png","question":"?","answer":"","dataset":"RSVQA","split":"test"}])
            report=validate_manifest(path,"vqa",check_images=False)
        self.assertFalse(report["valid"]); self.assertTrue(any("leakage" in x for x in report["errors"])); self.assertTrue(any("empty" in x for x in report["errors"]))
    def test_checkpoint_metadata_cannot_claim_adaptation_without_provenance(self):
        report=validate({"model_name":"x","base_model":"x","base_revision":"main","domain_adapted":True,"adaptation_method":"lora","adaptation_dataset":"BigEarthNet.txt","training_status":"planned","evaluation_status":"NOT_EVALUATED"})
        self.assertFalse(report["valid"])
    def test_checkpoint_checksum_is_verified(self):
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/"adapter.safetensors"; checkpoint.write_bytes(b"weights")
            metadata={"model_name":"x","base_model":"x","base_revision":"main","domain_adapted":True,"adaptation_method":"lora","adaptation_dataset":"BigEarthNet.txt","training_status":"completed","evaluation_status":"NOT_EVALUATED","checkpoint_sha256":sha256(checkpoint),"checkpoint_file":checkpoint.name}
            self.assertTrue(validate(metadata,checkpoint)["valid"])
            checkpoint.write_bytes(b"other")
            self.assertFalse(validate(metadata,checkpoint)["valid"])
    def test_unavailable_adapter_remains_training_required(self):
        model=RemoteSensingVLM(); self.assertEqual(model.get_capabilities()["remote_sensing_vqa"],"TRAINING_REQUIRED")
        with self.assertRaisesRegex(RuntimeError,"CHECKPOINT_NOT_AVAILABLE"): model.load()
    def test_cuda_required_training_is_explicitly_blocked_on_cpu(self):
        report=training_guard({"training":{"cuda_required":True}},smoke=True)
        if not report["hardware"]["cuda_available"]: self.assertEqual(report["status"],"TRAINING_BLOCKED_BY_HARDWARE")

if __name__=="__main__": unittest.main()
