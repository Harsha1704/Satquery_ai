"""Optional model integration test for a saved bi-temporal scene pair.

This used to execute model inference and assert at import time, which made
``unittest discover`` fail whenever model output changed. It remains available
for explicit integration validation and never runs in the default unit suite.
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    (
        ROOT / "outputs" / "temporal_high_change" / "noida_airport_jewar" / "noida_airport_jewar_2017.tif",
        ROOT / "outputs" / "temporal_high_change" / "noida_airport_jewar" / "noida_airport_jewar_2025.tif",
    ),
    (
        ROOT / "outputs" / "temporal_high_change" / "navi_mumbai_airport" / "navi_mumbai_airport_2017.tif",
        ROOT / "outputs" / "temporal_high_change" / "navi_mumbai_airport" / "navi_mumbai_airport_2025.tif",
    ),
)


@unittest.skipUnless(
    os.environ.get("SATQUERY_RUN_INTEGRATION") == "1",
    "Set SATQUERY_RUN_INTEGRATION=1 to run model-backed bi-temporal validation.",
)
class BitemporalSemanticUpgradeIntegrationTests(unittest.TestCase):
    def test_evidence_grounded_bitemporal_result(self):
        pair = next(((before, after) for before, after in CANDIDATES if before.is_file() and after.is_file()), None)
        self.assertIsNotNone(pair, "No saved bi-temporal test pair was found.")
        from ai.router.orchestrator import SatQueryOrchestrator

        before, after = pair
        execution = SatQueryOrchestrator().execute(
            "What changed between 2017 and 2025?", before_path=str(before), after_path=str(after)
        ).get("execution", {})
        self.assertTrue(execution.get("success"))
        self.assertIn(execution.get("primary_evidence"), {
            "spectral", "semantic", "structural", "visual_support_only", "qualified_supporting_evidence",
        })
        if execution.get("primary_evidence") in {"visual_support_only", "qualified_supporting_evidence"}:
            self.assertIn("supporting evidence", str(execution.get("answer", "")).lower())


if __name__ == "__main__":
    unittest.main()
