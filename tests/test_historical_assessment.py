from pathlib import Path
import unittest

if __name__ != "__main__":
    raise unittest.SkipTest("Manual historical-assessment script; run directly for fixture inspection.")

from gee_temporal import YearComposite, assess_historical_change


ROOT = Path(__file__).resolve().parents[1]

before_path = (
    ROOT
    / "outputs"
    / "temporal_high_change"
    / "navi_mumbai_airport"
    / "navi_mumbai_airport_2017.tif"
)

after_path = (
    ROOT
    / "outputs"
    / "temporal_high_change"
    / "navi_mumbai_airport"
    / "navi_mumbai_airport_2025.tif"
)

before = YearComposite(
    year=2017,
    path=before_path,
    collection_id="COPERNICUS/S2_SR_HARMONIZED",
    image_count=24,
    scale_m=10.0,
    stats={
        "ndvi": 0.25431352482351055,
        "ndbi": -0.05075964926517886,
        "water_fraction": 0.10864917351593327,
    },
)

after = YearComposite(
    year=2025,
    path=after_path,
    collection_id="COPERNICUS/S2_SR_HARMONIZED",
    image_count=100,
    scale_m=10.0,
    stats={
        "ndvi": 0.16462887062811735,
        "ndbi": -0.002706784177750041,
        "water_fraction": 0.05103180407229194,
    },
)

result = assess_historical_change(
    before=before,
    after=after,
    model_answer=(
        "No significant change was detected between "
        "2017 and 2025."
    ),
    changeformer_change_pct=0.0,
    cross_sensor=False,
)

print("=" * 78)
print("SATQUERY HISTORICAL ASSESSMENT TEST")
print("=" * 78)
print("Level:", result["level"])
print(
    "Spectral change detected:",
    result["spectral_change_detected"],
)
print(
    "ChangeFormer changed percentage:",
    result["changeformer_changed_percentage"],
)
print()
print("ANSWER")
print("-" * 78)
print(result["answer"])
print()
print("METRICS")
print("-" * 78)
print(result["metrics"])

assert result["level"] == "significant"
assert result["spectral_change_detected"] is True
assert (
    result["changeformer_changed_percentage"]
    == 0.0
)

print()
print("TEST PASSED")
