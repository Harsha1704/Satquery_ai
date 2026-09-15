from pathlib import Path
import json

import gee_temporal
from ai.router.orchestrator import SatQueryOrchestrator

BOUNDS = [77.20, 28.55, 77.24, 28.59]
BEFORE_YEAR = 2020
AFTER_YEAR = 2025

QUERY = "What changed here between 2020 and 2025?"

WORKDIR = Path("outputs") / "temporal_test"
WORKDIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("1. INITIALIZING EARTH ENGINE")
print("=" * 70)

gee_temporal.init_earth_engine()
print("Earth Engine: READY")

print()
print("=" * 70)
print("2. PLANNING HISTORICAL FETCH")
print("=" * 70)

plan = gee_temporal.plan_temporal_fetch(
    bounds_wgs84=BOUNDS,
    before_year=BEFORE_YEAR,
    after_year=AFTER_YEAR,
)

print("Before year:", plan.before_year)
print("After year :", plan.after_year)
print("Before collection:", plan.before_spec)
print("After collection :", plan.after_spec)
print("Before images:", plan.before_count)
print("After images :", plan.after_count)
print("Target scale:", plan.target_scale_m, "m")
print("Dimensions:", plan.width_px, "x", plan.height_px)
print("Cross sensor:", plan.cross_sensor)

dimensions = (
    plan.width_px,
    plan.height_px,
)

print()
print("=" * 70)
print("3. FETCHING BEFORE COMPOSITE")
print("=" * 70)

before = gee_temporal.fetch_year_composite(
    bounds_wgs84=BOUNDS,
    year=BEFORE_YEAR,
    workdir=WORKDIR,
    spec=plan.before_spec,
    dimensions=dimensions,
)

print("Path:", before.path)
print("Collection:", before.collection_id)
print("Images:", before.image_count)
print("Scale:", before.scale_m)
print("Stats:")
print(json.dumps(before.stats, indent=2))

print()
print("=" * 70)
print("4. FETCHING AFTER COMPOSITE")
print("=" * 70)

after = gee_temporal.fetch_year_composite(
    bounds_wgs84=BOUNDS,
    year=AFTER_YEAR,
    workdir=WORKDIR,
    spec=plan.after_spec,
    dimensions=dimensions,
)

print("Path:", after.path)
print("Collection:", after.collection_id)
print("Images:", after.image_count)
print("Scale:", after.scale_m)
print("Stats:")
print(json.dumps(after.stats, indent=2))

print()
print("=" * 70)
print("5. RUNNING CHANGEFORMER")
print("=" * 70)

orchestrator = SatQueryOrchestrator()

result = orchestrator.execute(
    QUERY,
    before_path=str(before.path),
    after_path=str(after.path),
)

execution = result.get("execution", {})

print("Success:", execution.get("success"))
print("Model:", execution.get("model"))
print("Training dataset:", execution.get("training_dataset"))
print("Answer:", execution.get("answer") or execution.get("description"))

analysis = execution.get("analysis") or {}

change_pct = analysis.get(
    "changed_percentage",
    execution.get(
        "execution_summary",
        {},
    ).get("changed_percentage"),
)

print("Changed percentage:", change_pct)

print()
print("=" * 70)
print("6. BUILDING COMBINED HISTORICAL NARRATIVE")
print("=" * 70)

narrative = gee_temporal.build_change_narrative(
    before=before,
    after=after,
    model_answer=(
        execution.get("answer")
        or execution.get("description")
    ),
    changeformer_change_pct=change_pct,
    cross_sensor=plan.cross_sensor,
)

print(narrative)

print()
print("=" * 70)
print("7. OUTPUT FILES")
print("=" * 70)

print("Before:", before.path)
print("After :", after.path)

visual = execution.get("visual_evidence")
print("Change evidence:", visual)

print()
print("=" * 70)
print("HISTORICAL COMPARISON TEST COMPLETE")
print("=" * 70)
