from pathlib import Path
import csv
import shutil
import traceback

from gee_temporal import plan_temporal_fetch, fetch_year_composite
from ai.change import ChangeFormerDetector, ChangeAnalyzer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "temporal_high_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEFORE_YEAR = 2017
AFTER_YEAR = 2025
MAX_CLOUD = 50.0

# Each ROI is kept under roughly 5 km x 5 km so the existing GEE safety limits
# remain compatible with Sentinel-2 10 m imagery.
CANDIDATES = [
    {
        "name": "greater_noida_west",
        "bounds": [77.405, 28.565, 77.448, 28.604],
    },
    {
        "name": "gurugram_dwarka_expressway",
        "bounds": [76.975, 28.435, 77.018, 28.474],
    },
    {
        "name": "noida_airport_jewar",
        "bounds": [77.590, 28.160, 77.633, 28.199],
    },
    {
        "name": "navi_mumbai_airport",
        "bounds": [73.045, 18.970, 73.087, 19.009],
    },
]


def stable_copy(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main():
    print("=" * 78)
    print("SATQUERY AI - FIND A HIGH-CHANGE REAL SATELLITE PAIR")
    print("=" * 78)
    print(f"Years: {BEFORE_YEAR} -> {AFTER_YEAR}")
    print("Source: Google Earth Engine")
    print("Ranking: ChangeFormerV6 changed percentage")
    print()

    detector = ChangeFormerDetector(tile_size=256, device="cpu")
    analyzer = ChangeAnalyzer()

    results = []

    for index, item in enumerate(CANDIDATES, start=1):
        name = item["name"]
        bounds = item["bounds"]

        print()
        print("-" * 78)
        print(f"[{index}/{len(CANDIDATES)}] {name}")
        print(f"Bounds: {bounds}")
        print("-" * 78)

        workdir = OUT_DIR / name
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            plan = plan_temporal_fetch(
                bounds_wgs84=bounds,
                before_year=BEFORE_YEAR,
                after_year=AFTER_YEAR,
                max_cloud_pct=MAX_CLOUD,
            )

            print("Before source :", plan.before_spec.get("label", plan.before_spec))
            print("After source  :", plan.after_spec.get("label", plan.after_spec))
            print("Before images :", plan.before_count)
            print("After images  :", plan.after_count)
            print("Target scale  :", plan.target_scale_m, "m")
            print("Dimensions    :", plan.width_px, "x", plan.height_px)
            print("Cross sensor  :", plan.cross_sensor)

            dimensions = (plan.width_px, plan.height_px)

            before = fetch_year_composite(
                bounds_wgs84=bounds,
                year=BEFORE_YEAR,
                workdir=workdir,
                max_cloud_pct=MAX_CLOUD,
                spec=plan.before_spec,
                dimensions=dimensions,
            )

            after = fetch_year_composite(
                bounds_wgs84=bounds,
                year=AFTER_YEAR,
                workdir=workdir,
                max_cloud_pct=MAX_CLOUD,
                spec=plan.after_spec,
                dimensions=dimensions,
            )

            before_path = stable_copy(
                before.path,
                workdir / f"{name}_{BEFORE_YEAR}.tif",
            )
            after_path = stable_copy(
                after.path,
                workdir / f"{name}_{AFTER_YEAR}.tif",
            )

            mask_path = workdir / f"{name}_change_mask.png"

            detection = detector.detect_with_result(
                before_path=str(before_path),
                after_path=str(after_path),
                output_path=str(mask_path),
            )

            analysis = analyzer.analyze(detection.mask)
            changed_percentage = float(
                analysis.get("changed_percentage", 0.0)
            )

            print("ChangeFormer changed % :", f"{changed_percentage:.4f}")
            print("Before stats           :", before.stats)
            print("After stats            :", after.stats)

            results.append(
                {
                    "name": name,
                    "bounds": bounds,
                    "before_path": str(before_path),
                    "after_path": str(after_path),
                    "mask_path": str(mask_path),
                    "changed_percentage": changed_percentage,
                    "before_ndvi": before.stats.get("ndvi"),
                    "after_ndvi": after.stats.get("ndvi"),
                    "before_ndbi": before.stats.get("ndbi"),
                    "after_ndbi": after.stats.get("ndbi"),
                    "before_water": before.stats.get("water_fraction"),
                    "after_water": after.stats.get("water_fraction"),
                }
            )

        except Exception as exc:
            print("FAILED:", type(exc).__name__, str(exc))
            traceback.print_exc()

    if not results:
        raise RuntimeError("No candidate ROI completed successfully.")

    results.sort(
        key=lambda row: row["changed_percentage"],
        reverse=True,
    )

    csv_path = OUT_DIR / "high_change_ranking.csv"
    fieldnames = [
        "rank",
        "name",
        "changed_percentage",
        "before_path",
        "after_path",
        "mask_path",
        "before_ndvi",
        "after_ndvi",
        "before_ndbi",
        "after_ndbi",
        "before_water",
        "after_water",
        "bounds",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                }
            )

    print()
    print("=" * 78)
    print("RANKING")
    print("=" * 78)

    for rank, row in enumerate(results, start=1):
        print(
            f"{rank}. {row['name']:<30} "
            f"{row['changed_percentage']:.4f}%"
        )

    best = results[0]

    best_before = OUT_DIR / f"BEST_{best['name']}_{BEFORE_YEAR}.tif"
    best_after = OUT_DIR / f"BEST_{best['name']}_{AFTER_YEAR}.tif"
    best_mask = OUT_DIR / f"BEST_{best['name']}_change_mask.png"

    shutil.copy2(best["before_path"], best_before)
    shutil.copy2(best["after_path"], best_after)
    shutil.copy2(best["mask_path"], best_mask)

    print()
    print("=" * 78)
    print("BEST PAIR")
    print("=" * 78)
    print("Location            :", best["name"])
    print(
        "ChangeFormer change :",
        f"{best['changed_percentage']:.4f}%",
    )
    print("Before GeoTIFF      :", best_before)
    print("After GeoTIFF       :", best_after)
    print("Change mask         :", best_mask)
    print("Ranking CSV         :", csv_path)
    print()
    print("Use these two BEST_*.tif files in SatQuery Bi-temporal Change.")


if __name__ == "__main__":
    main()
