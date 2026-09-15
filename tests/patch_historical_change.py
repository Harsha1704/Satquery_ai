from __future__ import annotations

from pathlib import Path
import shutil
import textwrap


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "gee_temporal.py"
BACKUP = ROOT / "gee_temporal_before_historical_fix.py"

MARKER = "# === SATQUERY HISTORICAL CHANGE ASSESSMENT V2 ==="

PATCH = """
# === SATQUERY HISTORICAL CHANGE ASSESSMENT V2 ===

def _satquery_metric_value(stats, key):
    try:
        value = stats.get(key)
        if value is None:
            return None
        value = float(value)
        if value != value:
            return None
        return value
    except Exception:
        return None


def _satquery_metric_level(delta, moderate, strong):
    magnitude = abs(float(delta))
    if magnitude >= strong:
        return "strong"
    if magnitude >= moderate:
        return "moderate"
    return "small"


def assess_historical_change(
    before,
    after,
    model_answer=None,
    changeformer_change_pct=None,
    cross_sensor=False,
):
    before_stats = getattr(before, "stats", {}) or {}
    after_stats = getattr(after, "stats", {}) or {}

    before_year = getattr(before, "year", None)
    after_year = getattr(after, "year", None)

    b_ndvi = _satquery_metric_value(before_stats, "ndvi")
    a_ndvi = _satquery_metric_value(after_stats, "ndvi")
    b_ndbi = _satquery_metric_value(before_stats, "ndbi")
    a_ndbi = _satquery_metric_value(after_stats, "ndbi")
    b_water = _satquery_metric_value(before_stats, "water_fraction")
    a_water = _satquery_metric_value(after_stats, "water_fraction")

    d_ndvi = (
        a_ndvi - b_ndvi
        if b_ndvi is not None and a_ndvi is not None
        else None
    )
    d_ndbi = (
        a_ndbi - b_ndbi
        if b_ndbi is not None and a_ndbi is not None
        else None
    )
    d_water = (
        a_water - b_water
        if b_water is not None and a_water is not None
        else None
    )

    signals = []

    if d_ndvi is not None:
        signals.append(
            {
                "metric": "ndvi",
                "delta": d_ndvi,
                "level": _satquery_metric_level(
                    d_ndvi,
                    moderate=0.04,
                    strong=0.08,
                ),
            }
        )

    if d_ndbi is not None:
        signals.append(
            {
                "metric": "ndbi",
                "delta": d_ndbi,
                "level": _satquery_metric_level(
                    d_ndbi,
                    moderate=0.025,
                    strong=0.05,
                ),
            }
        )

    if d_water is not None:
        signals.append(
            {
                "metric": "water_fraction",
                "delta": d_water,
                "level": _satquery_metric_level(
                    d_water,
                    moderate=0.02,
                    strong=0.05,
                ),
            }
        )

    strong_count = sum(
        1 for item in signals
        if item["level"] == "strong"
    )
    moderate_count = sum(
        1 for item in signals
        if item["level"] == "moderate"
    )

    if strong_count >= 2:
        spectral_level = "significant"
    elif strong_count >= 1 and moderate_count >= 1:
        spectral_level = "significant"
    elif strong_count >= 1:
        spectral_level = "meaningful"
    elif moderate_count >= 2:
        spectral_level = "meaningful"
    elif moderate_count >= 1:
        spectral_level = "limited"
    else:
        spectral_level = "stable"

    cf_pct = None
    try:
        if changeformer_change_pct is not None:
            cf_pct = max(
                0.0,
                float(changeformer_change_pct),
            )
    except Exception:
        cf_pct = None

    if cf_pct is None:
        structural_level = "unavailable"
    elif cf_pct < 0.01:
        structural_level = "very_low"
    elif cf_pct < 1.0:
        structural_level = "minor"
    elif cf_pct < 10.0:
        structural_level = "localized"
    elif cf_pct < 30.0:
        structural_level = "significant"
    else:
        structural_level = "major"

    if spectral_level == "significant":
        opening = (
            f"Historical comparison indicates significant spectral "
            f"change between {before_year} and {after_year}."
        )
    elif spectral_level == "meaningful":
        opening = (
            f"Historical comparison indicates meaningful spectral "
            f"change between {before_year} and {after_year}."
        )
    elif spectral_level == "limited":
        opening = (
            f"Historical comparison indicates limited spectral "
            f"change between {before_year} and {after_year}."
        )
    else:
        opening = (
            f"No strong spectral land-cover change signal was detected "
            f"between {before_year} and {after_year}."
        )

    details = []

    if d_ndvi is not None:
        if d_ndvi <= -0.04:
            ndvi_trend = "decreased"
        elif d_ndvi >= 0.04:
            ndvi_trend = "increased"
        else:
            ndvi_trend = "remained broadly stable"

        details.append(
            "Vegetation signal (NDVI) "
            f"{ndvi_trend} from {b_ndvi:.3f} to {a_ndvi:.3f} "
            f"({d_ndvi:+.3f})."
        )

    if d_ndbi is not None:
        if d_ndbi >= 0.025:
            ndbi_trend = "increased"
        elif d_ndbi <= -0.025:
            ndbi_trend = "decreased"
        else:
            ndbi_trend = "remained broadly stable"

        details.append(
            "Built-up spectral signal (NDBI) "
            f"{ndbi_trend} from {b_ndbi:.3f} to {a_ndbi:.3f} "
            f"({d_ndbi:+.3f})."
        )

    if d_water is not None:
        before_water_pct = b_water * 100.0
        after_water_pct = a_water * 100.0
        delta_water_pp = d_water * 100.0

        if delta_water_pp >= 2.0:
            water_trend = "increased"
        elif delta_water_pp <= -2.0:
            water_trend = "decreased"
        else:
            water_trend = "remained broadly stable"

        details.append(
            "Estimated water-covered fraction "
            f"{water_trend} from {before_water_pct:.2f}% "
            f"to {after_water_pct:.2f}% "
            f"({delta_water_pp:+.2f} percentage points)."
        )

    interpretation = []

    if (
        d_ndvi is not None
        and d_ndvi <= -0.04
        and d_ndbi is not None
        and d_ndbi >= 0.025
    ):
        interpretation.append(
            "The combined NDVI decrease and NDBI increase are "
            "consistent with increased surface development or loss "
            "of vegetated cover, but they do not by themselves prove "
            "a specific land-cover conversion."
        )

    if d_water is not None and abs(d_water) >= 0.02:
        interpretation.append(
            "The water metric also changed materially; this can reflect "
            "real shoreline/water-cover change as well as seasonal or "
            "compositing effects, so it should be interpreted with the "
            "before/after imagery."
        )

    if cf_pct is not None:
        interpretation.append(
            "ChangeFormerV6 detected approximately "
            f"{cf_pct:.2f}% LEVIR-style structural change. "
            "Because this checkpoint was trained on high-resolution "
            "LEVIR-CD building imagery rather than Sentinel/Landsat "
            "historical composites, this value is supporting structural "
            "evidence and does not override the spectral assessment."
        )
    elif model_answer:
        lower_answer = str(model_answer).lower()
        if (
            "failed" in lower_answer
            or "validation" in lower_answer
            or "error" in lower_answer
        ):
            interpretation.append(
                "ChangeFormer structural evidence was unavailable for "
                "this pair, so the historical conclusion is based on "
                "the Earth Engine spectral trends."
            )

    if cross_sensor:
        interpretation.append(
            "The dates use different sensor families, so both images "
            "were harmonized to a common comparison grid. Cross-sensor "
            "radiometric differences remain a source of uncertainty."
        )

    answer_parts = [opening]
    answer_parts.extend(details)
    answer_parts.extend(interpretation)

    return {
        "level": spectral_level,
        "spectral_change_detected": spectral_level
        in {"limited", "meaningful", "significant"},
        "structural_level": structural_level,
        "changeformer_changed_percentage": cf_pct,
        "metrics": {
            "ndvi": {
                "before": b_ndvi,
                "after": a_ndvi,
                "delta": d_ndvi,
            },
            "ndbi": {
                "before": b_ndbi,
                "after": a_ndbi,
                "delta": d_ndbi,
            },
            "water_fraction": {
                "before": b_water,
                "after": a_water,
                "delta": d_water,
            },
        },
        "answer": " ".join(
            part.strip()
            for part in answer_parts
            if part and str(part).strip()
        ),
        "method_note": (
            "Historical conclusion prioritizes Earth Engine spectral "
            "trends for medium-resolution Sentinel/Landsat composites. "
            "ChangeFormerV6 is reported separately as supporting "
            "building/structural evidence."
        ),
    }


def build_change_narrative(
    before,
    after,
    model_answer=None,
    changeformer_change_pct=None,
    cross_sensor=False,
):
    assessment = assess_historical_change(
        before=before,
        after=after,
        model_answer=model_answer,
        changeformer_change_pct=changeformer_change_pct,
        cross_sensor=cross_sensor,
    )
    return assessment["answer"]

# === END SATQUERY HISTORICAL CHANGE ASSESSMENT V2 ===
"""


def main():
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Cannot find {TARGET}"
        )

    source = TARGET.read_text(
        encoding="utf-8"
    )

    if MARKER in source:
        print(
            "Historical assessment V2 is already installed."
        )
        return

    if not BACKUP.exists():
        shutil.copy2(
            TARGET,
            BACKUP,
        )
        print(
            "Backup created:",
            BACKUP,
        )

    with TARGET.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n"
            + textwrap.dedent(PATCH)
            + "\n"
        )

    print(
        "Updated:",
        TARGET,
    )
    print(
        "Historical assessment V2 installed successfully."
    )


if __name__ == "__main__":
    main()
