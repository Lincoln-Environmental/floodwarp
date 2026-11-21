"""Generate the engineering decision from the integrated and archived matched runs."""

import csv
import json
from pathlib import Path

import numpy as np

from validation.hancock_long_storm_review.dataset import records
from validation.hancock_long_storm_review.results_text import table


def timing(*, runs, case, method, mode):
    return next(
        row
        for row in runs
        if row["case"] == case and row["method"] == method and row["mode"] == mode
    )


def physical_key(row):
    return (
        row["landform"],
        row["storm"],
        row["dx"],
        row["terrain_dx"],
        tuple(row["dimensions_m"]),
        row["hours"],
    )


def report(*, review, threshold):
    runs = records(review=review)
    controls = {physical_key(r): r for r in runs if r["method"] == "fv1"}
    pairs = []
    for r in runs:
        if r["mode"] != "tiles" or r["group"] not in (
            "primary",
            "large",
            "terrain_large",
            "moving",
        ):
            continue
        if r["timing_dataset"] != "integrated_events":
            raise ValueError("Incomplete verification for " + r["name"])
        f = controls[physical_key(r)]
        pairs.append((r, f, r["hydraulic_s"] / f["hydraulic_s"]))
    primary = [(r, f, t) for r, f, t in pairs if r["group"] == "primary"]
    larger = [(r, f, t) for r, f, t in pairs if r["group"] != "primary"]
    if len(primary) != 48 or len(larger) != 11:
        raise ValueError("Incomplete matched matrix")
    baseline = json.loads((review / "baseline.json").read_text())
    final = json.loads((review / "integrated_profile.json").read_text())
    lines = [
        "# Engineering decision for adaptive Hancock",
        "All wall times below use warmed, synchronised FP64 execution on the RTX 4070 Ti SUPER. The original development head was `0180327`. The tiled scheduler is available at `2914077` and the constant-capacity source operator at `e63c936`. The original timings and physical arrays remain available for comparison.",
        "## Fixed-timestep cost",
    ]
    short = []
    for case in ("stoker", "dry_patch"):
        values = []
        for source, method, mode in (
            (baseline, "contact_hll", "sparse"),
            (baseline, "hancock_adaptive", "dense"),
            (final, "contact_hll", "sparse"),
            (final, "hancock_adaptive", "dense"),
            (final, "hancock_adaptive", "tiles"),
        ):
            values.append(
                timing(runs=source, case=case, method=method, mode=mode)["median_s"]
                * 1000
            )
        short.append(
            [case, *[f"{v:.2f}" for v in values], f"{values[-1] / values[2]:.2f}×"]
        )
    lines.append(
        table(
            headers=[
                "Case",
                "Baseline FV1 ms",
                "Baseline dense ms",
                "Integrated FV1 ms",
                "Integrated dense ms",
                "Integrated tiles ms",
                "Tiles/FV1",
            ],
            rows=short,
        )
    )
    lines += [
        "## Production timestep cost",
        "Primary events cover eight landforms and six rainfall patterns on a common 384-cell grid. Each ran through six hours of rainfall and two hours of recession. The moderate plain ran for 24 hours.",
    ]
    ratios = np.array([t for _, _, t in primary])
    worst = sorted(primary, key=ratio)[-1]
    lines.append(
        f"The median tiled Hancock/FV1 runtime ratio was {np.median(ratios):.3f}×. {int((ratios <= threshold).sum())} of 48 pairs met {threshold:.1f}×. The largest ratio was {worst[2]:.3f}× for {worst[0]['landform']} terrain under {worst[0]['storm']} rainfall. It required {worst[0]['steps']:,} accepted steps versus {worst[1]['steps']:,} for FV1."
    )
    grouped = []
    for land in sorted({r["landform"] for r, _, _ in primary}):
        subset = [(r, f, t) for r, f, t in primary if r["landform"] == land]
        hi = [
            s["wet_froude_max"]
            for r, _, _ in subset
            for s in r["samples"]
            if s["wet_froude_max"] is not None
        ]
        grouped.append(
            [
                land,
                f"{np.median([t for _, _, t in subset]):.3f}",
                f"{np.median([r['steps'] / f['steps'] for r, f, _ in subset]):.3f}",
                f"{max(hi):.2f}",
            ]
        )
    lines.append(
        table(
            headers=[
                "Landform",
                "Median tiles/FV1 wall",
                "Median tiles/FV1 steps",
                "Largest sampled cell Froude",
            ],
            rows=grouped,
        )
    )
    detail = []
    for r, f, t in sorted(larger, key=pair_order):
        dense = next(
            x
            for x in runs
            if x["method"] == "hancock"
            and x["mode"] == "dense"
            and physical_key(x) == physical_key(r)
        )
        samples = r["samples"]
        detail.append(
            [
                r["landform"],
                r["storm"],
                f"{np.prod(r['dimensions_m']) / r['dx'] ** 2:,.0f}",
                f"{np.mean([s['wet_fraction'] for s in samples]) * 100:.1f}",
                f"{np.mean([s['active_fraction'] for s in samples]) * 100:.1f}",
                f"{f['hydraulic_s']:.2f}",
                f"{dense['hydraulic_s']:.2f}",
                f"{r['hydraulic_s']:.2f}",
                f"{t:.3f}",
            ]
        )
    lines.append(
        table(
            headers=[
                "Landform",
                "Rain",
                "Cells",
                "Wet %",
                "Scheduled %",
                "FV1 s",
                "Dense s",
                "Tiles s",
                "Tiles/FV1",
            ],
            rows=detail,
        )
    )
    with (review / "reference_uncertainty.csv").open() as stream:
        refinements = list(csv.DictReader(stream))
    good = sum(
        r["representation"] == "sampled" and r["both_resolved"] == "True"
        for r in refinements
    )
    verified = json.loads((review / "provenance.json").read_text())
    lines += [
        "## Hydraulic accuracy",
        f"The source specialisation matched the stored hydraulic outputs in {sum(s['count'] for s in verified['integrated_operator_checks'].values())} replays. The dense and sparse schedules agreed in all 96 small primary and 20 large pairs. All six independent accepted-timestep histories matched across dense, fixed-mask and tiles. Of the 48 finely resampled terrain/event references, {good} met both the depth and hydrograph refinement screens. The landform and 1, 5 and 10 m tables report errors and identify unresolved comparisons. [Accuracy and resolution tables](COMPARISONS.md)",
        "## Recommendation",
        "Keep adaptive Hancock as the development default. Select tiled execution explicitly for spatially restricted events on larger domains. Widespread rainfall, fine-grid outlet storage and shallow moving fronts can each move the full-event runtime above the target. FV1 remains appropriate where its errors meet the application tolerance. LI requires independent qualification of its reduced momentum and outlet boundary for the intended observable. The CPU and CUDA structure tests qualify quasi-steady links, with a first-order time split and a separate structure-volume ledger.",
        "A single-GPU strip crossover and one expanding footprint support an experimental switching policy. Additional GPU architectures, independent downstream controls and broader accuracy qualification are needed before changing public release or scheduling defaults. [Architecture, hydraulic mechanisms and limitations](REPORT.md) · [Complete operational metrics](event_summary.csv) · [Physical input and test provenance](provenance.json)",
    ]
    (review / "DECISION.md").write_text("\n\n".join(lines) + "\n")


def ratio(pair):
    return pair[2]


def pair_order(pair):
    row = pair[0]
    return row["landform"], row["storm"], np.prod(row["dimensions_m"])


if __name__ == "__main__":
    review = Path(__file__).parent
    preferred_ratio = 1.5
    report(review=review, threshold=preferred_ratio)
