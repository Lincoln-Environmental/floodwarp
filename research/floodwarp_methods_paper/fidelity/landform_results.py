"""Matched error distributions within each small-domain landform family."""

from collections import defaultdict
import numpy as np
from research.floodwarp_methods_paper import metrics


def collect(*, evidence, target, methods):
    groups = defaultdict(dict)
    for r in evidence:
        if r["cohort"] == "complete-legacy" and r["method"] in methods:
            groups[r["landform"], r["metric"], r["case"]][r["method"]] = r
    summaries = []
    for family, metric in sorted({(k[0], k[1]) for k in groups}):
        cases = [v for (f, m, _), v in groups.items() if (f, m) == (family, metric)]
        matched = [
            v
            for v in cases
            if all(name in v and v[name]["qualified"] for name in methods)
        ]
        for name in methods:
            points = [v[name] for v in matched]
            a = np.array([r["value_pct"] for r in points])
            summaries.append(
                dict(
                    landform=family,
                    metric=metric,
                    method=name,
                    candidate_events=len(cases),
                    paired_events=len(matched),
                    paired_terrains=len({r["seed"] for r in points}),
                    median_signed_pct=np.median(a) if a.size else None,
                    q25_pct=np.percentile(a, 25) if a.size else None,
                    q75_pct=np.percentile(a, 75) if a.size else None,
                    median_abs_pct=np.median(abs(a)) if a.size else None,
                    p90_abs_pct=np.percentile(abs(a), 90) if a.size else None,
                    median_cost_over_fv1=np.median(
                        [
                            v[name]["hydraulic_s"] / v["fv1"]["hydraulic_s"]
                            for v in matched
                        ]
                    )
                    if matched
                    else None,
                )
            )
    metrics.csv_write(file=target / "landform_fidelity.csv", rows=summaries)
    return summaries


def table(*, summaries, metric, methods):
    index = {
        (r["landform"], r["method"]): r for r in summaries if r["metric"] == metric
    }
    lines = [
        "| Landform | Matched / total storms | Terrain seeds in matched storms | FV1 | Hancock dense | FV2 RK2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in sorted({key[0] for key in index}):
        row = index[family, methods[0]]
        cells = []
        for name in methods:
            value = index[family, name]["median_abs_pct"]
            cells.append(f"{value:.2f}%" if value is not None else "unresolved")
        lines.append(
            f"| {family.capitalize()} | {row['paired_events']}/{row['candidate_events']} | {row['paired_terrains']} | "
            + " | ".join(cells)
            + " |"
        )
    return lines
