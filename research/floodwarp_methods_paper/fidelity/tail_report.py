"""Read-only reporting of the longer million-cell storage sensitivity experiment."""

import json
from pathlib import Path
import numpy as np
from research.floodwarp_methods_paper import metrics
from research.floodwarp_methods_paper.fidelity import (
    endpoints,
    receipts,
    recession_figures,
)


def generate(*, directory):
    folder = directory / "storage_tail"
    checks = list(folder.glob("*.json"))
    if len(checks) != 1:
        raise ValueError("Select a single completed storage sensitivity design")
    study = json.loads(checks[0].read_text())
    case = study["options"]["case"]
    end = study["common_end_s"]
    values = [
        r
        for r in metrics.csv_read(folder / "late_export.csv")
        if r["station"] == "outlet" and float(r["end_h"]) * 3600 == end
    ]
    rows = []
    items = []
    for v in values:
        ident = v["id"]
        data = receipts.read(root=directory / "extended_runs", ident=ident, end=end)
        path = endpoints.endpoint_path(directory=directory, ident=ident, end_s=end)
        with np.load(path) as fields:
            final = json.loads(fields["receipt"].item())
        rows.append(
            dict(
                method=v["method"],
                id=ident,
                end_h=end / 3600,
                hydraulic_s=data["hydraulic_s"],
                steps=sum(r["steps"] for r in data["frames"]),
                early_h=v["first_h"],
                early_outward_m3=v["first_outward_m3"],
                outward_m3=v["final_outward_m3"],
                additional_export_m3=float(v["final_outward_m3"])
                - float(v["first_outward_m3"]),
                remaining_storage_m3=final["storage_m3"],
                infiltration_m3=final["infiltration_m3"],
                last_window_pct=100 * float(v["late_fraction"]),
                signed_residual_m3=final["residual_m3"],
            )
        )
        items.append(
            dict(
                id=ident,
                factor=data["factor"],
                s=dict(data["spec"], hours=end / 3600),
                d=dict(
                    data, spatial=[r for r in data["frames"] if "wet_fraction" in r]
                ),
            )
        )
    baseline = next(r for r in rows if r["method"] == "fv1")
    for r in rows:
        r["runtime_over_fv1"] = r["hydraulic_s"] / baseline["hydraulic_s"]
    metrics.csv_write(file=folder / "runtime_and_storage.csv", rows=rows)
    title = f"# Residual storage after {end / 3600:g} hours"
    lines = [
        title,
        "",
        f"The 10 m mixed catchment received six hours of spatial rainfall. Both million-cell models first passed the two-hour transport check at {study['original_end_s'] / 3600:g} hours. The extended integrations below also passed the {study['options']['tail_window_hours']}-hour check at the internal section and outlet.",
        "",
        "| Method | Hydraulic time (s) | Accepted steps | Export since first check (m³) | Final outward volume (m³) | Last 12 h share | Infiltration (m³) | Remaining storage (m³) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = "FV1 sparse" if r["method"] == "fv1" else "Hancock tiled"
        lines.append(
            f"| {label} | {r['hydraulic_s']:.2f} | {r['steps']:,} | {r['additional_export_m3']:,.1f} | {float(r['outward_m3']):,.1f} | {r['last_window_pct']:.3f}% | {r['infiltration_m3']:,.1f} | {r['remaining_storage_m3']:,.1f} |"
        )
    h = next(r for r in rows if r["method"] == "hancock_adaptive")
    lines.extend(
        [
            "",
            f"Tiled Hancock took {h['runtime_over_fv1']:.3f}× FV1 at the longer endpoint. The two forecasts differ in both infiltration and retained surface storage. A difference in total export alone cannot establish numerical diffusion.",
            "",
            f"[Complete rain, discharge, wet fraction and cost curves](figures/{case}_cost.pdf) · [Water accounts and timings](runtime_and_storage.csv) · [Two-station tail test](late_export.csv)",
        ]
    )
    (folder / "README.md").write_text("\n".join(lines) + "\n")
    recession_figures.operational(
        groups={("complete-expanded", case, "sampled"): items},
        directory=folder,
        cases=[(case, study["options"]["dx"])],
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    generate(directory=directory)
