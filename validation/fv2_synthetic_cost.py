"""Replicated error versus runtime for the same seven synthetic hydraulic surfaces."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_accuracy import configured
from validation.fv2_cost import reset, timed
from validation.fv2_terrain import open_boundaries, terrain


def run(*, output, input_file, cases, spacings, device, samples, repeats, length, width):
    references = json.loads(input_file.read_text())
    rows = []
    for case in cases:
        for dx in spacings:
            d = terrain(name=case, dx=dx, length=length, width=width)
            for method in ("fv1", "mc"):
                s = configured(domain=d, method=method, cap=dx / 32, device=device)
                open_boundaries(solver=s, active=s._active, north_only=False)
                start = s.capture_state()
                warm = timed(solver=s, duration=d.duration, samples=samples)
                reps = []
                for _ in range(repeats):
                    reset(solver=s, state=start)
                    reps.append(timed(solver=s, duration=d.duration, samples=samples))
                control = next(
                    r
                    for r in references
                    if r["case"] == case
                    and r["dx"] == dx
                    and r["method"] == method
                    and not r.get("terrain_fixed_5m")
                )
                np.testing.assert_allclose(
                    s.get_water_depth().max(),
                    control["fine_comparisons"]["mc"]["depth_max_m"],
                    rtol=1.0e-13,
                    atol=1.0e-13,
                )
                gap = next(
                    r for r in references if r["case"] == case and r["dx"] == 1.25 and r["method"] == "mc"
                )
                rows.append(
                    dict(
                        case=case,
                        dx=dx,
                        method=method,
                        shape=list(d.bed.shape),
                        duration_s=d.duration,
                        warm=warm,
                        replicates=reps,
                        median_s=float(np.median([r["full_hydraulic_s"] for r in reps])),
                        metrics=control["fine_comparisons"],
                        last_mc_refinement_gap_m=gap["fine_comparisons"]["mc"]["depth_rmse_m"],
                    )
                )
        report_json(path=output, content=rows)
    return rows


def make_plot(*, rows, cases, output):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        for method, color in (("fv1", "#2166ac"), ("mc", "#11865a")):
            r = [v for v in rows if v["method"] == method and v["case"] == case]
            x = [v["median_s"] * 1000 for v in r]
            y = [1000 * v["metrics"]["mc"]["depth_rmse_m"] for v in r]
            ax.loglog(x, y, "o-", label=method, color=color, ms=4)
            for xx, yy, v in zip(x, y, r, strict=True):
                ax.annotate(f"{v['dx']:g}", (xx, yy), xytext=(2, 3), textcoords="offset points", fontsize=7)
        ax.axhline(1000 * r[0]["last_mc_refinement_gap_m"], ls=":", color="grey", label="MC refinement gap")
        ax.set(
            title=case.replace("_", " ").title(),
            xlabel="Median hydraulic time (ms)",
            ylabel="Depth RMSE (mm)",
        )
        ax.grid(alpha=0.15, which="both")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, loc="center")
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    cases = ("plane", "irregular", "steep", "expansion", "rainfall", "barrier", "oblique_channel")
    spacings = (20.0, 10.0, 5.0, 2.5)
    results = run(
        output=directory / "synthetic_cost.json",
        input_file=directory / "synthetic.json",
        cases=cases,
        spacings=spacings,
        device="cuda:0",
        samples=4,
        repeats=5,
        length=320.0,
        width=160.0,
    )
    make_plot(rows=results, cases=cases, output=directory / "synthetic_cost.png")
