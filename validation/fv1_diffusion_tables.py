"""Create measured tables for the hydraulic and computational questions."""

import json
from pathlib import Path
import numpy as np
from validation.fv1_diffusion_figures import load_rows


def table(*, columns, rows):
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append(
            "| "
            + " | ".join(f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in row)
            + " |"
        )
    return "\n".join(out)


def tables(*, directory, methods):
    data = load_rows(directory=directory)
    index = {(r["case"], r["dx"], r["method"], r["policy"]): r for r in data}
    lines = [
        "# Measured comparisons",
        "All timings are hydraulic seconds unless the column gives milliseconds. Section discharge is per unit width except in the radial-basin table.",
    ]
    times = []
    for p in ("common", "production"):
        for m in methods:
            ratios = [
                r["median_wall_s"] / index[c, dx, "fv1", p]["median_wall_s"]
                for (c, dx, method, policy), r in index.items()
                if policy == p and method == m
            ]
            times.append(
                [
                    p,
                    m,
                    len(ratios),
                    min(ratios),
                    float(np.median(ratios)),
                    max(ratios),
                    sum(i <= 1.5 for i in ratios),
                ]
            )
    lines += [
        "## Same-grid GPU cost",
        table(
            columns=[
                "Policy",
                "Method",
                "Pairs",
                "Minimum ×FV1",
                "Median ×FV1",
                "Maximum ×FV1",
                "At most 1.5×",
            ],
            rows=times,
        ),
    ]
    samples = []
    for c in (
        "front_plain",
        "front_steep",
        "gravity_pulse",
        "shear_pulse",
        "gentle_rise",
        "routing",
        "rain20",
        "damped_train",
    ):
        for m in methods:
            r = index[c, 5.0, m, "common"]
            samples.append(
                [
                    c,
                    m,
                    r["fr_wet_median"],
                    1000 * r["depth_rmse"],
                    r["extent"]["0.01"]["iou"],
                    r["hydro_rmse"],
                    r["peak_error_rmse"],
                    1000 * r["median_wall_s"],
                ]
            )
    lines += [
        "## Matched 5 m analytical fields and internal sections",
        table(
            columns=[
                "Case",
                "Method",
                "Median Fr",
                "Depth RMSE mm",
                "1 cm IoU",
                "Q RMSE m²/s",
                "Peak error RMS m²/s",
                "GPU ms",
            ],
            rows=samples,
        ),
    ]
    samples = []
    for r in json.loads((directory / "profile.json").read_text()):
        kernels = r["kernels"]
        prefixes = ("slopes", "predict", "faces", "split_sources", "euler", "limit_faces")
        groups = [
            np.median([sum(v for k, v in a.items() if k.startswith(pre)) for a in kernels])
            for pre in prefixes
        ]
        samples.append(
            [
                r["case"],
                r["mode"],
                r["method"],
                r["fused"],
                r["times"][0]["steps"],
                r["median_s"] * 1000,
                r["owned_device_bytes"] / 2**20,
                r["warm"]["graph_capture_seconds"] * 1000,
                *groups,
            ]
        )
    lines += [
        "## 400 × 240 cells, 128 accepted steps, seven warmed runs",
        "Uncaptured kernel groups are µs per accepted step. Fused prediction is included in slopes. Face kernels include their wave-speed calculation. FV1 uses different kernel names. Raw names and individual repeat values are in profile.json.",
        table(
            columns=[
                "Case",
                "Mode",
                "Method",
                "Fused",
                "Steps",
                "Wall ms",
                "Owned MiB",
                "Capture ms",
                *prefixes,
            ],
            rows=samples,
        ),
    ]
    hist = json.loads((directory / "pulse_histories.json").read_text())
    samples = []
    for r in hist:
        if r["dx"] != 5:
            continue
        diag = r["model_m2_s_diagnostics"]
        exact = r["exact_m2_s_diagnostics"]
        g = r["gauges"][1]
        samples.append(
            [
                r["case"],
                r["method"],
                g["exact_peak"],
                g["model_peak"],
                100 * (1 - g["model_peak"] / g["exact_peak"]),
                diag["peak_times_s"][1] - exact["peak_times_s"][1],
                diag["width_s"][1],
                exact["width_s"][1],
                diag["speed_m_s"],
                exact["speed_m_s"],
            ]
        )
    lines += [
        "## Transient peak at the central section, 5 m",
        table(
            columns=[
                "Case",
                "Method",
                "Exact peak",
                "Model peak",
                "Attenuation %",
                "Peak time error s",
                "Width s",
                "Exact width s",
                "Speed m/s",
                "Exact speed m/s",
            ],
            rows=samples,
        ),
    ]
    budget = json.loads((directory / "budgets.json").read_text())
    samples = []
    for r in budget:
        if (
            r.get("reference_dx") == 5
            and r["metric"] == "hydro_rmse"
            and r["case"] in ("front_plain", "front_steep", "gravity_pulse", "shear_pulse")
        ):
            samples.append(
                [
                    r["case"],
                    r["budget_s"] * 1000,
                    r["method"],
                    r["dx"],
                    r["runtime_s"] * 1000,
                    r["error"],
                    r["fv1_error"],
                ]
            )
    lines += [
        "## Affordable grids at the measured 5 m FV1 time budget",
        table(
            columns=["Case", "Budget ms", "Method", "Grid m", "Time ms", "Q RMSE", "FV1 5 m RMSE"],
            rows=samples,
        ),
    ]
    samples = []
    for f in ("radial.json", "radial_1m.json"):
        for r in json.loads((directory / f).read_text()):
            if r["dx"] in (1.0, 5.0, 10.0) and r["method"] in methods:
                g = r["gauges"][1]
                samples.append(
                    [
                        r["dx"],
                        r["method"],
                        r["depth_rmse_m"],
                        g["cycle2_peak_exact_m3_s"],
                        g["cycle2_peak_model_m3_s"],
                        g["hydrograph_rmse_m3_s"],
                        r["wet_symmetric_difference_m2"],
                        r["ledger"]["residual"],
                    ]
                )
    lines += [
        "## Circular Thacker basin after one cycle",
        "The selected gauge is 80 m from the centre. The inherited JSON fields named cycle2 refer here to the last complete cycle, which is cycle one.",
        table(
            columns=[
                "dx m",
                "Method",
                "Depth RMSE m",
                "Exact peak m³/s",
                "Model peak m³/s",
                "Q RMSE m³/s",
                "Wet symmetric difference m²",
                "Budget residual m³",
            ],
            rows=samples,
        ),
    ]
    samples = []
    for r in json.loads((directory / "shock_histories.json").read_text()):
        samples.append(
            [
                r["method"],
                r["dx"],
                r["late_spatial_q_rmse"],
                r["final_half_max_section_std"],
                r["final_half_max_departure"],
            ]
        )
    lines += [
        "## Jump discharge 40–80 s after initial projection",
        table(
            columns=[
                "Method",
                "dx m",
                "Section RMSE m²/s",
                "Max section std m²/s",
                "Max absolute error m²/s",
            ],
            rows=samples,
        ),
    ]
    (directory / "TABLES.md").write_text("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    directory = Path("validation/fv1_diffusion_review")
    methods = (
        "fv1",
        "low_froude_hll",
        "euler_mc",
        "hancock_mc",
        "hancock_adaptive",
        "hancock_low_froude",
        "mc",
        "adaptive",
        "li",
    )
    tables(directory=directory, methods=methods)
