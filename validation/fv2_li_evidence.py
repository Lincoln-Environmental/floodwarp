"""Render the matched local-inertia and full-momentum analytical results."""

from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from validation.fv1_riemann import report_json
from validation.fv2_figures import rows_table
from validation.fv2_local_inertia import build, domain, read_state


def profiles(*, directory, cases, dx, methods, duration, gravity, roughness, length, width, device, cap):
    fig, axes = plt.subplots(1, len(cases), figsize=(13, 4), layout="constrained")
    colors = {"li": "#b65a14", "contact_hll": "#2166ac", "mc": "#188754"}
    records = []
    for case, ax in zip(cases, axes, strict=True):
        d = domain(
            case=case,
            dx=dx,
            duration=duration,
            gravity=gravity,
            near_zero_n=roughness,
            length=length,
            width=width,
        )
        mid = d["x"].shape[0] // 2
        index = 2 if case == "damped_contact" else 0
        x = d["x"][mid]
        mask = d["window"][mid]
        exact = {name: ref[mid, :, index][mask] for name, ref in d["refs"].items()}
        for name, line in exact.items():
            ax.plot(
                x[mask],
                line,
                "--" if name == "swe" else ":",
                c="black",
                lw=2,
                label=f"Exact {name.upper()}",
                zorder=4,
            )
        values = {}
        for method in methods:
            solver = build(d=d, method=method, dx=dx, cap=cap, device=device, gravity=gravity)
            solver.run_device_loop(duration=duration)
            state = read_state(solver=solver, li=method == "li")
            v = state[mid, :, index][mask]
            values[method] = v.tolist()
            ax.plot(
                x[mask],
                v,
                c=colors[method],
                lw=1.5,
                label={"li": "FloodWarp LI", "contact_hll": "FV1 contact-HLL", "mc": "FV2 MC"}[method],
            )
        records.append(
            dict(
                case=case,
                dx=dx,
                x_m=x[mask].tolist(),
                quantity="hv_m2_s" if index == 2 else "depth_m",
                reference={k: v.tolist() for k, v in exact.items()},
                model=values,
            )
        )
        ax.set(
            title=case.replace("_", " ").capitalize(),
            xlabel="Position (m)",
            ylabel="Transverse momentum (m²/s)" if index == 2 else "Depth (m)",
        )
        ax.grid(alpha=0.2)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "local_inertia_profiles.png", dpi=175)
    plt.close(fig)
    report_json(path=directory / "local_inertia_profiles.json", content=records)


def cost_figure(*, primary, grids, directory):
    cases = ("stoker", "ritter", "damped_contact")
    methods = ("li", "contact_hll", "roe", "hllc", "minmod", "mc")
    colors = ("#94501c", "#1e5395", "#813f87", "#427880", "#758422", "#15804e")
    styles = ("-", "-", "--", ":", "--", "-")
    fig, axes = plt.subplots(2, len(cases), figsize=(13, 7), layout="constrained")
    for j, case in enumerate(cases):
        for method, color, style in zip(methods, colors, styles, strict=True):
            records = [primary[case, dx, method] for dx in grids]
            err = [
                r["errors"]["swe"]["momentum_rmse_m2_s"][1]
                if case == "damped_contact"
                else r["errors"]["swe"]["depth_rmse_m"]
                for r in records
            ]
            cost = [1000 * np.median(r["warm_wall_s"]) for r in records]
            for ax, x in ((axes[0, j], grids), (axes[1, j], cost)):
                ax.loglog(x, err, marker="o", lw=1.3, ms=4, color=color, linestyle=style, label=method)
        for ax in axes[:, j]:
            ax.grid(alpha=0.2, which="both")
            ax.set_ylabel("Transverse momentum RMSE (m²/s)" if case == "damped_contact" else "Depth RMSE (m)")
        axes[0, j].set(title=case.replace("_", " ").capitalize(), xlabel="Grid spacing (m)")
        axes[1, j].set_xlabel("Median GPU hydraulic wall time (ms)")
    axes[1, -1].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(directory / "li_accuracy_cost.png", dpi=175)
    plt.close(fig)


def publish(*, directory, methods, grids, dx):
    rows = json.loads((directory / "local_inertia.json").read_text())
    if len(rows) != 312:
        raise ValueError("Incomplete matched analytical study")
    primary = {
        (r["case"], r["dx"], r["method"]): r
        for r in rows
        if r["baseline_dt_cap_s"] == r["dt_cap_s"] and r["near_zero_manning"] != 1e-5
    }
    if len(primary) != len(methods) * len(grids) * 8:
        raise ValueError("Incomplete primary cases")
    for r in primary.values():
        if len(r["warm_loop_s"]) != 5:
            raise ValueError("Missing timing replicates")
    names = (
        "lake",
        "accelerating_plane",
        "thin_rain",
        "deep_rain",
        "weak_dam",
        "stoker",
        "ritter",
        "damped_contact",
    )

    def score(r):
        if r["case"] == "damped_contact":
            return r["errors"]["swe"]["momentum_rmse_m2_s"][1]
        if r["case"] in ("accelerating_plane", "thin_rain", "deep_rain"):
            return abs(r["gauge_volume_per_width_m2"] - r["exact"]["gauge_volume_per_width_m2"])
        return r["errors"]["swe"]["depth_rmse_m"]

    metrics = []
    costs = []
    refinement = []
    controls = []
    for case in names:
        metric = (
            "transport error (m²)"
            if case in ("accelerating_plane", "thin_rain", "deep_rain")
            else ("hv RMSE (m²/s)" if case == "damped_contact" else "h RMSE (m)")
        )
        metrics.append([case, metric, *[score(primary[case, dx, m]) for m in methods]])
        for m in methods:
            r = primary[case, dx, m]
            other = next(
                o
                for o in rows
                if o["case"] == case
                and o["dx"] == dx
                and o["method"] == m
                and o["dt_cap_s"] == 0.5 * r["dt_cap_s"]
            )
            finite = next(
                (
                    o
                    for o in rows
                    if o["case"] == case
                    and o["dx"] == dx
                    and o["method"] == m
                    and o["near_zero_manning"] == 1e-5
                ),
                None,
            )
            controls.append(
                [
                    case,
                    m,
                    metric,
                    score(r),
                    score(other) - score(r),
                    score(finite) - score(r) if finite else "—",
                ]
            )
        if case == "lake":
            continue
        times = [1000 * np.median(primary[case, dx, m]["warm_wall_s"]) for m in methods]
        baseline = primary[case, dx, "contact_hll"]
        affordable = [
            primary[case, d, "mc"]
            for d in grids
            if np.median(primary[case, d, "mc"]["warm_wall_s"]) <= np.median(baseline["warm_wall_s"])
        ]
        best = min(affordable, key=score) if affordable else None
        costs.append(
            [
                case,
                metric,
                score(baseline),
                times[1],
                best["dx"] if best else "—",
                score(best) if best else "—",
                1000 * np.median(best["warm_wall_s"]) if best else "—",
                times[-1] / times[1],
            ]
        )
        if case in ("weak_dam", "stoker", "ritter"):
            for ref in ("swe", "li"):
                refinement.append(
                    [case, ref, *[primary[case, d, "li"]["errors"][ref]["depth_rmse_m"] for d in grids]]
                )
    cost_figure(primary=primary, grids=grids, directory=directory)
    measured = rows_table(
        title="Accuracy against full-momentum analytical solutions at 5 m",
        columns=["Case", "Measure", "LI", "FV1 contact", "FV1 Roe", "FV1 HLLC", "FV2 minmod", "FV2 MC"],
        rows=metrics,
        note="Transport is the absolute error in cumulative signed discharge per unit width over 24 seconds at x = 0. LI cell-centred momenta are averages of the neighbouring staggered discharges. Primary measurements use a common maximum timestep of Δx/64 and near-inviscid n = 10⁻⁷ where needed.",
    )
    own = rows_table(
        title="Does FloodWarp LI converge to the exact local-inertia dam break?",
        columns=["Case", "Reference PDE", *[f"{d:g} m" for d in grids]],
        rows=refinement,
        note="Entries are LI depth RMSE in metres. Distinct full-SWE and reduced-system solutions use the same left/right depths. At strong shocks LI also has substantial numerical error against its own exact solution on these grids. Four grids do not establish convergence there.",
    )
    budget = rows_table(
        title="Measured error within the 5 m contact-HLL runtime budget",
        columns=[
            "Case",
            "Measure",
            "5 m FV1 error",
            "5 m FV1 (ms)",
            "Best affordable FV2 MC grid (m)",
            "FV2 error",
            "FV2 (ms)",
            "5 m FV2/FV1 cost",
        ],
        rows=costs,
        note="Five warmed synchronised wall times on one RTX 4070 Ti SUPER. Each times the same 24 s hydraulic simulation. Costs include work performed by the public run call and exclude setup, reference evaluation and retrieving diagnostic fields. Selection uses the measured grids only. The JSON separately gives device-loop times, graph capture, timesteps and full wall-time replicates for every method.",
    )
    sensitivity = rows_table(
        title="Timestep and friction sensitivity at 5 m",
        columns=["Case", "Method", "Measure", "Primary error", "Change with half dt", "Change with n = 10⁻⁵"],
        rows=controls,
        note="Signed changes in the scalar errors above. Friction sensitivity applies only to the three inviscid-limit Riemann problems. The independent positive-friction tests require no such extrapolation.",
    )
    archive = json.loads((directory / "hirds48/archived_li.json").read_text())
    event = rows_table(
        title="Archived Orton outlet response over 48 hours",
        columns=[
            "Method",
            "Peak (m³/s)",
            "Peak time",
            "Export (m³)",
            "RMSE from 20 m FV2 (m³/s)",
            "Correlation with 20 m FV2",
        ],
        rows=[
            [
                r["name"],
                r["peak_m3_s"],
                f"{r['peak_end_s'] // 3600} h {(r['peak_end_s'] % 3600) // 60} min",
                r["export_m3"],
                r["vs_fv2_hydrograph_rmse_m3_s"],
                r["vs_fv2_correlation"],
            ]
            for r in archive["hydrographs"]
        ],
        note="Existing 5 m FloodWarp LI and FV1 records have identical physical input signatures. All four complete records use the 578-node HIRDS storm. Metrics use one-minute gross outward volume differences. Only the 20 m corrected FV2 event completed. No matching 20 m LI event was established.",
    )
    intro = """# Does FloodWarp FV2 respond like FloodWarp local inertia?

**The archived outlet hydrographs resemble each other. The momentum-transport tests distinguish the solvers.** FloodWarp LI at 5 m peaked one minute before corrected 20 m FV2. The full-event hydrograph RMSE is smaller against 5 m LI than against either FV1 control. Spatial agreement depends on the quantity. At hour 48, maximum-depth-envelope RMSE from FV2 was 0.278 m for 5 m LI, 0.113 m for 5 m FV1 and 0.035 m for 20 m FV1, on the original 5 m active footprint. The 20 m and 5 m models represent different channels, masks and coastal openings. These archived simulations provide no observed outlet or native-grid reference.

![Complete and peak-window outlet hydrographs](hirds48/archived_li.png)

"""
    equations = """## Which tests separate the two momentum equations?

The comparison runs FloodWarp's public coarse-geometry LI solver with its resolved `physical_candidate` profile. FV1 and FV2 retain the divergence of the advective momentum flux. LI evolves the surface pressure gradient, local acceleration and Manning drag with staggered face discharge. The runs use FP64 on the same hardware, g = 9.80665 m/s², common 0.1 mm dry depth, the same physical domain and inputs, and reflecting outer boundaries. Errors are sampled in the interior half of each dimension, clear of the physical boundary waves. All 312 numerical configurations produced nonnegative final depths. For discharge errors in smooth shared solutions the actual face transport is used directly.

* **Positive-friction shear contact** isolates the omitted term with different exact predictions for the two equations. Start at constant h with u uniform and v switching between opposite values of equal magnitude. Both components decay by D = 1 + kt, k = gn²|u₀,v₀|/h^(4/3). SWE advects the contact to x = u₀ log(D)/k. The LI contact remains at its initial location. At 5 m after 24 s, exact SWE gives 32.22 m, FV2 MC gives 32.62 m, FV1 contact gives 31.46 m and FloodWarp LI gives −0.0004 m. LI's transverse-momentum RMSE against its own solution was 0.000033 m²/s. Against SWE it was 0.286 m²/s.
* **A partially dry lake and an accelerating uniform Manning plane** test the equations where both have the same exact solution. On the plane u = U tanh(gSt/U), U = h^(2/3)√S/n. The cumulative transport is the exact integral of hu. The lake tests equilibrium and conservation without advection.
* **Rain-fed slopes at 30 and 150 mm depth** test moving water with spatial rain, positive Manning friction and nonuniform momentum. The full-SWE reference comes from prescribing h and solving the exact steady momentum balance for z. The reduced momentum equation omits the gradient of q²/h, so the SWE target quantifies the combined effect of that approximation and LI discretisation.
* **Weak, Stoker and Ritter dam breaks** compare LI against both exact Riemann solutions. For flat inviscid LI the conservative flux is (q, gh²/2). Its left rarefaction gives q★ = 2√g(hL^(3/2) − h★^(3/2))/3. Its right shock gives q★ = (h★ − hR)√[g(h★ + hR)/2]. Their intersection sets h★, including for a dry downstream bed. Both models use analytical cell averages split exactly at waves. FloodWarp LI requires positive roughness. Every method runs with identical small positive roughness at two values, and the table below quantifies sensitivity to this inviscid limit. The dam-break references have zero friction. [Cozzolino et al., reduced-system Riemann solutions](https://doi.org/10.1016/j.advwatres.2019.103422) · [Delestre et al., analytical shallow-water benchmarks](https://doi.org/10.1002/fld.3741)

![Analytical and numerical centreline profiles](local_inertia_profiles.png)

![Analytical error as a function of spacing and GPU time](li_accuracy_cost.png)

"""
    (directory / "LOCAL_INERTIA.md").write_text(
        intro + event + "\n" + equations + measured + "\n" + own + "\n" + budget + "\n" + sensitivity
    )
    files = (
        "local_inertia.json",
        "local_inertia_profiles.json",
        "local_inertia_profiles.png",
        "li_accuracy_cost.png",
        "LOCAL_INERTIA.md",
        "hirds48/archived_li.json",
        "hirds48/archived_li_fields.json",
        "hirds48/archived_li.png",
        "flow_truth.json",
        "rain_flow.json",
        "radial_flow.json",
        "flow_reference_checks.json",
        "FLOW_TRUTH.md",
        "analytical_hydrographs.png",
    )
    checks = dict(
        configurations=len(rows),
        primary_configurations=len(primary),
        timed_repeats=sum(len(r["warm_wall_s"]) for r in rows),
        minimum_depth_m=min(r["domain_min_depth_m"] for r in rows),
        largest_relative_budget_residual=max(
            abs(r["budget_residual_m3"]) / r["initial_volume_m3"] for r in rows
        ),
        sha256={f: hashlib.sha256((directory / f).read_bytes()).hexdigest() for f in files},
    )
    sources = [
        Path("validation") / n
        for n in (
            "fv2_local_inertia.py",
            "fv2_li_evidence.py",
            "fv2_archived_li.py",
            "fv2_flow_truth.py",
            "fv2_radial_flow.py",
            "fv2_rain_flow.py",
            "fv2_flow_evidence.py",
            "fv2_accuracy.py",
            "fv1_riemann_cases.py",
            "fv1_riemann.py",
        )
    ]
    sources.extend(
        Path("src/floodwarp") / name
        for name in (
            "solver_factory.py",
            "fv2_solver.py",
            "fv2_kernels.py",
            "fv2_reference.py",
            "full_swe_solver.py",
            "full_swe_kernels.py",
            "local_inertial_solver.py",
            "local_inertial_kernels.py",
            "validation_profiles.py",
        )
    )
    checks["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report_json(path=directory / "analytical_evidence.json", content=checks)


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    selected = ("damped_contact", "stoker", "ritter")
    profiles(
        directory=directory,
        cases=selected,
        dx=5.0,
        methods=("li", "contact_hll", "mc"),
        duration=24.0,
        gravity=9.80665,
        roughness=1e-7,
        length=960.0,
        width=640.0,
        device="cuda:0",
        cap=5 / 64,
    )
    publish(
        directory=directory,
        methods=("li", "contact_hll", "roe", "hllc", "minmod", "mc"),
        grids=(20.0, 10.0, 5.0, 2.5),
        dx=5.0,
    )
