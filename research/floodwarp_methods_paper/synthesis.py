"""Results and remaining qualifications derived from hydraulic observations."""

from collections import Counter
from research.floodwarp_methods_paper.metrics import csv_read, OBJECTIVES


def case_evidence(*, study):
    p = study / "processed"
    rows = csv_read(p / "metrics/analytical.csv")
    lines = [
        "## Formulation at matched resolution",
        "",
        "The steep moving front and the smooth gravity pulse illustrate different limiter and timestep regimes. All values below are exact-solution RMSE at 5 m.",
        "",
        "| Case | Method | Depth (mm) | Face discharge (m²/s) | Accepted steps |",
        "|---|---|---:|---:|---:|",
    ]
    for c in ("front_steep", "gravity_pulse"):
        for m in ("fv1", "hancock_adaptive", "adaptive"):
            r = next(
                r
                for r in rows
                if (r["case"], r["stage"], r["method"], float(r["dx"]))
                == (c, "analytical", m, 5.0)
            )
            lines.append(
                f"| {c} | {m} | {float(r['depth_rmse_m']) * 1000:.4g} | {float(r['hydrograph_rmse_m2s']):.4g} | {r['steps']} |"
            )
    raw = csv_read(p / "metrics/archival_reduced_equations.csv")
    li = {
        r["reference_equations"]: r
        for r in raw
        if r["method"] == "li" and r["case"] == "weak_dam" and float(r["dx"]) == 1
    }
    lines += [
        "",
        f"On the archived 1 m weak dam, LI's depth RMSE was {float(li['li']['depth_rmse_m']) * 1000:.4g} mm against its own momentum equations and {float(li['swe']['depth_rmse_m']) * 1000:.4g} mm against full SWE. Each equation uses its own independent Riemann reference. [Equation-specific errors](../processed/metrics/archival_reduced_equations.csv)",
        "",
        "## Resolved differences on storm terrain",
        "",
        "A pairwise difference is labelled resolved here when its magnitude exceeds twice the observed reference-refinement change and the objective passes its individual refinement screen. The margin uses the observed change as an uncertainty estimate. It does not establish a rigorous bound on continuum error.",
        "",
        "| Observable, 10 m | Comparison method | Hancock lower error | Alternative lower error | Inside margin | Reference unresolved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    pairs = csv_read(p / "comparisons/reference_aware_pairs.csv")
    for obj in ("depth", "discharge", "inundation"):
        for m in ("fv1", "adaptive", "li"):
            counts = Counter(
                r["decision"]
                for r in pairs
                if r["objective"] == obj
                and r["formulation"] == m
                and float(r["dx"]) == 10
                and r["representation"] == "sampled"
            )
            lines.append(
                f"| {obj} | {m} | {counts['Hancock']} | {counts[m]} | {counts['within refinement difference']} | {counts['unresolved reference']} |"
            )
    lines += [
        "",
        "The fine-grid cross-formulation file reports each available Hancock–FV2 difference on identical 1 m geometry. The per-family file gives the operational timing and timestep ratios for all eight landforms. [Fine-grid comparisons](../processed/metrics/independent_fine_formulation.csv) · [Landform results](../processed/comparisons/landform_results.csv)",
    ]
    return "\n".join(lines) + "\n"


def limitations(*, study):
    p = study / "processed"
    ref = csv_read(p / "metrics/reference_uncertainty.csv")
    lines = [
        "# Unresolved and unqualified investigations",
        "",
        "A refinement screen applies to a physical event, terrain representation and hydraulic observable. The complete per-event record retains both grid identities and the magnitude of the change.",
        "",
        "| Terrain representation | Observable | Qualified | Unresolved |",
        "|---|---|---:|---:|",
    ]
    for rep in ("sampled", "fixed"):
        group = [r for r in ref if r["representation"] == rep]
        for obj in OBJECTIVES:
            if obj == "memory":
                continue
            ok = sum(r["qualified_" + obj] == "True" for r in group)
            lines.append(f"| {rep} | {obj} | {ok} | {len(group) - ok} |")
    lines += [
        "",
        "- Reference error on general catchments is estimated from numerical refinements. Independently exact storm-cycle evidence is limited to analytical geometries and forcing. Two formulations can share a topographic or outlet error.",
        "- Fixed and resampled terrain at 1 m and 5 m were compared on three development realisations. The remaining families lack a prospective matched 1 m operational matrix. The larger scaling runs quantify cost on the original qualification geometries.",
        "- Analytical timestep and spatial convergence rates depend on smoothness, source splitting, limiter activity and reference precision. Values near roundoff have limited order interpretation.",
        "- The 15-minute outlet interval bounds timing resolution and smooths subinterval peaks. Area averages to 10 m can obscure very narrow wet connections.",
        "- Roughness transfer uses one prospective fitted terrain and event. The original bounded analytical study covers additional training physics. Repeated objectives, bounds and timings share terrain and forcing.",
        "- The advisor is supported only within its measured grid, area, terrain and hardware range. Empirical prediction intervals can under-cover. Baseline rules and abstentions remain part of the reported assessment.",
        "- Native FREE outlet closures differ across formulations. Quasi-steady structures have separate split-operator CPU and GPU qualifications. Surveyed structure calibration and additional hydraulic controls require independent work.",
        "- No real-catchment qualification or multi-GPU architecture comparison was included. Public-release qualification remains open.",
        "",
        "[Reference changes and case IDs](processed/metrics/reference_uncertainty.csv) · [Held-out decisions](processed/advisor/heldout.csv) · [Structure qualifications](../../docs/HANCOCK_STRUCTURES.md)",
    ]
    return "\n".join(lines) + "\n"
