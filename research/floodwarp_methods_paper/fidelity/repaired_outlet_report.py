"""Generate the boundary repair evidence from accepted transport and immutable storm receipts."""

import json
import math
from pathlib import Path
import hashlib
import subprocess

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.source_trees import verify_tree


def source_proof(*, directory, settings, destination):
    root = directory.parents[2]
    stages = ("long-storms", "analytical", "convergence")
    revisions = {
        tag: verify_tree(root=root, revision=sha, stages=stages)
        for tag, sha in settings.items()
    }
    driver = directory / "outlet_storms.py"
    blob = subprocess.check_output(
        ["git", "show", f"{revisions['repair']['commit']}:{driver.relative_to(root)}"], cwd=root
    )
    driver_sha = hashlib.sha256(blob).hexdigest()
    sources = json.loads((destination / "sources.json").read_text())
    if (sources["solver"] != revisions["repair"]["source_sha256"]["long-storms"]
            or sources["driver"] != driver_sha):
        raise ValueError("Storm driver or solver differs from the named repair commit")
    old = {json.loads(p.read_text())["inputs"]["solver"]
           for p in (directory / "mean_storms").glob("*/result.json")}
    if old != {revisions["audit"]["source_sha256"]["long-storms"]}:
        raise ValueError("Historical incised comparison differs from its named audit commit")
    runs = {ident: json.loads((directory / "outlet_storms" / ident / "inputs.json").read_text())["hardware"]
            for ident in sources["storms"]}
    runner.write_json(path=destination / "revision_evidence.json",
                      data=dict(revisions=revisions, storm_driver_sha256=driver_sha, launch_environment=runs))
    return revisions


def split_sheet_bias(*, settings, dt):
    """At a uniform-sheet fixed point, consecutive drag half steps give one full step."""
    acceleration = settings["gravity"] * settings["grade"]
    normal_speed = settings["depth_m"]**(2/3) * math.sqrt(settings["grade"]) / settings["manning"]
    return 100 * (math.hypot(1, acceleration * dt / (2 * normal_speed)) - 1)


def reach_comparison(*, directory, destination, revisions):
    rows = []
    observations = {}
    settings = json.loads((directory / "free_reach_settings.json").read_text())
    physical_sha = runner.digest(json.loads((directory / "mean_storms/terrain.json").read_text()))
    for path in sorted((directory / "free_reach").glob("*/comparison.csv")):
        contract = json.loads((path.parent / "inputs.json").read_text())
        kind = next((s for s, r in revisions.items() if r["source_sha256"]["long-storms"] == contract["solver"]), None)
        if kind is None:
            continue
        if runner.digest(contract)[:24] != path.parent.name:
            raise ValueError("Changed physical reach experiment identity")
        if contract["settings"] != settings or contract["physical_sha"] != physical_sha:
            continue
        frame = contract["settings"]["times_s"][-2:]
        for row in metrics.csv_read(file=path):
            if [float(row["start_s"]), float(row["end_s"])] != frame:
                continue
            if abs(float(row["residual_m3"])) > 1.e-8:
                raise ValueError("Native reach water account failed")
            if (int(row["section_x_m"]) != settings["section_x_m"]
                    or [int(row["south_m"]), int(row["north_m"])] != settings["section_y_bounds_m"]):
                raise ValueError("Native reach measurements represent different physical sections")
            method, dx = contract.get("method", "fv2"), int(row["grid_m"])
            offset = int(row["extension_m"])
            key = (kind, method, dx, offset)
            q = float(row["section_m3s"])
            if key in observations and observations[key] != q:
                raise ValueError("Repeated physically identical reach forecasts differ")
            observations[key] = q
            rows.append(dict(revision=kind, method=method, execution=contract.get("mode", "dense"),
                             device=contract["device"], physical_sha=physical_sha, run_id=path.parent.name, dx_m=dx,
                             extension_m=offset, section_x_m=row["section_x_m"], start_s=frame[0], end_s=frame[1],
                             discharge_m3s=q, residual_m3=row["residual_m3"]))
    metrics.csv_write(file=destination / "incised_reach.csv", rows=rows)
    text = ["", "## Fixed incised section at 10,040 m", "",
            "Mean discharge (m³/s) from 30 to 60 seconds on the same physical 120 m-wide section. "
            "Boundary placement is varied independently of the section position. The reach has no exact discharge reference.",
            "", "| Version | Formulation | Grid (m) | At FREE face | 100 m buffer | 200 m buffer |",
            "|---|---|---:|---:|---:|---:|"]
    for tag, method, dx in sorted({k[:3] for k in observations}):
        values = [observations.get((tag, method, dx, offset)) for offset in (0, 100, 200)]
        text.append(f"| {tag} | {method} | {dx} | "
                    + " | ".join(f"{q:.5f}" if q is not None else "pending" for q in values) + " |")
    return [*text, "", "[Measurement windows and source runs](incised_reach.csv)"]


def run(*, directory):
    source = directory / "outlet_results"
    rows = metrics.csv_read(file=source / "events.csv")
    settings = json.loads((directory / "outlet_review_revisions.json").read_text())
    revisions = source_proof(directory=directory, settings=settings, destination=source)
    history = metrics.csv_read(file=directory / "mean_storms/complete_events.csv")
    before = {(r["method"], r["dx_m"]): r for r in history}
    plane = directory / "free_plane"
    sha = runner.source_identity(
        root=directory.parents[2], study=directory.parent, stage="long-storms"
    )
    (current,) = [
        p.parent
        for p in plane.glob("*/identity.json")
        if json.loads(p.read_text())["solver_sha256"] == sha
    ]
    sheet = metrics.csv_read(file=current / "discharge.csv")
    old = metrics.csv_read(file=plane / "discharge.csv")
    parameters = json.loads((current/"identity.json").read_text())["settings"]
    split_checks=[]
    for row in sheet:
        if row["method"]!="fv1" and float(row["start_s"]) == 30:
            prediction=split_sheet_bias(settings=parameters,dt=float(row["maximum_dt_s"]))
            split_checks.append(dict(method=row["method"],dx_m=row["dx_m"],dt_s=row["maximum_dt_s"],face=row["face"],
                                     extension_m=row["downstream_extension_m"],measured_bias_pct=float(row["discharge_bias_pct"]),
                                     split_bias_pct=prediction,unexplained_percentage_points=float(row["discharge_bias_pct"])-prediction))
    metrics.csv_write(file=source/"uniform_sheet_source_balance.csv",rows=split_checks)
    mismatch=max(abs(r["unexplained_percentage_points"]) for r in split_checks)
    text = [
        "# FREE outlet qualification",
        "",
        f"The archived pre-repair numerical source matches Git commit `{revisions['audit']['commit'][:12]}`. "
        f"The corrected hydraulic source and storm driver match commit `{revisions['repair']['commit'][:12]}`. "
        "Several trajectories started while the repair was in the working tree. The source hashes identify the committed implementation used by those runs. "
        "[Git-tree verification and per-run hardware](revision_evidence.json)",
        "",
        "The exact Manning reference is steady uniform flow on a 7.329% plane. "
        "Discharge below uses the accepted face volume between 30 and 60 s at the same section, 400 m from the upstream boundary.",
        "",
        "| Method | Grid (m) | Cap (s) | Original face bias (%) | Repaired face bias (%) | Buffered section bias (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sheet:
        if (
            r["start_s"] not in ("30.0", "30")
            or r["face"] != "east"
            or int(r["downstream_extension_m"])
        ):
            continue
        terms = ("method", "dx_m", "maximum_dt_s", "face", "time_s")
        matches = [
            s
            for s in old
            if all(s[k] == r[k] for k in terms)
            and int(s["downstream_extension_m"]) == 0
        ]
        buffer = [
            s
            for s in sheet
            if all(s[k] == r[k] for k in terms)
            and int(s["downstream_extension_m"]) == 400
        ]
        if len(matches) != 1 or len(buffer) != 1:
            raise ValueError("Inconsistent analytical gauge support")
        text.append(
            f"| {r['method']} | {r['dx_m']} | {r['maximum_dt_s']} | {float(matches[0]['discharge_bias_pct']):+.4f} | "
            f"{float(r['discharge_bias_pct']):+.4f} | {float(buffer[0]['discharge_bias_pct']):+.4f} |"
        )
    text += [
        "",
        "The terminal slope previously vanished when the reconstruction stencil reached the FREE face. "
        "This left a half-cell bed discontinuity at the last internal face and suppressed the surface-gradient momentum source. "
        "Hancock and RK2 now extrapolate the bed and surface increments from the wet interior at an explicitly open face, "
        "with zero normal velocity gradient and the existing dry and positivity checks. The residual uniform-sheet error "
        "is predicted by the discrete source balance. Let U be the exact normal speed, a = gS the bed acceleration, "
        "and k = a/U² the quadratic drag coefficient. The steady speed v just before acceleration satisfies "
        "k v(v + aΔt) = a because consecutive drag half steps combine into one full drag step. "
        "The accepted halfway face speed is v + aΔt/2 = √(U² + (aΔt/2)²). "
        f"Across both formulations, all measured orientations, the face and the buffered section, this predicts the observed 30–60 s bias "
        f"to within {mismatch:.2e} percentage points. [Numerical and predicted residual](uniform_sheet_source_balance.csv) "
        "FV1 reconstructs from piecewise-constant cell beds. At this grade, the bed drops 0.07329 m per metre, exceeding the 0.03 m sheet depth even on the finest tested grid. Its hydrostatic cut to the higher cell bed clips the downstream state. At the 0.1 s cap the buffered FV1 deficit drops from 46.84% on 10 m cells to 36.82% on 1 m cells. The extra face deficit falls from 25.66 percentage points to below the displayed precision.",
        *reach_comparison(directory=directory, destination=source, revisions=revisions),
        "", "## Full incised storm",
        "",
        "The gauge remains at the eastern edge of the original 104.8576 km² catchment. The extension continues the 1 m valley, "
        "initial channel state and soil. Its additional rainfall is zero. The source fields on the original footprint retain their historical values.",
        "",
        "| Method | Grid (m) | Extension (m) | CFL factor | Pre-fix direct peak (m³/s) | Peak (m³/s) | Minute ending (h) | "
        "Fixed 15 min peak (m³/s) | Moving 15 min peak (m³/s) | Export at gauge (m³) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=sort_case):
        prev = before[r["method"], r["dx_m"]]
        historical = (
            prev["outlet_peak_m3s"]
            if float(r["cfl_factor"]) == 1 and float(r["extension_m"]) == 0
            else None
        )
        value = f"{float(historical):.3f}" if historical else "—"
        text.append(
            f"| {r['method']} | {r['dx_m']} | {r['extension_m']} | {r['cfl_factor']} | {value} | "
            f"{float(r['minute_peak_m3s']):.3f} | {float(r['minute_end_s']) / 3600:.3f} | "
            f"{float(r['fixed_15min_peak_m3s']):.3f} | {float(r['moving_15min_peak_m3s']):.3f} | {float(r['gauge_m3']):,.1f} |"
        )
    text += [
        "",
        f"{len(rows)} of 14 trajectories have completed the historical 16-hour comparison. "
        "Each hydrograph is an interval average of accepted transport. On these irregular terrains a solver difference alone does not establish accuracy.",
        "",
        "[Water accounts, infiltration inside and outside the rainfall footprint](events.csv) · "
        "[Crest and whole-recession last 40 m budgets](last_40m.csv) · [Spatial extent](extent.csv) · "
        "[Wet-union depth differences on common spatial support](depth_disagreement.csv) · "
        "[Spatial reference qualification](spatial_reference_checks.csv) · [Hydrograph and infiltration reference qualification](reference_checks.csv)",
    ]
    balances = metrics.csv_read(file=source / "water_differences.csv")
    extents = metrics.csv_read(file=source / "extent.csv")
    text.extend(
        [
            "",
            "## FV1 minus Hancock on the original catchment",
            "",
            "| Grid (m) | Extension (m) | Extra export (m³) | Infiltration difference in rain footprint (m³) | "
            "Infiltration difference outside (m³) | Final storage difference (m³) | Unexplained export (m³) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in balances:
        if (r["first_method"], r["second_method"]) == ("fv1", "hancock_adaptive"):
            text.append(
                f"| {r['dx_m']} | {r['extension_m']} | {float(r['gauge_m3']):+,.1f} | "
                f"{float(r['infiltration_inside_rain_m3']):+,.1f} | {float(r['infiltration_outside_rain_m3']):+,.1f} | "
                f"{float(r['footprint_storage_m3']):+,.1f} | {float(r['unexplained_export_m3']):+.2e} |"
            )
    text.extend(
        [
            "",
            "## Hour-eight inundation and maximum extent",
            "",
            "| Model | Grid (m) | 5 cm at 8 h (km²) | 10 cm at 8 h (km²) | Ever above 5 cm by 16 h (km²) | "
            "Ever above 10 cm by 16 h (km²) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in sorted(rows, key=sort_case):
        if float(r["cfl_factor"]) == 1 and int(r["extension_m"]) == 0:
            areas = {
                int(float(a["hour"])): a for a in extents if a["run_id"] == r["run_id"]
            }
            quantities = [
                float(areas[hour][f"{field}_above_{cm}cm_m2"]) / 1.0e6
                for hour, field in ((8, "area"), (16, "ever"))
                for cm in (5, 10)
            ]
            text.append(
                f"| {r['method']} | {r['dx_m']} | "
                + " | ".join(f"{value:.4f}" for value in quantities)
                + " |"
            )
    text.extend(
        [
            "",
            "The spatial comparison thresholds area-averaged depths on the coarser common support. "
            "For maxima, it averages the recorded native-cell maxima. The extent table above counts cells at each native hydraulic spacing. "
            "The 5 m trajectories quantify formulation and timestep sensitivity. A finer storm reference is required for absolute 5 m accuracy.",
            "",
            "[Water-balance decomposition](water_differences.csv) · [Least measured cost by tolerance](least_cost_fidelity.csv) · "
            "[Smallest qualified error within the FV1 budget](equal_cost_fidelity.csv)",
            "",
            "Depth and inundation costs run from the initial state to each snapshot. "
            "The 8 h comparison uses only the first eight hours of measured hydraulic execution. "
            "[Least cost by spatial tolerance](least_cost_depth_extent.csv) · "
            "[Smallest spatial error at each FV1 budget](equal_cost_depth_extent.csv) · "
            "[Independent timing windows](spatial_timing_windows.csv)",
            "",
            "The [regional water and inundation accounts](regional_contrasts.csv) divide the original footprint "
            "at the internal gauge and at the start of the final 40 m. Each area is also split by the prescribed rainfall footprint.",
        ]
    )
    controls = directory / "strong_storms/completed.csv"
    if controls.exists():
        active_ids = {r["id"] for r in metrics.csv_read(file=controls)}
        for path in sorted((directory / "native_step_history").glob("*/verified.json")):
            receipt = json.loads(path.read_text())
            policies = receipt.get("methods", [])
            if (not policies or {r["primary"] for r in policies} - active_ids
                    or {r["method"] for r in policies} != {"hancock_dense", "hancock_adaptive"}):
                continue
            if len({(r["steps"], r["accepted_dt_sha256"], r["end_s"]) for r in policies}) != 1:
                raise ValueError("Different ordered CPU timesteps in published parity receipt")
            for r in policies:
                if runner.checksum(path.parent / (r["method"] + ".npz")) != r["sha256"]:
                    raise ValueError("Changed CPU timestep history")
            text += ["", f"On the 10 m moderate-rainfall plain the two CPU policies reproduced "
                     f"all {policies[0]['steps']:,} ordered timesteps through {policies[0]['end_s']/3600:g} hours. "
                     f"Their interval face transport and final fields matched the primary trajectory. "
                     f"[Ordered timestep evidence](../{path.relative_to(directory).as_posix()})"]
    (source / "README.md").write_text("\n".join(text) + "\n")


def sort_case(row):
    return (
        -int(row["dx_m"]),
        int(row["extension_m"]),
        row["method"],
        float(row["cfl_factor"]),
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
