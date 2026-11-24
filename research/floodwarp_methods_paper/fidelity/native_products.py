"""Read-only hydraulic qualification on native metre DEMs and matched physical-event windows."""

import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    budgets,
    catalogue,
    device_controls,
    extended_analysis,
    endpoints,
    final_depth,
    native_archive,
    native_matrix,
    native_policies,
    phase_cost,
    parity,
    recession,
    recession_figures,
    repeat_costs,
    statistics,
    terrain_evidence,
)
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.terrain.generators.generate import parameters


def prefixes(*, directory):
    records = {}
    paths = [
        *(directory / "mean_storms").glob("*/result.json"),
        *(directory / "native_matrix").glob("*/*/result.json"),
    ]
    for path in paths:
        raw = json.loads(path.read_text())
        definition, data = raw["inputs"], raw["data"]
        spec = definition["spec"]
        if (
            spec["dem"] != spec["dx"]
            or spec.get("initial_grid") != "native_1m_block_mean"
            or spec.get("terrain_aggregation") != "native_1m_block_mean"
        ):
            raise ValueError(f"Derived coarse-grid input in practical archive {path}")
        terrain = definition.get("terrain")
        if terrain is None:
            terrain = json.loads(
                (path.parents[1] / (definition["physical_sha"] + ".json")).read_text()
            )
            if runner.digest(terrain) != definition["physical_sha"]:
                raise ValueError("Physical input receipt changed")
        inputs = definition.get("physical", definition.get("inputs"))
        registered = parameters(
            design=inputs["cfg"]["synthetic"],
            template=inputs["template"],
            family=spec["landform"],
            seed=spec["seed"],
            initial=spec["initial"],
        )
        if (
            terrain["inputs"]["parameters"] != registered
            or terrain["inputs"]["dimensions_m"] != spec["dimensions"]
        ):
            raise ValueError(
                "Native terrain seed or footprint differs from the registered event"
            )
        forcing = data["forcing"]
        if (
            terrain["inputs"]["native_m"] != 1
            or forcing["native_inputs"]["native_sha256"] != terrain["native_sha256"]
        ):
            raise ValueError("Hydraulic forcing and native input receipt disagree")
        ident = path.parent.name
        if runner.digest(definition)[:24] != ident:
            raise ValueError("Changed experiment identity for " + ident)
        item = assessment.item(
            spec=spec,
            data=data,
            ident=ident,
            path=path.parent,
            cohort="native-metre",
            factor=spec.get("factor", 1.0),
        )
        item["native_sha"] = runner.digest(terrain["native_sha256"])
        item["solver_source"] = definition["solver"]
        item["hardware"] = raw["hardware"]["gpu"]
        records[ident] = item
    return records


def gauge_observation(*, item, station):
    edges, discharge = assessment.curve(it=item, station=station)
    peak = float(discharge.max())
    # Each ordinate is accepted net face volume divided by its own interval.
    crest = np.flatnonzero(discharge == peak)
    return dict(
        peak_m3s=peak,
        unique_outward_peak_interval_end_s=float(edges[crest[0] + 1])
        if peak > 0 and crest.size == 1 else None,
        net_m3=float(discharge @ np.diff(edges)),
        end_m3s=float(discharge[-1]),
        end_s=float(edges[-1]),
    )


def threshold_areas(*, archive, support_m, dx_m):
    areas = {}
    for cm in (5, 10):
        fractions = archive[f"wet_fraction_{cm}cm"]
        if not np.all(np.isfinite(fractions) & (fractions >= 0) & (fractions <= 1)):
            raise ValueError("Invalid model-grid wet fraction")
        cells = float(fractions.sum()) * (support_m / dx_m)**2
        count = round(cells)
        if not np.isclose(cells, count, rtol=0., atol=1e-6):
            raise ValueError("Fractional observations disagree with the model grid cell area")
        areas[f"model_grid_wet_area_{cm}cm_m2"] = count * dx_m**2
    return areas


def section_storage(*, depths, support_m, section_x_m, storage_m3):
    """Integrate observed mean depths on each side of an exactly represented section."""
    totals = depths.sum(axis=0, dtype=np.float64) * support_m**2
    if not np.isclose(float(totals.sum()), storage_m3, rtol=2e-11, atol=1e-8):
        raise ValueError("Archived depth field disagrees with the endpoint water account")
    split = round(section_x_m / support_m)
    if split * support_m != section_x_m:
        return dict(upstream_storage_m3=None, downstream_storage_m3=None)
    if not 0 < split < len(totals):
        raise ValueError("Internal gauge must divide the observed catchment")
    return dict(upstream_storage_m3=float(totals[:split].sum()), downstream_storage_m3=float(totals[split:].sum()))


def endpoint_observations(*, item, directory, section_x_m):
    spec = item["s"]
    end = round(spec["hours"] * 3600)
    ident = item["id"]
    path = endpoints.endpoint_path(directory=directory, ident=ident, end_s=end)
    has_receipt = path.is_file()
    if not has_receipt:
        if item.get("parent_id", ident) != ident:
            raise ValueError("Missing completion fields for model-grid inundation measurement")
        path = item["path"] / "spatial.npz"
    with np.load(path, allow_pickle=False) as archive:
        support = spec["support"]
        if has_receipt:
            receipt = json.loads(archive["receipt"].item())
            if receipt["id"] != ident or receipt["end_s"] != end:
                raise ValueError("Extent measurement belongs to a different forecast window")
            support = receipt["support"]
        regions = section_storage(depths=archive["depth"], support_m=support, section_x_m=section_x_m,
                                  storage_m3=item["d"]["final_storage_m3"])
        return dict(**regions, **threshold_areas(archive=archive, support_m=support, dx_m=spec["dx"]))


def describe(*, item, settings, status, directory):
    d, s = item["d"], item["s"]
    outlet = gauge_observation(item=item, station="outlet")
    section = gauge_observation(item=item, station="section")
    section_x_m = (round(s["dimensions"][0] / s["dx"]) // 2) * s["dx"]
    matched, stations = recession.settled(
        frames=d["frames"],
        storage=d["final_storage_m3"],
        supply=d["initial_m3"] + d["rain_m3"],
        config=settings,
    )
    method = s.get("requested_method", s["method"])
    policy = assessment.numerical_identity(method=method)["execution_mode"]
    selection = d.get("forcing", {}).get("execution_selection", {})
    return dict(
        case=s["case"],
        landform=s["landform"],
        method=method,
        dx_m=s["dx"],
        factor=item["factor"],
        source_id=item["id"],
        native_sha=item["native_sha"],
        horizon=status,
        end_h=outlet["end_s"] / 3600,
        interval_s=s["interval"],
        cells=round(np.prod(s["dimensions"]) / s["dx"] ** 2),
        initial_m3=d["initial_m3"],
        rain_m3=d["rain_m3"],
        infiltration_m3=d["infiltration_m3"],
        storage_m3=d["final_storage_m3"],
        outlet_peak_m3s=outlet["peak_m3s"],
        outlet_unique_outward_peak_interval_end_s=outlet["unique_outward_peak_interval_end_s"],
        end_outlet_m3s=outlet["end_m3s"],
        net_outlet_m3=outlet["net_m3"],
        section_x_m=section_x_m,
        section_peak_m3s=section["peak_m3s"],
        section_unique_outward_peak_interval_end_s=section["unique_outward_peak_interval_end_s"],
        net_section_m3=section["net_m3"],
        late_outward_volume_fraction=stations["outlet"]["recent_to_volume"],
        both_stations_passed=matched,
        residual_m3=d["residual_m3"],
        hydraulic_s=d["hydraulic_s"],
        hardware=item["hardware"],
        solver_source=item["solver_source"],
        timing_repeat=s.get("timing_repeat"),
        execution_mode=policy,
        resolved_execution_mode=selection.get("selected") if method == "hancock_auto" else policy,
        end_to_end_s=d.get("end_to_end_s"),
        hydraulic_s_per_step=d["hydraulic_s"]/d["steps"],
        steps=d["steps"],
        **endpoint_observations(item=item, directory=directory, section_x_m=section_x_m),
    )


def compare_incised_windows(*, directory, records, target):
    """Report matched cross-grid disagreement without treating a forecast as truth."""
    design = json.loads((directory / "mean_settings.json").read_text())
    entries = [
        r
        for r in records.values()
        if r["s"]["case"] == design["case"]
        and r["path"].parent == directory / "mean_storms"
    ]
    differences = []
    areas = []
    for item in entries:
        s = item["s"]
        with np.load(item["path"] / "spatial.npz") as state:
            for cm in (5, 10):
                fraction = state[f"wet_fraction_{cm}cm"]
                areas.append(
                    dict(
                        id=item["id"],
                        method=s["method"],
                        dx_m=s["dx"],
                        depth_cm=cm,
                        hours=s["hours"],
                        inundated_km2=float(fraction.sum()) * s["support"] ** 2 / 1e6,
                    )
                )
    for a, b in combinations(entries, 2):
        sa, sb = a["s"], b["s"]
        if (
            a["native_sha"] != b["native_sha"]
            or sa["support"] != sb["support"]
            or sa["hours"] != sb["hours"]
        ):
            raise ValueError(
                "Incised grid comparisons need matched physical source, support and time"
            )
        if sa["method"] != sb["method"] and sa["dx"] != sb["dx"]:
            continue
        if (sa["dx"], sa["method"]) < (sb["dx"], sb["method"]):
            a, b = b, a
            sa, sb = sb, sa
        with (
            np.load(a["path"] / "spatial.npz") as first,
            np.load(b["path"] / "spatial.npz") as second,
        ):
            h1, h2 = first["depth"], second["depth"]
            for cm in (0, 5, 10):
                mask = np.maximum(h1, h2) >= cm / 100
                field = h1 - h2
                delta = field[mask]
                squared = float(np.sum(field**2))
                differences.append(
                    dict(
                        left_id=a["id"],
                        right_id=b["id"],
                        left_method=sa["method"],
                        right_method=sb["method"],
                        left_dx_m=sa["dx"],
                        right_dx_m=sb["dx"],
                        common_support_m=sa["support"],
                        hours=sa["hours"],
                        wet_union_cm=cm,
                        blocks=int(mask.sum()),
                        eastern_edge_blocks=int(mask[:, -1].sum()),
                        eastern_strip_fraction_squared_difference=float(
                            np.sum(field[:, -1] ** 2) / squared
                        )
                        if squared
                        else None,
                        eastern_strip_signed_storage_difference_m3=float(
                            field[:, -1].sum() * sa["support"] ** 2
                        ),
                        rmse_cm=float(100 * np.sqrt(np.mean(delta**2)))
                        if delta.size
                        else None,
                        signed_mean_cm=float(100 * delta.mean())
                        if delta.size
                        else None,
                    )
                )
    metrics.csv_write(file=target / "incised_state_disagreement.csv", rows=differences)
    metrics.csv_write(file=target / "incised_practical_extent.csv", rows=areas)


def merged_repeat_receipt(*, receipts):
    """Join subsets of one independent timing repeat at the same physical endpoint."""
    first = receipts[0]
    series = {}
    for receipt in receipts:
        if any(receipt[k] != first[k] for k in
               ("case", "representation", "cohort", "end_s", "extended", "settings", "source")):
            raise ValueError("Timing repeat subsets do not share the same physical endpoint and settings")
        if any(receipt["hardware"][k] != first["hardware"][k] for k in
               ("gpu", "cpu_model", "warp", "python", "numpy", "precision")):
            raise ValueError("Timing repeat subsets use different hardware or numerical environments")
        parents = receipt["sources"]
        children = receipt["series"] if receipt["extended"] else [None] * len(parents)
        if not receipt["extended"] and receipt["series"]:
            raise ValueError("Unexpected continuation in unextended repeat")
        for parent, child in zip(parents, children, strict=True):
            if parent in series and series[parent] != child:
                raise ValueError("Different recession histories for the same timing repeat")
            series[parent] = child
    parents = sorted(series)
    return dict(first, sources=parents, series=[series[p] for p in parents] if first["extended"] else [])


def grouped(*, directory, records, target, expected):
    groups = {}
    repeat_receipts = {}
    for path in (directory / "extended_runs").glob("group_*.json"):
        r = json.loads(path.read_text())
        if (
            (r["cohort"] not in ("area-mean", "native-metre") and not r["cohort"].startswith("native-repeat-"))
            or not set(r["sources"]) <= records.keys()
        ):
            continue
        items = [records[k] for k in r["sources"]]
        if ({i["s"]["case"] for i in items} != {r["case"]}
                or len({i["native_sha"] for i in items}) != 1):
            raise ValueError("Matched recession forecasts must share a case and native physical inputs")
        if (
            r["cohort"] == "native-metre"
            and set(
                (
                    i["s"]["dx"],
                    i["s"].get("requested_method", i["s"]["method"]),
                    i["factor"],
                )
                for i in items
            )
            != expected[r["case"]]
        ):
            continue
        key = r["cohort"], r["case"], r["representation"]
        if r["cohort"].startswith("native-repeat-"):
            repeat_receipts.setdefault(key, []).append(r)
        elif key in groups:
            raise ValueError("Multiple matched source sets " + str(key))
        else:
            groups[key] = items
    merged = []
    for key, receipts in repeat_receipts.items():
        receipt = merged_repeat_receipt(receipts=receipts)
        items = [records[parent] for parent in receipt["sources"]]
        if len({i["native_sha"] for i in items}) != 1:
            raise ValueError("Different native physical inputs across timing repeat subsets")
        groups[key] = items
        merged.append(receipt)
    from unittest.mock import patch

    with patch.object(recession, "candidates", native_matrix.all_items):
        joined = extended_analysis.attach(
            directory=directory, groups=groups, target=target, canonical_native_cost=True, merged_receipts=merged
        )
    result = {}
    for (cohort, case, _), items in joined.items():
        for item in items:
            parent = records[item.get("parent_id", item["id"])]
            # A policy may share a verified prefix across groups; retain the measured requested policy.
            s = dict(parent["s"], hours=item["s"]["hours"])
            s["method"] = s.get("requested_method", s["method"])
            item["s"] = s
            item["native_sha"] = parent["native_sha"]
            item["hardware"] = parent["hardware"]
            item["solver_source"] = parent["solver_source"]
            if item["id"] != parent["id"]:
                elapsed = sorted((directory / "extension_timings" / item["id"]).glob("*.json"))
                timings = [json.loads(path.read_text()) for path in elapsed]
                ends = [r["end_s"] for r in timings if r["end_s"] <= s["hours"] * 3600]
                if (ends and ends == sorted(set(ends)) and ends[-1] == s["hours"] * 3600
                        and parent["d"].get("end_to_end_s") is not None):
                    item["d"]["end_to_end_s"] = parent["d"]["end_to_end_s"] + sum(
                        r["elapsed_wall_s"] for r in timings if r["end_s"] <= ends[-1])
        result[cohort, case, "native metre area means"] = items
    return result


def plot_curves(*, directory, records):
    import matplotlib.pyplot as plt

    target = directory / "native_results/figures"
    target.mkdir(exist_ok=True)
    groups = {}
    for item in records:
        groups.setdefault(item["s"]["case"], []).append(item)
    for case, items in groups.items():
        fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True, layout="constrained")
        for column, (station, title) in enumerate((("section", "Internal section"), ("outlet", "Outlet"))):
            flow, net, fraction = axes[:, column]
            flow.set_title(title)
            for item in items:
                s = item["s"]
                edges, q = assessment.curve(it=item, station=station)
                t_h = edges / 3600
                method = assessment.numerical_identity(method=s.get("requested_method", s["method"]))
                label = f"{method['formulation']} {method['execution_mode']} · {s['dx']:g} m"
                if item["factor"] != 1:
                    label += f" · CFL ×{item['factor']:g}"
                line = flow.stairs(q, t_h, label=label)
                volumes = np.r_[0, np.cumsum(q * np.diff(edges))]
                net.plot(t_h, volumes / 1000, color=line.get_edgecolor())
                if volumes[-1] > 0 and q.min() >= 0:
                    fraction.plot(t_h, volumes / volumes[-1], color=line.get_edgecolor())
            if not fraction.lines:
                fraction.text(.5, .5, "No positive unidirectional series",
                              ha="center", va="center", transform=fraction.transAxes)
            fraction.set_xlabel("Time since storm start (h)")
        axes[0, 0].set_ylabel("Discharge (m³/s)")
        axes[1, 0].set_ylabel("Net transport (10³ m³)")
        axes[2, 0].set_ylabel("Fraction of event transport")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", fontsize=7, ncols=3)
        for ax in axes.flat:
            ax.axvline(6, ls=":", lw=1, color="0.4")
        metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}
        with plt.rc_context({"svg.hashsalt": case}):
            for suffix, tags in metadata.items():
                path = target / f"{case}.{suffix}"
                fig.savefig(path, dpi=300, metadata=tags)
                if suffix == "svg":
                    path.write_text(
                        "\n".join(
                            line.rstrip() for line in path.read_text().splitlines()
                        )
                        + "\n"
                    )
        plt.close(fig)


def event_figures(*, directory, groups, target):
    feature_path = directory / "native_features/observations.csv"
    features = {
        (r["case"], int(r["dx_m"])): float(r["spatial_mean_multiplier"])
        for r in metrics.csv_read(file=feature_path)
    } if feature_path.exists() else {}
    index = []
    for key, items in sorted(groups.items()):
        cohort, case, representation = key
        if not items or any(i["s"].get("timing_repeat") for i in items):
            continue
        for dx in sorted({i["s"]["dx"] for i in items}):
            subset = [i for i in items if i["s"]["dx"] == dx and i["factor"] == 1]
            if not subset:
                continue
            end = {i["s"]["hours"] for i in subset}
            if len(end) != 1 or len({i["native_sha"] for i in subset}) != 1:
                raise ValueError("Figure requires matched forcing, terrain and physical end time")
            tag = cohort.replace("-", "_")
            recession_figures.operational(
                groups={key: subset}, directory=target, cases=[(case, dx)],
                cohort=cohort, representation=representation,
                policies=tuple(i["s"]["method"] for i in subset),
                rain_multiplier=features.get((case, dx)), output_tag=tag,
            )
            index.append(dict(cohort=cohort, case=case, dx_m=dx, end_h=end.pop(),
                              ids=",".join(i["id"] for i in subset),
                              pdf=f"figures/{case}_{dx:g}m_{tag}_cost.pdf"))
    metrics.csv_write(file=target / "event_figures.csv", rows=index)


def run(*, directory, require_complete):
    target = directory / "native_results"
    target.mkdir(exist_ok=True)
    cfg = read_config(path=directory.parent / "configs/study.json")
    settings = json.loads((directory / "recession_settings.json").read_text())
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    expected = {}
    for s in native_matrix.specifications(
        study=directory.parent, config=cfg, design=design
    ):
        expected.setdefault(s["case"], set()).add(
            (s["dx"], s["requested_method"], s["factor"])
        )
    records = prefixes(directory=directory)
    compare_incised_windows(directory=directory, records=records, target=target)
    current = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    sources = {i["solver_source"] for i in records.values()}
    if current in sources:
        records = {key:i for key,i in records.items() if i["solver_source"] == current}
    elif require_complete or len(sources) > 1:
        raise ValueError("No current matched numerical implementation for native qualification")
    device_controls.collect(directory=directory, records=records, target=target)
    prefix_groups = {}
    for i in records.values():
        spec = i["s"]
        key = "original-window", spec["case"], f"{i['native_sha']}, {spec['hours']} h, repeat {spec.get('timing_repeat', 0)}"
        member = dict(i,s=dict(spec,method=spec.get("requested_method",spec["method"])))
        prefix_groups.setdefault(key,[]).append(member)
    initial_parity = parity.compare(directory=directory,groups=prefix_groups,target=target,original_window=True)
    original = [
        describe(item=i, settings=settings, status="original_window", directory=directory)
        for i in records.values()
    ]
    groups = grouped(
        directory=directory, records=records, target=target, expected=expected
    )
    verification = native_archive.observations(directory=directory, records=records, groups=groups)
    metrics.csv_write(file=target / "verified_archives.csv", rows=verification)
    timing_sources = [dict(case=case, id=item["id"], parent_id=item["parent_id"], cohort=cohort,
                           method=item["s"]["method"], dx_m=item["s"]["dx"], end_h=item["s"]["hours"],
                           corrected_hydraulic_s=item["d"]["hydraulic_s"], **item["d"]["native_timing_provenance"])
                      for (cohort, case, _), items in groups.items() for item in items
                      if "native_timing_provenance" in item["d"]]
    metrics.csv_write(file=target / "timing_provenance.csv", rows=timing_sources)
    final_parity = parity.compare(directory=directory,groups=groups,target=target)
    complete = [
        describe(item=i, settings=settings, status=key[0], directory=directory)
        for key, items in groups.items()
        for i in items
    ]
    metrics.csv_write(file=target / "measured_windows.csv", rows=original + complete)
    timings = repeat_costs.collect(rows=original + complete, target=target)
    for members in groups.values():
        for item in members:
            repeated = timings.get(item["id"])
            if repeated:
                item["cost_for_comparison_s"] = repeated["hydraulic_median_s"]
    hydro = json.loads((directory / "design.json").read_text())
    application = json.loads((directory / "application_settings.json").read_text())
    native_policies.collect(
        groups={**prefix_groups, **groups}, verified=initial_parity + final_parity,
        cadence=application["cost_interval_s"], target=target,
    )
    phase_cost.collect(
        groups=groups, directory=directory, target=target, representation="native metre area means",
        cadence=application["cost_interval_s"], edges=application["wet_fraction_edges"],
        phase_edges=application["phase_end_hours"],
    )
    additional = json.loads((directory/"native_advisor_settings.json").read_text())["reference_limits"]
    hydro = dict(hydro, reference_limits={**hydro["reference_limits"], **additional})
    primary = {key: members for key, members in groups.items() if not key[0].startswith("complete-native-repeat-")}
    rows, checks = assessment.storm_rows(groups=primary, cfg=hydro)
    metrics.csv_write(file=target / "hydraulic_fidelity.csv", rows=rows)
    metrics.csv_write(file=target / "reference_checks.csv", rows=checks)
    front = statistics.frontiers(rows=rows, objectives=tuple(hydro["reference_limits"]))
    metrics.csv_write(file=target / "hydrograph_pareto.csv", rows=front)
    controls = terrain_evidence.forecast_controls(directory=directory, rows=complete)
    budgets.native_cost(
        rows=front, name="fidelity", controls=controls,
        target=target,
        tolerances=application["tolerances_pct"],
        spacings=application["fv1_budget_grids_m"],
    )
    final_depth.process(
        groups=primary,
        directory=directory,
        target=target,
        settings=application["endpoint_depth"], terrain_controls=controls,
    )
    event_figures(directory=directory, groups={**prefix_groups, **groups}, target=target)
    plot_curves(
        directory=directory,
        records=list(records.values())
        if not groups
        else [i for items in primary.values() for i in items],
    )
    catalogue.database(target=target)
    counted = {r["case"] for r in complete if r["horizon"] == "complete-native-metre"}
    text = [
        "# Native-metre storm qualification",
        "",
        f"{len(records)} original forecast windows have been measured. {len(counted)} of {len(expected)} physical domains have a matched recession receipt for every scheduled configuration.",
        "",
        (f"The numerical source matches the current hydraulic implementation `{current}`."
         if current in sources else
         "The wide-matrix records in this product are historical. Corrected incised results are in "
         "[the repaired outlet study](../outlet_results/README.md). The current native matrix is pending."),
        "",
        "Outlet peaks are interval means of accepted face transport. The original-window table may end while substantial flow continues. Numerical error rankings require common recession endpoints and separate spatial, temporal and formulation checks for the observable.",
        "",
        "[Hydraulic windows and water budgets](measured_windows.csv) · [Hydrograph comparisons](hydraulic_fidelity.csv) · [Reference checks](reference_checks.csv) · [Depth and extent](recession_depth.csv) · [Recorded hydrographs](figures) · [5 cm and 10 cm native extent](incised_practical_extent.csv) · [Matched-grid depth differences](incised_state_disagreement.csv) · [Accepted steps, retries, memory and accounting](whole_storm_diagnostics.csv) · [Costs by phase and footprint](cost_by_phase.csv) · [Original policy parity](original_execution_parity.csv) · [Recession policy parity](recession_execution_parity.csv)",
        "",
        "[Rainfall, outlet discharge, wet area and cumulative GPU runtime by event](event_figures.csv) · [Verified dense, tiled and automatic policy costs](policy_cost_events.csv) · [Matched 15-minute timing windows](policy_cost_windows.csv) · [Independent timings, run IDs and spread](timing_distribution.csv) · [Verified raw files and matched horizon receipts](verified_archives.csv)",
    ]
    (target / "README.md").write_text("\n".join(text) + "\n")
    print(text[2], flush=True)
    if require_complete and (
        counted != set(expected) or not all(r["both_stations_passed"] for r in complete)
    ):
        raise ValueError("Native storm matrix or recession qualification incomplete")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    require_complete = sys.argv[1:] == ["verify-native"]
    run(directory=directory, require_complete=require_complete)
