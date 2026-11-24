"""Read-only hydraulic qualification on native metre DEMs and matched physical-event windows."""

import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    assessment,
    budgets,
    catalogue,
    extended_analysis,
    final_depth,
    native_matrix,
    recession,
    statistics,
)
from research.floodwarp_methods_paper.plan import read_config


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
        records[ident] = item
    return records


def describe(*, item, settings, status):
    d, s = item["d"], item["s"]
    edges, q = assessment.curve(it=item, station="outlet")
    matched, stations = recession.settled(
        frames=d["frames"],
        storage=d["final_storage_m3"],
        supply=d["initial_m3"] + d["rain_m3"],
        config=settings,
    )
    return dict(
        case=s["case"],
        landform=s["landform"],
        method=s.get("requested_method", s["method"]),
        dx_m=s["dx"],
        factor=item["factor"],
        source_id=item["id"],
        native_sha=item["native_sha"],
        horizon=status,
        end_h=edges[-1] / 3600,
        interval_s=s["interval"],
        cells=round(np.prod(s["dimensions"]) / s["dx"] ** 2),
        initial_m3=d["initial_m3"],
        rain_m3=d["rain_m3"],
        infiltration_m3=d["infiltration_m3"],
        storage_m3=d["final_storage_m3"],
        outlet_peak_m3s=float(q.max()),
        end_outlet_m3s=float(q[-1]),
        net_outlet_m3=float(q @ np.diff(edges)),
        late_outward_volume_fraction=stations["outlet"]["recent_to_volume"],
        both_stations_passed=matched,
        residual_m3=d["residual_m3"],
        hydraulic_s=d["hydraulic_s"],
        steps=d["steps"],
    )


def grouped(*, directory, records, target, expected):
    groups = {}
    for path in (directory / "extended_runs").glob("group_*.json"):
        r = json.loads(path.read_text())
        if (
            r["cohort"] not in ("area-mean", "native-metre")
            or not set(r["sources"]) <= records.keys()
        ):
            continue
        items = [records[k] for k in r["sources"]]
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
        if key in groups:
            raise ValueError("Multiple matched source sets " + str(key))
        groups[key] = items
    from unittest.mock import patch

    with patch.object(recession, "candidates", native_matrix.all_items):
        joined = extended_analysis.attach(
            directory=directory, groups=groups, target=target
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
        fig, axes = plt.subplots(
            3, 1, figsize=(7.2, 6.5), sharex=True, layout="constrained"
        )
        for item in items:
            s = item["s"]
            edges, q = assessment.curve(it=item, station="outlet")
            label = f"{s.get('requested_method', s['method'])}, {s['dx']:g} m, CFL ×{item['factor']:g}"
            axes[0].stairs(q, edges / 3600, label=label)
            volumes = np.cumsum(q * np.diff(edges))
            axes[1].plot(edges / 3600, np.r_[0, volumes] / 1000.0)
            if volumes[-1] > 0 and q.min() >= 0:
                axes[2].plot(edges / 3600, np.r_[0, volumes] / volumes[-1])
        axes[0].set_ylabel("Outlet flow (m³/s)")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].set_ylabel("Net export (10³ m³)")
        axes[2].set_ylabel("Fraction exported")
        axes[2].set_xlabel("Time since storm start (h)")
        for ax in axes:
            ax.axvline(6, ls=":", lw=1, color="0.4")
        metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": {}}
        with plt.rc_context({"svg.hashsalt": case}):
            for suffix, tags in metadata.items():
                path = target / f"{case}.{suffix}"
                fig.savefig(path, dpi=300, metadata=tags)
                if suffix == "svg":
                    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)


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
    original = [
        describe(item=i, settings=settings, status="original_window")
        for i in records.values()
    ]
    groups = grouped(
        directory=directory, records=records, target=target, expected=expected
    )
    complete = [
        describe(item=i, settings=settings, status=key[0])
        for key, items in groups.items()
        for i in items
    ]
    metrics.csv_write(file=target / "measured_windows.csv", rows=original + complete)
    hydro = json.loads((directory / "design.json").read_text())
    application = json.loads((directory / "application_settings.json").read_text())
    rows, checks = assessment.storm_rows(groups=groups, cfg=hydro)
    metrics.csv_write(file=target / "hydraulic_fidelity.csv", rows=rows)
    metrics.csv_write(file=target / "reference_checks.csv", rows=checks)
    front = statistics.frontiers(rows=rows, objectives=tuple(hydro["reference_limits"]))
    metrics.csv_write(file=target / "hydrograph_pareto.csv", rows=front)
    budgets.feasible_cost(
        rows=front,
        target=target,
        tolerances=application["tolerances_pct"],
        spacings=application["fv1_budget_grids_m"],
    )
    final_depth.process(
        groups=groups,
        directory=directory,
        target=target,
        settings=application["endpoint_depth"],
    )
    plot_curves(
        directory=directory,
        records=list(records.values())
        if not groups
        else [i for items in groups.values() for i in items],
    )
    catalogue.database(target=target)
    counted = {r["case"] for r in complete if r["horizon"] == "complete-native-metre"}
    text = [
        "# Native-metre storm qualification",
        "",
        f"{len(records)} original forecast windows have been measured. {len(counted)} of {len(expected)} physical domains have a matched recession receipt for every scheduled configuration.",
        "",
        "Outlet peaks are interval means of accepted face transport. The original-window table may end while substantial flow continues. Numerical error rankings require common recession endpoints and separate spatial, temporal and formulation checks for the observable.",
        "",
        "[Hydraulic windows and water budgets](measured_windows.csv) · [Hydrograph comparisons](hydraulic_fidelity.csv) · [Reference checks](reference_checks.csv) · [Depth and extent](recession_depth.csv) · [Recorded hydrographs](figures)",
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
