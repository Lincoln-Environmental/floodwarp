"""Analyse fixed physical sections and separately conserved regions in completed storm windows."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    final_depth,
    catalogue,
    native_tail,
    outlet_observables,
)
from research.floodwarp_methods_paper.terrain.generators.mean_dem import mean_cells


def crest(*, q, times, count, stride):
    starts = np.arange(0, len(q) - count + 1, stride)
    if len(starts) == 0:
        raise ValueError("Incomplete peak reporting interval")
    sums = np.r_[0.0, np.cumsum(q)]
    means = (sums[starts + count] - sums[starts]) / count
    k = int(np.argmax(means))
    return float(means[k]), int(times[starts[k] + count - 1])


def strip(*, row):
    return {
        0: dict(transfer_m3=row["gauge"]),
        40: dict(
            transfer_m3=row["tail_in"],
            storage_m3=row["last40m_storage_m3"],
            rain_m3=row["last40m_rain_m3"],
            infiltration_m3=row["last40m_infiltration_m3"],
        ),
    }


def strip_change(*, before, after):
    a, b = strip(row=before), strip(row=after)
    for data in (a, b):
        data[0].update(storage_m3=0.0, rain_m3=0.0, infiltration_m3=0.0)
    return native_tail.changes(before=a, after=b)[1]


def checkpoint_fields(*, state_dir, hours, grid, footprint):
    import hashlib

    with np.load(state_dir / f"{int(hours * 3600)}.npz") as archive:
        width = footprint // grid
        metadata = json.loads(archive["metadata"].item())
        ny = metadata["configuration"]["ny"]
        hashes = json.loads(archive["_direct_manifest"].item())["sha256"]
        if metadata["time"] != hours * 3600:
            raise ValueError("Depth snapshot is from another physical time")
        q, maximum = archive["q"], archive["maximum"]
        for name, field in (("q", q), ("maximum", maximum)):
            if hashlib.sha256(memoryview(field).cast("B")).hexdigest() != hashes[name]:
                raise ValueError("Changed depth evidence in " + name)
        return (
            q[:, 0].reshape(ny, -1)[:, :width],
            maximum.reshape(ny, -1)[:, :width],
        )


def water_differences(*, summaries):
    differences = []
    for first, second in combinations(summaries, 2):
        if (first["dx_m"], first["extension_m"]) != (
            second["dx_m"],
            second["extension_m"],
        ):
            continue
        if second["method"] == "fv1":
            first, second = second, first
        row = dict(
            first=first["run_id"],
            second=second["run_id"],
            dx_m=first["dx_m"],
            extension_m=first["extension_m"],
            first_method=first["method"],
            second_method=second["method"],
        )
        for key in (
            "initial_storage_m3",
            "footprint_rain_m3",
            "infiltration_inside_rain_m3",
            "infiltration_outside_rain_m3",
            "footprint_infiltration_m3",
            "footprint_storage_m3",
            "gauge_m3",
        ):
            row[key] = first[key] - second[key]
        row["unexplained_export_m3"] = (
            row["initial_storage_m3"]
            + row["footprint_rain_m3"]
            - row["infiltration_inside_rain_m3"]
            - row["infiltration_outside_rain_m3"]
            - row["footprint_storage_m3"]
            - row["gauge_m3"]
        )
        differences.append(row)
    return differences


def regional_contrasts(*, directory, peers, hour, thresholds, strip_m):
    import hashlib

    rows = []
    for first in peers:
        if first["method"] != "fv1" or first["cfl_factor"] != 1:
            continue
        for second in peers:
            dx, extension = first["dx_m"], first["extension_m"]
            if (second["method"], second["cfl_factor"], second["dx_m"], second["extension_m"]) != (
                    "hancock_adaptive", 1, dx, extension):
                continue
            width = first["footprint_width_m"] // dx
            states = []
            for member in (first, second):
                path = directory.parents[2] / ".fidelity-restarts/outlet_storms" / member["run_id"] / f"{int(hour * 3600)}.npz"
                with np.load(path) as saved:
                    sealed = json.loads(saved["_direct_manifest"].item())["sha256"]
                    metadata = json.loads(saved["metadata"].item())
                    if metadata["time"] != hour * 3600:
                        raise ValueError("Regional contrast references the wrong hour")
                    state = {}
                    for name in ("q", "soil", "rain_sum"):
                        field = saved[name]
                        if hashlib.sha256(memoryview(field).cast("B")).hexdigest() != sealed[name]:
                            raise ValueError("Changed archived regional field " + name)
                        values = field[:, 0] if name == "q" else field
                        state[name] = values.reshape(metadata["configuration"]["ny"], -1)[:, :width]
                    states.append(state)
            a, b = states
            footprint = a["rain_sum"] > 0
            np.testing.assert_array_equal(footprint, b["rain_sum"] > 0)
            soil_delta = a["soil"] - b["soil"]
            disagreements = {}
            for depth in thresholds:
                wet_a, wet_b = a["q"] > depth, b["q"] > depth
                disagreements[round(depth * 100)] = wet_a & ~wet_b, wet_b & ~wet_a
            column = np.arange(width)[None, :]
            regions = (
                ("upstream of internal gauge", column < width // 2),
                ("internal gauge to final strip", (column >= width // 2) & (column < width - strip_m // dx)),
                ("final strip", column >= width - strip_m // dx),
            )
            for label, reach in regions:
                for side, rain in (("inside", footprint), ("outside", ~footprint)):
                    cells = reach & rain
                    record = dict(fv1_id=first["run_id"], hancock_id=second["run_id"], hour=hour,
                                  dx_m=dx, extension_m=extension, reach=label, rainfall_footprint=side,
                                  region_m2=int(cells.sum()) * dx**2,
                                  fv1_minus_hancock_infiltration_m3=float(np.sum(soil_delta, where=cells) * dx**2))
                    for cm, (fv1_only, hancock_only) in disagreements.items():
                        record[f"fv1_only_above_{cm}cm_m2"] = int(np.count_nonzero(cells & fv1_only)) * dx**2
                        record[f"hancock_only_above_{cm}cm_m2"] = int(np.count_nonzero(cells & hancock_only)) * dx**2
                    rows.append(record)
    return rows


def spatial_differences(*, directory, peers, hour, thresholds):
    differences = []
    states = {}
    for peer in peers:
        cache = (
            directory.parents[2] / ".fidelity-restarts/outlet_storms" / peer["run_id"]
        )
        states[peer["run_id"], peer["dx_m"]] = checkpoint_fields(
            state_dir=cache,
            hours=hour,
            grid=peer["dx_m"],
            footprint=peer["footprint_width_m"],
        )
    for a, b in combinations(peers, 2):
        if a["extension_m"] != b["extension_m"] and (
            a["dx_m"],
            a["method"],
            a["cfl_factor"],
        ) != (b["dx_m"], b["method"], b["cfl_factor"]):
            continue
        grid = max(a["dx_m"], b["dx_m"])
        fields = []
        for peer in (a, b):
            key = peer["run_id"], grid
            if key not in states:
                states[key] = tuple(
                    mean_cells(values=v, factor=grid // peer["dx_m"])
                    for v in states[peer["run_id"], peer["dx_m"]]
                )
            fields.append(states[key])
        for label, first, second in zip(("instant", "maximum"), *fields, strict=True):
            for threshold in thresholds:
                wet = (first > threshold) | (second > threshold)
                differences.append(
                    dict(
                        a_id=a["run_id"],
                        b_id=b["run_id"],
                        hour=hour,
                        field=label,
                        a_method=a["method"],
                        b_method=b["method"],
                        a_dx_m=a["dx_m"],
                        b_dx_m=b["dx_m"],
                        a_extension_m=a["extension_m"],
                        b_extension_m=b["extension_m"],
                        dx_m=grid,
                        threshold_m=threshold,
                        common_wet_union_m2=int(wet.sum()) * grid**2,
                        wet_union_depth_rmse_m=float(
                            np.sqrt(np.mean((first[wet] - second[wet]) ** 2))
                        )
                        if wet.any()
                        else None,
                    )
                )
    return differences, states


def spatial_qualification(*, peers, states, hour, thresholds, settings):
    rows = []
    for extension in sorted({p["extension_m"] for p in peers}):
        group = [p for p in peers if p["extension_m"] == extension]
        by_key = {(p["method"], p["dx_m"], p["cfl_factor"]): p for p in group}
        spacing = min(p["dx_m"] for p in group)
        support = max(p["dx_m"] for p in group)
        reference = by_key.get(("adaptive", spacing, 1.0))
        if reference is None:
            continue
        controls = dict(
            space=by_key.get(("adaptive", support, 1.0)) if support > spacing else None,
            time=by_key.get(("adaptive", spacing, 0.5)),
            cross=by_key.get(("hancock_adaptive", spacing, 1.0)),
        )
        if controls["time"] and controls["time"]["steps"] <= reference["steps"]:
            controls["time"] = None
        for p in group:
            key = p["run_id"], support
            if key not in states:
                states[key] = tuple(
                    mean_cells(values=v, factor=support // p["dx_m"])
                    for v in states[p["run_id"], p["dx_m"]]
                )
        truth = states[reference["run_id"], support]
        depth_limit = max(
            settings["depth_floor_m"],
            settings["depth_relative"] * float(np.sqrt(np.mean(truth[0] ** 2))),
        )
        limits = dict(
            depth_rmse_m=depth_limit,
            wet_depth_rmse_m=depth_limit,
            maximum_depth_rmse_m=settings["maximum_depth_m"],
        )
        for threshold in thresholds:
            for prefix in ("", "maximum_"):
                limits[f"{prefix}extent_error_{round(100 * threshold)}cm"] = settings[
                    "extent_error"
                ]
        values = {
            p["run_id"]: final_depth.difference(
                candidate=states[p["run_id"], support],
                reference=truth,
                support=support,
                thresholds=thresholds,
            )
            for p in group
        }
        for observable, limit in limits.items():
            spreads = {
                name: abs(values[p["run_id"]][observable])
                if p and values[p["run_id"]][observable] is not None
                else None
                for name, p in controls.items()
            }
            sensitivity = (
                max(spreads.values())
                if all(value is not None for value in spreads.values())
                else None
            )
            for p in group:
                error = values[p["run_id"]][observable]
                represented = (
                    "extent_error_" not in observable
                    or values[p["run_id"]][
                        observable.replace("extent_error", "reference_wet_area") + "_m2"
                    ]
                    > 0
                )
                rows.append(
                    dict(
                        method=p["method"],
                        dx_m=p["dx_m"],
                        cfl_factor=p["cfl_factor"],
                        extension_m=extension,
                        run_id=p["run_id"],
                        hour=hour,
                        support_m=support,
                        reference_id=reference["run_id"],
                        metric=observable,
                        error=error,
                        reference_sensitivity=sensitivity,
                        criterion=limit,
                        qualified=bool(
                            p["dx_m"] > spacing
                            and p["cfl_factor"] == 1
                            and error is not None
                            and sensitivity is not None
                            and sensitivity <= limit
                            and represented
                        ),
                        **spreads,
                        **{
                            name + "_id": c["run_id"] if c else None
                            for name, c in controls.items()
                        },
                    )
                )
    return rows


def run(*, directory, dest, end, rainfall_end, extent_hour, thresholds):
    summaries, strips, snapshots, peers = [], [], [], []
    experiments = []
    references = {}
    sources = {}
    current = runner.source_identity(
        root=directory.parents[2], study=directory.parent, stage="long-storms"
    )
    driver = runner.checksum(directory / "outlet_storms.py")
    for path in sorted((directory / "outlet_storms").glob(f"*/summary_{end}.json")):
        run = path.parent
        contract = json.loads((run / "inputs.json").read_text())
        identity, initial = contract["identity"], contract["initial"]
        if runner.digest(identity)[:24] != run.name:
            raise ValueError("Storm inputs do not reproduce the archived run ID")
        if (identity["numerical"], identity["source"]) != (current, driver):
            continue
        physical = dict(
            terrain=identity["terrain"],
            fields=contract["fields"],
            forcing=identity["forcing"],
        )
        key = identity["dx"], identity["extension_m"]
        if references.setdefault(key, physical) != physical:
            raise ValueError("Physical storm inputs changed between methods")
        files = [
            p
            for p in sorted(run.glob("[0-9]*.json"), key=window_end)
            if int(p.stem) <= end
        ]
        sources[run.name] = {
            p.name: runner.checksum(p) for p in [run / "inputs.json", path, *files]
        }
        case = dict(
            run_id=run.name,
            method=identity["method"],
            dx_m=identity["dx"],
            extension_m=identity["extension_m"],
            cfl_factor=identity["cfl_factor"],
            footprint_width_m=identity["forcing"]["shape"][1] * identity["dx"],
            end_h=end / 3600,
        )
        rows = [
            r
            for p in sorted(run.glob("[0-9]*.json"), key=window_end)
            if int(p.stem) <= end
            for r in json.loads(p.read_text())["frames"]
        ]
        times = np.array([r["t_s"] for r in rows])
        cadence = identity["settings"]["interval_s"]
        np.testing.assert_array_equal(times, np.arange(cadence, end + 1, cadence))
        q = np.array([r["outlet_m3s"] for r in rows])
        peaks = {}
        for label, minutes, stride in (
            ("minute", 1, 1),
            ("fixed_15min", 15, 15),
            ("moving_15min", 15, 1),
        ):
            value, t = crest(
                q=q,
                times=times,
                count=minutes * 60 // cadence,
                stride=stride * 60 // cadence,
            )
            peaks[label + "_peak_m3s"], peaks[label + "_end_s"] = value, t
        frames = {r["t_s"]: r for r in rows}
        frames[0] = initial
        for name, start, stop in (
            ("crest_minute", peaks["minute_end_s"] - cadence, peaks["minute_end_s"]),
            ("post_rain_recession", rainfall_end, end),
        ):
            delta = strip_change(before=frames[start], after=frames[stop])
            strips.append(dict(case, period=name, start_s=start, end_s=stop, **delta))
        final = rows[-1]
        stats = json.loads(path.read_text())
        if (
            stats["final"] != final
            or stats["steps"] != sum(row["steps"] for row in rows)
            or stats["end_s"] != end
            or stats["frames"] != len(rows)
        ):
            raise ValueError("Storm summary differs from the accepted transport record")
        experiments.append((contract, rows, stats, run))
        summaries.append(
            dict(
                case,
                **peaks,
                initial_storage_m3=initial["footprint_storage_m3"],
                recession_passed=stats["recession_passed"],
                hydraulic_s=stats["hydraulic_s"],
                steps=stats["steps"],
                retries=sum(r["rejected"] for r in rows),
                **{
                    k: v
                    for k, v in final.items()
                    if k.startswith(("extension_", "footprint_", "infiltration_"))
                },
                gauge_m3=final["gauge"],
                outer_boundary_m3=final["boundary"],
            )
        )
        for hour in (extent_hour, end / 3600):
            record = frames[int(hour * 3600)]
            snapshots.append(
                dict(
                    case,
                    hour=hour,
                    **{k: v for k, v in record.items() if "_above_" in k},
                )
            )
        peers.append(dict(case, steps=stats["steps"]))
    dest.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("events", summaries),
        ("last_40m", strips),
        ("extent", snapshots),
    ):
        metrics.csv_write(file=dest / f"{name}.csv", rows=rows)
    differences, spatial_checks, contrasts = [], [], []
    settings = json.loads((directory / "application_settings.json").read_text())[
        "endpoint_depth"
    ]
    for hour in (extent_hour, end / 3600):
        delta, states = spatial_differences(
            directory=directory, peers=peers, hour=hour, thresholds=thresholds
        )
        differences.extend(delta)
        spatial_checks.extend(
            spatial_qualification(
                peers=peers,
                states=states,
                hour=hour,
                thresholds=thresholds,
                settings=settings,
            )
        )
        del states
        if experiments:
            contrasts.extend(regional_contrasts(directory=directory, peers=peers, hour=hour, thresholds=thresholds,
                                                strip_m=experiments[0][0]["identity"]["settings"]["strip_m"]))
    metrics.csv_write(file=dest / "regional_contrasts.csv", rows=contrasts)
    metrics.csv_write(file=dest / "spatial_reference_checks.csv", rows=spatial_checks)
    outlet_observables.spatial_decisions(
        directory=directory, destination=dest, experiments=experiments, checks=spatial_checks
    )
    metrics.csv_write(file=dest / "depth_disagreement.csv", rows=differences)
    metrics.csv_write(
        file=dest / "water_differences.csv", rows=water_differences(summaries=summaries)
    )
    outlet_observables.qualify(
        directory=directory, destination=dest, experiments=experiments
    )
    runner.write_json(
        path=dest / "sources.json",
        data=dict(solver=current, driver=driver, end_s=end, storms=sources),
    )


def window_end(path):
    return int(path.stem)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    end = 16 * 3600
    rainfall_end = 6 * 3600
    extent_hour = 8
    thresholds = (0.05, 0.10)
    from research.floodwarp_methods_paper.fidelity import repaired_outlet_report

    destination = directory / "outlet_results"
    run(
        directory=directory,
        dest=destination,
        end=end,
        rainfall_end=rainfall_end,
        extent_hour=extent_hour,
        thresholds=thresholds,
    )
    repaired_outlet_report.run(directory=directory)
    catalogue.database(target=destination)
    for endpoint in sorted(
        {
            int(p.stem.split("_")[1])
            for p in (directory / "outlet_storms").glob("*/summary_*.json")
        }
    ):
        if endpoint > end:
            run(
                directory=directory,
                dest=destination / f"end_{endpoint}",
                end=endpoint,
                rainfall_end=rainfall_end,
                extent_hour=extent_hour,
                thresholds=thresholds,
            )
            catalogue.database(target=destination / f"end_{endpoint}")
