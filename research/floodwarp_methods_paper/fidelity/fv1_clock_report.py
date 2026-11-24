"""Render the FV1 time-refinement experiment from accepted CPU transport records."""

import json
from pathlib import Path

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import assessment, fv1_clock, hydrograph


def end_comparison(*, observations, source):
    peers = {r["id"]: r for r in observations if int(r["end_s"]) == source["end_s"]}
    normal, reduced = peers[source["normal"]], peers[source["reduced"]]
    for field in ("case", "dx_m", "device", "method", "interval_s", "end_s"):
        if normal[field] != reduced[field]:
            raise ValueError(f"Incompatible FV1 temporal controls in {field}")
    if any(r["both_stations_passed"] != "True" for r in peers.values()):
        raise ValueError("Unsettled FV1 temporal comparison")
    if not 0 < source["reduced_cfl"] < source["normal_cfl"]:
        raise ValueError("Invalid FV1 timestep refinement")
    result = dict(case=normal["case"], dx_m=normal["dx_m"], interval_s=normal["interval_s"],
                  end_h=int(normal["end_s"])/3600, device=normal["device"], method=normal["method"],
                  normal_cfl=source["normal_cfl"], reduced_cfl=source["reduced_cfl"],
                  normal_id=normal["id"], reduced_id=reduced["id"])
    for field in ("peak_m3s", "net_export_m3", "storage_m3", "infiltration_m3", "rain_m3", "residual_m3", "steps"):
        first, second = float(normal[field]), float(reduced[field])
        result["normal_" + field] = first
        result["reduced_" + field] = second
        result["change_" + field] = second - first
        if field in ("peak_m3s", "net_export_m3"):
            result["pct_change_" + field] = 100 * (second / first - 1) if first > 0 else None
    result["peak_interval_shift_s"] = (
        int(reduced["peak_interval_end_s"]) - int(normal["peak_interval_end_s"])
        if reduced["peak_interval_end_s"] and normal["peak_interval_end_s"] else None
    )
    return result


def report(*, directory, output, write_table):
    program = directory / "fv1_clock.py"
    program_hash = runner.checksum(path=program)
    design_path = directory / "design.json"
    design = json.loads(design_path.read_text())
    inputs, comparisons = [program, design_path, Path(hydrograph.__file__), Path(assessment.__file__)], {}
    controls = {r["id"] for r in metrics.csv_read(file=directory / "strong_storms/completed.csv")}
    for manifest in sorted((directory / "fv1_temporal").glob("*/sources.json")):
        source = json.loads(manifest.read_text())
        if source["implementation"] != program_hash or source["historical_normal"] not in controls:
            continue
        if runner.digest(source)[:24] != manifest.parent.name:
            raise ValueError("FV1 temporal comparison source identity mismatch")
        observations = manifest.parent / "observations.csv"
        rows = metrics.csv_read(file=observations)
        entries = {(r["id"], int(r["end_s"])): r for r in rows}
        if len(entries) != len(rows):
            raise ValueError("Duplicate FV1 temporal observation")
        trajectories = {}
        for ident, windows in source["windows"].items():
            folder = directory / "strong_storms" / ident
            contract = folder / "inputs.json"
            inputs.append(contract)
            setup = json.loads(contract.read_text())
            if runner.digest(setup["identity"])[:24] != ident:
                raise ValueError("FV1 input identity mismatch")
            frames = []
            for end in sorted(map(int, windows)):
                expected = windows[str(end)]
                raw = folder / f"{end}.json"
                if runner.checksum(path=raw) != expected:
                    raise ValueError("Changed FV1 temporal observation window")
                inputs.append(raw)
                block = json.loads(raw.read_text())
                if block["start_s"] != (frames[-1]["t_s"] if frames else 0) or block["end_s"] != end:
                    raise ValueError("Discontinuous FV1 temporal observations")
                frames.extend(block["frames"])
                values = fv1_clock.observe(
                    forecast=dict(folder=folder, frames=frames, budget=block["budget"], prescribed=setup["prescribed_m3"]),
                    stopping=setup["identity"]["stopping"],
                )
                saved = entries.pop((ident, end))
                spec = setup["identity"]["spec"]
                for field, value in dict(case=spec["case"], method=spec["method"], dx_m=spec["dx"],
                                         factor=spec["factor"], interval_s=spec["interval"],
                                         device=setup["identity"]["settings"]["device"], **values).items():
                    if isinstance(value, float):
                        np.testing.assert_allclose(float(saved[field]), value, atol=1e-10, rtol=1e-12)
                    elif saved[field] != ("" if value is None else str(value)):
                        raise ValueError(f"FV1 temporal observation differs from accepted record in {field}")
            trajectories[ident] = frames
        if entries:
            raise ValueError("FV1 temporal observations lack source windows")
        row = end_comparison(observations=rows, source=source)
        normal = trajectories[source["normal"]]
        reduced = trajectories[source["reduced"]]
        ends = [r["t_s"] for r in normal]
        np.testing.assert_array_equal(ends, [r["t_s"] for r in reduced])
        for station in ("outlet", "section"):
            field = station + "_m3s"
            shape = hydrograph.evaluate(edges=[0, *ends], model=[r[field] for r in reduced],
                                        reference=[r[field] for r in normal], config=design)
            row[station + "_status"] = shape["reference_status"]
            for key in ("shape_tv_pct", "width80_bias_pct", "centroid_delay_s", "wasserstein_s"):
                row[station + "_" + key] = shape.get(key) if assessment.limits(row=shape, key=key) else None
        if row["case"] in comparisons:
            raise ValueError("Duplicate current FV1 temporal comparison")
        comparisons[row["case"]] = row
        inputs.extend((manifest, observations))
    rows = list(comparisons.values())
    result = output / "fv1_temporal_sensitivity.csv"
    if write_table:
        metrics.csv_write(file=result, rows=rows)
    elif metrics.csv_read(file=result) != [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]:
        raise ValueError("Archived FV1 sensitivity table differs from its inputs; regenerate native-products")
    if not rows:
        return [], inputs + [result]
    text = ["", "### FV1 sensitivity to the accepted timestep", "",
            "The same 1 km² plain, 10 m represented bed, initial state and six-hour rainfall were run on CPU "
            "at two CFL values. Changes below give the reduced-CFL result minus the ordinary result. "
            "Each endpoint passed the two-hour recession checks at both gauges.", "",
            "| Storm | End (h) | CFL | Accepted steps | Reduced-CFL steps | Peak change (%) | Peak interval shift (s) | Export change (m³) |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in sorted(comparisons):
        r = comparisons[row]
        peak = r["pct_change_peak_m3s"]
        shift = r["peak_interval_shift_s"]
        text.append(f"| {r['case']} | {r['end_h']:g} | {r['normal_cfl']:g} → {r['reduced_cfl']:g} | "
                    f"{r['normal_steps']:,.0f} | {r['reduced_steps']:,.0f} | "
                    f"{format(peak, '+.4f') if peak is not None else 'unresolved'} | "
                    f"{format(shift, '+d') if shift is not None else 'unresolved'} | {r['change_net_export_m3']:+.2f} |")
    text.extend(["", "The observed sensitivity applies to these 10 m runs. FV1 temporal sensitivity at 1 m "
                 "requires a separate refinement at that spacing. "
                 "[Water accounts, width, shape, timing and paired run identities](../native_results/fv1_temporal_sensitivity.csv)"])
    return text, inputs + [result]
