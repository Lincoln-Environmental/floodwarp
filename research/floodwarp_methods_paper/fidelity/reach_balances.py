"""Measure signed water accounts upstream and downstream of the physical gauge."""

import getopt
import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_map_endpoints
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def integrate(*, initial, depth, rain, soil, volume_x, volume_y, split, cell_area):
    ny, nx = initial.shape
    if (not 0 < split < nx or any(field.shape != initial.shape for field in (depth, rain, soil))
            or volume_x.shape != (ny, nx + 1) or volume_y.shape != (ny + 1, nx)):
        raise ValueError("Inconsistent section location, hydraulic state or face geometry")
    rows = []
    for name, start, stop in (("upstream", 0, split), ("downstream", split, nx)):
        columns = slice(start, stop)
        row = dict(reach=name)
        for key, field in (("initial", initial), ("rain", rain), ("infiltration", soil), ("storage", depth)):
            row[key + "_m3"] = float(field[:, columns].sum() * cell_area)
        row["west_eastward_m3"] = float(volume_x[:, start].sum())
        row["east_eastward_m3"] = float(volume_x[:, stop].sum())
        row["south_northward_m3"] = float(volume_y[0, columns].sum())
        row["north_northward_m3"] = float(volume_y[-1, columns].sum())
        row["residual_m3"] = (row["initial_m3"] + row["rain_m3"] - row["infiltration_m3"] - row["storage_m3"]
                              + row["west_eastward_m3"] - row["east_eastward_m3"]
                              + row["south_northward_m3"] - row["north_northward_m3"])
        rows.append(row)
    return rows


def verify_totals(*, reaches, event):
    for key in ("initial_m3", "rain_m3", "storage_m3", "infiltration_m3"):
        np.testing.assert_allclose(sum(r[key] for r in reaches), float(event[key]), rtol=2e-11, atol=1e-8)
    np.testing.assert_allclose(
        [r["east_eastward_m3"] for r in reaches],
        [float(event[key]) for key in ("net_section_m3", "net_outlet_m3")], rtol=2e-11, atol=1e-8,
    )
    for row in reaches:
        supply = sum(abs(row[key]) for key in ("initial_m3", "rain_m3", "west_eastward_m3", "east_eastward_m3",
                                               "north_northward_m3", "south_northward_m3"))
        if abs(row["residual_m3"]) > 1e-8 + 2e-11 * supply:
            raise ValueError(f"Open {row['reach']} reach water balance")


def measure(*, directory, row, rec, initial_path, initial_hash):
    evidence = dict(series_id=row["source_id"], end_s=rec["end_s"])
    path = native_map_endpoints.checkpoint(directory=directory, evidence=evidence)
    if runner.checksum(initial_path) != initial_hash:
        raise ValueError("Initial water field differs from its archived raster")
    initial = np.load(initial_path, allow_pickle=False)
    ny, nx = initial.shape
    dx = rec["dx"]
    gauge = float(row["section_x_m"])
    split = round(gauge / dx)
    if gauge != split * dx:
        raise ValueError("Internal section does not coincide with a hydraulic face")
    with np.load(path, allow_pickle=False) as state:
        direct = json.loads(state["_direct_manifest"].item())
        if (runner.digest(direct["sha256"]) != rec["checkpoint_array_set_sha256"]
                or runner.digest(direct["identity"]) != rec["identity_sha256"]):
            raise ValueError("Reach checkpoint and recorded endpoint identity differ")
        values = {key: state[key] for key in ("q", "rain_sum", "soil", "volume_x", "volume_y", "control")}
        for key, value in values.items():
            actual = array_sha(value)
            if actual != direct["sha256"][key] or (key != "control" and actual != rec["hashes"][key]):
                raise ValueError(f"Changed reach accounting field {key}")
        if values["control"][0] != rec["end_s"]:
            raise ValueError("Reach fields have the wrong physical endpoint")
    reaches = integrate(initial=initial, depth=values["q"][:, 0].reshape(ny, nx),
                       rain=values["rain_sum"].reshape(ny, nx), soil=values["soil"].reshape(ny, nx),
                       volume_x=values["volume_x"].reshape(ny, nx + 1),
                       volume_y=values["volume_y"].reshape(ny + 1, nx), split=split, cell_area=dx**2)
    verify_totals(reaches=reaches, event=row)
    return reaches


def archive(*, directory, event, recompute):
    cache = directory.parents[2] / ".fidelity-restarts/native_inputs"
    ident = event["source_id"]
    end = int(float(event["end_h"]) * 3600)
    field = directory / "endpoint_fields" / ident / f"{end}.npz"
    with np.load(field, allow_pickle=False) as data:
        rec = json.loads(data["receipt"].item())
    if (rec["id"] != ident or rec["case"] != event["case"] or rec["end_s"] != end
            or rec["dx"] != float(event["dx_m"]) or rec["other_sources_m3"] != 0):
        raise ValueError("Reach water account does not match the completed surface-water run")
    base = directory / "native_matrix" / rec["case"]
    parent = json.loads((base / rec["parent_id"] / "result.json").read_text())
    physical = parent["inputs"]["physical_sha"]
    terrain = json.loads((base / (physical + ".json")).read_text())
    if runner.digest(terrain) != physical:
        raise ValueError("Changed physical input receipt")
    name = f"depth_{int(rec['dx'])}m.npy"
    keys = ("case", "method", "factor", "dx_m", "end_h", "interval_s", "section_x_m", "source_id")
    identity = dict(event={k: event[k] for k in keys}, hydraulic_source=event["solver_source"],
                    physical_sha=physical, initial_raster_sha=terrain["files"][name],
                    endpoint_sha=runner.checksum(field), analysis_sha=runner.checksum(Path(__file__)))
    dest = directory / "reach_balances" / runner.digest(identity)[:24]
    receipt, table = dest / "receipt.json", dest / "water.csv"
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity or saved["water_sha"] != runner.checksum(table):
            raise ValueError("Archived reach balances changed")
        if not recompute:
            return
    reaches = measure(directory=directory, row=event, rec=rec,
                      initial_path=cache / runner.digest(terrain["inputs"]) / name, initial_hash=terrain["files"][name])
    if receipt.is_file():
        if metrics.csv_read(file=table) != [{k: str(v) for k, v in r.items()} for r in reaches]:
            raise ValueError("Archived reach balance does not reproduce")
    else:
        metrics.csv_write(file=table, rows=reaches)
        runner.write_json(path=receipt, data=dict(identity=identity, water_sha=runner.checksum(table)))
    print(event["case"], event["method"], event["factor"], event["end_h"], "h verified", dest.name)


def run(*, directory, cases, dx, method, dry, recompute):
    rows = metrics.csv_read(file=directory / "native_results/measured_windows.csv")
    current = runner.source_identity(root=directory.parents[2], study=directory.parent, stage="long-storms")
    for case in cases:
        selected = [r for r in rows if r["case"] == case and r["horizon"] == "complete-native-metre"
                    and float(r["dx_m"]) == dx and r["solver_source"] == current
                    and r["method"] in ("fv1", "adaptive", "hancock_dense", "hancock_tiles")
                    and (method is None or r["method"] == method)]
        print(case, dx, "m:", len(selected), "complete forecasts with the selected formulation")
        for row in selected:
            if not dry:
                archive(directory=directory, event=row, recompute=recompute)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    flags, extra = getopt.getopt(sys.argv[1:], "", ["case=", "dx=", "method=", "resume", "dry-run", "recompute"])
    if extra:
        raise ValueError(f"Unexpected reach-balance arguments {extra}")
    options = dict(flags)
    cases = ([options["--case"]] if "--case" in options else
             ["plain-709-moderate-1000x1000", "plain-709-intense-1000x1000"])
    run(directory=directory, cases=cases, dx=float(options.get("--dx", 1)), method=options.get("--method"),
        dry="--dry-run" in options, recompute="--recompute" in options)
