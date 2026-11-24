"""Measure accepted transport in and outside the surveyed opening's transverse band."""

import getopt
import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import native_archive, native_map_endpoints, restore_native
from research.floodwarp_methods_paper.terrain.generators.generate import array_sha


def profiles(*, native_bed, bed, volume, dx, window, face):
    west, south, width, height = window
    if (not west <= face <= west + width or any(v % dx for v in (west, south, width, height, face))
            or native_bed.shape != tuple(dx * n for n in bed.shape)
            or volume.shape != (bed.shape[0], bed.shape[1] + 1)
            or min(west, south) < 0 or south + height > native_bed.shape[0] or west + width > native_bed.shape[1]):
        raise ValueError("Transect, control window and hydraulic faces must coincide in physical space")
    native_crest = native_bed[:, west:west + width].max(axis=1)
    crest = bed[:, west // dx:(west + width) // dx].max(axis=1)
    return dict(accepted_eastward_m3=volume[:, face // dx].tolist(),
                crest_m=crest.tolist(), native_crest_m=native_crest.tolist())


def summarize(*, data, dx, window):
    _, south, _, height = window
    flux = np.asarray(data["accepted_eastward_m3"], dtype=float)
    crest = np.asarray(data["crest_m"], dtype=float)
    native = np.asarray(data["native_crest_m"], dtype=float)
    if (flux.ndim != 1 or crest.shape != flux.shape or native.shape != (len(flux) * dx,)
            or south < 0 or height <= 0 or (south + height) > len(native)
            or south % dx or height % dx or not np.isfinite(flux).all()
            or not np.isfinite(crest).all() or not np.isfinite(native).all()):
        raise ValueError("Invalid face or transverse crest profile")
    band = np.zeros(flux.size, dtype=bool)
    band[south // dx:(south + height) // dx] = True
    if band.all():
        raise ValueError("The control band spans the entire section")
    outside = (~band).repeat(dx)
    shift_cm = 100 * (crest.repeat(dx) - native)[outside]
    return dict(net_section_m3=float(flux.sum()), net_in_band_m3=float(flux[band].sum()),
                net_outside_band_m3=float(flux[~band].sum()),
                outside_median_crest_shift_cm=float(np.median(shift_cm)),
                outside_min_crest_shift_cm=float(shift_cm.min()), outside_max_crest_shift_cm=float(shift_cm.max()))


def contracts(*, directory, event, feature, face):
    case, ident = event["case"], event["source_id"]
    end = int(float(event["end_h"]) * 3600)
    parent = ident
    endpoint = None
    if event["horizon"] == "complete-native-metre":
        endpoint = directory / "endpoint_fields" / ident / f"{end}.npz"
        if endpoint.is_file():
            with np.load(endpoint, allow_pickle=False) as saved:
                settled = json.loads(saved["receipt"].item())
            if settled["id"] != ident or settled["end_s"] != end or settled["case"] != case:
                raise ValueError("Settled transect endpoint has different provenance")
            parent = settled["parent_id"]
        else:
            endpoint = None
    prefix = directory / "native_matrix" / case / parent
    proof = native_archive.prefix_evidence(folder=prefix)
    raw = json.loads((prefix / "result.json").read_text())
    if endpoint is None and end != raw["data"]["hours_completed"] * 3600:
        raise ValueError("Missing extended checkpoint for this transect time")
    dx = int(event["dx_m"])
    pre = directory / "native_features" / case / f"{dx}m" / feature["identity"] / "pre_run.json"
    geom = json.loads(pre.read_text())
    if (runner.digest(geom["identity"])[:24] != feature["identity"]
            or geom["bed_sha256"] != raw["data"]["forcing"]["terrain_sha256"]
            or raw["inputs"]["solver"] != event["solver_source"] or geom["case"] != case
            or raw["inputs"]["spec"]["dx"] != dx):
        raise ValueError("Surveyed window does not match the simulated bed")
    phy = raw["inputs"]["physical_sha"]
    terrain = json.loads((prefix.parent / (phy + ".json")).read_text())
    if runner.digest(terrain) != phy:
        raise ValueError("Changed terrain raster manifest")
    hashes = raw["data"]["hashes"]
    keys = ("case", "method", "factor", "dx_m", "end_h", "interval_s", "horizon", "source_id")
    identity = dict(event={k: event[k] for k in keys}, face_m=face, window_m=geom["control_window_m"],
                    physical_sha=phy, solver_source=event["solver_source"], feature_sha=runner.checksum(path=pre),
                    prefix_sha=proof["result_sha256"], volume_x_sha=settled["hashes"]["volume_x"] if endpoint else hashes["volume_x"],
                    endpoint_sha=runner.checksum(path=endpoint) if endpoint else None, analysis_sha=runner.checksum(path=Path(__file__)))
    return identity, terrain, hashes, bool(endpoint)


def measure(*, directory, event, identity, terrain, hashes, extended, index):
    cache = directory.parents[2] / ".fidelity-restarts"
    dx = int(event["dx_m"])
    beds = {}
    for spacing in {1, dx}:
        name = f"bed_{spacing}m.npy"
        f = cache / "native_inputs" / runner.digest(terrain["inputs"]) / name
        if runner.checksum(path=f) != terrain["files"][name]:
            raise ValueError("Changed elevation raster used for the crest profile")
        beds[spacing] = np.load(f, mmap_mode="r", allow_pickle=False)
    end = int(float(event["end_h"]) * 3600)
    if extended:
        checkpoint = native_map_endpoints.checkpoint(directory=directory, evidence=dict(series_id=event["source_id"], end_s=end))
    else:
        checkpoint = index[hashes["state"], hashes["volume_x"], hashes["soil"]]
    with np.load(checkpoint, allow_pickle=False) as state:
        v = state["volume_x"]
        if array_sha(v) != identity["volume_x_sha"] or state["control"][0] != end:
            raise ValueError("Changed accepted face volumes or physical endpoint")
    ny, nx = beds[dx].shape
    volume = v.reshape(ny, nx + 1)
    gauge = float(event["section_x_m"])
    if gauge % dx:
        raise ValueError("Published gauge does not coincide with a face")
    np.testing.assert_allclose(volume[:, [int(gauge / dx), nx]].sum(axis=0),
                               [float(event["net_section_m3"]), float(event["net_outlet_m3"])], rtol=2e-11, atol=1e-8)
    return profiles(native_bed=beds[1], bed=beds[dx], volume=volume, dx=dx, window=identity["window_m"], face=identity["face_m"])


def run(*, directory, cases, spacing, method, face, dry, recompute):
    products = directory / "native_results"
    features = {(r["case"], int(r["dx_m"])): r for r in metrics.csv_read(file=directory / "native_features/observations.csv")}
    events = [r for r in metrics.csv_read(file=products / "measured_windows.csv")
              if r["case"] in cases and (spacing is None or int(r["dx_m"]) == spacing)
              and r["horizon"] in ("original_window", "complete-native-metre") and not int(r["timing_repeat"] or 0)
              and r["method"] in ("fv1", "hancock_dense", "adaptive") and (method is None or r["method"] == method)]
    print(len(events), "measured forecast windows at the requested fixed transect", face, "m")
    if dry:
        return
    index = None
    for event in events:
        dx = int(event["dx_m"])
        identity, terrain, hashes, extended = contracts(directory=directory, event=event, feature=features[event["case"], dx], face=face)
        dest = directory / "embankment_transport" / runner.digest(identity)[:24] / "profile.json"
        if dest.exists():
            archive = json.loads(dest.read_text())
            if archive["identity"] != identity or runner.digest(archive["profile"]) != archive["profile_sha"]:
                raise ValueError("Changed archived embankment transport profile")
            if not recompute:
                continue
        if not extended and index is None:
            index = restore_native.prefix_index(root=directory.parents[2] / ".fidelity-restarts")
        data = measure(directory=directory, event=event, identity=identity, terrain=terrain, hashes=hashes, extended=extended, index=index)
        summarize(data=data, dx=dx, window=identity["window_m"])
        if dest.exists():
            if data != archive["profile"]:
                raise ValueError("Recomputed embankment profile differs from the archived measurement")
        else:
            runner.write_json(path=dest, data=dict(identity=identity, profile=data, profile_sha=runner.digest(data)))


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    flags, extra = getopt.getopt(sys.argv[1:], "", ["case=", "dx=", "method=", "face-m=", "dry-run", "resume", "recompute", "force"])
    if extra:
        raise ValueError(f"Unexpected transect arguments {extra}")
    options = dict(flags)
    cases = [options["--case"]] if "--case" in options else ["plain-709-moderate-1000x1000", "plain-709-intense-1000x1000"]
    run(directory=directory, cases=cases, spacing=int(options["--dx"]) if "--dx" in options else None,
        method=options.get("--method"), face=int(options.get("--face-m", 880)), dry="--dry-run" in options,
        recompute="--recompute" in options or "--force" in options)
