"""Portable depth, maximum and mass-accounting evidence from a completed restart."""

from pathlib import Path
import json

import numpy as np
from scipy.ndimage import label
from research.floodwarp_methods_paper import runner
from validation.hancock_long_storm_review.accuracy import block_mean


def endpoint_path(*, directory, ident, end_s):
    return directory / "endpoint_fields" / ident / f"{end_s}.npz"


def capture(*, directory, data, supports):
    ident, end = data["id"], data["end_s"]
    target = endpoint_path(directory=directory, ident=ident, end_s=end)
    if target.is_file():
        return target
    path = directory.parents[2] / ".fidelity-restarts" / ident / f"{end}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing completed hydraulic restart for {ident} at {end} s"
        )
    spec = data["spec"]
    dx = spec["dx"]
    support = supports.get(spec["case"], spec["support"])
    factor = round(support / dx)
    if not np.isclose(factor * dx, support):
        raise ValueError("Endpoint observation grid must nest the native grid")
    output = {}
    with np.load(path, allow_pickle=False) as state:
        meta = json.loads(state["metadata"].item())
        verified = json.loads(state["_direct_manifest"].item())
        names = verified["sha256"]
        if "q" in names:
            time = meta["time"]
            shape = meta["configuration"]["ny"], meta["configuration"]["nx"]
            h = state["q"][:, 0].reshape(shape)
            maximum = state["maximum"].reshape(shape)
            boundary_inflow = float(state["inflow"].sum())
            boundary_outflow = float(state["outflow"].sum())
            others = 0.0
            hashes = {
                k: names[k]
                for k in (
                    "q",
                    "maximum",
                    "volume_x",
                    "volume_y",
                    "rain_sum",
                    "soil",
                    "dt_histogram",
                )
            }
        else:
            time = meta["host"]["simulation_time"]
            h = state["water_depth"]
            maximum = state["max_water_depth_m"].reshape(h.shape)
            rain, infiltrated, other, boundary_inflow, boundary_outflow, additional = (
                state["cumulative_totals"]
            )
            others = float(other + additional)
            np.testing.assert_allclose(
                [rain, infiltrated],
                [data["rain_m3"], data["infiltration_m3"]],
                rtol=1e-10,
                atol=1e-10,
            )
            hashes = {
                k: names[k]
                for k in (
                    "water_depth",
                    "qx",
                    "qy",
                    "max_water_depth_m",
                    "cumulative_net_qx_volume_m3",
                    "cumulative_net_qy_volume_m3",
                    "cumulative_infiltration",
                )
            }
        if (
            time != end
            or np.min(h) < 0.0
            or not np.isfinite(h).all()
            or not np.isfinite(maximum).all()
        ):
            raise ValueError("Invalid endpoint time or depth")
        storage = float(np.sum(h) * dx**2)
        np.testing.assert_allclose(storage, data["storage_m3"], rtol=2e-11, atol=1e-9)
        residual = (
            meta["initial_volume"]
            + data["rain_m3"]
            + others
            + boundary_inflow
            - data["infiltration_m3"]
            - boundary_outflow
            - storage
        )
        scale = max(1.0, meta["initial_volume"] + data["rain_m3"] + boundary_inflow)
        if abs(residual - data["residual_m3"]) > 1e-11 * scale:
            raise ArithmeticError(
                "Saved boundary and storage accounting disagree with the hydraulic receipt"
            )
        output["depth"] = block_mean(values=h, factor=factor)
        output["maximum"] = block_mean(values=maximum, factor=factor)
        components = {}
        for cm in (1, 5, 10):
            wet = h >= cm / 100
            output[f"wet_fraction_{cm}cm"] = block_mean(values=wet, factor=factor)
            components[str(cm)] = int(label(wet)[1])
        description = dict(
            id=ident,
            parent_id=data["parent_id"],
            end_s=end,
            case=spec["case"],
            dx=dx,
            method=spec["method"],
            support=support,
            initial_m3=meta["initial_volume"],
            rain_m3=data["rain_m3"],
            infiltration_m3=data["infiltration_m3"],
            inflow_m3=float(boundary_inflow),
            outflow_m3=float(boundary_outflow),
            other_sources_m3=others,
            storage_m3=storage,
            residual_m3=residual,
            native_depth_min_m=float(h.min()),
            native_depth_max_m=float(h.max()),
            components=components,
            hashes=hashes,
            identity_sha256=runner.digest(verified["identity"]),
            checkpoint_array_set_sha256=runner.digest(names),
        )
    output["receipt"] = np.asarray(
        json.dumps(description, sort_keys=True, allow_nan=False)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".pending.npz")
    np.savez_compressed(tmp, **output)
    tmp.replace(target)
    return target


def capture_existing(*, directory, supports):
    count = 0
    for receipt in sorted((directory / "extended_runs").glob("*/series.json")):
        data = json.loads(receipt.read_text())
        capture(directory=directory, data=data, supports=supports)
        count += 1
    return count


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    supports = json.loads((directory / "spatial_settings.json").read_text())[
        "support_m"
    ]
    print(
        capture_existing(directory=directory, supports=supports),
        "final states and water budgets archived",
    )
