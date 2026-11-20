"""Measure whole-catchment 1 m wet-state feasibility on the installed GPU."""

from pathlib import Path
import gc
import json

import numpy as np
import warp as wp

from floodwarp.fv2_solver import FV2Solver
from floodwarp.full_swe_solver import FullSWESolver
from validation.fv1_riemann import report_json
from validation.fv2_terrain import blocks, prolong


def distribute(*, bed, coarse, factor, dry):
    """Level water over each fine bed patch at fixed parent volume and both discharges."""
    ny, nx = coarse.shape[:2]
    total = factor**2
    patches = bed.reshape(ny, factor, nx, factor).transpose(0, 2, 1, 3).reshape(ny, nx, total)
    ordered = np.sort(patches, axis=-1)
    relative = ordered - ordered[:, :, :1]
    levels = (relative.cumsum(axis=-1) + total * coarse[:, :, 0:1]) / np.arange(1, total + 1)
    count = np.maximum(1, np.sum(levels > relative, axis=-1))
    surface = np.take_along_axis(levels, count[:, :, None] - 1, axis=2) + ordered[:, :, :1]
    h = np.maximum(0.0, surface - patches)
    h = np.where(coarse[:, :, 0:1] > 0.0, h, 0.0)
    eligible = np.where(h > dry, h, 0.0)
    supplied = eligible.sum(axis=2, keepdims=True)
    share = np.divide(total * eligible, supplied, where=supplied > 0.0, out=np.zeros_like(h))
    fine = np.stack((h, share * coarse[:, :, 1:2], share * coarse[:, :, 2:3]), axis=-1)
    fine = (
        fine.reshape(ny, nx, factor, factor, 3).transpose(0, 2, 1, 3, 4).reshape(ny * factor, nx * factor, 3)
    )
    np.testing.assert_allclose(blocks(value=fine, factor=factor), coarse, atol=1.0e-12, rtol=1.0e-13)
    return fine


def native_state(*, source, checkpoint, factor):
    with np.load(checkpoint) as a:
        meta = json.loads(str(a["metadata"]))
        c = meta["configuration"]
        shape = (c["ny"], c["nx"])
        mask = prolong(value=a["active"].reshape(shape).astype(bool), factor=factor)
        with np.load(source) as n:
            bed = np.where(mask, n["subgrid_bed_elevation"][::-1], 0.0).astype(float)
            rough = np.where(mask, n["subgrid_manning_n"][::-1], 0.0).astype(float)
        q = distribute(bed=bed, coarse=a["q"].reshape(*shape, 3), factor=factor, dry=c["dry_depth"])
        fields = {
            k: prolong(value=a[k].reshape(shape), factor=factor)
            for k in ("f0", "fd", "conductivity", "soil", "infil", "rain", "multiplier")
        }
        by = np.zeros((bed.shape[0] + 1, bed.shape[1]), np.int32)
        coarse = a["types_y"].reshape(shape[0] + 1, shape[1])
        by[::factor] = np.repeat(np.where(coarse == 3, 3, 0), factor, axis=1)
        if np.any(a["stages_y"][a["types_y"] == 3]):
            raise ValueError("Expected stage-zero coastal boundary")
        return dict(
            config=dict(c, nx=bed.shape[1], ny=bed.shape[0], dx=c["dx"] / factor, dy=c["dy"] / factor),
            active=mask,
            bed=bed,
            manning=rough,
            q=q,
            by=by,
            time=meta["time"],
            fields=fields,
            rain_times=a["rain_times"],
            rain_rates=a["rain_rates"],
        )


def probe(*, inputs, method, device, warmup, duration, repeats):
    config = inputs["config"]
    cls = FullSWESolver if method == "fv1" else FV2Solver
    s = cls(
        **config,
        active_mask=inputs["active"],
        bed_elevation=inputs["bed"],
        manning_n=inputs["manning"],
        execution_mode="dense",
        device=device,
    )
    s.set_boundary_type_qy(boundary_type=inputs["by"])
    s.set_rainfall_hyetograph(times_s=inputs["rain_times"], rates_m_s=inputs["rain_rates"])
    for k, v in inputs["fields"].items():
        s._fields[k].assign(v.ravel())
    s.q.assign(inputs["q"].reshape(-1, 3))
    s._fields["maximum"].assign(np.ascontiguousarray(inputs["q"][..., 0]).ravel())
    s.time = inputs["time"]
    warm = s.run_device_loop(duration=warmup)
    intervals = []
    for _ in range(repeats):
        intervals.append(s.run_device_loop(duration=duration))
    q = s.get_conservative_state()
    if not np.isfinite(q).all() or q[..., 0].min() < 0.0:
        raise ArithmeticError("Invalid native wet probe")
    return dict(
        method=method,
        nx=s.nx,
        ny=s.ny,
        active_cells=int(inputs["active"].sum()),
        terrain_dx_m=s.dx,
        native_start_s=inputs["time"],
        elapsed_model_s=s.time - inputs["time"],
        owned_pool_bytes=wp.get_mempool_used_mem_current(device=device),
        process_pool_high_water_bytes=wp.get_mempool_used_mem_high(device=device),
        warmup=warm,
        intervals=intervals,
        wall_seconds_per_model_second=float(np.median([i["loop_seconds"] for i in intervals]) / duration),
        final_volume_m3=float(np.sum(q[..., 0]) * s.dx * s.dy),
        deepest_water_m=float(q[..., 0].max()),
    )


def run(*, source, checkpoint, output, device, factor, warmup, duration, repeats):
    inputs = native_state(source=source, checkpoint=checkpoint, factor=factor)
    records = []
    for method in ("fv1", "mc"):
        gc.collect()
        records.append(
            probe(
                inputs=inputs, method=method, device=device, warmup=warmup, duration=duration, repeats=repeats
            )
        )
        report_json(path=output, content=dict(source=str(source), checkpoint=str(checkpoint), rows=records))


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    source = Path("../floodfm_data/examples/purau_point_1575902_5166615_catchment_model/model_inputs.npz")
    run(
        source=source,
        checkpoint=checkpoint,
        output=directory / "whole_native_probe.json",
        device="cuda:0",
        factor=5,
        warmup=0.01,
        duration=0.5,
        repeats=3,
    )
