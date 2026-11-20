"""Full archived Orton HIRDS storm on a nested native coastal tile, with closed cut edges."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from floodwarp.full_swe_solver import FullSWESolver
from floodwarp.fv2_solver import FV2Solver
from validation.fv1_riemann import errors, report_json
from validation.fv2_native import aggregate
from validation.fv2_terrain import blocks, prolong


@dataclass
class Tile:
    fields: dict
    coast: np.ndarray
    time: np.ndarray
    rates: np.ndarray
    configuration: dict
    original_volume: float


def tile_inputs(*, checkpoint, native_file, row, col, width):
    native_window = np.s_[row * 5 : (row + width) * 5, col * 5 : (col + width) * 5]
    with np.load(native_file) as a:
        fields = dict(
            bed=a["subgrid_bed_elevation"][native_window][::-1].astype(float),
            manning=a["subgrid_manning_n"][native_window][::-1].astype(float),
        )
    with np.load(checkpoint) as a:
        config = json.loads(str(a["metadata"]))["configuration"]
        ny, nx = config["ny"], config["nx"]
        bottom = ny - row - width
        window = np.s_[bottom : bottom + width, col : col + width]
        for key in ("active", "multiplier", "f0", "fd", "conductivity", "infil"):
            value = a[key].reshape(ny, nx)[window]
            fields[key] = prolong(value=value, factor=5)
        fields["active"] = fields["active"].astype(bool)
        time = a["rain_times"]
        rates = a["rain_rates"]
        boundary = a["types_y"].reshape(ny + 1, nx)[bottom : bottom + width + 1, col : col + width]
        iy, ix = np.where(boundary == 3)
        coast = np.column_stack(
            (np.repeat(iy * 5, 5), np.repeat(ix * 5, 5) + np.tile(np.arange(5) + 0.5, len(ix)))
        )
        if np.any(a["stages_y"][a["types_y"] == 3]) or len(coast) == 0:
            raise ValueError("Expected stage-zero coastal openings on the source mask")
    physical_volume = float(np.trapezoid(rates, time) * fields["multiplier"][fields["active"]].sum())
    return Tile(
        fields=fields,
        coast=coast,
        time=time,
        rates=rates,
        configuration=config,
        original_volume=physical_volume,
    )


def native_geometry(*, tile, terrain_dx, factor):
    f = tile.fields
    fraction = blocks(value=f["active"].astype(float), factor=terrain_dx)
    mask = fraction > 0.0
    data = {"active": prolong(value=mask, factor=factor)}
    for k in ("bed", "manning", "multiplier", "f0", "fd", "conductivity", "infil"):
        if k in ("bed", "manning"):
            value = aggregate(value=f[k], mask=f["active"], factor=terrain_dx)
        else:
            value = blocks(value=np.where(f["active"], f[k], 0.0), factor=terrain_dx)
        if k == "fd":
            value[~mask] = 1.0
        data[k] = prolong(value=value, factor=factor)
    # Reproduce the original coastline on each represented DEM. Artificial cut edges are closed.
    south = np.pad(mask, ((1, 0), (0, 0)))
    north = np.pad(mask, ((0, 1), (0, 0)))
    y, x = np.where(south & ~north)
    face_centres = np.column_stack((y * terrain_dx, (x + 0.5) * terrain_dx))
    distance, nearest = cKDTree(face_centres).query(tile.coast)
    coarse = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    coarse[y[nearest], x[nearest]] = 3
    fine = np.zeros((data["active"].shape[0] + 1, data["active"].shape[1]), np.int32)
    fine[::factor, :] = np.repeat(coarse, factor, axis=1)
    return (
        data,
        fine,
        dict(
            stage_width_m=int(np.sum(coarse == 3) * terrain_dx),
            max_stage_displacement_m=float(distance.max()),
        ),
    )


def step_event(*, tile, terrain_dx, factor, method, cap, sample, work, device, wet, suffix):
    d, by, geometry = native_geometry(tile=tile, terrain_dx=terrain_dx, factor=factor)
    dx = terrain_dx / factor
    filename = work / f"dem{terrain_dx}_factor{factor}_{method}_{suffix}.npz"
    fingerprint = hashlib.sha256()
    for array in (*d.values(), by, tile.time, tile.rates):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    fingerprint.update(str((tile.configuration, dx, method, cap, sample, wet)).encode())
    signature = fingerprint.hexdigest()
    if filename.exists():
        with np.load(filename) as a:
            record = json.loads(str(a["record"]))
            if record.get("input_sha256") != signature:
                raise ValueError(
                    f"Existing tile result has different inputs or numerical settings: {filename}"
                )
            return record, a["q"], a["maximum"]
    config = dict(
        tile.configuration, nx=d["bed"].shape[1], ny=d["bed"].shape[0], dx=dx, dy=dx, maximum_dt=cap
    )
    cls = FV2Solver if method == "mc" else FullSWESolver
    s = cls(
        **config,
        device=device,
        bed_elevation=d["bed"],
        manning_n=d["manning"],
        active_mask=d["active"],
        execution_mode="dense",
    )
    s.set_boundary_type_qy(boundary_type=by)
    s.set_boundary_stage_qy(stage=np.zeros_like(by, float))
    s.set_source_rates(rain=0.0, infil=d["infil"])
    s.set_exponential_infiltration(f0=d["f0"], fd=d["fd"], conductivity=d["conductivity"])
    s.set_rainfall_hyetograph(times_s=tile.time, rates_m_s=tile.rates)
    s.set_rainfall_spatial_multiplier(multiplier=d["multiplier"])
    peaks = []
    ledger_before = 0.0
    elapsed = 0.0
    for target in np.arange(sample, tile.time[-1] + 1, sample):
        stats = s.run_device_loop(duration=float(target - s.time))
        elapsed += stats["loop_seconds"]
        balance = s.cumulative_accounting()
        peaks.append((balance["boundary_outflow"] - ledger_before) / sample)
        ledger_before = balance["boundary_outflow"]
    np.testing.assert_allclose(balance["rain"], tile.original_volume, rtol=1.0e-9)
    np.testing.assert_allclose(balance["residual"], 0.0, atol=1.0e-8 * tile.original_volume)
    record = dict(
        input_sha256=signature,
        method=method,
        terrain_dx_m=terrain_dx,
        hydraulic_dx_m=dx,
        cap_s=cap,
        time_s=s.time,
        geometry=geometry,
        steps=s.steps,
        sample_s=sample,
        hydrograph_m3_s=peaks,
        ledger=balance,
        peak_m3_s=max(peaks),
        peak_interval_end_s=(int(np.argmax(peaks)) + 1) * sample,
        kernel_loop_s=elapsed,
    )
    q = s.get_conservative_state()
    maximum = s.get_max_water_depth()
    record.update(wet_area_m2=float(np.sum(q[..., 0] > wet) * dx**2), peak_envelope_m=float(maximum.max()))
    work.mkdir(parents=True, exist_ok=True)
    temp = filename.with_suffix(".partial.npz")
    np.savez_compressed(temp, q=q, maximum=maximum, record=json.dumps(record))
    temp.replace(filename)
    return record, q, maximum


def support(*, field, dx):
    if dx >= 1:
        return prolong(value=field, factor=round(dx))
    return blocks(value=field, factor=round(1 / dx))


def experiment(
    *, checkpoint, native_file, row, col, width, spacings, refinements, cap, sample, device, wet, work, output
):
    tile = tile_inputs(checkpoint=checkpoint, native_file=native_file, row=row, col=col, width=width)
    runs = {}
    cases = [(dx, 1, method, cap, "base") for dx in spacings for method in ("fv1", "mc")]
    cases.extend(
        (1, refine, method, cap / refine, "fine") for refine in refinements for method in ("fv1", "mc")
    )
    cases.append((1, refinements[-1], "mc", cap / (2 * refinements[-1]), "half_dt"))
    for dx, factor, method, dt, suffix in cases:
        a, q, maximum = step_event(
            tile=tile,
            terrain_dx=dx,
            factor=factor,
            method=method,
            cap=dt,
            sample=sample,
            work=work,
            device=device,
            wet=wet,
            suffix=suffix,
        )
        grid = dx / factor
        runs[grid, method, suffix] = a, support(field=q, dx=grid), support(field=maximum, dx=grid)
        print(
            f"Native Orton HIRDS tile, DEM {dx} m, hydraulic grid {grid} m, {method}, {suffix}, 48 h complete",
            flush=True,
        )
    target = 1 / refinements[-1]
    mask = tile.fields["active"]
    rows = []
    for a, q, maximum in runs.values():
        a = a.copy()
        for name, key in (
            ("vs_fine_mc", (target, "mc", "fine")),
            ("vs_fine_fv1", (target, "fv1", "fine")),
            ("vs_half_dt", (target, "mc", "half_dt")),
        ):
            r, v, envelope = runs[key]
            metrics = errors(result=q, expected=v, wet_depth=wet, dx=1.0, window=mask)
            both = (q[..., 0] > wet) & (v[..., 0] > wet) & mask
            union = ((q[..., 0] > wet) | (v[..., 0] > wet)) & mask
            metrics.update(
                wet_iou=float(both.sum() / union.sum()) if union.any() else 1.0,
                peak_envelope_rmse_m=float(np.sqrt(np.mean((maximum - envelope)[mask] ** 2))),
                peak_delta_m3_s=a["peak_m3_s"] - r["peak_m3_s"],
                hydrograph_rmse_m3_s=float(
                    np.sqrt(np.mean((np.array(a["hydrograph_m3_s"]) - r["hydrograph_m3_s"]) ** 2))
                ),
            )
            a[name] = metrics
        rows.append(a)
    report_json(
        path=output,
        content=dict(
            checkpoint=str(checkpoint),
            source=str(native_file),
            terrain_sha256=hashlib.sha256(tile.fields["bed"].tobytes()).hexdigest(),
            north_first_5m_window=[row, col, width, width],
            tile_rain_volume_m3=tile.original_volume,
            active_area_m2=int(mask.sum()),
            rows=rows,
        ),
    )


if __name__ == "__main__":
    evidence = Path("validation/fv2_review")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    source = Path("../floodfm_data/examples/purau_point_1575902_5166615_catchment_model/model_inputs.npz")
    experiment(
        checkpoint=checkpoint,
        native_file=source,
        row=112,
        col=172,
        width=40,
        spacings=(20, 10, 5, 2, 1),
        refinements=(2, 4),
        cap=1.0,
        sample=900,
        device="cuda:0",
        wet=0.01,
        work=Path("outputs/fv2_hirds_native_tile"),
        output=evidence / "native_hirds48.json",
    )
