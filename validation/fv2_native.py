"""Real terrain, matched numerical refinement and separate terrain aggregation controls."""

from pathlib import Path
import hashlib
import json

import numpy as np

from floodwarp.fv2_solver import FV2Solver
from validation.fv1_riemann import report_json
from validation.fv1_riemann_cases import Domain
from validation.fv1_riemann_catchment import checkpoint_solver
from validation.fv2_terrain import blocks, compare, evaluate, prolong, refined, run_domain


def source_fields(*, site, crop_dir, purau, purau_window, duration):
    if site == "orton":
        z = np.load(crop_dir / "native_terrain.npy").astype(float)
        active = prolong(value=np.load(crop_dir / "active.npy"), factor=5)
        rain = np.loadtxt(crop_dir / "rainfall.csv", delimiter=",", skiprows=1)
        return (
            z,
            active,
            np.full_like(z, 0.035),
            np.zeros_like(z),
            rain,
            dict(
                input=str(crop_dir),
                rain="complete installation storm and recession",
                bounds="north-facing active/exterior FREE",
            ),
        )
    rows, cols = purau_window
    with np.load(purau / "model_inputs.npz") as a:
        native = np.s_[rows.start * 5 : rows.stop * 5, cols.start * 5 : cols.stop * 5]
        coarse = np.s_[rows, cols]
        z = a["subgrid_bed_elevation"][native][::-1].astype(float)
        roughness = a["subgrid_manning_n"][native][::-1].astype(float)
        active = prolong(value=a["active_mask"][coarse][::-1].astype(bool), factor=5)
        terminal = prolong(value=a["infiltration_final_rate_m_per_s"][coarse][::-1], factor=5)
        times = a["rainfall_node_times_s"]
        rates = a["rainfall_node_rates_m_per_s"]
        peak = int(np.argmax(rates))
        start = times[peak] - duration / 2
        interior = (times > start) & (times < start + duration)
        t = np.r_[start, times[interior], start + duration]
        rain = np.column_stack((t - start, np.interp(t, times, rates)))
    return (
        z,
        active,
        roughness,
        terminal,
        rain,
        dict(
            input=str(purau),
            row_start=rows.start,
            col_start=cols.start,
            model_hash=json.loads((purau / "model_inputs.json").read_text())["model_hash"],
            storm_start_s=float(start),
            rain="archived peak segment, initially dry, terminal mapped infiltration",
            bounds="outward-only cut edges",
        ),
    )


def aggregate(*, value, mask, factor):
    # Mask weighting avoids treating nodata elevation/roughness as physical zero.
    divisor = blocks(value=mask.astype(float), factor=factor)
    return np.divide(
        blocks(value=np.where(mask, value, 0.0), factor=factor),
        divisor,
        out=np.zeros_like(divisor),
        where=divisor > 0.0,
    )


def native_support(*, result, hydraulic_dx):
    row, states, maximum = result

    def transfer(*, value):
        if hydraulic_dx >= 1.0:
            return prolong(value=value, factor=int(hydraulic_dx))
        return blocks(value=value, factor=round(1.0 / hydraulic_dx))

    return row, [transfer(value=q) for q in states], transfer(value=maximum)


def site_runs(*, site, fields, spacings, reference_factors, output, duration, device, samples, wet):
    z, active, n, infil, rain, metadata = fields
    for name, array in (("terrain", z), ("active", active), ("manning", n), ("infil", infil), ("rain", rain)):
        metadata[name + "_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    results = {}
    geometries = {}

    def execute(*, dx, factor, method, cap, diagnostic=False):
        bed, mask, roughness, loss = geometries[dx]
        d = Domain(dx=dx, bed=bed, initial=np.zeros((*bed.shape, 3)), duration=duration, manning=roughness)
        d = refined(domain=d, factor=factor)
        k = prolong(value=loss, factor=factor)
        return run_domain(
            domain=d,
            method=method,
            cap=cap,
            device=device,
            samples=samples,
            wet=wet,
            mask=prolong(value=mask, factor=factor),
            rain_series=rain,
            north_only=site == "orton",
            infiltration=dict(f0=k, fd=1.0, conductivity=k),
            diagnostic=diagnostic,
        )

    for dx in spacings:
        mask = blocks(value=active.astype(float), factor=dx) >= 0.5
        bed = aggregate(value=z, mask=active, factor=dx)
        rough = aggregate(value=n, mask=active, factor=dx)
        loss = aggregate(value=infil, mask=active, factor=dx)
        geometries[dx] = bed, mask, rough, loss
        for method in ("fv1", "mc"):
            results[dx, 1, method] = execute(dx=dx, factor=1, method=method, cap=2.0)
        print(site, dx, flush=True)
    for factor in reference_factors:
        for method in ("fv1", "mc"):
            results[1, factor, method] = execute(dx=1, factor=factor, method=method, cap=2.0 / factor)
        print(site, "fixed 1m subcells", factor, flush=True)
    ref = results[1, reference_factors[-1], "mc"]
    finest_dx = 1 / reference_factors[-1]
    common = native_support(result=ref, hydraulic_dx=finest_dx)
    rows = []
    for (dx, factor, method), result in results.items():
        step = dx / factor
        row = dict(terrain_dx=dx, hydraulic_dx=step, method=method, **result[0])
        row["vs_fine"] = compare(result=result, reference=ref, wet=wet, dx=step, duration=duration)
        row["vs_other_fine"] = compare(
            result=result,
            reference=results[1, reference_factors[-1], "fv1"],
            wet=wet,
            dx=step,
            duration=duration,
        )
        row["common_1m"] = compare(
            result=native_support(result=result, hydraulic_dx=step),
            reference=common,
            wet=wet,
            dx=1.0,
            duration=duration,
        )
        if factor == 1:
            bed, mask, _, _ = geometries[dx]
            delta = (prolong(value=bed, factor=dx) - z)[active]
            row["bed_rmse_m"] = float(np.sqrt(np.mean(delta**2)))
            row["mask_area_change_m2"] = int(mask.sum() * dx * dx - active.sum())
        rows.append(row)
    for dx in (5, 10):
        # Refine the coarse DEM down to 1 m with no new elevation data.
        same = execute(dx=dx, factor=dx, method="mc", cap=0.5)
        for method in ("mc", "fv1"):
            row = next(r for r in rows if r["terrain_dx"] == dx and r["method"] == method)
            row["refined_coarse_terrain_vs_native"] = compare(
                result=same, reference=common, wet=wet, dx=1.0, duration=duration
            )
            row["vs_fixed_terrain_subcells"] = compare(
                result=results[dx, 1, method], reference=same, wet=wet, dx=dx, duration=duration
            )
    small = execute(dx=1, factor=reference_factors[-1], method="mc", cap=1.0 / reference_factors[-1])
    temporal = compare(result=ref, reference=small, wet=wet, dx=1 / reference_factors[-1], duration=duration)
    tracked = execute(dx=1, factor=1, method="mc", cap=2.0, diagnostic=True)
    np.testing.assert_array_equal(tracked[1][-1], results[1, 1, "mc"][1][-1])
    report_json(
        path=output,
        content=dict(
            site=site,
            metadata=metadata,
            duration=duration,
            wet_m=wet,
            reference_timestep_check=temporal,
            reconstruction=tracked[0]["reconstruction"],
            rows=rows,
        ),
    )


def full_window(*, path, output, device, duration, samples):
    original = checkpoint_solver(scheme="contact_hll", device=device, path=path)
    state = original.capture_state()
    f = state["arrays"]
    options = state["configuration"]
    shape = original._shape
    second = FV2Solver(
        device=device,
        bed_elevation=f["bed"].reshape(shape),
        manning_n=f["manning"].reshape(shape),
        active_mask=f["active"].reshape(shape),
        **options,
    )
    state["configuration"] = second._configuration()
    state["arrays"]["fv2_cfl_predictor"] = np.zeros(1)
    state["control"] = np.zeros(second.ctrl.size)
    second.restore_state(state=state)
    cases = []
    for method, solver in (("fv1", original), ("mc", second)):
        result = evaluate(solver=solver, duration=duration, samples=samples, wet=0.01)
        if method == "fv1":
            first = result
        row = dict(
            method=method,
            **result[0],
            vs_fv1=compare(result=result, reference=first, wet=0.01, dx=solver.dx, duration=duration),
        )
        cases.append(row)
    report_json(
        path=output,
        content=dict(
            checkpoint=str(path),
            start=state["time"],
            shape=shape,
            sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            rows=cases,
        ),
    )


if __name__ == "__main__":
    directory = Path("validation/fv2_review")
    crop = Path("examples/purau")
    projects = Path.cwd().parent
    purau = projects / "floodfm_data/outputs/purau_model_100_48hr_guided"
    purau_window = (slice(1, 41), slice(316, 356))
    archives = (
        (
            "orton",
            Path(
                json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())[
                    "checkpoint"
                ]
            ),
        ),
        ("nz_demo", projects / "floodwarp-nz-demo/outputs/nz_demo/purau_6h_fv1/checkpoint.npz"),
    )
    device = "cuda:0"
    for site, path in archives:
        if path.is_file():
            full_window(
                path=path,
                output=directory / (site + "_full_window.json"),
                device=device,
                duration=120.0,
                samples=4,
            )
    for site, duration, wet in (("orton", 240.0, 0.001), ("purau", 600.0, 0.001)):
        if site == "purau" and not purau.is_dir():
            continue
        fields = source_fields(
            site=site, crop_dir=crop, purau=purau, purau_window=purau_window, duration=duration
        )
        site_runs(
            site=site,
            fields=fields,
            spacings=(1, 2, 5, 10, 20),
            reference_factors=(2, 4, 8),
            output=directory / (site + "_terrain.json"),
            duration=duration,
            device=device,
            samples=6,
            wet=wet,
        )
