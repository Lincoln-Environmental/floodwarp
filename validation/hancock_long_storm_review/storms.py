"""Complete, separately timed physical storm runs with recoverable per-run output."""

import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import label
import warp as wp

from floodwarp.model import advance, diagnostics
from validation.fv2_local_inertia import read_state
from validation.hancock_long_storm_review.landforms import storm_model


def sections(*, solver, li):
    key = "cumulative_net_qx_volume_m3" if li else "qx"
    qx = solver.get_cumulative_face_volumes()[key]
    return qx[:, [solver.nx // 2, -1]].sum(axis=0)


def event(
    *,
    config,
    landform,
    storm,
    method,
    mode,
    dx,
    terrain_dx,
    dimensions,
    hours,
    interval,
    device,
    directory,
    fields,
    label_name,
):
    name = f"{label_name}_{dimensions[0]}x{dimensions[1]}_{landform}_{storm}_{method}_{mode}_{dx}m_{terrain_dx}dem_{hours}h"
    dest = directory / (name + ".json")
    archive = fields / (name + ".npz")
    pending_archive = fields / (name + ".pending.npz")
    if dest.exists() and (label_name in ("large", "terrain_large") or archive.exists()):
        return
    print(name, flush=True)
    started = perf_counter()
    duration = hours * 3600.0
    model, expected_rain = storm_model(
        config=config,
        landform=landform,
        storm=storm,
        method=method,
        mode=mode,
        dx=dx,
        terrain_dx=terrain_dx,
        dimensions=dimensions,
        duration=duration,
        device=device,
        directional=label_name == "directional",
    )
    s = model.solver
    li = method == "li"
    wet = s.get_water_depth() > 0.01
    first = np.where(wet, 0.0, np.nan)
    last = first.copy()
    volumes = sections(solver=s, li=li)
    samples = []
    snapshots = []
    total_extract = 0.0
    previous = 0.0
    if storm == "moving":
        expected_rain = 0.0
        y, x = np.indices(s._shape, dtype=float)
        rate_times = np.r_[config["hours"], hours] * 3600.0
        rates = np.r_[config["storms_mm_h"]["spatial"], 0.0] / 3.6e6
    wp.synchronize_device(device=device)
    for t in np.arange(interval, duration + 1.0, interval):
        if storm == "moving":
            centre = 0.2 + 0.35 * min(previous / 21600.0, 1.0)
            pattern = np.maximum(
                0.0,
                1.0
                - (((x + 0.5) / s.nx - centre) / 0.023) ** 2
                - (((y + 0.5) / s.ny - 0.6) / 0.032) ** 2,
            )
            s.set_rainfall_spatial_multiplier(multiplier=pattern)
            knots = np.r_[
                previous, rate_times[(rate_times > previous) & (rate_times < t)], t
            ]
            expected_rain += (
                pattern.sum()
                * dx**2
                * np.trapezoid(np.interp(knots, rate_times, rates), knots)
            )
        tick = perf_counter()
        stats = advance(model=model, end_time=t)
        wp.synchronize_device(device=device)
        wall = perf_counter() - tick
        tick = perf_counter()
        state = read_state(solver=s, li=li)
        h = state[..., 0]
        if not np.isfinite(state).all() or h.min() < 0.0:
            raise ArithmeticError(f"Invalid state in {name} at {t}")
        wet = h > 0.01
        first[wet & np.isnan(first)] = t
        last[wet] = t
        if label_name not in ("large", "terrain_large"):
            snapshots.append(h.copy())
        transport = sections(solver=s, li=li)
        discharge = (transport - volumes) / (t - previous)
        volumes = transport
        ledger = diagnostics(model=model)
        active = {} if li else s.execution_diagnostics()
        w = h > s.dry_depth
        depths = h[w]
        speed = np.hypot(state[..., 1][w], state[..., 2][w]) / depths
        froude = speed / np.sqrt(float(s.gravity) * depths)
        samples.append(
            dict(
                t_s=t,
                section_m3s=discharge[0],
                outlet_m3s=discharge[1],
                wet_fraction=float((h > 0.0).mean()),
                wet_cm_fraction=float(wet.mean()),
                active_fraction=active.get("evaluated_cells", h.size) / h.size,
                active_tiles=active.get("work_tiles"),
                storage_m3=ledger["storage_m3"],
                rain_m3=ledger["rain"],
                loss_m3=ledger["infiltration"],
                max_speed_ms=speed.max(initial=0.0),
                wet_froude_p95=float(np.quantile(froude, 0.95))
                if froude.size
                else None,
                wet_froude_max=float(froude.max()) if froude.size else None,
                supercritical_wet_fraction=float((froude > 1.0).mean())
                if froude.size
                else None,
                max_abs_residual_m3=abs(ledger["residual_m3"]),
                steps=stats["steps"],
                dt_min_s=stats["dt_min_s"],
                dt_max_s=stats["dt_max_s"],
                dt_mean_s=stats["dt_mean_s"],
                rejected=stats.get("rejected_cfl_trials", 0),
                loop_s=stats["loop_seconds"],
                capture_s=stats["graph_capture_seconds"],
                preload_s=stats.get("compilation_seconds"),
                hydraulic_wall_s=wall
                - stats["graph_capture_seconds"]
                - stats.get("compilation_seconds", 0.0),
                graph=stats.get(
                    "cuda_graph", stats.get("execution") == "conditional_cuda"
                ),
                fallback_cells_last_trial=int(s.predictor_fallback_cells().sum())
                if method == "hancock"
                else None,
            )
        )
        total_extract += perf_counter() - tick
        previous = t
    if previous != duration:
        raise ValueError("Sampling must include the exact common end time")
    ledger = diagnostics(model=model)
    wp.synchronize_device(device=device)
    tick = perf_counter()
    if label_name not in ("large", "terrain_large"):
        np.savez_compressed(
            pending_archive,
            depth=h,
            maximum=s.get_max_water_depth(),
            arrival_s=first,
            last_wet_s=last,
            snapshots=np.stack(snapshots),
        )
    write_fields = perf_counter() - tick
    supply = max(1.0, s.initial_volume + ledger["rain"] + ledger["boundary_inflow"])
    residual = max(row["max_abs_residual_m3"] for row in samples)
    row = dict(
        name=name,
        group=label_name,
        landform=landform,
        storm=storm,
        method=method,
        mode=mode,
        dx=dx,
        terrain_dx=terrain_dx,
        dimensions_m=dimensions,
        hours=hours,
        sample_seconds=interval,
        initial_m3=s.initial_volume,
        expected_rain_m3=expected_rain,
        rain_m3=ledger["rain"],
        infiltration_m3=ledger["infiltration"],
        outflow_m3=ledger["boundary_outflow"],
        inflow_m3=ledger["boundary_inflow"],
        final_m3=ledger["storage_m3"],
        residual_m3=ledger["residual_m3"],
        balance_pass=bool(residual <= 1e-8 * supply),
        extraction_s=total_extract,
        field_write_s=write_fields,
        process_pool_mib=wp.get_mempool_used_mem_current(device=device) / 2**20,
        final_components_1cm=label(wet)[1],
        hydraulic_s=sum(r["hydraulic_wall_s"] for r in samples),
        loop_s=sum(r["loop_s"] for r in samples),
        steps=sum(r["steps"] for r in samples),
        end_to_end_s=perf_counter() - started,
        samples=samples,
    )
    if label_name == "moving":
        row["final_sha256"] = {
            key: hashlib.sha256(s._fields[key].numpy().tobytes()).hexdigest()
            for key in (
                "q",
                "soil",
                "rain_sum",
                "maximum",
                "flux_x",
                "flux_y",
                "pressure_x",
                "pressure_y",
                "volume_x",
                "volume_y",
                "dt_histogram",
            )
        }
    if dest.exists():
        # A published run may only need its raw arrays regenerated.
        old = json.loads(dest.read_text())
        for key in (
            "steps",
            "final_m3",
            "rain_m3",
            "infiltration_m3",
            "outflow_m3",
            "inflow_m3",
        ):
            np.testing.assert_allclose(
                row[key], old[key], rtol=2e-13, atol=1e-11, err_msg=key
            )
        pending_archive.replace(archive)
        return
    if label_name not in ("large", "terrain_large"):
        pending_archive.replace(archive)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(dest)
    del model, s
    gc.collect()


def experiment_cases(*, config, suite):
    extended = config["extended_event"]
    for spec in config["suites"][suite]:
        for land in spec.get("landforms", config["landforms"]):
            for storm in spec.get("storms", config["storms_mm_h"]):
                end = (
                    extended["hours"]
                    if (land, storm) == (extended["landform"], extended["storm"])
                    else config["end_hours"]
                )
                for method, mode in config["solver_groups"][spec["solvers"]]:
                    for dx, dem in spec.get(
                        "grids", config["default_hydraulic_dem_grids_m"]
                    ):
                        for size in spec.get("dimensions_m", [config["dimensions_m"]]):
                            yield dict(
                                landform=land,
                                storm=storm,
                                method=method,
                                mode=mode,
                                dx=dx,
                                terrain_dx=dem,
                                dimensions=size,
                                hours=spec.get("hours", end),
                                interval=spec.get(
                                    "interval_s", config["sample_seconds"]
                                ),
                            )


def run(*, config, directory, fields, suite, device):
    directory.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    for case in experiment_cases(config=config, suite=suite):
        event(
            config=config,
            directory=directory,
            fields=fields,
            device=device,
            label_name=suite,
            **case,
        )


if __name__ == "__main__":
    review = Path(__file__).parent
    config = json.loads((review / "cases.json").read_text())
    directory = Path(os.environ.get("FLOODWARP_STORM_RESULTS", str(review / "events")))
    fields = Path(
        os.environ.get(
            "FLOODWARP_STORM_FIELDS", "outputs/hancock_long_storm_review/fields"
        )
    )
    suite = os.environ.get("FLOODWARP_STORM_SUITE", "primary")
    device = os.environ.get("FLOODWARP_STORM_DEVICE", "cuda:0")
    run(config=config, directory=directory, fields=fields, suite=suite, device=device)
