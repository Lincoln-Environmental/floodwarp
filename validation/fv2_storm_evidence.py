"""Summarise matched HIRDS runs on the archived 5 m footprint and at the outlet."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from validation.fv1_riemann import errors, report_json
from validation.fv2_orton_matrix import dry_initial, geometry, replay_solver
from validation.fv2_terrain import prolong
from validation.sparse_fv1_benchmark import select_checkpoints


def read_snapshot(*, path):
    with np.load(path) as archive:
        config = json.loads(str(archive["metadata"]))["configuration"]
        shape = (config["ny"], config["nx"])
        return (
            archive["q"].reshape(*shape, 3),
            archive["maximum"].reshape(shape),
            archive["active"].reshape(shape).astype(bool),
        )


def on_footprint(*, field, dx, base, shape):
    return prolong(value=field, factor=round(dx / base))[: shape[0], : shape[1]]


def overlap(*, a, b, mask):
    inside = ((a | b) & mask).sum()
    return float(((a & b) & mask).sum() / inside) if inside else 1.0


def fallback(*, row):
    samples = []
    for minute in row["hydrograph"]:
        if "interior_transport" not in minute:
            continue
        categories = {}
        for key, bands in minute["interior_transport"]["groups"].items():
            counts = np.array(bands).sum(axis=0)
            total = counts.sum(axis=0)
            fractions = np.divide(counts[1:3].sum(axis=0), total, where=total > 0.0, out=np.zeros_like(total))
            categories[key] = [float(f) if n else None for f, n in zip(fractions, total, strict=True)]
        samples.append(dict(hour=minute["t"] / 3600, fallback_fraction_by_counts_volume_impulse=categories))
    return samples


def threshold_overlap(*, predicted, reference, active, thresholds, cell_area):
    rows = []
    for limit in thresholds:
        a = (predicted > limit) & active
        b = (reference > limit) & active
        rows.append(
            dict(
                depth_m=limit,
                predicted_m2=int(a.sum() * cell_area),
                reference_m2=int(b.sum() * cell_area),
                both_m2=int((a & b).sum() * cell_area),
                predicted_only_m2=int((a & ~b).sum() * cell_area),
                reference_only_m2=int((b & ~a).sum() * cell_area),
                iou=overlap(a=a, b=b, mask=active),
            )
        )
    return rows


def summarize(*, directory, grids, methods, wet, sample_hours, thresholds):
    base = min(grids)
    records = {}
    for dx in grids:
        for method in methods:
            path = directory / f"{dx}m_{method}.json"
            row = json.loads(path.read_text())
            if row["time_s"] != 48 * 3600 or len(row["hydrograph"]) != 2880:
                raise ValueError(f"Incomplete event in {path}")
            records[dx, method] = row
    baseline = records[base, "contact_hll"]
    q_ref, max_ref, active = read_snapshot(path=Path(baseline["origin"]))
    base_path = Path(baseline["origin"])
    archive_hours = (
        select_checkpoints(directory=base_path.parent, hours=sample_hours)
        if base_path.name.startswith("checkpoint-")
        else [base_path.parent / f"hour-{hour:02d}.npz" for hour in sample_hours]
    )
    wet_ref = first_wet(paths=archive_hours, wet=wet, hours=sample_hours)
    href = np.array([r["outflow_m3_s"] for r in baseline["hydrograph"]])
    output = []
    for (dx, method), row in records.items():
        path = Path(row["origin"])
        state, maximum, _ = read_snapshot(path=path)
        final = on_footprint(field=state, dx=dx, base=base, shape=active.shape)
        peak = on_footprint(field=maximum, dx=dx, base=base, shape=active.shape)
        final_wet = final[..., 0] > wet
        observed = final_wet & active
        ref_wet = q_ref[..., 0] > wet
        metrics = errors(result=final, expected=q_ref, wet_depth=wet, dx=base, window=active)
        hydro = np.array([r["outflow_m3_s"] for r in row["hydrograph"]])
        paths = (
            archive_hours
            if (dx, method) == (base, "contact_hll")
            else [path.parent / f"hour-{hour:02d}.npz" for hour in sample_hours]
        )
        arrived = on_footprint(
            field=first_wet(paths=paths, wet=wet, hours=sample_hours), dx=dx, base=base, shape=active.shape
        )
        common = (arrived > 0) & (wet_ref > 0) & active
        metrics.update(
            peak_envelope_rmse_m=float(np.sqrt(np.mean((peak - max_ref)[active] ** 2))),
            final_wet_iou=overlap(a=final_wet, b=ref_wet, mask=active),
            envelope_wet_iou=overlap(a=peak > wet, b=max_ref > wet, mask=active),
            final_wet_components=int(label(observed)[1]),
            arrival_mae_h=float(np.mean(np.abs(arrived[common] - wet_ref[common]))) if common.any() else None,
            common_arrived_area_m2=int(common.sum() * base**2),
            sampled_arrival_iou=overlap(a=arrived > 0, b=wet_ref > 0, mask=active),
            hydrograph_rmse_m3_s=float(np.sqrt(np.mean((hydro - href) ** 2))),
            volume_delta_m3=float(row["ledger"]["boundary_outflow"] - baseline["ledger"]["boundary_outflow"]),
            peak_delta_m3_s=float(row["peak_outflow_m3_s"] - baseline["peak_outflow_m3_s"]),
            peak_time_delta_min=(row["peak_interval_end_s"] - baseline["peak_interval_end_s"]) / 60,
        )
        output.append(
            dict(
                dx=dx,
                method=method,
                wet_m=wet,
                sample_hours=sample_hours,
                original_active_area_m2=int(active.sum() * base**2),
                vs_5m_contact=metrics,
                envelope_thresholds=threshold_overlap(
                    predicted=peak, reference=max_ref, active=active, thresholds=thresholds, cell_area=base**2
                ),
                fallback_samples=fallback(row=row),
            )
        )
    report_json(path=directory / "common_support.json", content=output)


def first_wet(*, paths, wet, hours):
    first = None
    for hour, path in zip(hours, paths, strict=True):
        with np.load(path) as archive:
            if first is None:
                c = json.loads(str(archive["metadata"]))["configuration"]
                first = np.zeros((c["ny"], c["nx"]), dtype=np.int16)
            envelope = archive["maximum"].reshape(first.shape)
            first[(first == 0) & (envelope > wet)] = hour
    return first


def plot_hydrographs(*, directory, grids, methods, checkpoint, colors):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    with np.load(checkpoint) as archive:
        import hashlib

        forcing = json.loads((directory / "5m_contact_hll.json").read_text())["forcing"]
        t, rate = archive["rain_times"], archive["rain_rates"]
        if hashlib.sha256(rate.tobytes()).hexdigest() != forcing["rate_nodes_sha256"]:
            raise ValueError("Rainfall series hash mismatch")
        axes.flat[0].fill_between(t / 3600, rate * 3.6e6, color="#007d77")
        axes.flat[0].set(title="Archived HIRDS v4 historical 100-year storm", ylabel="Rainfall rate (mm/h)")
    labels = {
        "contact_hll": "FV1 contact-HLL",
        "roe": "FV1 Roe / HLL",
        "hllc": "FV1 HLLC / HLL",
        "fv2_mc": "FV2 MC",
    }
    styles = {"contact_hll": "-", "roe": "--", "hllc": ":", "fv2_mc": "-"}
    highest = 0.0
    for axis, dx in zip(axes.flat[1:], grids, strict=True):
        for method, color in zip(methods, colors, strict=True):
            path = directory / f"{dx}m_{method}.json"
            if not path.exists():
                continue
            a = json.loads(path.read_text())
            x = np.array([r["t"] for r in a["hydrograph"]]) / 3600
            y = [r["outflow_m3_s"] for r in a["hydrograph"]]
            highest = max(highest, max(y))
            axis.plot(
                x, y, label=labels[method], color=color, linestyle=styles[method], linewidth=1.3, alpha=0.9
            )
        title = (
            f"{dx} m Orton Bradley"
            if len(axis.lines) == len(methods)
            else f"{dx} m, {len(axis.lines)}/{len(methods)} complete"
        )
        axis.set(title=title, ylabel="Gross outlet flow (m³/s)", ylim=(0, None))
    for ax in axes.flat:
        ax.set(xlabel="Storm time (hours)", xlim=(0, 48))
        ax.grid(alpha=0.18)
    for axis in axes.flat[1:]:
        axis.set_ylim(0.0, 1.05 * highest)
    axes.flat[-1].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "storm.png", dpi=160)
    plt.close(fig)


def depth_maps(*, directory, grids, methods, difference_limit, depth_limit):
    base = grids[0]
    reference = json.loads((directory / f"{base}m_contact_hll.json").read_text())
    _, expected, active = read_snapshot(path=Path(reference["origin"]))
    fig, axes = plt.subplots(
        len(grids), len(methods), figsize=(12, 12), layout="constrained", sharex=True, sharey=True
    )
    extent = (0, expected.shape[1] * base / 1000, 0, expected.shape[0] * base / 1000)
    for i, dx in enumerate(grids):
        for j, method in enumerate(methods):
            row = json.loads((directory / f"{dx}m_{method}.json").read_text())
            _, envelope, _ = read_snapshot(path=Path(row["origin"]))
            value = on_footprint(field=envelope, dx=dx, base=base, shape=active.shape)
            ax = axes[i, j]
            if i == 0 and j == 0:
                im = ax.imshow(
                    np.ma.masked_where(~active, value),
                    origin="lower",
                    extent=extent,
                    vmin=0.0,
                    vmax=depth_limit,
                    cmap="Blues",
                )
                fig.colorbar(im, ax=ax, shrink=0.6, label="Depth (m)")
                ax.set_title("5 m contact-HLL envelope")
            else:
                diff = ax.imshow(
                    np.ma.masked_where(~active, value - expected),
                    origin="lower",
                    extent=extent,
                    vmin=-difference_limit,
                    vmax=difference_limit,
                    cmap="RdBu",
                )
                ax.set_title(f"{dx} m {method}")
            if j == 0:
                ax.set_ylabel("Northing from origin (km)")
            if i == len(grids) - 1:
                ax.set_xlabel("Easting from origin (km)")
    fig.colorbar(
        diff, ax=list(axes.flat[1:]), shrink=0.6, label="Maximum-depth difference from 5 m contact-HLL (m)"
    )
    fig.savefig(directory / "depth_maps.png", dpi=165)
    plt.close(fig)


def reconstruction_sample(*, directory, work, checkpoint, grids, hours, duration, device, edges):
    initial = dry_initial(checkpoint=checkpoint)
    masks = []
    volumes = []
    output = []
    for dx in grids:
        factor = dx // 5
        data, _ = geometry(state=initial, factor=factor)
        solver = replay_solver(data=data, state=initial, factor=factor, method="fv2_mc", device=device)
        for hour in hours:
            path = work / f"{dx}m_fv2_mc" / f"hour-{hour:02d}.npz"
            solver.load_checkpoint(path=path)
            diagnostic = solver.track_reconstruction(depth_edges=edges)
            starting_export = solver.cumulative_accounting()["boundary_outflow"]
            stats = solver.run_device_loop(duration=duration)
            flow = (solver.cumulative_accounting()["boundary_outflow"] - starting_export) / duration
            archive = json.loads((directory / f"{dx}m_fv2_mc.json").read_text())
            minute = next(r for r in archive["hydrograph"] if r["t"] == hour * 3600 + duration)
            np.testing.assert_allclose(flow, minute["outflow_m3_s"], rtol=1.0e-10, atol=1.0e-8)
            if stats["steps"] != minute["steps"]:
                raise ValueError("Monitored replay changed accepted step count")
            result = diagnostic.summary()
            result["outlet_replay_difference_m3_s"] = flow - minute["outflow_m3_s"]
            result.update(dx=dx, start_h=hour, sample_s=duration)
            output.append(result)
            if hour == hours[0]:
                x = diagnostic.maps[0].numpy().reshape(solver.ny, solver.nx + 1)
                y = diagnostic.maps[1].numpy().reshape(solver.ny + 1, solver.nx)
                volumes.append(0.5 * (x[:, 1:] + x[:, :-1] + y[1:] + y[:-1]) / solver.dx**2)
                masks.append(solver._active.copy())
    report_json(path=directory / "muscl_fallback.json", content=output)
    fig, axes = plt.subplots(1, len(grids), figsize=(10, 5), layout="constrained")
    from matplotlib.colors import LogNorm

    positive = np.concatenate([v[v > 0] for v in volumes])
    norm = LogNorm(
        vmin=max(1.0e-10, float(np.percentile(positive, 5))), vmax=float(np.percentile(positive, 99.9))
    )
    for ax, dx, v, mask in zip(axes, grids, volumes, masks, strict=True):
        raster = np.ma.masked_where(~mask | (v <= 0), v)
        im = ax.imshow(
            raster, origin="lower", norm=norm, extent=(0, dx * v.shape[1], 0, dx * v.shape[0]), cmap="viridis"
        )
        ax.set(title=f"{dx} m", xlabel="Easting from grid origin (m)")
    axes[0].set_ylabel("Northing from grid origin (m)")
    fig.colorbar(im, ax=axes, label="Absolute volume on fallback faces per cell area (m)", shrink=0.8)
    fig.suptitle(f"Orton Bradley reconstruction fallback, {duration:g} s from storm hour {hours[0]}")
    fig.savefig(directory / "fallback.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    directory = Path("validation/fv2_review/hirds48")
    work = Path("outputs/fv2_hirds48_v1")
    checkpoint = Path(
        json.loads(Path("validation/fv1_riemann_review/corrected/catchment.json").read_text())["checkpoint"]
    )
    grids = (5, 10, 20)
    methods = ("contact_hll", "roe", "hllc", "fv2_mc")
    colors = ("#2166ac", "#a45b00", "#8b52a5", "#11865a")
    wet = 0.01
    thresholds = (0.001, 0.01, 0.03, 0.10, 0.30, 1.0)
    sample_hours = tuple(range(2, 49, 2))
    if not checkpoint.is_file():
        checkpoint = Path(json.loads((directory / "5m_contact_hll.json").read_text())["origin"])
    summarize(
        directory=directory,
        grids=grids,
        methods=methods,
        wet=wet,
        sample_hours=sample_hours,
        thresholds=thresholds,
    )
    plot_hydrographs(directory=directory, grids=grids, methods=methods, checkpoint=checkpoint, colors=colors)
    depth_maps(directory=directory, grids=grids, methods=methods, difference_limit=0.25, depth_limit=1.5)
    reconstruction_sample(
        directory=directory,
        work=work,
        checkpoint=checkpoint,
        grids=grids,
        hours=(24, 12, 36),
        duration=60.0,
        device="cuda:0",
        edges=(0.001, 0.01, 0.1, 1.0),
    )
