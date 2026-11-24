"""Partition late outlet export using accepted face volumes and same-grid native storm checkpoints."""

import json
from pathlib import Path
import sys

import numpy as np
from floodwarp.model import advance, load_checkpoint
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import (
    continuation,
    native_event,
    native_products,
)
from research.floodwarp_methods_paper.terrain.generators import native_inputs


def measurements(*, h, soil, rain, volumes, distances, dx):
    ny, nx = h.shape
    if (
        soil.shape != h.shape
        or rain.shape != h.shape
        or dx <= 0
        or volumes.size != ny * (nx + 1)
        or 0 not in distances
        or len(set(distances)) != len(distances)
    ):
        raise ValueError("Incompatible fields or missing unique outlet face")
    if any(m % dx or m < 0 or m > nx * dx for m in distances):
        raise ValueError("A diagnostic face lies outside or between native faces")
    columns = [nx - m // dx for m in distances]
    transfers = volumes.reshape(ny, nx + 1)[:, columns].sum(axis=0)
    out = {}
    for metres, transfer in zip(distances, transfers, strict=True):
        j = nx - metres // dx
        out[metres] = dict(
            transfer_m3=transfer,
            storage_m3=h[:, j:].sum() * dx**2,
            rain_m3=rain[:, j:].sum() * dx**2,
            infiltration_m3=soil[:, j:].sum() * dx**2,
        )
    return out


def changes(*, before, after):
    rows = []
    export = after[0]["transfer_m3"] - before[0]["transfer_m3"]
    for m, b in before.items():
        delta = {k: after[m][k] - b[k] for k in b}
        source = delta["rain_m3"] - delta["infiltration_m3"] - delta["storage_m3"]
        residual = delta["transfer_m3"] + source - export
        scale = max(1, abs(export), abs(delta["transfer_m3"]), b["storage_m3"])
        if not np.isfinite(residual) or abs(residual) > 1e-8 * scale:
            raise ArithmeticError(f"Strip {m} m water balance residual {residual} m³")
        rows.append(
            dict(
                strip_width_m=m,
                net_export_m3=export,
                net_into_strip_m3=delta["transfer_m3"],
                start_storage_m3=b["storage_m3"],
                end_storage_m3=after[m]["storage_m3"],
                rain_m3=delta["rain_m3"],
                infiltration_m3=delta["infiltration_m3"],
                net_storage_contribution_m3=source,
                fraction_from_strip_storage=source / export if export > 0 else None,
                residual_m3=residual,
            )
        )
    return rows


def from_state(*, path, shape, distances, dx):
    with np.load(path) as state:
        return measurements(
            h=state["q"][:, 0].reshape(shape),
            soil=state["soil"].reshape(shape),
            rain=state["rain_sum"].reshape(shape),
            volumes=state["volume_x"],
            distances=distances,
            dx=dx,
        )


def from_model(*, model, distances):
    solver = model.solver
    shape = solver.ny, solver.nx
    return measurements(
        h=solver.get_water_depth(),
        soil=solver._fields["soil"].numpy().reshape(shape),
        rain=solver._fields["rain_sum"].numpy().reshape(shape),
        volumes=solver._fields["volume_x"].numpy(),
        distances=distances,
        dx=int(solver.dx),
    )


def replay(*, directory, prefix, series, settings, start_path, output):
    root = directory.parents[2]
    opts = json.loads((directory / "recession_settings.json").read_text())
    definition = json.loads((prefix["path"] / "result.json").read_text())["inputs"]
    physical = definition["physical"]
    receipt = definition["terrain"]
    bundle = (
        root / ".fidelity-restarts/native_inputs" / runner.digest(receipt["inputs"])
    )
    native_inputs.verify(folder=bundle, identity=receipt["inputs"])
    frames = series["frames"]
    original_end = round(prefix["s"]["hours"] * 3600)
    peak_time = max((f for f in frames if f["t_s"] > original_end), key=outlet)["t_s"]
    stop = min(
        series["end_s"],
        max(
            settings["minimum_replay_end_h"] * 3600,
            peak_time + 60 * settings["minutes_after_crest"],
        ),
    )
    stamp = runner.digest(
        dict(
            series=series["id"],
            stop=stop,
            source=runner.checksum(Path(__file__)),
            settings=settings,
        )
    )[:24]
    target = output / stamp
    if (target / "replay.json").exists():
        return
    target.mkdir(exist_ok=True)
    spec = dict(prefix["s"], hours=opts["forcing_hours"])
    model, _, _ = native_event.create(
        config=physical["cfg"],
        template=physical["template"],
        spec=spec,
        device=opts["device"],
        bundle=bundle,
        receipt=receipt,
    )
    if (
        continuation.seed_path(directory=directory, model=model, end=original_end)
        / "state.npz"
        != start_path
    ):
        raise ValueError(
            "Full native input identity does not match the same-grid checkpoint"
        )
    load_checkpoint(model=model, path=start_path)
    prev = from_model(model=model, distances=settings["upstream_distances_m"])
    rows = []
    for frame in frames:
        end = frame["t_s"]
        if end <= original_end or end > stop:
            continue
        start = model.solver.time
        stat = advance(model=model, end_time=end)
        if stat["steps"] != frame["steps"]:
            raise ValueError("Diagnostic replay changed the accepted timestep count")
        current = from_model(model=model, distances=settings["upstream_distances_m"])
        delta = changes(before=prev, after=current)
        dt = end - start
        for m, key in ((0, "outlet_m3s"), (max(prev), "section_m3s")):
            measured = (current[m]["transfer_m3"] - prev[m]["transfer_m3"]) / dt
            np.testing.assert_allclose(measured, frame[key], rtol=2e-11, atol=1e-10)
        for r in delta:
            rows.append(
                dict(
                    start_s=start,
                    end_s=end,
                    accepted_steps=stat["steps"],
                    net_into_strip_m3s=r["net_into_strip_m3"] / dt,
                    **r,
                )
            )
        prev = current
    if model.solver.time != stop:
        raise ValueError("Incomplete diagnostic horizon")
    metrics.csv_write(file=target / "minute_strips.csv", rows=rows)
    runner.write_json(
        path=target / "replay.json",
        data=dict(
            parent=prefix["id"],
            series=series["id"],
            dx_m=spec["dx"],
            method=spec["method"],
            settings=settings,
            start_s=original_end,
            end_s=stop,
            highest_post_window_outlet_interval_end_s=peak_time,
            original_checkpoint_sha256=runner.checksum(start_path),
            result_sha256=runner.checksum(target / "minute_strips.csv"),
            code=runner.checksum(Path(__file__)),
            comparison="Every accepted interval count and both original station volumes verified",
        ),
    )


def outlet(row):
    return row["outlet_m3s"]


def crest_windows(*, histories, rain_end_s, original_end_s):
    """Compare the same recorded horizon on every available policy and grid."""
    rows = []
    horizons = {original_end_s}
    for history in histories:
        horizons.update(history.get("horizons", ()))
        horizons.add(history["frames"][-1]["t_s"])
    for end in sorted(horizons):
        for history in histories:
            frames = history["frames"]
            if frames[-1]["t_s"] < end:
                continue
            t = np.asarray([r["t_s"] for r in frames])
            q = np.asarray([r["outlet_m3s"] for r in frames])
            dt = np.diff(t, prepend=0)
            if not np.all(dt == dt[0]) or end not in t or rain_end_s not in t:
                raise ValueError(
                    "Outlet comparisons require aligned accepted intervals"
                )
            for label, start, stop in (
                ("through_rain", 0, rain_end_s),
                ("after_rain", rain_end_s, end),
                ("after_original_window", original_end_s, end),
                ("whole_record", 0, end),
            ):
                selection = (t > start) & (t <= stop)
                if not selection.any():
                    continue
                peak = q[selection].max()
                times = t[selection & (q == peak)]
                rows.append(
                    dict(
                        id=history["id"],
                        method=history["method"],
                        dx_m=history["dx"],
                        window=label,
                        start_s=start,
                        end_s=stop,
                        comparison_end_s=end,
                        recession_passed_at_end=history["settled"] and end == t[-1],
                        interval_s=int(dt[0]),
                        peak_interval_mean_m3s=float(peak),
                        first_peak_interval_end_s=int(times[0]),
                        last_peak_interval_end_s=int(times[-1]),
                        net_volume_m3=float(q[selection] @ dt[selection]),
                    )
                )
    return rows


def cadence_peaks(*, histories, windows, minutes):
    """Coarsen accepted interval volumes with no reconstruction of peaks between observations."""
    records = {v["id"]: v["frames"] for v in histories}
    rows = []
    for window in windows:
        if window["window"] != "whole_record":
            continue
        original = window["interval_s"]
        for m in minutes:
            period = 60 * m
            duration = window["end_s"]
            if period % original or duration % period:
                raise ValueError(
                    "Sampling period must combine complete recorded intervals"
                )
            frames = records[window["id"]][: duration // original]
            q = np.array([f["outlet_m3s"] for f in frames])
            coarse = q.reshape(-1, period // original).mean(axis=1)
            count = period // original
            moving = np.convolve(q, np.ones(count) / count, mode="valid")
            peak = float(coarse.max())
            np.testing.assert_allclose(
                coarse.sum() * period, window["net_volume_m3"], rtol=2e-13, atol=1e-9
            )
            rows.append(
                dict(
                    id=window["id"],
                    dx_m=window["dx_m"],
                    method=window["method"],
                    end_h=duration / 3600,
                    interval_s=period,
                    peak_m3s=peak,
                    first_peak_end_s=int((np.argmax(coarse) + 1) * period),
                    moving_peak_m3s=float(moving.max()),
                    first_moving_peak_end_s=int((np.argmax(moving) + count) * original),
                    net_volume_m3=window["net_volume_m3"],
                )
            )
    return rows


def plot_outlet(*, dest, histories, forcing):
    import matplotlib.pyplot as plt
    from research.floodwarp_methods_paper.fidelity.figures import save

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 6), layout="constrained")
    hmax = max(v["frames"][-1]["t_s"] for v in histories) / 3600
    hours = np.asarray(forcing["rain_times_s"]) / 3600
    rates = np.asarray(forcing["rain_rates_ms"]) * 3.6e6
    if rates[-1] != 0:
        raise ValueError("Rainfall forcing is incomplete")
    axes[0].plot(np.r_[hours, max(hours[-1], hmax)], np.r_[rates, 0], color="#22658a")
    axes[0].set_ylabel("Rain multiplier = 1\nintensity (mm/h)")
    colour = {"fv1": "#ab5f16", "hancock_adaptive": "#00775d", "adaptive": "#343b89"}
    names = {"fv1": "FV1", "hancock_adaptive": "Hancock", "adaptive": "FV2 RK2"}
    for v in sorted(histories, key=history_order):
        frames = v["frames"]
        times = np.r_[0, [f["t_s"] for f in frames]] / 3600
        for ax, name in zip(axes[1:], ("outlet_m3s", "section_m3s"), strict=True):
            ax.stairs(
                [f[name] for f in frames],
                times,
                baseline=None,
                color=colour[v["method"]],
                linestyle="-" if v["dx"] == 5 else "--",
                linewidth=1,
                label=f"{names[v['method']]}, {v['dx']} m",
            )
    axes[1].set_ylabel("Outlet (m³/s)")
    axes[2].set_ylabel("Mid-catchment\nsection (m³/s)")
    axes[2].set_xlabel("Hours from storm onset")
    axes[0].set_xlim(0, hmax)
    axes[1].legend(fontsize=7, ncols=3)
    save(fig=fig, folder=dest, name="incised_rain_and_routed_discharge")


def history_order(v):
    return v["method"], v["dx"]


def report(*, dest, rows, strip_m, crests, cadence):
    summary = [r for r in rows if r["strip_width_m"] == strip_m]
    lines = [
        "# Discharge after rainfall in the incised catchment",
        "",
        "Rainfall ended at hour six. Routed runoff can peak during recession. "
        "These budgets compare measured transport through two sections with changes in storage and infiltration between them. "
        "The downstream section is the open outlet. The upstream section spans the entire domain width.",
        "",
        f"## Water balance over the last {strip_m} metres",
        "",
        "| Method | Grid (m) | Window (h) | Outlet volume (m³) | Net volume entering reach (m³) | Storage released less infiltration plus rain (m³) | Reach contribution (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "adaptive": "FV2 RK2",
        "hancock_adaptive": "Hancock tiled",
        "fv1": "FV1 sparse",
    }
    for row in summary:
        fraction = row["fraction_from_strip_storage"]
        share = "undefined" if fraction is None else f"{100 * fraction:.1f}"
        lines.append(
            f"| {names[row['method']]} | {row['dx_m']} | {row['start_h']:g}–{row['end_h']:g} | "
            f"{row['net_export_m3']:,.1f} | {row['net_into_strip_m3']:,.1f} | "
            f"{row['net_storage_contribution_m3']:,.1f} | {share} |"
        )
    matched = {(r["method"], r["dx_m"], r["start_h"], r["end_h"]): r for r in summary}
    paired = []
    for (method, dx, start, end), coarse in matched.items():
        if dx != 10 or (method, 5, start, end) not in matched:
            continue
        fine = matched[method, 5, start, end]
        paired.append(
            dict(
                method=method,
                start_h=start,
                end_h=end,
                strip_width_m=strip_m,
                coarse_id=coarse["id"],
                fine_id=fine["id"],
                extra_net_export_m3=coarse["net_export_m3"] - fine["net_export_m3"],
                extra_upstream_supply_m3=coarse["net_into_strip_m3"]
                - fine["net_into_strip_m3"],
                extra_storage_contribution_m3=coarse["net_storage_contribution_m3"]
                - fine["net_storage_contribution_m3"],
            )
        )
    metrics.csv_write(file=dest / "paired_reach_budgets.csv", rows=paired)
    for pair in paired:
        lines.extend(
            [
                "",
                f"Between hours {pair['start_h']:g} and {pair['end_h']:g}, the 10 m minus 5 m "
                f"{names[pair['method']]} difference in outlet volume was {pair['extra_net_export_m3']:+,.0f} m³. "
                f"The upstream face supplied {pair['extra_upstream_supply_m3']:+,.0f} m³ of this difference. "
                f"Storage release in the reach, including net rain and infiltration, supplied {pair['extra_storage_contribution_m3']:+,.0f} m³.",
            ]
        )
    lines.extend(
        [
            "",
            "Each row uses the same run’s eight-hour checkpoint and latest saved endpoint. "
            "Rows with different endpoints describe different time windows. The source split for the whole window cannot establish "
            "the origin of a one-minute peak. Agreement between accepted volumes and storage changes checks conservation. "
            "The magnitude of the physical crest requires a separately qualified hydraulic reference.",
            "",
            "[All section budgets](strip_budgets.csv) · [Boundary location controls](../native_outlet/README.md)",
        ]
    )
    lines.extend(
        [
            "",
            "## Outlet maxima by observation horizon",
            "",
            "Each row covers the interval from storm onset to the stated end time. Times refer to the end of the highest measured discharge interval. "
            "Only rows sharing an end time are directly comparable. A later maximum can arise from routed runoff, stored water or both.",
            "",
            "| Method | Grid (m) | End (h) | Max interval mean (m³/s) | Peak interval ends (h:min) | Recession verified at end |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in crests:
        if row["window"] != "whole_record":
            continue
        hours, remaining = divmod(row["first_peak_interval_end_s"], 3600)
        lines.append(
            f"| {names[row['method']]} | {row['dx_m']} | {row['end_s'] / 3600:g} | "
            f"{row['peak_interval_mean_m3s']:.3f} | {hours}:{remaining // 60:02d} | "
            f"{'yes' if row['recession_passed_at_end'] else 'unverified'} |"
        )
    lines.extend(
        [
            "",
            "[Peaks at different observation intervals](interval_peaks.csv) · [Rainfall and recession windows](crest_windows.csv) · [Rainfall and both discharge sections](incised_rain_and_routed_discharge.pdf)",
        ]
    )
    latest = max(r["end_h"] for r in cadence)
    lines.extend(
        [
            "",
            f"## Peak sensitivity to sampling through hour {latest:g}",
            "",
            "The fixed intervals start at storm onset. The moving interval can end at any recorded minute. "
            "Both average accepted discharge volumes over the full stated duration.",
            "",
            "| Method | Grid (m) | Interval (min) | Fixed peak (m³/s) | Moving peak (m³/s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in cadence:
        if r["end_h"] == latest:
            lines.append(
                f"| {names[r['method']]} | {r['dx_m']} | {r['interval_s'] / 60:g} | "
                f"{r['peak_m3s']:.2f} | {r['moving_peak_m3s']:.2f} |"
            )
    (dest / "README.md").write_text("\n".join(lines) + "\n")


def run(*, directory, settings, replay_peak):
    root = directory.parents[2] / ".fidelity-restarts"
    prefixes = native_products.prefixes(directory=directory)
    checkpoints = {}
    for path in (root / "prefixes").glob("*/state.npz"):
        with np.load(path) as state:
            sha = json.loads(state["_direct_manifest"].item())["sha256"].get("q")
            if sha:
                checkpoints[sha] = path
    dest = directory / "native_tail"
    dest.mkdir(exist_ok=True)
    total = []
    histories = {
        ident: dict(
            id=ident,
            method=v["s"]["method"],
            dx=v["s"]["dx"],
            frames=v["d"]["frames"],
            settled=False,
        )
        for ident, v in prefixes.items()
        if v["path"].parent == directory / "mean_storms"
        and v["s"]["case"] == settings["case"]
    }
    for p in sorted((directory / "extended_runs").glob("*/series.json")):
        s = json.loads(p.read_text())
        if s["cohort"] != "area-mean" or s["spec"]["case"] != settings["case"]:
            continue
        histories[s["parent_id"]].update(
            id=s["id"],
            frames=s["frames"],
            settled=s["settled"],
            horizons=[
                int(f.name.split(".")[0])
                for f in (p.parent / "horizons").glob("*.json.gz")
            ],
        )
        prefix = prefixes[s["parent_id"]]
        dx = s["spec"]["dx"]
        start = checkpoints[prefix["d"]["hashes"]["state"]]
        end = root / s["id"] / f"{int(s['end_s'])}.npz"
        if not end.is_file():
            raise ValueError("Missing latest native recession checkpoint")
        args = dict(
            shape=tuple(prefix["d"]["forcing"]["shape"]),
            dx=dx,
            distances=settings["upstream_distances_m"],
        )
        with np.load(end) as state:
            if json.loads(state["metadata"].item())["time"] != s["end_s"]:
                raise ValueError("Wrong final time")
        diff = changes(
            before=from_state(path=start, **args), after=from_state(path=end, **args)
        )
        crest = max(s["frames"], key=outlet)
        observed = []
        for row in diff:
            observed.append(
                dict(
                    id=s["id"],
                    method=s["spec"]["method"],
                    dx_m=dx,
                    start_h=prefix["s"]["hours"],
                    end_h=s["end_s"] / 3600.0,
                    crest_minute_end_s=crest["t_s"],
                    peak_outlet_m3s=crest["outlet_m3s"],
                    recession_passed=s["settled"],
                    section_at_crest_m3s=crest["section_m3s"],
                    **row,
                )
            )
        budget_dir = dest / "horizons" / s["id"]
        budget_dir.mkdir(exist_ok=True, parents=True)
        metrics.csv_write(file=budget_dir / f"{s['end_s']}.csv", rows=observed)
        total.extend(observed)
        if (
            replay_peak
            and s["settled"]
            and [dx, s["spec"]["method"]] in settings["replay"]
        ):
            replay(
                directory=directory,
                prefix=prefix,
                series=s,
                settings=settings,
                start_path=start,
                output=dest,
            )
    metrics.csv_write(file=dest / "strip_budgets.csv", rows=total)
    original_ends = {prefixes[k]["s"]["hours"] * 3600 for k in histories}
    if len(original_ends) != 1:
        raise ValueError("Missing or mismatched original native forecast windows")
    rain = json.loads((directory / "recession_settings.json").read_text())
    crests = crest_windows(
        histories=histories.values(),
        rain_end_s=rain["rain_end_s"],
        original_end_s=original_ends.pop(),
    )
    metrics.csv_write(file=dest / "crest_windows.csv", rows=crests)
    cadence = cadence_peaks(
        histories=histories.values(),
        windows=crests,
        minutes=settings["integration_minutes"],
    )
    metrics.csv_write(file=dest / "interval_peaks.csv", rows=cadence)
    report(
        dest=dest,
        rows=total,
        strip_m=settings["report_strip_m"],
        crests=crests,
        cadence=cadence,
    )
    plot_outlet(
        dest=dest,
        histories=histories.values(),
        forcing=prefixes[next(iter(histories))]["d"]["forcing"],
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "native_tail_settings.json").read_text())
    replay_peak = sys.argv[1:] == ["native-peak-replay"]
    run(directory=directory, settings=settings, replay_peak=replay_peak)
