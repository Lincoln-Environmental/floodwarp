"""Continue matched storms on their native output cadence until late exported transport subsides."""

import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import warp as wp

from floodwarp.model import advance, diagnostics, load_checkpoint, save_checkpoint
from research.floodwarp_methods_paper import events, runner
from research.floodwarp_methods_paper.fidelity import assessment
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix


def settled(*, frames, storage, supply, config):
    """Test a complete trailing window, separately at the internal section and outlet."""
    t = np.asarray([r["t_s"] for r in frames])
    dt = np.diff(np.r_[0.0, t])
    if not len(t) or t[-1] < config["rain_end_s"] + config["window_s"]:
        return False, {}
    late = t > t[-1] - config["window_s"]
    details = {}
    ok = True
    for station in ("outlet", "section"):
        q = np.asarray([r[station + "_m3s"] for r in frames])
        out = np.maximum(q, 0.0)
        volume = out @ dt
        peak = max(out)
        if peak < config["minimum_peak_m3s"] or volume < config["minimum_volume_m3"]:
            # Unexported water upstream can still form a delayed pulse.
            dry = storage <= max(
                config["minimum_volume_m3"], config["dry_supply_fraction"] * supply
            )
            sustained_inflow = np.all(q[late] < -config["minimum_peak_m3s"])
            passed = bool(dry or sustained_inflow)
            peak_ratio = volume_ratio = None
        else:
            peak_ratio = max(out[late]) / peak
            volume_ratio = out[late] @ dt[late] / volume
            passed = bool(
                peak_ratio <= config["tail_fraction"]
                and volume_ratio <= config["late_volume_fraction"]
            )
        details[station] = dict(
            passed=passed,
            terminal_m3s=float(q[-1]),
            peak_m3s=float(peak),
            recent_to_peak=peak_ratio,
            recent_to_volume=volume_ratio,
        )
        ok &= passed
    return bool(ok), details


def atomic_json(*, path, payload):
    pending = path.with_suffix(".pending")
    runner.write_json(path=pending, data=payload)
    pending.replace(path)


def build(*, it, study, config, settings):
    spec = dict(it["s"])
    if "dimensions" not in spec:
        spec.update(
            dimensions=config["synthetic"]["dimensions_m"],
            interval=900,
            support=10,
            initial_grid="sampled",
            controls=False,
            coastal_stage=False,
        )
    template = json.loads(
        (
            study.parents[1] / "validation/hancock_long_storm_review/cases.json"
        ).read_text()
    )
    template["storms_mm_h"]["double"] = settings["double_mm_h"]
    options = matrix.design(study=study)
    normal = events.numerical

    def numeric(*, method):
        if method in ("roe", "hllc"):
            return "fv1", "sparse", dict(flux_scheme=method, cfl=0.4 * it["factor"])
        if method == "hancock_auto":
            raise ValueError(
                "Use its hydraulically identical explicit policy for continuation"
            )
        key, mode, opts = normal(method=method)
        if method != "li":
            opts = dict(opts, cfl=0.4 * it["factor"])
        return key, mode, opts

    def geometry(**opts):
        return hydraulics.represented(options=options, settings=spec, **opts)

    # The last input node remains zero throughout this horizon. The stopping decision uses actual state.
    with (
        patch.object(events, "fields", geometry),
        patch.object(events, "numerical", numeric),
    ):
        model, expected, forcing = events.create_event(
            config=config,
            template=template,
            spec=dict(spec, hours=settings["forcing_hours"]),
            device=settings["device"],
        )
    return spec, model, expected, forcing, options


def advance_frames(*, model, spec, start, end, options, device):
    li = spec["method"] == "li"
    solver = model.solver
    prev = hydraulics.volumes(solver=solver, li=li)
    rows = []
    for t in range(start + spec["interval"], end + 1, spec["interval"]):
        last = t - spec["interval"]
        if spec["coastal_stage"] and last % 900 == 0:
            level = np.interp(
                last / 3600, options["coastal_stage_hours"], options["coastal_stage_m"]
            )
            solver.set_boundary_stage_qx(
                stage=np.full((solver.ny, solver.nx + 1), level)
            )
        wp.synchronize_device(device=device)
        tick = perf_counter()
        stats = advance(model=model, end_time=float(t))
        wp.synchronize_device(device=device)
        elapsed = (
            perf_counter()
            - tick
            - stats["graph_capture_seconds"]
            - stats.get("compilation_seconds", 0.0)
        )
        volumes = hydraulics.volumes(solver=solver, li=li)
        section, outlet = (volumes - prev) / spec["interval"]
        rows.append(
            dict(
                t_s=t,
                section_m3s=float(section),
                outlet_m3s=float(outlet),
                hydraulic_s=elapsed,
                steps=stats["steps"],
            )
        )
        prev = volumes
    return rows


def verify_prefix(*, old, new):
    reference = old.get("frames", old.get("samples"))
    if len(reference) != len(new):
        raise ValueError("Different prefix time grids")
    for field in ("t_s", "section_m3s", "outlet_m3s", "steps"):
        a = np.array([r[field] for r in reference])
        b = np.array([r[field] for r in new])
        if field in ("t_s", "steps"):
            np.testing.assert_array_equal(a, b, err_msg=field)
        else:
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=1e-10, err_msg=field)


def prolong(*, it, directory, config, settings, floor_s, run_sha):
    ident = runner.digest(
        dict(parent=it["id"], factor=it["factor"], config=settings, source=run_sha)
    )[:24]
    dest = directory / "extended_runs" / ident
    dest.mkdir(exist_ok=True, parents=True)
    path = dest / "series.json"
    archived = json.loads(path.read_text()) if path.exists() else None
    if archived and archived["end_s"] >= floor_s and archived["settled"]:
        return archived
    spec, model, expected, forcing, options = build(
        it=it, study=directory.parent, config=config, settings=settings
    )
    cache = directory.parents[2] / ".fidelity-restarts" / ident
    cache.mkdir(exist_ok=True, parents=True)
    old = it["d"]
    base = int(it["s"]["hours"] * 3600)
    if archived:
        start = archived["end_s"]
        checkpoint = cache / f"{start}.npz"
        if checkpoint.is_file():
            load_checkpoint(model=model, path=checkpoint)
            frames = archived["frames"]
        else:
            # A portable checkout can reproduce the same prefix without a workstation-only restart.
            frames = advance_frames(
                model=model,
                spec=spec,
                start=0,
                end=start,
                options=options,
                device=settings["device"],
            )
            verify_prefix(old=archived, new=frames)
    else:
        start = base
        frames = advance_frames(
            model=model,
            spec=spec,
            start=0,
            end=base,
            options=options,
            device=settings["device"],
        )
        verify_prefix(old=old, new=frames)
    maximum = settings["forcing_hours"] * 3600
    target = max(floor_s, start)
    while start < maximum:
        if start < target or (archived and not archived["settled"]):
            stop = min(maximum, start + settings["checkpoint_hours"] * 3600)
            stop = min(stop, target) if start < target else stop
            frames.extend(
                advance_frames(
                    model=model,
                    spec=spec,
                    start=start,
                    end=stop,
                    options=options,
                    device=settings["device"],
                )
            )
        else:
            stop = start
        budget = diagnostics(model=model)
        supply = expected + model.solver.initial_volume + budget["boundary_inflow"]
        if abs(budget["rain"] - expected) > 1e-8 * max(1, expected) or abs(
            budget["residual_m3"]
        ) > 1e-8 * max(1, supply):
            raise ArithmeticError("Extended storm failed its rain or mass balance")
        passed, diagnosis = settled(
            frames=frames,
            storage=budget["storage_m3"],
            supply=max(expected + model.solver.initial_volume, 1),
            config=settings,
        )
        archive = dict(
            id=ident,
            parent_id=it["id"],
            spec=spec,
            cohort=it["cohort"],
            factor=it["factor"],
            end_s=stop,
            settled=passed,
            checks=diagnosis,
            frames=frames,
            forcing=forcing,
            storage_m3=budget["storage_m3"],
            rain_m3=budget["rain"],
            infiltration_m3=budget["infiltration"],
            residual_m3=budget["residual_m3"],
            hydraulic_s=sum(r["hydraulic_s"] for r in frames),
            hardware=settings["device"],
            source_sha256=run_sha,
        )
        save_checkpoint(model=model, path=cache / f"{stop}.npz")
        atomic_json(path=path, payload=archive)
        if start != stop:
            (cache / f"{start}.npz").unlink(missing_ok=True)
        if passed and stop >= floor_s:
            return archive
        start = stop
        archived = archive
    raise RuntimeError(
        f"{spec['case']} still has unresolved transport after {maximum / 3600:g} h; inspect the boundary and remaining storage"
    )


def candidates(*, items, cohort):
    if cohort in ("legacy", "double"):
        return items
    # Exact FP64 dense/tiled/automatic parity is qualified in the existing storm archive.
    # Retain measured cost of the explicit method actually continued.
    selected = {}
    for it in items:
        s = it["s"]
        if s["method"] == "hancock_auto":
            continue
        key = (
            s["dx"],
            "hancock"
            if s["method"] in ("hancock_dense", "hancock_adaptive")
            else s["method"],
            it["factor"],
        )
        if (
            key not in selected
            or it["d"]["hydraulic_s"] < selected[key]["d"]["hydraulic_s"]
        ):
            selected[key] = it
    return list(selected.values())


def run(*, directory, settings, cohorts):
    study = directory.parent
    cfg = read_config(path=study / "configs/study.json")
    groups = assessment.archive(directory=directory, config=cfg, partial=True)
    machine = runner.machine(root=study.parents[1])
    source = runner.digest(
        dict(
            implementation=runner.checksum(Path(__file__)),
            solver=runner.source_identity(
                root=study.parents[1], study=study, stage="long-storms"
            ),
            adapter=runner.checksum(study / "qualification/hydraulics.py"),
            hardware=[machine["gpu"], machine["warp"]],
            template=runner.checksum(
                study.parents[1] / "validation/hancock_long_storm_review/cases.json"
            ),
            config=cfg,
        )
    )
    output = directory / "extended_runs"
    output.mkdir(exist_ok=True)
    for (cohort, case, rep), items in sorted(groups.items()):
        if (
            cohort not in cohorts
            or rep not in ("sampled", "10m bed")
            or cohort == "fixed"
        ):
            continue
        if (
            cohort == "legacy"
            and any(
                i["s"].get("dimensions") == [120, 80] and i["factor"] != 1
                for i in items
            )
            is False
        ):
            raise ValueError("Missing temporal controls for " + case)
        selected = candidates(items=items, cohort=cohort)
        # Same experiment ID can occur in distinct immutable stages, never count it twice.
        selected = list({i["id"]: i for i in selected}.values())
        need = False
        for it in selected:
            d = it["d"]
            ok, _ = settled(
                frames=d.get("samples", d.get("frames")),
                storage=d.get("final_m3", d.get("final_storage_m3")),
                supply=d["initial_m3"] + d["rain_m3"],
                config=settings,
            )
            need |= not ok
        key = runner.digest(
            dict(
                case=case,
                cohort=cohort,
                rep=rep,
                source=source,
                settings=settings,
                inputs=sorted(i["id"] for i in selected),
            )
        )[:22]
        receipt = output / f"group_{key}.json"
        if receipt.exists():
            continue
        end = max(int(i["s"]["hours"] * 3600) for i in selected)
        records = []
        repeat_end = settings.get("timing_repeat_end_s", end)
        if repeat_end < end:
            raise ValueError("Timing repeat ends before the primary window")
        need |= repeat_end > end
        if need:
            floor = max(end + settings["first_extension_h"] * 3600, repeat_end)
            while True:
                records = [
                    prolong(
                        it=i,
                        directory=directory,
                        config=cfg,
                        settings=settings,
                        floor_s=floor,
                        run_sha=source,
                    )
                    for i in selected
                ]
                floor = max(r["end_s"] for r in records)
                if all(r["end_s"] == floor and r["settled"] for r in records):
                    break
            end = floor
        atomic_json(
            path=receipt,
            payload=dict(
                case=case,
                cohort=cohort,
                representation=rep,
                end_s=end,
                extended=need,
                sources=[i["id"] for i in selected],
                series=[r["id"] for r in records],
                hardware=machine,
                settings=settings,
                source=source,
            ),
        )
        print(
            cohort,
            case,
            end / 3600,
            "hours",
            len(selected),
            "matched configurations",
            flush=True,
        )


if __name__ == "__main__":
    import sys

    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "recession_settings.json").read_text())
    cohorts = tuple(sys.argv[1:]) or ("legacy", "expanded", "double")
    run(directory=directory, settings=settings, cohorts=cohorts)
