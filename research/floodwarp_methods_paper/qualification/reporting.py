"""Publication and policy summaries from measured, physically matched experiments."""

from collections import defaultdict
import fnmatch
import json
from pathlib import Path
from statistics import median
from unittest.mock import patch

import numpy as np
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.qualification import observations, stages


def policies(*, rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[r["case"], r["representation"], float(r["dx"])][r["method"]] = r
    result = []
    for (case, rep, dx), g in sorted(groups.items()):
        if not {"fv1", "hancock_dense", "hancock_adaptive"} <= g.keys():
            continue
        fv, d, t = (g[k] for k in ("fv1", "hancock_dense", "hancock_adaptive"))
        best = min((d, t), key=runtime)
        row = dict(
            case=case,
            landform=fv["landform"],
            representation=rep,
            dx=dx,
            cells=int(fv.get("cells") or 9600 / dx**2),
            split=fv["split"],
            hours=fv["hours"],
            fv1_s=float(fv["runtime_s"]),
            dense_s=float(d["runtime_s"]),
            tiles_s=float(t["runtime_s"]),
            fv1_steps=fv["steps"],
            hancock_steps=t["steps"],
            rejected=t["rejected"],
            step_ratio=float(t["steps"]) / float(fv["steps"]),
            dense_ratio=runtime(d) / runtime(fv),
            tiles_ratio=runtime(t) / runtime(fv),
            fv1_ms_per_step=1000 * runtime(fv) / float(fv["steps"]),
            fastest_hancock_ms_per_step=1000 * runtime(best) / float(best["steps"]),
            per_step_ratio=(runtime(best) / float(best["steps"]))
            / (runtime(fv) / float(fv["steps"])),
            fastest_mode=best["method"],
            fastest_ratio=runtime(best) / runtime(fv),
            fv1_id=fv["experiment_id"],
            dense_id=d["experiment_id"],
            tiles_id=t["experiment_id"],
            wet_fraction=t["wet_fraction"],
            work_fraction=t.get("scheduled_fraction", t.get("active_fraction")),
            memory_mib=t["memory_mib"],
        )
        if "hancock_auto" in g:
            auto = g["hancock_auto"]
            row.update(
                auto_s=runtime(auto),
                auto_id=auto["experiment_id"],
                selected_policy=auto["selected_mode"],
                auto_ratio=runtime(auto) / runtime(fv),
                auto_regret_pct=100 * (runtime(auto) / runtime(best) - 1),
            )
        fastest = min(
            (
                g[k]
                for k in ("hancock_dense", "hancock_adaptive", "hancock_auto")
                if k in g
            ),
            key=runtime,
        )
        row.update(
            fastest_all_policy=fastest["method"],
            fastest_all_id=fastest["experiment_id"],
            fastest_all_ratio=runtime(fastest) / runtime(fv),
        )
        result.append(row)
    return result


def runtime(row):
    return float(row["runtime_s"])


def aggregate(*, study, config):
    results = observations.build(study=study, config=config)
    from research.floodwarp_methods_paper.qualification import attribution

    attribution.build(study=study, config=config)
    p = study / "processed"
    small = [
        r
        for r in metrics.csv_read(p / "metrics/storms.csv")
        if r["stage"] in ("long-storms", "resolution")
        and r["reference_setup"] == "False"
    ]
    primary = [
        r
        for r in results
        if r["stage"] in ("million-storms", "million-auto", "recession-controls")
    ]
    policy = policies(rows=small) + policies(rows=primary)
    metrics.csv_write(file=p / "million/policies.csv", rows=policy)
    geo = []
    for rec in observations.current(
        study=study, config=config, stage="million-terrain"
    ):
        d = json.loads((study / rec["directory"] / "result.json").read_text())
        for dx, v in d["window"].items():
            geo.append(
                dict(
                    case=rec["spec"]["case"],
                    model_dx=rec["spec"]["dx"],
                    diagnostic_dx=float(dx),
                    terrain_id=rec["experiment_id"],
                    domain_area_m2=d["domain"]["area_m2"],
                    domain_cells=d["domain"]["cells"],
                    terrain_sha256=d["terrain_sha256"],
                    **v["comparison"],
                )
            )
    metrics.csv_write(file=p / "million/terrain.csv", rows=geo)
    profiles = []
    from validation.hancock_long_storm_review.graph_profile import category

    for rec in observations.current(
        study=study, config=config, stage="million-profiles"
    ):
        d = json.loads((study / rec["directory"] / "result.json").read_text())
        s = rec["spec"]
        for name, val in d["stage_seconds"].items():
            profiles.append(
                dict(
                    case=s["case"],
                    method=s["method"],
                    kernel=name,
                    category="flux and update"
                    if name == "donor_budget"
                    else category(name=name),
                    kernel_seconds=val["seconds"],
                    launches=val["launches"],
                    steps=d["steps"],
                    experiment_id=rec["experiment_id"],
                )
            )
    metrics.csv_write(file=p / "million/graph_profile.csv", rows=profiles)
    repetitions = defaultdict(list)
    for stage in ("million-storms", "million-repeats"):
        for rec in observations.current(study=study, config=config, stage=stage):
            s = rec["spec"]
            if s["seed"] not in (701, 702) or s["method"] not in (
                "fv1",
                "hancock_adaptive",
                "hancock_dense",
            ):
                continue
            data = json.loads((study / rec["directory"] / "result.json").read_text())
            repetitions[s["case"], s["method"]].append(
                (
                    data["hydraulic_s"],
                    data["hashes"],
                    data["steps"],
                    rec["experiment_id"],
                )
            )
    repeats = []
    for (case, method), runs in repetitions.items():
        if any((h, n) != (runs[0][1], runs[0][2]) for _, h, n, _ in runs):
            raise ValueError(f"Repeat changed hydraulic result for {case} {method}")
        times = [r[0] for r in runs]
        repeats.append(
            dict(
                case=case,
                method=method,
                n=len(times),
                min_s=min(times),
                median_s=median(times),
                max_s=max(times),
                range_pct=100 * (max(times) - min(times)) / median(times),
                run_ids=" ".join(r[3] for r in runs),
            )
        )
    metrics.csv_write(file=p / "million/repeat_times.csv", rows=repeats)
    return dict(
        primary=len(primary),
        policy_comparisons=len(policy),
        terrain_records=len(geo),
        kernel_profiles=len(profiles),
    )


def offline(*, study, root, stage, config, flags, grid):
    match = dict(case=flags["case"], method=flags["method"], grid=grid)

    def specs(**_):
        yield dict(case=match["case"], method=match["method"], grid=grid)

    def identity(**_):
        from research.floodwarp_methods_paper.controller import derived_identity

        return derived_identity(root=root, study=study, stage="terrain-features")

    def calculate(*, directory, **_):
        archive = study / "processed/million"
        aggregate(study=study, config=config)
        name = "policies" if stage == "execution-policy" else "pareto"
        rows = metrics.csv_read(archive / (name + ".csv"))
        selected = []
        for r in rows:
            if not fnmatch.fnmatchcase(r["case"], match["case"]):
                continue
            if "dx" in grid and float(r["dx"]) != grid["dx"]:
                continue
            if "dem" in grid:
                representation = r["representation"]
                dem = (
                    float(r["dx"])
                    if representation == "sampled"
                    else 10
                    if representation == "fixed"
                    else float(representation.split("-")[1].removesuffix("m"))
                )
                if dem != grid["dem"]:
                    continue
            if stage == "pareto" and not fnmatch.fnmatchcase(
                r["method"], match["method"]
            ):
                continue
            if stage == "execution-policy" and match["method"] not in (
                "*",
                "fv1",
                "hancock_dense",
                "hancock_auto",
                "hancock_adaptive",
            ):
                continue
            selected.append(r)
        metrics.csv_write(file=directory / (name + ".csv"), rows=selected)
        return dict(rows=len(selected), selection=match)

    system = {} if flags["dry"] else offline_system(root=root, study=study)
    with (
        patch.object(runner, "SIMULATIONS", (*runner.SIMULATIONS, stage)),
        patch.object(runner, "execute", calculate),
        patch.object(runner, "specs", specs),
        patch.object(runner, "source_identity", identity),
    ):
        runner.run_stage(
            stage=stage,
            opts=flags,
            root=root,
            study=study,
            config=config,
            system=system,
        )


def offline_system(*, root, study):
    from research.floodwarp_methods_paper.controller import offline_machine

    return offline_machine(root=root, study=study)


def verification(*, study, config):
    counts = {}
    pairs = 0
    for stage in stages.NUMERICAL:
        expected = {
            runner.digest(dict(stage=stage, settings=s, config=config))[:24]
            for s in stages.specifications(
                stage=stage, config=config, root=study.parents[1]
            )
        }
        found = observations.current(study=study, stage=stage, config=config)
        if {r["experiment_id"] for r in found} != expected:
            raise ValueError(
                f"{stage}: {len(expected)} expected, {len(found)} complete"
            )
        for r in found:
            runner.verify_artifacts(record=r, study=study)
        counts[stage] = len(found)
    groups = observations.load_events(
        study=study,
        config=config,
        include=(
            "million-storms",
            "million-auto",
            "million-profiles",
            "recession-controls",
        ),
    )
    for items in groups.values():
        by_dx = defaultdict(list)
        for i in items:
            by_dx[i["spec"]["dx"]].append(i)
            s = i["spec"]
            d = i["data"]
            if d["hours_completed"] != s["hours"] or d["hours_completed"] < 8:
                raise ValueError("Incomplete storm " + i["id"])
            t = np.array(d["forcing"]["rain_times_s"])
            v = np.array(d["forcing"]["rain_rates_ms"])
            if (
                np.diff(t)[v[1:] + v[:-1] > 0].sum() < 21600
                or d["frames"][-1]["t_s"] < 28800
            ):
                raise ValueError("Incomplete rainfall or recession")
        for cohort in by_dx.values():
            if len({runner.digest(i["data"]["forcing"]) for i in cohort}) != 1:
                raise ValueError("Mismatched physical input")
            targets = [i for i in cohort if i["spec"]["method"].startswith("hancock")]
            base = next(
                i["data"] for i in targets if i["spec"]["method"] == "hancock_dense"
            )
            for i in targets:
                d = i["data"]
                for key in (
                    "hashes",
                    "final_storage_m3",
                    "rain_m3",
                    "infiltration_m3",
                    "steps",
                    "rejected",
                    "components",
                ):
                    if d[key] != base[key]:
                        raise ValueError(f"{key} parity failed for {i['id']}")
                for a, b in zip(d["frames"], base["frames"], strict=True):
                    for k in (
                        "steps",
                        "dt_min_s",
                        "dt_max_s",
                        "outlet_m3s",
                        "section_m3s",
                        "rejected",
                    ):
                        if a[k] != b[k]:
                            raise ValueError(
                                f"Interval parity failed for {i['id']} ({k})"
                            )
                pairs += int(d is not base)
    return dict(counts=counts, scheduler_comparisons=pairs)
