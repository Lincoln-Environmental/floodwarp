"""Compare fixed and resampled terrain with a common, measured end to their recessions."""

from collections import defaultdict
import json
from pathlib import Path
from research.floodwarp_methods_paper import evidence, metrics, runner
from research.floodwarp_methods_paper.fidelity import assessment, recession
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import observations


def inputs(*, directory, config):
    study = directory.parent
    groups = defaultdict(list)
    for rec in evidence.records(study=study, stages=("resolution",)):
        s = rec["spec"]
        path = study / rec["directory"]
        spec = dict(
            s,
            dimensions=config["synthetic"]["dimensions_m"],
            interval=900,
            support=10,
            initial_grid="parent",
            controls=False,
            coastal_stage=False,
        )
        groups[s["case"]].append(
            assessment.item(
                spec=spec,
                data=json.loads((path / "result.json").read_text()),
                ident=rec["experiment_id"],
                path=path,
                cohort="geometry",
            )
        )
    expanded = observations.load_events(study=study, config=config)
    required = defaultdict(set)
    for (case, rep), items in expanded.items():
        if rep.startswith("fixed-"):
            required[case].update(
                (i["spec"]["dx"], method)
                for i in items
                for method in (i["spec"]["method"], "fv1")
            )
    for (case, rep), items in expanded.items():
        for i in items:
            s = i["spec"]
            if (s["dx"], s["method"]) in required[case]:
                groups[case].append(
                    assessment.item(
                        spec=s,
                        data=i["data"],
                        ident=i["id"],
                        path=i["file"].parent,
                        cohort="geometry",
                    )
                )
    return groups


def source_id(*, directory, cfg, settings):
    study = directory.parent
    machine = runner.machine(root=study.parents[1])
    return runner.digest(
        dict(
            implementation=[
                runner.checksum(p)
                for p in (
                    Path(__file__),
                    directory / "recession.py",
                    study / "qualification/hydraulics.py",
                    study / "qualification/matrix.py",
                )
            ],
            hydraulics=runner.source_identity(
                root=study.parents[1], study=study, stage="resolution"
            ),
            config=cfg,
            settings=settings,
            hardware=[machine["gpu"], machine["warp"]],
        )
    )


def run(*, directory, settings):
    cfg = read_config(path=directory.parent / "configs/study.json")
    source = source_id(directory=directory, cfg=cfg, settings=settings)
    for case, raw in sorted(inputs(directory=directory, config=cfg).items()):
        items = list({i["id"]: i for i in raw}.values())
        key = runner.digest(dict(source=source, inputs=sorted(i["id"] for i in items)))[
            :22
        ]
        receipt = directory / "bed_runs" / f"{key}.json"
        if receipt.is_file():
            continue
        end = max(round(i["s"]["hours"] * 3600) for i in items)
        failed = False
        for i in items:
            d = i["d"]
            ok, _ = recession.settled(
                frames=d.get("frames", d.get("samples")),
                storage=d.get("final_m3", d.get("final_storage_m3")),
                supply=d["initial_m3"] + d["rain_m3"],
                config=settings,
            )
            failed |= not ok
        series = []
        if failed:
            end += 3600 * settings["first_extension_h"]
            while True:
                series = [
                    recession.prolong(
                        it=i,
                        directory=directory,
                        config=cfg,
                        settings=settings,
                        floor_s=end,
                        run_sha=source,
                    )
                    for i in items
                ]
                end = max(s["end_s"] for s in series)
                if all(s["end_s"] == end and s["settled"] for s in series):
                    break
        runner.write_json(
            path=receipt,
            data=dict(
                case=case,
                end_s=end,
                source=source,
                originals=[i["id"] for i in items],
                series=[s["id"] for s in series],
                extended=bool(series),
            ),
        )
        print(
            "bed comparison",
            case,
            end / 3600,
            "h",
            len(items),
            "configurations",
            flush=True,
        )


def compare_beds(*, directory, config):
    cfg = read_config(path=directory.parent / "configs/study.json")
    groups = inputs(directory=directory, config=cfg)
    observations_out = []
    for file in sorted((directory / "bed_runs").glob("*.json")):
        receipt = json.loads(file.read_text())
        items = groups.pop(receipt["case"])
        chosen = {i["id"]: i for i in items}
        if set(chosen) != set(receipt["originals"]):
            raise ValueError("Bed archive input mismatch")
        for ident in receipt["series"]:
            s = json.loads(
                (directory / "extended_runs" / ident / "series.json").read_text()
            )
            if not s["settled"] or s["end_s"] != receipt["end_s"]:
                raise ValueError("Incomplete terrain comparison")
            item = chosen[s["parent_id"]]
            item["d"] = {"frames": s["frames"], "initial_m3": item["d"]["initial_m3"]}
            item["id"] = ident
        for item in chosen.values():
            s = item["s"]
            if s["dem"] == s["dx"]:
                continue
            comparable = [
                i
                for i in chosen.values()
                if i["s"]["dx"] == s["dx"] and i["s"]["dem"] == s["dx"]
            ]
            same = next(
                (i for i in comparable if i["s"]["method"] == s["method"]), None
            )
            fv1 = next((i for i in comparable if i["s"]["method"] == "fv1"), None)
            if not same or not fv1:
                continue
            for station in ("outlet", "section"):
                terrain = assessment.compare(
                    it=item, ref=same, station=station, cfg=config
                )
                method = assessment.compare(
                    it=fv1, ref=same, station=station, cfg=config
                )
                for k in config["reference_limits"]:
                    observations_out.append(
                        dict(
                            case=s["case"],
                            method=s["method"],
                            station=station,
                            dx=s["dx"],
                            dem=s["dem"],
                            initial_grid=s["initial_grid"],
                            hours=receipt["end_s"] / 3600,
                            metric=k,
                            bed_difference=terrain.get(k),
                            formulation_difference=method.get(k),
                            observable=assessment.limits(row=terrain, key=k)
                            and assessment.limits(row=method, key=k),
                            bed_id=item["id"],
                            native_id=same["id"],
                            fv1_id=fv1["id"],
                            initial_volume_change_m3=item["d"]["initial_m3"]
                            - same["d"]["initial_m3"],
                        )
                    )
    if groups:
        raise ValueError(
            "Missing completed terrain comparisons: " + ", ".join(sorted(groups))
        )
    metrics.csv_write(
        file=directory / "results/bed_contrasts_complete.csv", rows=observations_out
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "recession_settings.json").read_text())
    run(directory=directory, settings=settings)
