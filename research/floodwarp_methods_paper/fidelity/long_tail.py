"""Test whether remaining mixed-catchment storage materially changes the late hydrograph."""

import json
import numpy as np
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import assessment, receipts, recession
from research.floodwarp_methods_paper.plan import read_config


def late_export(*, frames, window, fraction):
    t = np.r_[0, [r["t_s"] for r in frames]]
    duration = np.maximum(0, t[1:] - np.maximum(t[:-1], t[-1] - window))
    result = {}
    for name in ("outlet", "section"):
        q = np.maximum([r[name + "_m3s"] for r in frames], 0.0)
        whole = float(q @ np.diff(t))
        late = float(q @ duration)
        result[name] = dict(
            final_outward_m3=whole,
            late_outward_m3=late,
            late_fraction=late / whole if whole else None,
            passed=late <= fraction * whole,
        )
    return result


def run(*, directory, settings):
    options = json.loads((directory / "long_tail_settings.json").read_text())
    config = read_config(path=directory.parent / "configs/study.json")
    archive = assessment.archive(directory=directory, config=config, partial=True)
    key = ("expanded", options["case"], "sampled")
    original = [
        i
        for i in archive[key]
        if i["s"]["method"] in options["methods"]
        and i["s"]["dx"] == options["dx"]
        and i["factor"] == 1
    ]
    ids = {i["id"] for i in original}
    folder = directory / "extended_runs"
    measured = [json.loads(f.read_text()) for f in folder.glob("group_*.json")]
    matches = [
        r
        for r in measured
        if set(r["sources"]) == ids
        and (r["cohort"], r["case"], r["representation"]) == key
    ]
    if len(matches) != 1 or len(original) != len(options["methods"]):
        raise ValueError(
            "The native comparison must finish before checking residual storage"
        )
    first = matches[0]
    target = directory / "storage_tail"
    target.mkdir(exist_ok=True)
    signature = runner.digest(
        dict(
            first=first["series"],
            end=first["end_s"],
            settings=options,
            implementation=runner.checksum(directory / "long_tail.py"),
        )
    )
    path = target / f"{signature}.json"
    if path.exists():
        return
    floor = max(options["minimum_hours"] * 3600, first["end_s"])
    parents = {i["id"]: i for i in original}
    rows = []
    while floor <= settings["forcing_hours"] * 3600:
        results = [
            recession.prolong(
                it=item,
                directory=directory,
                config=config,
                settings=settings,
                floor_s=floor,
                run_sha=first["source"],
            )
            for item in original
        ]
        end = max(r["end_s"] for r in results)
        if any(r["end_s"] != end for r in results):
            floor = end
            continue
        passed = True
        for r in results:
            early = receipts.read(root=folder, ident=r["id"], end=first["end_s"])
            for station, tail in late_export(
                frames=r["frames"],
                window=options["tail_window_hours"] * 3600,
                fraction=options["late_volume_fraction"],
            ).items():
                passed = passed and tail["passed"] and r["checks"][station]["passed"]
                before = late_export(
                    frames=early["frames"],
                    window=options["tail_window_hours"] * 3600,
                    fraction=options["late_volume_fraction"],
                )[station]
                rows.append(
                    dict(
                        case=r["spec"]["case"],
                        method=r["spec"]["method"],
                        id=r["id"],
                        first_h=first["end_s"] / 3600,
                        end_h=end / 3600,
                        station=station,
                        first_outward_m3=before["final_outward_m3"],
                        first_storage_m3=early["storage_m3"],
                        remaining_storage_m3=r["storage_m3"],
                        infiltration_m3=r["infiltration_m3"],
                        initial_m3=parents[r["parent_id"]]["d"]["initial_m3"],
                        rain_m3=r["rain_m3"],
                        residual_m3=r["residual_m3"],
                        **tail,
                    )
                )
        metrics.csv_write(file=target / "late_export.csv", rows=rows)
        if passed:
            runner.write_json(
                path=path,
                data=dict(
                    options=options,
                    original_end_s=first["end_s"],
                    common_end_s=end,
                    series=first["series"],
                    source=first["source"],
                ),
            )
            print(
                "Residual-storage check",
                options["case"],
                end / 3600,
                "hours",
                flush=True,
            )
            return
        floor = end + options["extension_hours"] * 3600
    raise RuntimeError("Residual-storage check exceeded the dry forcing coverage")
