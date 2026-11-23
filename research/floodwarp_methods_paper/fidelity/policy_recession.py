"""Time every operational Hancock policy to the same end used by the completed hydraulic group."""

import json
from pathlib import Path
from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity import assessment, receipts, recession
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import automatic, launch, stages


def unmatched(*, items, selected):
    spacings = {i["s"]["dx"] for i in items if i["s"]["method"] == "hancock_auto"}
    seen = {
        (i["s"]["dx"], i["s"]["method"], i["factor"])
        for i in items
        if i["id"] in selected
    }
    missing = []
    for i in items:
        s = i["s"]
        key = s["dx"], s["method"], i["factor"]
        if (
            s["dx"] not in spacings
            or i["factor"] != 1
            or s["method"] not in ("hancock_auto", "hancock_dense", "hancock_adaptive")
        ):
            continue
        if key not in seen:
            missing.append(i)
            seen.add(key)
    return missing


def receipt_key(*, directory, config, base, end):
    study = directory.parent
    source = stages.source(root=study.parents[1], study=study, stage="million-auto")
    return runner.digest(
        dict(
            parent=base["id"],
            source=source,
            end=end,
            configuration=config,
            driver=runner.checksum(Path(__file__)),
        )
    )[:24]


def run(*, directory):
    study = directory.parent
    config = read_config(path=study / "configs/study.json")
    settings = json.loads((directory / "recession_settings.json").read_text())
    from research.floodwarp_methods_paper.fidelity.endpoints import endpoint_path
    import numpy as np

    groups = assessment.archive(directory=directory, config=config)
    entries = index(directory=directory)
    machine = runner.machine(root=study.parents[1])
    for (cohort, case, rep), items in sorted(groups.items()):
        if cohort != "expanded" or rep != "sampled":
            continue
        selected = {
            i["id"]: i for i in recession.candidates(items=items, cohort=cohort)
        }
        missing = unmatched(items=items, selected=selected)
        if not missing:
            continue
        receipt = entries[case, tuple(sorted(selected))]
        end = receipt["end_s"]
        for it in missing:
            dx = it["s"]["dx"]
            method = it["s"]["method"]
            ident = receipt_key(directory=directory, config=config, base=it, end=end)
            folder = directory / "policy_runs" / ident
            path = folder / "result.json"
            if path.exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            base = it["d"]
            matched = next(
                i
                for i in selected.values()
                if i["s"]["dx"] == dx
                and i["s"]["method"] in ("hancock_adaptive", "hancock_dense")
                and i["factor"] == 1
            )
            frame_source = matched["d"]
            expected_hashes = frame_source.get("hashes", {})
            if receipt["extended"]:
                for run_id in receipt["series"]:
                    data = receipts.read(
                        root=directory / "extended_runs", ident=run_id, end=end
                    )
                    if data["parent_id"] == matched["id"]:
                        frame_source = data
                        path_state = endpoint_path(
                            directory=directory, ident=run_id, end_s=end
                        )
                        with np.load(path_state) as fields:
                            hashes = json.loads(fields["receipt"].item())["hashes"]
                        expected_hashes = dict(hashes, state=hashes["q"])
                        break
            spec = dict(it["s"], hours=end / 3600)
            if int(base["hours_completed"] * 3600) == end:
                data = base
            else:
                calculate = (
                    automatic.calculate
                    if method == "hancock_auto"
                    else launch.calculate
                )
                data = calculate(
                    stage="million-auto"
                    if method == "hancock_auto"
                    else "million-storms",
                    spec=spec,
                    study=study,
                    root=study.parents[1],
                    config=config,
                    context={"device": settings["device"]},
                    directory=folder,
                )
            recession.verify_prefix(old=base, new=data["frames"][: len(base["frames"])])
            recession.verify_prefix(old=frame_source, new=data["frames"])
            fields = ("state", "maximum", "volume_x", "volume_y", "soil", "rain_sum")
            for field in fields:
                if (
                    field not in expected_hashes
                    or data["hashes"].get(field) != expected_hashes[field]
                ):
                    raise ArithmeticError(
                        f"Hancock final {field} differs across policies at {case} {dx} m"
                    )
            passed, checks = recession.settled(
                frames=data["frames"],
                storage=data["final_storage_m3"],
                supply=data["initial_m3"] + data["rain_m3"],
                config=settings,
            )
            if not passed:
                raise ValueError("Hancock policies differ at the recession endpoint")
            runner.write_json(
                path=path,
                data=dict(
                    id=ident,
                    parent_id=it["id"],
                    peer=matched["id"],
                    spec=spec,
                    end_s=end,
                    exact_fields=list(fields),
                    checks=checks,
                    hardware=machine,
                    data=data,
                    reused=end == round(it["s"]["hours"] * 3600),
                ),
            )
            print(
                "execution parity",
                case,
                dx,
                method,
                end / 3600,
                "h",
                round(data["hydraulic_s"], 2),
                "s",
                flush=True,
            )


def index(*, directory):
    out = {}
    for file in (directory / "extended_runs").glob("group_*.json"):
        data = json.loads(file.read_text())
        if data["cohort"] == "expanded":
            out[data["case"], tuple(sorted(data["sources"]))] = data
    return out


def attach(*, directory, groups, completed):
    config = read_config(path=directory.parent / "configs/study.json")
    for (cohort, case, rep), items in groups.items():
        if cohort != "expanded" or rep != "sampled":
            continue
        primary = completed["complete-expanded", case, rep]
        unique = {i.get("parent_id", i["id"]) for i in primary}
        end = round(primary[0]["s"]["hours"] * 3600)
        for i in unmatched(items=items, selected=unique):
            ident = receipt_key(directory=directory, config=config, base=i, end=end)
            file = directory / "policy_runs" / ident / "result.json"
            data = json.loads(file.read_text())
            if data["end_s"] != end:
                raise ValueError("Policy end differs")
            primary.append(
                dict(
                    i,
                    s=data["spec"],
                    d=data["data"],
                    id=ident,
                    parent_id=i["id"],
                    cohort="complete-expanded",
                )
            )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    run(directory=directory)
