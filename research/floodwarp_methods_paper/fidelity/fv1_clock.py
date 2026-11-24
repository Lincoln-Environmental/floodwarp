"""Test FV1's own temporal sensitivity on the archived widespread-rain CPU controls."""

import fnmatch
import getopt
import json
from pathlib import Path
import sys

import numpy as np

from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import device_controls, recession, strong_storms
from research.floodwarp_methods_paper.plan import read_config


def observe(*, forecast, stopping):
    frames = forecast["frames"]
    budget = forecast["budget"]
    ends = np.array([r["t_s"] for r in frames])
    q = np.array([r["outlet_m3s"] for r in frames])
    crest = np.flatnonzero(q == q.max())
    passed, _ = recession.settled(frames=frames, storage=budget["storage_m3"], supply=forecast["prescribed"], config=stopping)
    export = float(q @ np.diff(ends, prepend=0))
    np.testing.assert_allclose(export, budget["boundary_outflow"] - budget["boundary_inflow"], rtol=1e-10, atol=1e-7)
    return dict(id=forecast["folder"].name, end_s=int(ends[-1]), both_stations_passed=passed,
                steps=sum(r["steps"] for r in frames), peak_m3s=float(q.max()),
                peak_interval_end_s=int(ends[crest[0]]) if len(crest) == 1 and q.max() > 0 else None,
                net_export_m3=export, storage_m3=budget["storage_m3"], infiltration_m3=budget["infiltration"],
                rain_m3=budget["rain"], residual_m3=budget["residual_m3"])


def run_control(*, directory, control, settings, config, design, stopping, factor):
    root = directory / "strong_storms"
    identity = json.loads((root / control["id"] / "inputs.json").read_text())["identity"]
    peers = [strong_storms.forecast(directory=directory, spec=dict(identity["spec"], factor=value),
                                    design=design, config=config, settings=settings, stopping=stopping)
             for value in (1., factor)]
    normal, reduced = peers
    manifests = [json.loads((p["folder"] / "inputs.json").read_text()) for p in peers]
    archived = json.loads((root / control["id"] / "inputs.json").read_text())
    if (runner.digest(identity)[:24] != control["id"] or identity["numerical"] != manifests[0]["identity"]["numerical"]
            or archived["physical_fields"] != manifests[0]["physical_fields"]
            or archived["forcing"] != manifests[0]["forcing"] or identity["spec"] != normal["spec"]):
        raise ValueError("Archived FV1 baseline differs in solver, physical inputs or numerical settings")
    base, other = [r["forcing"]["numerical"] for r in manifests]
    if (base["solver_method"] != "fv1" or other != dict(base, cfl=base["cfl"] * factor)
            or manifests[0]["physical_fields"] != manifests[1]["physical_fields"]):
        raise ValueError("FV1 temporal control must change only the CFL factor")
    earliest = float(control["end_h"])
    observations = []
    for hour in range(settings["first_check_h"], stopping["forcing_hours"] + 1, stopping["checkpoint_hours"]):
        end = hour * 3600
        views = []
        for peer, manifest in zip(peers, manifests, strict=True):
            if (peer["folder"] / f"{end}.json").is_file():
                frames, block, _ = device_controls.cpu_recession(
                    folder=peer["folder"], end=end, numerical=manifest["forcing"]["numerical"]
                )
                views.append(dict(peer, frames=frames, budget=block["budget"]))
            else:
                for elapsed in range(settings["first_check_h"], hour + 1, stopping["checkpoint_hours"]):
                    if elapsed * 3600 > peer["start"]:
                        strong_storms.advance_window(forecast=peer, end=elapsed * 3600, directory=directory, settings=settings)
                views.append(peer)
        if (root / control["id"] / f"{end}.json").is_file():
            old_frames, old, _ = device_controls.cpu_recession(folder=root / control["id"], end=end, numerical=base)
            device_controls.interval_differences(cpu=old_frames, gpu=views[0]["frames"])
            for field in ("steps", "section_m3s", "outlet_m3s"):
                np.testing.assert_array_equal([r[field] for r in old_frames], [r[field] for r in views[0]["frames"]])
            for field in ("storage_m3", "infiltration", "boundary_outflow", "boundary_inflow"):
                np.testing.assert_allclose(old["budget"][field], views[0]["budget"][field], rtol=1e-12, atol=1e-8)
        device_controls.same_forcing(cpu=manifests[0]["forcing"], gpu=manifests[1]["forcing"], end=end)
        measurements = [observe(forecast=peer, stopping=stopping) for peer in views]
        for p, values in zip(peers, measurements, strict=True):
            observations.append(dict(case=control["case"], dx_m=control["dx_m"], method="fv1", device=settings["device"],
                                     factor=p["spec"]["factor"], interval_s=p["spec"]["interval"], **values))
        print(control["case"], hour, "h", "FV1 peak at normal and reduced CFL",
              *(m["peak_m3s"] for m in measurements), flush=True)
        if hour >= earliest and all(m["both_stations_passed"] for m in measurements):
            break
    else:
        raise ValueError("FV1 temporal comparison needs a longer common recession")
    output = directory / "fv1_temporal"
    source = dict(historical_normal=control["id"], normal=normal["folder"].name, reduced=reduced["folder"].name,
                  normal_cfl=base["cfl"], reduced_cfl=other["cfl"], end_s=end,
                  implementation=runner.checksum(Path(__file__)),
                  windows={p["folder"].name: {str(r["end_s"]): runner.checksum(p["folder"] / f"{r['end_s']}.json")
                           for r in observations if r["id"] == p["folder"].name} for p in peers})
    dest = output / runner.digest(source)[:24]
    runner.write_json(path=dest / "sources.json", data=source)
    metrics.csv_write(file=dest / "observations.csv", rows=observations)


def run(*, directory, arguments, factor):
    flags, extra = getopt.getopt(arguments, "", ["dry-run", "resume", "case=", "method=", "dx="])
    if extra:
        raise ValueError(f"Unexpected FV1 temporal control arguments {extra}")
    options = dict(flags)
    settings = json.loads((directory / "strong_storm_settings.json").read_text())
    controls = [r for r in metrics.csv_read(file=directory / "strong_storms/completed.csv")
                if fnmatch.fnmatchcase(r["case"], options.get("--case", "*")) and r["method"] == "fv1"
                and options.get("--method", "fv1") == "fv1" and float(options.get("--dx", r["dx_m"])) == float(r["dx_m"])]
    if not controls:
        raise ValueError("No matching archived widespread-rain FV1 temporal controls")
    for r in controls:
        print(r["case"], r["dx_m"], "m", r["device"], "FV1 CFL factor", factor,
              "paired with", r["id"], "through at least", r["end_h"], "h", flush=True)
    if "--dry-run" in options:
        return
    config = read_config(path=directory.parent / "configs/study.json")
    design = json.loads((directory / "native_matrix_settings.json").read_text())
    stopping = json.loads((directory / "recession_settings.json").read_text())
    for row in controls:
        run_control(directory=directory, control=row, settings=settings, config=config,
                    design=design, stopping=stopping, factor=factor)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    factor = 0.5
    run(directory=directory, arguments=sys.argv[1:], factor=factor)
