"""Manually request a recommendation for one terrain raster."""

import getopt
import json
from pathlib import Path
import sys
from floodwarp.solver_advisor import load_model, recommend_dem


def run(*, arguments):
    flags, _ = getopt.getopt(
        arguments,
        "h",
        [
            "help",
            "dem=",
            "resolution=",
            "objective=",
            "output=",
            "model=",
            "tolerance=",
            "hydraulic-json=",
            "hardware=",
            "memory-mib=",
            "compare-resolutions=",
        ],
    )
    opt = dict(flags)
    if "--help" in opt or "-h" in opt or not opt:
        print(
            "python -m floodwarp.solver_advisor --dem terrain.tif --resolution 5 --objective discharge --output outputs/advisor [--tolerance 0.01] [--hydraulic-json hydraulic.json] [--hardware GPU] [--memory-mib 8192] [--compare-resolutions 1,10] [--model model.json]"
        )
        return
    options = dict(
        dem=Path(opt["--dem"]),
        resolution=float(opt["--resolution"]),
        objective=opt["--objective"],
        output=Path(opt["--output"]),
        model=load_model(path=opt.get("--model")),
    )
    if "--tolerance" in opt:
        options["tolerance"] = float(opt["--tolerance"])
    if "--hydraulic-json" in opt:
        options["hydraulic"] = json.loads(Path(opt["--hydraulic-json"]).read_text())
    if "--hardware" in opt:
        options["hardware"] = opt["--hardware"]
    if "--memory-mib" in opt:
        options["memory_mib"] = float(opt["--memory-mib"])
    if "--compare-resolutions" in opt:
        options["compare_resolutions"] = tuple(
            map(float, opt["--compare-resolutions"].split(","))
        )
    print(json.dumps(recommend_dem(**options)["selected"], indent=2))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    run(arguments=arguments)
