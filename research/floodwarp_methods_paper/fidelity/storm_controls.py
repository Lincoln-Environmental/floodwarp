"""Additional matched hydrographs for temporal sensitivity and a two-pulse rain event."""
import json
from pathlib import Path
from unittest.mock import patch

from research.floodwarp_methods_paper import evidence, events, runner
from research.floodwarp_methods_paper.plan import read_config
from research.floodwarp_methods_paper.qualification import hydraulics, matrix, observations


def specs(*, study, config, extra):
    records = evidence.records(study=study, stages=("long-storms",))
    for rec in records:
        s = rec["spec"]
        if s["method"] == "adaptive" and s["dx"] == 2.5:
            base = dict(s, dimensions=config["synthetic"]["dimensions_m"], interval=900,
                        support=10, initial_grid="sampled", controls=False, coastal_stage=False)
            for method, factor in (("adaptive", .5), ("hancock_dense", 1.)):
                yield dict(base, method=method), factor, "legacy"
    for (_, rep), group in observations.load_events(study=study, config=config).items():
        refs = [i for i in group if i["spec"]["method"] == "adaptive"]
        if not refs or rep != "sampled":
            continue
        ref = min(refs, key=observations.spacing)
        if ref["spec"]["dimensions"] == config["synthetic"]["dimensions_m"] or ref["spec"]["case"] in extra:
            yield ref["spec"], .5, "expanded"
    base = dict(case="floodplain-double-rain", landform="floodplain", seed=18217, initial="dry",
                split="diagnostic", hours=12, dimensions=[120,80], interval=60, support=10,
                controls=False, coastal_stage=False, initial_grid="parent", multiplier=1., storm="double")
    for dx, method, factor in ((1,"adaptive",1.),(2,"adaptive",1.),(1,"adaptive",.5),(1,"hancock_dense",1.),
                               (5,"adaptive",1.),(10,"adaptive",1.),(5,"fv1",1.),(10,"fv1",1.),
                               (5,"hancock_dense",1.),(10,"hancock_dense",1.),(10,"roe",1.),(10,"hllc",1.)):
        yield dict(base, dx=dx, dem=10, method=method), factor, "double"


def run(*, directory, cfg, extra, rates):
    study = directory.parent
    root = study.parents[1]
    template = json.loads((root/"validation/hancock_long_storm_review/cases.json").read_text())
    template["storms_mm_h"]["double"] = rates
    options = matrix.design(study=study)
    hardware = runner.machine(root=root)
    source = runner.source_identity(root=root, study=study, stage="long-storms")
    code = [Path(__file__), study/"qualification/hydraulics.py", study/"qualification/matrix.py"]
    folder = directory/"storm_runs"
    folder.mkdir(exist_ok=True)
    index = []
    normal = events.numerical
    for spec, factor, cohort in specs(study=study, config=cfg, extra=extra):
        def numerical(*, method):
            if method in ("roe","hllc"):
                return "fv1","sparse",dict(flux_scheme=method, cfl=.4*factor)
            key, mode, opt = normal(method=method)
            return key, mode, dict(opt, cfl=.4*factor)
        identity = dict(spec=spec, factor=factor, cohort=cohort, source=source,
                        code=[runner.checksum(p) for p in code], config=cfg, template=template,
                        options=options, device=[hardware["gpu"],hardware["warp"]])
        ident = runner.digest(identity)[:20]
        dest = folder/ident
        dest.mkdir(exist_ok=True)
        receipt = dest/"result.json"
        if not receipt.exists():
            with patch.object(events,"numerical",numerical):
                data = hydraulics.run_event(spec=spec, config=cfg, template=template, options=options,
                                            directory=dest, device="cuda:0")
            tmp = dest/"pending.json"
            runner.write_json(path=tmp, data=dict(spec=spec, factor=factor, cohort=cohort, data=data,
                                                  inputs=identity, hardware=hardware))
            tmp.replace(receipt)
            print(cohort, spec["case"], spec["dx"], spec["method"], factor, data["steps"], flush=True)
        index.append(ident)
    runner.write_json(path=folder/"index.json", data=index)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    config = read_config(path=directory.parent/"configs/study.json")
    extra = ("mixed-708-spatial-10240x10240",)
    rain_rates_mm_h = [4,50,6,4,55,8,4,0]
    run(directory=directory, cfg=config, extra=extra, rates=rain_rates_mm_h)

