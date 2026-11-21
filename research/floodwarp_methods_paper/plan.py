"""Expand the prospective design to explicit physical and numerical cases."""

import json
from pathlib import Path


def read_config(*, path):
    index = json.loads(path.read_text())
    return {k: json.loads((path.parent / v).read_text()) for k,v in index.items()}


def storm_cases(*, config):
    land=config["synthetic"]
    rain=config["storms"]
    for i,family in enumerate(land["landforms"]):
        for j,seed in enumerate(land["seeds"]):
            for k in range(2):
                storm=rain["events"][(i+2*j+k)%len(rain["events"])]
                yield dict(case=f"{family}-{seed}-{storm}",landform=family,seed=seed,split=land["splits"][j],storm=storm,initial=land["initial_states"][(i+j)%3],hours=rain["extended_plain_hours"] if i==j==k==0 else rain["end_hours"])


def specs(*, stage, config, root):
    a=config["analytical"]
    rain=config["storms"]
    land=config["synthetic"]
    if stage=="check":
        yield dict(case="environment-and-integrity",method="all")
    elif stage in ("analytical","convergence"):
        domains=json.loads((root/"validation/fv1_diffusion_review/domains.json").read_text())
        for domain in domains:
            if stage=="convergence" and domain["family"] not in ("pulse","front","gravity_pulse"):
                continue
            if stage=="convergence" and domain["name"] not in ("front_steep","gravity_pulse","shear_pulse","front_gentle"):
                continue
            for method in a["methods"]:
                for dx in a["spacings"] if stage=="analytical" else a["formal_spacings"]:
                    yield dict(case=domain["name"],method=method,dx=dx,kind="exact",policy="production" if stage=="analytical" else "common")
        if stage=="analytical":
            yield dict(case="independent-waves-equilibria-and-jumps",method="all",kind="archived-controls")
        else:
            yield dict(case="smooth-manufactured-and-fixed-grid",method="all",kind="temporal")
    elif stage=="synthetic":
        for family in land["landforms"]:
            for seed in land["seeds"]:
                yield dict(case=f"{family}-{seed}",landform=family,seed=seed,method="terrain")
    elif stage in ("long-storms","resolution"):
        seen=set()
        for c in storm_cases(config=config):
            if stage=="resolution":
                if c["split"]!="development" or c["landform"] not in land["resolution_families"] or c["landform"] in seen:
                    continue
                seen.add(c["landform"])
                for dx in (5,1):
                    for dem in (dx,10):
                        for m in rain["methods"]:
                            yield dict(**c,method=m,dx=dx,dem=dem,reference=False,multiplier=1.0)
            else:
                for dx in rain["reference_grids"]:
                    yield dict(**c,method="adaptive",dx=dx,dem=dx,reference=True,multiplier=1.0)
                for m in rain["methods"]:
                    yield dict(**c,method=m,dx=rain["dx"],dem=rain["dx"],reference=False,multiplier=1.0)
    elif stage=="calibration":
        cal=config["calibration"]
        for c in storm_cases(config=config):
            fit=c["seed"]==cal["fit_seed"] and c["storm"]==cal["fit_event"] and c["landform"]==cal["fit_family"]
            test=c["seed"] in cal["test_seeds"] and c["landform"] in cal["test_families"]
            if fit or test:
                for dx in (10,5) if fit or c["landform"]=="plain" else (10,):
                    for multiplier in cal["multipliers"]:
                        yield dict(**c,method="fv1",dx=dx,dem=dx,reference=False,multiplier=multiplier,role="fit" if fit and dx==10 else "transfer")
    elif stage=="performance":
        yield dict(case="paired-control-and-scale",method="all",kind="scale")
        c=next(s for s in storm_cases(config=config) if s["landform"]=="plain" and s["storm"]==config["performance"]["operational_storm"])
        for method in rain["methods"]:
            for rep in range(config["performance"]["timing_repeats"]):
                yield dict(**c,method=method,dx=10,dem=10,multiplier=1.0,reference=False,repeat=rep)
    else:
        yield dict(case=stage,method="all")
