"""Prospective complete-event schedules. Distances and datum elevations are metres."""
import json
from research.floodwarp_methods_paper.plan import storm_cases


def design(*, study):
    return json.loads((study / "qualification/design.json").read_text())


def cases(*, stage, config, root):
    study = root / "research/floodwarp_methods_paper"
    plan = design(study=study)
    common = plan["production_methods"]
    for family, seed, rain, initial, dx, nx, ny, hours, split, interval in plan["events"]:
        length, width = nx * dx, ny * dx
        name = f"{family}-{seed}-{rain}-{length}x{width}"
        base = dict(case=name, landform=family, seed=seed, storm=rain, initial=initial,
                    hours=hours, split=split, dimensions=[length,width], dx=dx, dem=dx,
                    multiplier=1.0, interval=interval, controls=True, initial_grid="sampled",
                    coastal_stage=family=="coastal", support=10 if seed==709 else 40)
        if stage == "million-terrain":
            yield dict(base,method="terrain")
        elif stage == "million-pilots" and seed in (701,702) and dx==10:
            for method in common:
                yield dict(base, method=method, end_s=plan["pilot_end_s"])
        elif stage == "million-profiles" and nx==1024 and seed in (701,702,708):
            for method in common:
                yield dict(base,method=method)
        elif stage == "million-storms":
            for method in common + plan["subset_methods"].get(str(seed),[]):
                yield dict(base,method=method)
        elif stage == "large-references" and seed==709 and dx==10:
            for spacing in (5, 2.5, 1):
                for method in plan["reference_methods"]:
                    yield dict(base,dx=spacing,dem=10,method=method)
            for method in plan["reference_methods"]:
                yield dict(base,dx=2.5,dem=2.5,method=method)
        elif stage == "large-references" and seed in (701,705,715) and nx==1024:
            # Same complete catchment. Fine FV2 plus independent Hancock on the
            # shared fine grid, including a held-out family/event realisation.
            for ratio in (1.25,2):
                for method in plan["reference_methods"] if ratio==2 else ["adaptive"]:
                    yield dict(base,dx=dx/ratio,dem=dx/ratio,method=method)
    if stage=="large-references":
        chosen=set()
        for small in storm_cases(config=config):
            family=small["landform"]
            if family not in plan["small_reference_families"] or family in chosen:
                continue
            chosen.add(family)
            for dx in plan["small_reference_grids"]:
                methods=common+["adaptive"] if dx==10 else plan["reference_methods"]
                for method in methods:
                    yield dict(small,dimensions=config["synthetic"]["dimensions_m"],dx=dx,dem=dx,
                               method=method,multiplier=1.,interval=plan["reference_interval_s"],
                               controls=False,initial_grid="sampled",coastal_stage=False,support=10)

