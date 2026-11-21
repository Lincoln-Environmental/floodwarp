"""Additional reference evidence and matched wet initial-state controls."""
from research.floodwarp_methods_paper.qualification.matrix import cases
from research.floodwarp_methods_paper.qualification import launch
from research.floodwarp_methods_paper import runner


def expanded(*,stage,config,root):
    for s in cases(stage="million-storms",config=config,root=root):
        if s["method"]!="fv1":
            continue
        if s["seed"] in (710,711,703):
            for ratio in (1.25,2):
                for method in ("adaptive","hancock_dense") if ratio==2 else ("adaptive",):
                    yield dict(s,dx=s["dx"]/ratio,dem=s["dx"]/ratio,method=method)
        if s["seed"] in (702,703):
            for method in ("fv1","hancock_dense","adaptive"):
                yield dict(s,method=method,dx=20,dem=20)
                for initial in ("sampled","parent"):
                    yield dict(s,method=method,dem=20,initial_grid=initial)
        if s["seed"]==709 and s["dx"]==1:
            # Independent refinement on the same 1 km² for the metre-grid forecasts.
            for method in ("adaptive","hancock_dense"):
                yield dict(s,dx=.5,dem=.5,method=method)


def source(*,root,study,stage):
    return runner.digest(dict(base=launch.identity_original(root=root,study=study,stage=stage),
                              refinements=runner.checksum(study/"qualification/refinements.py")))

