"""Freeze a pre-flight schedule from pre-existing controls and measure complete automatic forecasts."""
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from research.floodwarp_methods_paper import metrics,runner
from research.floodwarp_methods_paper.qualification import launch,matrix
from floodwarp.solver_advisor.execution import configure_hancock_execution


def fit(*,root,output):
    directory=root/"validation/hancock_long_storm_review"
    original=directory/"event_summary.csv"
    data=metrics.csv_read(original)
    groups=defaultdict(dict)
    for r in data:
        if r["method"]=="hancock" and r["mode"] in ("dense","tiles") and r["balance_pass"]=="True" and r["group"] in ("large","terrain_large"):
            groups[r["landform"],r["storm"],int(r["cells"])][r["mode"]]=float(r["hydraulic_s"])
    winning=[cells for (_,_,cells),runs in groups.items() if "tiles" in runs and "dense" in runs and runs["tiles"]<runs["dense"]]
    strip=directory/"occupancy.json"
    records=json.loads(strip.read_text())
    by_depth=defaultdict(dict)
    for r in records:
        if r["method"]=="hancock_adaptive":
            by_depth[r["seed_fraction"]][r["mode"]]=r
    total=max(r["tiles"]["active"]["evaluated_cells"] for r in by_depth.values())
    ranked=sorted((r["tiles"]["active"]["evaluated_cells"]/total,
                   r["tiles"]["graph_s"]<r["dense"]["graph_s"]) for r in by_depth.values())
    upper=min(frac for frac,won in ranked if not won)
    lower=max(frac for frac,won in ranked if won and frac<upper)
    policy=dict(device_name="NVIDIA GeForce RTX 4070 Ti SUPER",smallest_measured_sparse_win_cells=min(winning),
                dense_work_fraction=(upper+lower)*.5,measured_transition=[lower,upper],
                sources={p.name:runner.checksum(p) for p in (original,strip)},domain_pairs=len(groups))
    runner.write_json(path=output,data=policy)
    return policy


def cases(*,stage,config,root):
    for s in matrix.cases(stage="million-storms",config=config,root=root):
        if s["method"]=="hancock_dense":
            yield dict(s,method="hancock_auto")


def source(*,root,study,stage):
    path=root/"src/floodwarp/solver_advisor"
    return runner.digest(dict(base=launch.identity_original(root=root,study=study,stage=stage),
             policy=runner.checksum(path/"execution_policy.json"),implementation=runner.checksum(path/"execution.py"),
             adapter=runner.checksum(study/"qualification/automatic.py")))


def calculate(*,stage,spec,study,root,config,context,directory):
    from research.floodwarp_methods_paper import events
    from research.floodwarp_methods_paper.qualification.hydraulics import run_event
    build=events.create_event
    policy={}
    def prepare(**kwargs):
        model,expected,forcing=build(**kwargs)
        policy.update(configure_hancock_execution(solver=model.solver,execution_mode="auto"))
        return model,expected,forcing
    template=json.loads((root/"validation/hancock_long_storm_review/cases.json").read_text())
    with patch.object(events,"create_event",prepare):
        result=run_event(spec=dict(spec,method="hancock_dense"),config=config,template=template,
            options=matrix.design(study=study),directory=directory,device=context["device"])
    result.update(execution="auto",preflight=policy)
    return result


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[3]
    output=root/"src/floodwarp/solver_advisor/execution_policy.json"
    fit(root=root,output=output)

