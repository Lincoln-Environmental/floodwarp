"""Corrected frontier, physical identity, domain support and schedule qualification."""
import json
from pathlib import Path

import numpy as np
import pytest

from floodwarp.solver_advisor.execution import select,coverage,configure_hancock_execution
from floodwarp.solver_advisor.model import recommend
from floodwarp.fv2_solver import HancockSolver
from research.floodwarp_methods_paper import metrics,runner
from research.floodwarp_methods_paper.qualification import advisor_data,matrix,observations
from research.floodwarp_methods_paper.qualification.terrain import connection
from research.floodwarp_methods_paper.plan import read_config


def study_paths():
    root=Path(__file__).resolve().parents[1]
    study=root/"research/floodwarp_methods_paper"
    return root,study


def test_frozen_schedule_domain_and_resources():
    root,_=study_paths()
    policy=json.loads((root/"src/floodwarp/solver_advisor/execution_policy.json").read_text())
    args=dict(shape=(1024,1024),active_fraction=1.,device_name=policy["device_name"],free_bytes=2**30,policy=policy)
    assert select(**args,scheduled=.1)=="tiles"
    assert select(**args,scheduled=1.)=="dense"
    for changes in (dict(shape=(8,12)),dict(free_bytes=0),dict(device_name="unmeasured")):
        assert select(**dict(args,**changes),scheduled=.05)=="dense"
    s=np.zeros((16,160),bool);s[4,32]=1
    assert 0<coverage(seeds=s)<1


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_automatic_checkpoint_parity_after_rain_and_drying(device):
    import warp as wp
    if device.startswith("cuda") and not wp.is_cuda_available():pytest.skip("CUDA unavailable")
    args=dict(nx=161,ny=17,dx=1.,dy=1.,active_mask=np.ones((17,161),bool),bed_elevation=0.,manning_n=.035,device=device,maximum_dt=.5)
    base=HancockSolver(**args)
    selected=HancockSolver(**args)
    explicit=HancockSolver(**args,execution_mode="tiles")
    rain=np.zeros((17,161));rain[8,96]=3.e-6
    zs=(base,selected,explicit)
    for s in zs:s.set_source_rates(rain=rain)
    # Exercise the tiles choice on a small regression fixture. The measured
    # size gate is separately tested above with its shipped calibration.
    policy=dict(device_name=wp.get_device(device).name,smallest_measured_sparse_win_cells=0,dense_work_fraction=.99)
    choice=configure_hancock_execution(solver=selected,execution_mode="auto",distributed_rain_expected=False,policy=policy)
    if device.startswith("cuda"):assert choice["selected"]=="tiles"
    for s in zs:s.run_device_loop(duration=1.)
    assert 0<base.get_water_depth()[8,96]<base.dry_depth
    base_state=base.capture_state()
    for s in zs[1:]:
        for k,v in s.capture_state()["arrays"].items():np.testing.assert_array_equal(v,base_state["arrays"][k])
    selected.restore_state(state=explicit.capture_state())
    configure_hancock_execution(solver=selected,execution_mode="dense")
    for s in zs:s.set_source_rates(rain=np.zeros_like(rain));s.run_device_loop(duration=2.)
    for s in zs[1:]:
        for k,v in s.capture_state()["arrays"].items():np.testing.assert_array_equal(v,base.capture_state()["arrays"][k])


def test_reference_flags_survive_csv_and_policy_ties():
    base=dict(case="x",method="hancock_dense",dx=5,qualified_depth="True",reference_self="False",err=".01",wall="1")
    rows=[dict(base,experiment_id="a"),dict(base,method="hancock_adaptive",experiment_id="b"),
          dict(base,qualified_depth="False",experiment_id="c",err="0",wall=".1")]
    front=metrics.pareto(rows=rows,keys=("case",),objectives=dict(depth="err"),cost="wall",include_unresolved=True)
    assert [r["nondominated"] for r in front]==[True,True,None]
    assert front[0]["configuration"]!=front[1]["configuration"]


def test_separate_landforms_and_reject_invalid_grid():
    shared=dict(terrain_id="same",landform="plain")
    with pytest.raises(ValueError,match="leakage"):
        advisor_data.split_groups(rows=[dict(shared,split="test"),dict(shared,split="development")],heldout_family="mixed")
    model=dict(hardware="x",revision="r",training=[],validation_spacings=[10],tolerances=dict(discharge=.01))
    assert recommend(model=model,features=dict(resolution=1),objective="discharge")["status"]=="insufficient_evidence"
    assert recommend(model=model,features=dict(resolution=10,critical_connection_lost=True),objective="discharge")["status"]=="refine_terrain"


def test_barrier_connection_and_actual_represented_bed():
    from floodwarp.terrain_advisor.diagnostics import analyse_array
    bed=np.zeros((20,20));bed[:,9:11]=1;bed[9:11,9:11]=0
    coarse=np.zeros((4,4));coarse[:,2]=.4
    assert connection(elevation=bed,level=.1)
    assert not connection(elevation=coarse,level=.1)
    stats,_,maps=analyse_array(elevation=bed,native_resolution=1,resolution=5,represented_elevation=coarse)
    assert stats["comparison"]["elevation_rmse_m"]>0
    np.testing.assert_array_equal(maps["elevation"],coarse)


def test_fine_hydrograph_aggregation_preserves_signed_transport():
    row=dict(frames=[dict(t_s=i*60,outlet_m3s=float(i-10)) for i in range(1,31)])
    aggregates=observations.integral(record=row,seconds=900)
    np.testing.assert_array_equal(aggregates,[-2,13])
    assert aggregates.sum()*900==sum(f["outlet_m3s"]*60 for f in row["frames"])


def test_scaled_configuration_identity_and_sampling():
    root,study=study_paths()
    config=read_config(path=study/"configs/study.json")
    cases=list(matrix.cases(stage="million-storms",config=config,root=root))
    missing=set(config["synthetic"]["landforms"])-{s["landform"] for s in cases if np.prod(s["dimensions"])/s["dx"]**2==1024**2}
    assert not missing
    assert all(s["hours"]>=8 and 900%s["interval"]==0 for s in cases)
    a=cases[0]
    assert runner.digest(a)==runner.digest(dict(reversed(tuple(a.items()))))
    assert runner.digest(a)!=runner.digest(dict(a,seed=a["seed"]+1))

