"""Manually run the archived Orton Bradley 48 h HIRDS storm with matched physical sections."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp

from floodwarp.fv2_solver import FV2Solver, HancockSolver
from research.floodwarp_methods_paper import metrics, runner
from research.floodwarp_methods_paper.fidelity import recession
from validation import fv2_orton_matrix as archive


def discharge_volumes(*, solver, gauges):
    field = solver._fields["volume_y"].numpy().reshape(solver.ny+1, solver.nx)
    positions = np.array(gauges)/solver.dy
    if not np.array_equal(positions, np.rint(positions)) or positions.max() >= len(field):
        raise ValueError("Gauges must lie on represented faces")
    return field[positions.astype(int)].sum(axis=1)


def simulate(*, directory, settings, state, data, geometry, forcing, dx, method, identity):
    dest = directory/"orton_transfer"/runner.digest(identity)[:24]
    dest.mkdir(parents=True,exist_ok=True)
    cache = directory.parents[2]/".fidelity-restarts/orton_transfer"/dest.name
    cache.mkdir(parents=True,exist_ok=True)
    result = dest/"complete.json"
    if result.exists():
        return json.loads(result.read_text())
    names = {"fv1":("contact_hll","sparse"),"hancock":("fv2_adaptive","tiles"),"rk2":("fv2_adaptive","dense")}
    label, mode = names[method]
    s = archive.replay_solver(data=data,state=state,factor=dx//5,method=label,device=settings["device"],
                              execution_mode=mode,fv2_class=HancockSolver if method=="hancock" else FV2Solver)
    actual = s.numerical_configuration()
    if actual["solver_method"] != {"fv1":"fv1","hancock":"fv1_hancock","rk2":"fv2"}[method]:
        raise ValueError("Orton transfer formulation mismatch")
    manifest=dest/"inputs.json"
    if not manifest.exists():
        recession.atomic_json(path=manifest,payload=dict(identity=identity,numerical=actual,policy=mode,
                  hardware=runner.machine(root=directory.parents[2]),forcing=forcing,geometry=geometry,
                  gauges_above_southern_origin_m=settings["gauges_m"]))
    interval=settings["cadence_s"]
    records=[]
    previous=0.
    initial=discharge_volumes(solver=s,gauges=settings["gauges_m"])
    for hour in range(settings["checkpoint_h"],forcing["duration_h"]+1,settings["checkpoint_h"]):
        path=dest/f"hour_{hour}.json"
        checkpoint=cache/f"hour_{hour}.npz"
        if path.exists():
            record=json.loads(path.read_text())
            if runner.checksum(checkpoint)!=record["checkpoint_sha256"]:
                raise ValueError("Orton transfer checkpoint changed")
            s.load_checkpoint(path=checkpoint)
            records.extend(record["frames"])
            previous=records[-1]["boundary_net_m3"]
            initial=discharge_volumes(solver=s,gauges=settings["gauges_m"])
            continue
        new=[]
        tick=perf_counter()
        for target in range(int(s.time)+interval,hour*3600+1,interval):
            wp.synchronize_device(device=s.device)
            start=perf_counter()
            info=s.run_device_loop(duration=target-s.time)
            wp.synchronize_device(device=s.device)
            elapsed=perf_counter()-start
            net=float(s._fields["outflow"].numpy().sum()-s._fields["inflow"].numpy().sum())
            current=discharge_volumes(solver=s,gauges=settings["gauges_m"])
            capture=info["graph_capture_seconds"]
            compile_s=info.get("compilation_seconds",0.)
            new.append(dict(t_s=target,net_outlet_m3s=(net-previous)/interval,boundary_net_m3=net,
                            gauge_m3s=((current-initial)/interval).tolist(), steps=info["steps"],
                            retries=info.get("rejected_cfl_trials",0),dt_min_s=info["dt_min_s"],
                            hydraulic_s=elapsed-capture-compile_s, graph_capture_s=capture,compilation_s=compile_s,
                            pool_mib=wp.get_mempool_used_mem_current(device=s.device)/2**20))
            previous,initial=net,current
        s.save_checkpoint(path=checkpoint)
        recession.atomic_json(path=path,payload=dict(frames=new,checkpoint_sha256=runner.checksum(checkpoint),
                                                      window_wall_s=perf_counter()-tick))
        records.extend(new)
        print("Orton Bradley", dx,"m",method,hour,"of",forcing["duration_h"],"h",flush=True)
    ledger=s.cumulative_accounting()
    h=s.get_water_depth()
    maximum=s.get_max_water_depth()
    np.testing.assert_allclose(ledger["rain"],forcing["integrated_volume_m3"],rtol=1e-9)
    if h.min()<0 or not np.isfinite(h).all() or abs(ledger["residual"])>1e-8*ledger["rain"]:
        raise ArithmeticError("Orton transfer failed conservation or positivity")
    peak=max(records,key=peak_flow)
    answer=dict(id=dest.name,method=method,dx_m=dx,policy=mode,peak_m3s=peak["net_outlet_m3s"],peak_end_s=peak["t_s"],
                storage_m3=float(h.sum()*dx**2),steps=sum(r["steps"] for r in records),retries=sum(r["retries"] for r in records),
                hydraulic_s=sum(r["hydraulic_s"] for r in records),sampled_peak_pool_mib=max(r["pool_mib"] for r in records),
                **ledger)
    for cm in settings["thresholds_cm"]:
        answer[f"final_area_{cm}cm_m2"]=int((h>cm/100).sum())*dx**2
        answer[f"maximum_area_{cm}cm_m2"]=int((maximum>cm/100).sum())*dx**2
    recession.atomic_json(path=result,payload=answer)
    return answer


def peak_flow(frame):
    return frame["net_outlet_m3s"]


def run(*, directory, settings, dry):
    release=directory.parents[2]/settings["release"]
    identity=dict(source=runner.checksum(Path(__file__)),inputs=runner.checksum(release/"inputs/manifest.json"),
                  dem=runner.checksum(release/"outputs/orton_bradley_dem_5m.tif"),settings=settings,
                  numerical=runner.source_identity(root=directory.parents[2],study=directory.parent,stage="long-storms"))
    if dry:
        for dx in settings["grids_m"]:
            print("Orton Bradley",dx,"m,",settings["methods"],"48 h HIRDS design forcing, external transfer")
        return
    state=archive.release_initial(directory=release)
    rows=[]
    for dx in settings["grids_m"]:
        data,geometry=archive.geometry(state=state,factor=dx//5)
        forcing=archive.forcing_check(state=state,data=data,dx=dx)
        for method in settings["methods"]:
            rows.append(simulate(directory=directory,settings=settings,state=state,data=data,geometry=geometry,
                                 forcing=forcing,dx=dx,method=method,identity=dict(identity,dx=dx,method=method)))
            metrics.csv_write(file=directory/"orton_transfer/transfer.csv",rows=rows)


if __name__ == "__main__":
    directory=Path(__file__).resolve().parent
    settings=json.loads((directory/"orton_transfer_settings.json").read_text())
    options=set(sys.argv[1:])
    if options.difference({"--dry-run","--resume"}):
        raise ValueError("Unsupported transfer options")
    run(directory=directory,settings=settings,dry="--dry-run" in options)
