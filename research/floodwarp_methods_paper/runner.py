"""Manual, resumable scientific execution with immutable numerical attempts."""

from contextlib import contextmanager
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import fcntl

from research.floodwarp_methods_paper.plan import read_config, specs


STAGES = ("check","analytical","convergence","synthetic","long-storms","resolution","calibration","performance","terrain-features","advisor-train","advisor-validate","tables","figures","manuscript","verify")
SIMULATIONS = STAGES[:8]


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def write_json(*, path, data):
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def machine(*, root):
    return dict(git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),cpu=platform.processor() or platform.machine(),cpu_model=next((s.partition(":")[2].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),"unknown"),python=platform.python_version(),gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip(),warp=importlib.metadata.version("warp-lang"),numpy=importlib.metadata.version("numpy"),precision="FP64")


def source_identity(*, root, study, stage):
    paths=[*sorted((root/"src/floodwarp").glob("*.py")),*sorted((root/"validation").glob("*.py")),*sorted((root/"validation/hancock_long_storm_review").glob("*.py")),study/"plan.py",study/"runner.py"]
    if stage in ("analytical","convergence"): paths += [study/"exact.py",study/"history.py"]
    if stage in ("synthetic","long-storms","resolution","calibration","performance"): paths += [study/"events.py",study/"history.py",study/"terrain/generators/generate.py"]
    if stage not in SIMULATIONS:
        paths += list(study.glob("*.py"))+list((study/"tables").glob("*.py"))+list((study/"figures").glob("*.py"))
        paths += list((root/"src/floodwarp/solver_advisor").glob("*.py"))+list((root/"src/floodwarp/terrain_advisor").glob("*.py"))
    paths += [study/"PROTOCOL.md",root/"validation/fv1_diffusion_review/domains.json",root/"validation/hancock_long_storm_review/cases.json"]
    return digest({str(p.relative_to(root)):checksum(p) for p in set(paths) if p.exists()})


def archived(*, study, stages):
    result=[]
    for stage in stages:
        for p in sorted((study/"manifests"/stage).glob("*.json")):
            data=json.loads(p.read_text())
            if data["status"]=="complete": result.append(data)
    return result


@contextmanager
def log_output(*, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    sys.stdout.flush(); sys.stderr.flush()
    with path.open("x") as log:
        old=[os.dup(fd) for fd in (1,2)]
        try:
            for fd in (1,2): os.dup2(log.fileno(),fd)
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            for fd,original in zip((1,2),old,strict=True): os.dup2(original,fd);os.close(original)


def verify_artifacts(*, record, study):
    base=study/record["directory"]
    for name,sha in record["artifacts"].items():
        file=base/name
        if not file.is_file() or checksum(file)!=sha: raise ValueError("Archive missing or corrupt: "+str(file))


def execute(*, stage, spec, study, root, config, context, directory):
    if stage=="check":
        import warp as wp
        if sys.version_info[:2]!=(3,14) or not wp.get_device(context["device"]).is_cuda: raise RuntimeError("Python 3.14 and CUDA required")
        env=dict(os.environ,OVERLANDER_TEST_FULL_SWE_CUDA="1",FLOODWARP_TEST_RENDER="1")
        result=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout)
        result.check_returncode()
        return dict(summary=result.stdout.strip().splitlines()[-1],cuda_driver=wp.get_cuda_driver_version(),cuda_toolkit=wp.get_cuda_toolkit_version())
    if stage in ("analytical","convergence"):
        from research.floodwarp_methods_paper.exact import evaluate
    elif stage in ("synthetic","long-storms","resolution","calibration","performance"):
        from research.floodwarp_methods_paper.events import evaluate
    else:
        from research.floodwarp_methods_paper.analysis import evaluate
    return evaluate(stage=stage,spec=spec,study=study,root=root,config=config,context=context,directory=directory)


def run_stage(*, stage, opts, root, study, config, system):
    todo=[s for s in specs(stage=stage,config=config,root=root) if fnmatch.fnmatchcase(s["case"],opts["case"]) and (s["method"]==opts["method"] or opts["method"]=="*")]
    if not todo:
        raise ValueError("No cases matched stage "+stage)
    if opts["dry"]:
        for s in todo: print(stage,json.dumps(s,sort_keys=True))
        print(f"{stage} planned={len(todo)}")
        return
    source=source_identity(root=root,study=study,stage=stage)
    from time import perf_counter
    start=perf_counter()
    counts=dict(completed=0,failed=0,skipped=0,pending=len(todo))
    context=dict(**system,device=opts["device"],source_sha256=source)
    prerequisites=[] if stage in SIMULATIONS else [r["attempt"] for r in archived(study=study,stages=STAGES[:STAGES.index(stage)])]
    for s in todo:
        case_id=digest(dict(stage=stage,settings=s,config=config))[:24]
        execution_id=digest(dict(case=case_id,source=source,hardware=[system["gpu"],system["warp"],system["python"],opts["device"]],inputs=prerequisites))[:24]
        manifest=study/"manifests"/stage/(execution_id+".json")
        manifest.parent.mkdir(parents=True,exist_ok=True)
        with (manifest.with_suffix(".lock")).open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if manifest.exists():
                prior=json.loads(manifest.read_text())
                if prior["status"]=="complete" and not opts["force"]:
                    verify_artifacts(record=prior,study=study)
                    counts["skipped"]+=1;counts["pending"]-=1
                    continue
            attempt=uuid.uuid4().hex[:16]
            category={"long-storms":"storms","convergence":"analytical","resolution":"synthetic"}.get(stage,stage)
            directory=study/"raw_results"/category/case_id/attempt
            directory.mkdir(parents=True)
            log=study/"logs"/stage/(attempt+".log")
            base=dict(stage=stage,spec=s,experiment_id=case_id,execution_id=execution_id,attempt=attempt,context=context,directory=str(directory.relative_to(study)),log=str(log.relative_to(study)),status="running",config_sha256=digest(config))
            write_json(path=directory/"attempt.json",data=base)
            write_json(path=manifest,data=base)
            tick=perf_counter()
            try:
                with log_output(path=log):
                    result=execute(stage=stage,spec=s,study=study,root=root,config=config,context=context,directory=directory)
                write_json(path=directory/"result.json",data=result)
                base["artifacts"]={str(p.relative_to(directory)):checksum(p) for p in directory.rglob("*") if p.is_file() and p.name!="attempt.json"}
                base["status"]="complete"
                counts["completed"]+=1
            except BaseException as error:
                import traceback
                with log.open("a") as output: traceback.print_exc(file=output)
                base.update(status="failed",error=repr(error))
                counts["failed"]+=1
                raise
            finally:
                counts["pending"]-=1
                base["wall_s"]=perf_counter()-tick
                write_json(path=directory/"receipt.json",data=base)
                write_json(path=manifest,data=base)
                print(stage,s["case"],s["method"],base["status"],counts,"elapsed_s",round(perf_counter()-start,2),flush=True)
    print(stage,counts,"elapsed_s",round(perf_counter()-start,2),flush=True)


def options(*, args):
    result=dict(dry=False,force=False,case="*",method="*",device=os.environ.get("FLOODWARP_PAPER_DEVICE","cuda:0"))
    it=iter(args)
    for text in it:
        if text in ("--dry-run","--force"): result["dry" if text=="--dry-run" else "force"]=True
        elif text in ("--case","--method"): result[text[2:]]=next(it)
        else: raise ValueError("Unknown option "+text)
    return result


if __name__ == "__main__":
    study=Path(__file__).resolve().parent
    root=study.parents[1]
    config_file=Path(os.environ.get("FLOODWARP_PAPER_CONFIG",study/"configs/study.json"))
    config=read_config(path=config_file)
    stage=sys.argv[1]
    selected=STAGES if stage=="all" else (stage,)
    if set(selected)-set(STAGES): raise ValueError("Unknown study stage")
    opts=options(args=sys.argv[2:])
    system={} if opts["dry"] else machine(root=root)
    started=time.perf_counter()
    for group in selected: run_stage(stage=group,opts=opts,root=root,study=study,config=config,system=system)
    print("total_experiment_wall_s",round(time.perf_counter()-started,3))
