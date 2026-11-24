"""Compare archived CPU and GPU gauge observations on an identical initial storm window."""

import json
import numpy as np

from research.floodwarp_methods_paper import metrics, runner


def interval_differences(*, cpu, gpu):
    ends = np.array([row["t_s"] for row in cpu])
    np.testing.assert_array_equal(ends, [row["t_s"] for row in gpu])
    widths = np.diff(ends, prepend=0.)
    if not len(ends) or np.any(widths <= 0):
        raise ValueError("Invalid gauge intervals for the CPU and CUDA comparison")
    result = dict(intervals=len(ends), end_s=int(ends[-1]),
                  max_interval_s=float(widths.max()), min_interval_s=float(widths.min()),
                  intervals_with_different_step_count=sum(a["steps"] != b["steps"] for a, b in zip(cpu, gpu, strict=True)))
    for station in ("outlet", "section"):
        delta = np.array([g[station+"_m3s"] - c[station+"_m3s"] for c, g in zip(cpu, gpu, strict=True)])
        if not np.all(np.isfinite(delta)):
            raise ValueError("Nonfinite gauge transport in the CPU and CUDA comparison")
        result[station+"_max_abs_difference_m3s"] = float(abs(delta).max())
        result[station+"_net_difference_m3"] = float(delta @ widths)
    return result


def same_forcing(*, cpu, gpu, end):
    times = "rain_times_s"
    fields = (cpu.keys() | gpu.keys()) - {times, "numerical"}
    if any(cpu[k] != gpu[k] for k in fields):
        raise ValueError("Different physical storm inputs between CPU and CUDA")
    if cpu[times] != gpu[times]:
        if (cpu[times][:-1] != gpu[times][:-1] or min(cpu[times][-1], gpu[times][-1]) < end
                or cpu["rain_rates_ms"][-2:] != [0, 0]):
            raise ValueError("Different rainfall over the matched CPU and CUDA observation window")


def collect(*, directory, records, target):
    index = directory / "strong_storms/completed.csv"
    rows = []
    if index.exists():
        for control in metrics.csv_read(file=index):
            folder = index.parent / control["id"]
            manifest = json.loads((folder / "inputs.json").read_text())
            identity = manifest["identity"]
            if runner.digest(identity)[:24] != control["id"]:
                raise ValueError("Invalid CPU storm source identity")
            for item in records.values():
                if item["s"] != identity["spec"] or item["solver_source"] != identity["numerical"]:
                    continue
                data = item["d"]
                end = int(item["s"]["hours"] * 3600)
                window = folder / f"{end}.json"
                observed = json.loads(window.read_text())
                if (observed["start_s"] != 0 or observed["end_s"] != end
                        or data["forcing"]["native_inputs"]["native_sha256"] != manifest["physical_fields"]
                        or observed["budget"]["numerical"] != data["numerical"]):
                    raise ValueError("CPU and CUDA controls differ in physical inputs, window or formulation")
                same_forcing(cpu=manifest["forcing"], gpu=data["forcing"], end=end)
                delta = interval_differences(cpu=observed["frames"], gpu=data["frames"])
                ledger = observed["budget"]
                rows.append(dict(case=control["case"], method=control["method"], dx_m=control["dx_m"],
                                 cpu_id=control["id"], gpu_id=item["id"], cpu_window_sha256=runner.checksum(window),
                                 native_fields_sha256=item["native_sha"], solver_source=item["solver_source"],
                                 rain_difference_m3=data["rain_m3"]-ledger["rain"],
                                 infiltration_difference_m3=data["infiltration_m3"]-ledger["infiltration"],
                                 storage_difference_m3=data["final_storage_m3"]-ledger["storage_m3"], **delta))
    metrics.csv_write(file=target / "cpu_cuda_initial_controls.csv", rows=rows)
