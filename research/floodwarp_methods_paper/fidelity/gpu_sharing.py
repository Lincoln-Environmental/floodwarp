"""Associate sampled simultaneous GPU compute with measured native forecast clocks."""

import getopt
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

from research.floodwarp_methods_paper import runner


def process_samples(*, lines, zone, gpu, solver_pid):
    samples = {}
    for line in lines:
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        day, clock, device, pid, kind, sm = fields[:6]
        if int(device) != gpu:
            continue
        timestamp = datetime.strptime(f"{day} {clock}", "%Y%m%d %H:%M:%S").replace(tzinfo=zone)
        second = int(timestamp.timestamp())
        sample = samples.setdefault(second, [0, 0])
        if "C" in kind and sm != "-":
            value = int(sm)
            if not 0 <= value <= 100:
                raise ValueError("Invalid GPU SM utilisation")
            index = 0 if int(pid) == solver_pid else 1
            sample[index] = max(sample[index], value)
    return [[second, *samples[second]] for second in sorted(samples)]


def save_capture(*, directory, log, zone, gpu, solver_pid, device_uuid, case, before):
    # Keep complete sampling ticks. A live writer may still be appending processes
    # to its last timestamp, even when its last line ends with a newline.
    lines = log.read_bytes().splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b"\n")]
    samples = process_samples(lines=[line.decode() for line in complete],
                              zone=zone, gpu=gpu, solver_pid=solver_pid)
    if before is not None:
        if not samples or samples[-1][0] < before:
            raise ValueError("Capture has not reached the exclusive cutoff")
        samples = [s for s in samples if s[0] < before]
        selected = []
        for line in complete:
            fields = line.split()
            if fields and not fields[0].startswith(b"#"):
                dt = datetime.strptime(b" ".join(fields[:2]).decode(), "%Y%m%d %H:%M:%S")
                if dt.replace(tzinfo=zone).timestamp() >= before:
                    break
            selected.append(line)
        complete = selected
    if not samples or not any(s[1] for s in samples):
        raise ValueError("No measured solver compute in the selected capture")
    evidence = dict(case=case, gpu_index=gpu, device_uuid=device_uuid, solver_pid=solver_pid,
                    source_zone=zone.key, source_sha256=sha256(b"".join(complete)).hexdigest(),
                    columns=["utc_epoch_s", "solver_sm_pct", "largest_other_compute_sm_pct"],
                    samples=samples)
    path = directory / "gpu_samples" / (runner.digest(evidence)[:24] + ".json")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != evidence:
            raise ValueError("GPU capture identity already exists with different observations")
    else:
        with path.open("x") as output:
            json.dump(evidence, output, separators=(",", ":"))
            output.write("\n")
    print(f"{path.name}: {len(samples)} samples, {sum(a > 0 and b > 0 for _, a, b in samples)} with simultaneous compute")


def read_samples(*, directory):
    cases = defaultdict(dict)
    for path in sorted((directory / "gpu_samples").glob("*.json")):
        record = json.loads(path.read_text())
        if path.stem != runner.digest(record)[:24]:
            raise ValueError(f"Altered GPU sample receipt {path}")
        observed = cases[record["case"]]
        for second, solver, other in record["samples"]:
            if second in observed and observed[second] != (solver, other):
                raise ValueError("Conflicting GPU samples for the same forecast and second")
            observed[second] = (solver, other)
    return {case: (sorted(data), data) for case, data in cases.items()}


def window_status(*, samples, windows):
    times, observations = samples
    ticks = set()
    for start, end in windows:
        # Require a whole timestamp second inside the recorded invocation.
        i = bisect_left(times, start)
        ticks.update(t for t in times[i:bisect_left(times, end)] if t + 1 <= end)
    active = [t for t in sorted(ticks) if observations[t][0] > 0]
    shared = [t for t in active if observations[t][1] > 0]
    return dict(gpu_active_samples=len(active), gpu_shared_samples=len(shared),
                first_shared_utc_s=shared[0] if shared else None,
                last_shared_utc_s=shared[-1] if shared else None,
                raw_timing_status=("shared GPU observed" if shared else
                                   "no shared compute in sampled ticks" if active else "unmonitored"))


def assess(*, directory, records, groups, verification):
    samples = read_samples(directory=directory)
    attempts = {r["id"]: r["attempt"] for r in verification if r.get("attempt")}
    windows = {}
    for ident, item in records.items():
        case = item["s"]["case"]
        spans = []
        if case in samples and ident in attempts:
            attempt = attempts[ident]
            start = int(attempt.removeprefix("attempt_")) / 1e9
            receipt = json.loads((item["path"] / attempt / "complete.json").read_text())
            spans.append((start, start + receipt["seconds"]))
        windows[ident] = spans
        item.update(window_status(samples=samples.get(case, ([], {})), windows=spans))
    for (_, case, _), peers in groups.items():
        for item in peers:
            parent = item.get("parent_id", item["id"])
            spans = list(windows[parent])
            if case in samples and parent != item["id"]:
                end_s = item["s"]["hours"] * 3600
                for path in (directory / "extension_timings" / item["id"]).glob("*.json"):
                    receipt = json.loads(path.read_text())
                    if receipt["case"] != case or receipt["id"] != item["id"]:
                        raise ValueError("GPU clock window belongs to another forecast")
                    if receipt["end_s"] <= end_s:
                        stop = receipt["recorded_ns"] / 1e9
                        spans.append((stop - receipt["elapsed_wall_s"], stop))
            item.update(window_status(samples=samples.get(case, ([], {})), windows=spans))


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    flags, extra = getopt.getopt(sys.argv[1:], "", ["log=", "zone=", "gpu=", "pid=", "device-uuid=",
                                                    "case=", "before=", "closed-log", "dry-run", "resume"])
    options = dict(flags)
    if extra:
        raise ValueError(f"Unexpected GPU sample arguments {extra}")
    if "--dry-run" in options:
        print("Import a finished pmon log with --closed-log or bound a live sample with --before UTC_EPOCH_SECOND.")
    else:
        if ("--closed-log" in options) == ("--before" in options):
            raise ValueError("Specify either --closed-log or an exclusive epoch-second cutoff")
        save_capture(directory=directory, log=Path(options["--log"]), zone=ZoneInfo(options["--zone"]),
                     gpu=int(options["--gpu"]), solver_pid=int(options["--pid"]), case=options["--case"],
                     device_uuid=options["--device-uuid"], before=int(options["--before"]) if "--before" in options else None)
