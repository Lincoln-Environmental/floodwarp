"""Right-running simple waves before characteristic crossing, with accepted section transport."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import warp as wp

from research.floodwarp_methods_paper import runner
from validation.fv2_local_inertia import build


def state(*, x, time, settings, lobes):
    g = settings["gravity"]
    c0 = np.sqrt(g * settings["depth"])
    slope_bound = sum(amp / sigma for _, sigma, amp in lobes) * np.exp(-0.5)
    if 3 * slope_bound * np.max(time) >= 1:
        raise ValueError("Analytical wave may have broken")
    low = x - (c0 + 3 * sum(lobe[2] for lobe in lobes)) * time
    high = x - c0 * time
    for _ in range(42):
        origin = (high + low) / 2
        dc = sum(amp * np.exp(-0.5 * ((origin - settings["centre"] - offset) / sigma) ** 2)
                 for offset, sigma, amp in lobes)
        left = origin + (c0 + 3 * dc) * time < x
        low = np.where(left, origin, low)
        high = np.where(left, high, origin)
    h = (c0 + dc) ** 2 / g
    return h, 2 * dc * h


def initial(*, dx, cfg, lobes, nodes, weights):
    x = (np.arange(round(cfg["length"] / dx)) + 0.5) * dx - cfg["length"] / 2
    shape = (round(cfg["width"] / dx), len(x))
    h, q = state(x=x[:, None] + dx * nodes / 2, time=0, settings=cfg, lobes=lobes)
    u = np.zeros((*shape, 3))
    u[..., 0] = h @ weights / 2
    u[..., 1] = q @ weights / 2
    return dict(z=np.zeros(shape), n=np.zeros(shape), rain=np.zeros(shape), initial=u)


def forecast(*, cfg, method, dx, courant, profile):
    nodes, weights = np.polynomial.legendre.leggauss(cfg["order"])
    lobes = cfg["profiles"][profile]
    data = initial(dx=dx, cfg=cfg, lobes=lobes, nodes=nodes, weights=weights)
    t = np.arange(0, cfg["duration"] + cfg["interval"], cfg["interval"])
    tx = (t[:-1, None] + (nodes + 1) * np.diff(t)[:, None] / 2)[..., None]
    loc = np.asarray(cfg["stations"])
    _, q = state(x=loc, time=tx, settings=cfg, lobes=lobes)
    truth = np.einsum("tng,n->tg", q, weights) * cfg["width"] / 2
    idx = (loc + cfg["length"] / 2) / dx
    np.testing.assert_array_equal(idx, idx.astype(int))
    cap = courant * dx / np.sqrt(cfg["gravity"] * cfg["depth"])
    solver = build(d=data, method=method, dx=dx, cap=cap, device=cfg["device"], gravity=cfg["gravity"])
    if method in ("fv1", "roe", "hllc"):
        solver.configure_execution(mode="sparse")
    previous = np.zeros(len(loc))
    flow = []
    steps = 0
    elapsed = 0.
    minimum_dt = np.inf
    for dt in np.diff(t):
        wp.synchronize_device(device=cfg["device"])
        tick = perf_counter()
        s = solver.run_device_loop(duration=float(dt))
        wp.synchronize_device(device=cfg["device"])
        elapsed += perf_counter() - tick - s["graph_capture_seconds"] - s.get("compilation_seconds", 0.)
        current = solver.get_cumulative_face_volumes()["qx"][:, idx.astype(int)].sum(axis=0)
        flow.append((current - previous) / dt)
        previous = current
        steps += s["steps"]
        minimum_dt = min(minimum_dt, s["dt_min_s"])
    depth = solver.get_water_depth()
    residual = float(depth.sum() * dx**2 - solver.initial_volume)
    if not np.isfinite(depth).all() or depth.min() < 0 or abs(residual) > 1.e-9 * solver.initial_volume:
        raise ArithmeticError("Pulse conservation or positivity failure")
    return dict(edges=t, flow=np.array(flow), truth=truth, stations=loc,
                runtime_s=elapsed, steps=steps, minimum_dt=minimum_dt, residual_m3=residual,
                initial_sha256=hashlib.sha256(data["initial"].tobytes()).hexdigest())


def run(*, cfg, output, root):
    hardware = runner.machine(root=root)
    source = {str(p.relative_to(root)): runner.checksum(p) for p in
              [Path(__file__), root / "validation/fv2_local_inertia.py", *sorted((root / "src/floodwarp").glob("*.py"))]}
    fingerprint = runner.digest(dict(cfg=cfg, source=source, hardware=[hardware["gpu"], hardware["warp"], hardware["python"]]))
    target = output / fingerprint[:16]
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "inputs.json"
    if not manifest.exists():
        runner.write_json(path=manifest, data=dict(cfg=cfg, source=source, hardware=hardware))
    for profile in cfg["profiles"]:
        for dx in cfg["dx"]:
            for courant in cfg["cap_courant"]:
                for method in cfg["methods"]:
                    file = target / f"{profile}_{dx}_{courant}_{method}.npz"
                    if file.exists():
                        continue
                    obs = forecast(cfg=cfg, method=method, dx=dx, courant=courant, profile=profile)
                    temp = file.with_suffix(".pending.npz")
                    np.savez_compressed(temp, **obs)
                    os.replace(temp, file)
            print(profile, dx, "m complete", flush=True)
    (output / "current.txt").write_text(target.name + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "design.json").read_text())["pulse"]
    run(cfg=settings, output=directory / "pulse_runs", root=directory.parents[2])

