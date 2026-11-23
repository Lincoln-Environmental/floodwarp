"""Exact oblique wave crossing a fixed, physical x-face section in a two-dimensional box."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity.pulse import state
from validation.fv2_local_inertia import build


def solve(*, cfg, dx, normal, method, length, aperture, cap_courant):
    nodes, weights = np.polynomial.legendre.leggauss(cfg["order"])
    lobes = cfg["profiles"]["smooth"]
    angle_x, angle_y = normal
    axis = (np.arange(round(length / dx)) + 0.5) * dx - length / 2
    along = axis[None, :] * angle_x + axis[:, None] * angle_y
    # At time zero the free surface is explicit. Integrate square cells in both directions.
    init = np.zeros((*along.shape, 3))
    for i, wx in zip(nodes, weights, strict=True):
        for j, wy in zip(nodes, weights, strict=True):
            x = along + dx * (i * angle_x + j * angle_y) / 2
            dc = sum(
                a * np.exp(-0.5 * ((x - cfg["centre"] - off) / s) ** 2)
                for off, s, a in lobes
            )
            h = (np.sqrt(cfg["depth"] * cfg["gravity"]) + dc) ** 2 / cfg["gravity"]
            w = wx * wy / 4
            init[..., 0] += w * h
            init[..., 1] += w * 2 * dc * h * angle_x
            init[..., 2] += w * 2 * dc * h * angle_y
    data = dict(
        initial=init,
        z=np.zeros_like(along),
        n=np.zeros_like(along),
        rain=np.zeros_like(along),
    )
    solver = build(
        d=data,
        method=method,
        dx=dx,
        cap=cap_courant * dx / np.sqrt(cfg["gravity"] * cfg["depth"]),
        device=cfg["device"],
        gravity=cfg["gravity"],
    )
    if method == "fv1":
        solver.configure_execution(mode="sparse")
    t = np.arange(0, cfg["duration"] + cfg["interval"], cfg["interval"])
    faces = np.array(cfg["stations"])
    xidx = ((faces + length / 2) / dx).astype(int)
    assert np.array_equal(xidx * dx - length / 2, faces)
    ymask = np.abs(axis) < aperture / 2
    assert ymask.sum() * dx == aperture
    # No reflected wall signal can reach an observed section before the end.
    speed = np.sqrt(cfg["depth"] * cfg["gravity"]) + 3 * sum(l[2] for l in lobes)
    if speed * t[-1] >= length / 2 - max(aperture / 2, max(abs(faces))):
        raise ValueError("Gauge intersects wall signal cone")
    tt = (t[:-1, None] + np.diff(t)[:, None] * (nodes + 1) / 2)[:, :, None, None]
    proj = faces[None, :] * angle_x + nodes[:, None] * aperture / 2 * angle_y
    _, q = state(x=proj, time=tt, settings=cfg, lobes=lobes)
    truth = np.einsum("tnyg,n,y->tg", q, weights, weights) * aperture * angle_x / 4
    prev = np.zeros(len(faces))
    flow = []
    cost = 0.0
    for dt in np.diff(t):
        wp.synchronize_device(device=cfg["device"])
        tick = perf_counter()
        stats = solver.run_device_loop(duration=float(dt))
        wp.synchronize_device(device=cfg["device"])
        cost += (
            perf_counter()
            - tick
            - stats["graph_capture_seconds"]
            - stats.get("compilation_seconds", 0)
        )
        curr = solver.get_cumulative_face_volumes()["qx"][ymask][:, xidx].sum(axis=0)
        flow.append((curr - prev) / dt)
        prev = curr
    for series in (np.asarray(flow), truth):
        if np.any(series[-1] > cfg["tail_fraction"] * series.max(axis=0)):
            raise ValueError("The orientation experiment needs a longer recession")
    h = solver.get_water_depth()
    residual = float(h.sum() * dx**2 - solver.initial_volume)
    if h.min() < 0 or abs(residual) > 1e-9 * solver.initial_volume:
        raise ArithmeticError("Oblique wave balance failed")
    return dict(
        edges=t,
        truth=truth,
        flow=flow,
        stations=faces,
        distance=faces * angle_x - cfg["centre"],
        normal=normal,
        runtime_s=cost,
        residual_m3=residual,
    )


def run(*, directory, cfg, spacings, normals, methods, length, aperture, cap):
    machine = runner.machine(root=directory.parents[2])
    spec = dict(
        cfg=cfg,
        spacings=spacings,
        normals=normals,
        methods=methods,
        length=length,
        aperture=aperture,
        cap=cap,
        code=runner.checksum(Path(__file__)),
        source=runner.source_identity(
            root=directory.parents[2], study=directory.parent, stage="analytical"
        ),
        pulse_code=runner.checksum(directory / "pulse.py"),
        hardware=[machine["gpu"], machine["warp"]],
    )
    output = directory / "oblique_runs" / runner.digest(spec)[:16]
    output.mkdir(parents=True, exist_ok=True)
    runner.write_json(path=output / "inputs.json", data=spec)
    for dx in spacings:
        for label, normal in normals.items():
            for method in methods:
                file = output / f"{label}_{dx}_{method}.npz"
                if file.exists():
                    continue
                result = solve(
                    cfg=cfg,
                    dx=dx,
                    normal=normal,
                    method=method,
                    length=length,
                    aperture=aperture,
                    cap_courant=cap,
                )
                temporary = file.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **result)
                temporary.replace(file)
        print(dx, "m orientation pairs complete", flush=True)
    (output.parent / "current.txt").write_text(output.name + "\n")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    design = json.loads((directory / "design.json").read_text())
    config = dict(design["pulse"], duration=176, tail_fraction=design["tail_fraction"])
    spacings = (5, 10, 20)
    normals = {"aligned": (1.0, 0.0), "oblique": (0.8, 0.6)}
    methods = ("fv1", "hancock_adaptive", "adaptive")
    length = 1600
    aperture = 80
    cap = 0.13
    run(
        directory=directory,
        cfg=config,
        spacings=spacings,
        normals=normals,
        methods=methods,
        length=length,
        aperture=aperture,
        cap=cap,
    )
