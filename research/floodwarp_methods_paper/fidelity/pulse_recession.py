"""Extend the channel controls until even the coarse, broadened double pulse has passed."""

import json
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
from research.floodwarp_methods_paper.fidelity import pulse
from research.floodwarp_methods_paper import runner


@lru_cache
def maximum_slope(lobes):
    # Bound the maximum slope between sampled points by the sup of the Gaussian second derivative.
    sigma_min = min(s for _, s, _ in lobes)
    left = min(offset - 12 * s for offset, s, _ in lobes)
    right = max(offset + 12 * s for offset, s, _ in lobes)
    nodes = np.linspace(left, right, int(np.ceil(128 * (right - left) / sigma_min)) + 1)
    derivative = np.zeros_like(nodes)
    for offset, s, a in lobes:
        u = (nodes - offset) / s
        derivative -= a * u * np.exp(-u * u / 2) / s
    curvature = sum(a / s**2 for _, s, a in lobes)
    inside = np.max(abs(derivative)) + 0.5 * (nodes[1] - nodes[0]) * curvature
    outside = sum(a / s for _, s, a in lobes) * 12 * np.exp(-72)
    return max(inside, outside)


def state(*, x, time, settings, lobes):
    waves = tuple(tuple(v) for v in lobes)
    if 3 * np.max(time) * maximum_slope(waves) >= 1:
        raise ValueError("Exact pulse reaches characteristic crossing")
    base = np.sqrt(settings["gravity"] * settings["depth"])
    left = x - (base + 3 * sum(a for _, _, a in waves)) * time
    right = x - base * time
    for _ in range(46):
        origin = (left + right) / 2
        dc = sum(
            a * np.exp(-0.5 * ((origin - settings["centre"] - offset) / sigma) ** 2)
            for offset, sigma, a in waves
        )
        below = origin + (base + 3 * dc) * time < x
        left = np.where(below, origin, left)
        right = np.where(below, right, origin)
    h = (base + dc) ** 2 / settings["gravity"]
    return h, 2 * dc * h


def run(*, directory, duration, length, tail):
    cfg = json.loads((directory / "design.json").read_text())["pulse"]
    largest = sum(a for _, _, a in cfg["profiles"]["double"])
    speed = np.sqrt(cfg["gravity"] * cfg["depth"]) + 3 * largest
    if speed * duration >= length / 2 - max(abs(x) for x in cfg["stations"]):
        raise ValueError("Move end walls outside the analytical domain of dependence")
    cfg = dict(
        cfg,
        length=length,
        duration=duration,
        extended_truth_source=runner.checksum(Path(__file__)),
    )
    with patch.object(pulse, "state", state):
        pulse.run(cfg=cfg, output=directory / "long_pulses", root=directory.parents[2])
    folder = directory / "long_pulses"
    target = folder / (folder / "current.txt").read_text().strip()
    for file in target.glob("*.npz"):
        with np.load(file) as d:
            for k in ("flow", "truth"):
                q = d[k]
                if np.any(q[-1] > tail * q.max(axis=0)):
                    raise ValueError(f"Recession unresolved in {file.name}")


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    duration = 176
    length = 1600
    tail = 0.05
    run(directory=directory, duration=duration, length=length, tail=tail)
