"""Independent integration of simple-wave section transport in initial-position coordinates."""

from math import exp, sqrt
from pathlib import Path
import json

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from research.floodwarp_methods_paper import metrics, runner


def section_flux(*, edges, station, config, profile):
    c0 = sqrt(config["gravity"] * config["depth"])
    waves = config["profiles"][profile]
    centre = config["centre"]
    fastest = c0 + 3 * sum(amplitude for _, _, amplitude in waves)

    def celerity_increment(origin):
        return sum(
            amplitude * exp(-0.5 * ((origin - centre - offset) / sigma) ** 2)
            for offset, sigma, amplitude in waves
        )

    def crossing_time(origin):
        return (station - origin) / (c0 + 3 * celerity_increment(origin))

    def residual(origin, time):
        return crossing_time(origin) - time

    def volume_per_initial_metre(origin):
        increment = celerity_increment(origin)
        slope = sum(
            -amplitude
            * (origin - centre - offset)
            / sigma**2
            * exp(-0.5 * ((origin - centre - offset) / sigma) ** 2)
            for offset, sigma, amplitude in waves
        )
        jacobian = (1 + 3 * slope * crossing_time(origin)) / (c0 + 3 * increment)
        if jacobian <= 0:
            raise ArithmeticError("Characteristic crossing in reference quadrature")
        return (
            config["width"]
            * 2
            * increment
            * (c0 + increment) ** 2
            / config["gravity"]
            * jacobian
        )

    bounds = [
        station
        if t == 0
        else brentq(
            residual,
            station - fastest * edges[-1] - config["length"],
            station,
            args=(t,),
        )
        for t in edges
    ]
    integrals = [
        quad(volume_per_initial_metre, lo, hi, epsabs=1e-13, epsrel=1e-13)[0]
        for hi, lo in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    return np.asarray(integrals) / np.diff(edges)


def verify(*, directory, absolute_tolerance, relative_tolerance):
    folder = directory / "long_pulses"
    archive = folder / (folder / "current.txt").read_text().strip()
    config = json.loads((archive / "inputs.json").read_text())["cfg"]
    rows = []
    for profile in config["profiles"]:
        files = sorted(archive.glob(profile + "_*.npz"))
        with np.load(files[0]) as source:
            time = source["edges"]
            stations = source["stations"]
        independent = np.column_stack(
            [
                section_flux(edges=time, station=x, config=config, profile=profile)
                for x in stations
            ]
        )
        for file in files:
            with np.load(file) as data:
                np.testing.assert_array_equal(data["edges"], time)
                np.testing.assert_array_equal(data["stations"], stations)
                np.testing.assert_allclose(
                    data["truth"],
                    independent,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                for j, x in enumerate(stations):
                    q = independent[:, j]
                    curve = data["truth"][:, j]
                    rows.append(
                        dict(
                            source_id=file.stem,
                            source_sha256=runner.checksum(file),
                            profile=profile,
                            station_m=x,
                            distance_m=x - config["centre"],
                            interval_count=len(q),
                            independent_peak_m3s=q.max(),
                            independent_volume_m3=q @ np.diff(time),
                            max_absolute_difference_m3s=np.max(abs(curve - q)),
                        )
                    )
    metrics.csv_write(file=directory / "results/pulse_reference_checks.csv", rows=rows)
    largest = max(r["max_absolute_difference_m3s"] for r in rows)
    print(
        f"Checked {len(rows)} archived section records. Largest difference {largest:.3g} m³/s."
    )


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    absolute_tolerance = 1e-11
    relative_tolerance = 1e-11
    verify(
        directory=directory,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
