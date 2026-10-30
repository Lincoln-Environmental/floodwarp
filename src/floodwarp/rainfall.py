"""Native rainfall hyetograph construction and serialization for Overlander.

Dynamic hyetographs are continuous, piecewise-linear rainfall-rate functions.
Their node times and rates are the canonical forcing shared by Overlander and
LISFLOOD-FP. Rates are stored in m/s and serialized for LISFLOOD-FP in mm/h.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


RAINFALL_INTERPOLATION = "piecewise_linear"


@dataclass(frozen=True)
class Hyetograph:
    """A continuous piecewise-linear rainfall-rate function at time nodes."""

    node_times_s: np.ndarray
    node_rates_m_per_s: np.ndarray
    storm_type: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.node_times_s.ndim != 1 or self.node_rates_m_per_s.ndim != 1:
            raise ValueError("hyetograph arrays must be one-dimensional")
        if self.node_times_s.size != self.node_rates_m_per_s.size:
            raise ValueError("hyetograph needs one rate at each time node")
        if self.node_times_s.size < 2 or self.node_times_s[0] != 0.0:
            raise ValueError("hyetograph must start at zero and contain at least two nodes")
        if not np.all(np.diff(self.node_times_s) > 0.0):
            raise ValueError("hyetograph node times must be strictly increasing")
        if not np.all(np.isfinite(self.node_rates_m_per_s)) or np.any(self.node_rates_m_per_s < 0.0):
            raise ValueError("hyetograph rates must be finite and non-negative")

    @property
    def duration_s(self) -> float:
        return float(self.node_times_s[-1])

    @property
    def integrated_depth_m(self) -> float:
        return float(np.trapezoid(self.node_rates_m_per_s, self.node_times_s))

    @property
    def interpolation(self) -> str:
        return RAINFALL_INTERPOLATION

    def rate_at(self, time_s: float) -> float:
        """Return the linearly interpolated rainfall rate at a simulation time."""
        if time_s < 0.0 or time_s > self.duration_s:
            raise ValueError("time is outside the hyetograph duration")
        return float(np.interp(time_s, self.node_times_s, self.node_rates_m_per_s))

    def integrated_depth_between(self, start_time_s: float, end_time_s: float) -> float:
        """Return the exact piecewise-linear rainfall depth over an interval.

        Rainfall is zero outside the storm.  This makes a variable physical
        timestep a quadrature choice only: callers can use the returned depth
        to set an interval-average explicit source without changing total
        rainfall volume.
        """
        if not np.isfinite(start_time_s) or not np.isfinite(end_time_s):
            raise ValueError("interval times must be finite")
        if end_time_s < start_time_s:
            raise ValueError("end_time_s must not precede start_time_s")
        if end_time_s == start_time_s:
            return 0.0

        start_s = max(float(start_time_s), 0.0)
        end_s = min(float(end_time_s), self.duration_s)
        if end_s <= start_s:
            return 0.0

        interior_nodes = self.node_times_s[
            (self.node_times_s > start_s) & (self.node_times_s < end_s)
        ]
        sample_times = np.concatenate((np.array([start_s]), interior_nodes, np.array([end_s])))
        sample_rates = np.interp(
            sample_times,
            self.node_times_s,
            self.node_rates_m_per_s,
        )
        return float(np.trapezoid(sample_rates, sample_times))

    def mean_rate_between(self, start_time_s: float, end_time_s: float) -> float:
        """Return the interval-average rate, including zero forcing after rain."""
        duration_s = float(end_time_s) - float(start_time_s)
        if duration_s <= 0.0:
            raise ValueError("end_time_s must be greater than start_time_s")
        return self.integrated_depth_between(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        ) / duration_s

    def next_breakpoint_after(self, time_s: float) -> float:
        """Return the next node after a simulation time."""
        index = int(np.searchsorted(self.node_times_s, time_s, side="right"))
        return float(self.node_times_s[min(index, self.node_times_s.size - 1)])


def interval_rainfall_rate(
    *,
    hyetograph: Hyetograph,
    start_time_s: float,
    end_time_s: float,
    sample_at_interval_start: bool,
) -> float:
    """Return the uniform forcing rate for one accepted solver interval.

    Profiles that apply sources after momentum (the LISFLOOD-FP ACC
    convention) sample the piecewise-linear forcing at the interval start,
    matching the validated explicit host/device-loop paths.  Other profiles
    use the exact interval mean, which equals the explicit midpoint sample
    for piecewise-linear forcing and keeps applied volume quadrature-exact
    under timestep refinement.  The rate is reevaluated for every trial
    interval, so a retry that shortens dt can never double-apply rainfall.
    """
    if end_time_s <= start_time_s:
        raise ValueError("end_time_s must be greater than start_time_s")
    if sample_at_interval_start:
        # Rainfall is zero outside the storm; the interval-start sample must
        # not clamp onto the terminal rate once the storm has passed.
        if float(start_time_s) >= hyetograph.duration_s:
            return 0.0
        return hyetograph.rate_at(time_s=float(start_time_s))
    return hyetograph.mean_rate_between(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )


def build_uniform_hyetograph(duration_s: float, total_depth_m: float) -> Hyetograph:
    """Build the existing constant-intensity storm representation."""
    _validate_storm_inputs(duration_s=duration_s, total_depth_m=total_depth_m)
    intensity = total_depth_m / duration_s
    return Hyetograph(
        node_times_s=np.array([0.0, duration_s]),
        node_rates_m_per_s=np.array([intensity, intensity]),
        storm_type="uniform",
        metadata={"interpolation": RAINFALL_INTERPOLATION},
    )


def build_chicago_normalized_hyetograph(
    duration_s: float,
    total_depth_m: float,
    peak_fraction: float,
    interval_s: float,
    pre_peak_exponent: float,
    post_peak_exponent: float,
) -> Hyetograph:
    """Build a normalized Chicago-style storm without regional IDF assumptions."""
    _validate_storm_inputs(duration_s=duration_s, total_depth_m=total_depth_m)
    if not 0.0 < peak_fraction < 1.0:
        raise ValueError("peak_fraction must be strictly between zero and one")
    if min(interval_s, pre_peak_exponent, post_peak_exponent) <= 0.0:
        raise ValueError("Chicago interval and exponents must be positive")

    peak_time_s = duration_s * peak_fraction
    node_times_s = np.unique(
        np.concatenate((np.arange(0.0, duration_s, interval_s), np.array([peak_time_s, duration_s])))
    )
    node_times_s.sort()
    pre_peak = np.power(node_times_s / peak_time_s, pre_peak_exponent)
    post_peak = np.power(
        (duration_s - node_times_s) / (duration_s - peak_time_s),
        post_peak_exponent,
    )
    normalized_rates = np.where(node_times_s <= peak_time_s, pre_peak, post_peak)
    normalized_depth = float(np.trapezoid(normalized_rates, node_times_s))
    node_rates_m_per_s = normalized_rates * total_depth_m / normalized_depth
    return Hyetograph(
        node_times_s=node_times_s,
        node_rates_m_per_s=node_rates_m_per_s,
        storm_type="chicago_normalized",
        metadata={
            "interpolation": RAINFALL_INTERPOLATION,
            "peak_fraction": peak_fraction,
            "interval_s": interval_s,
            "pre_peak_exponent": pre_peak_exponent,
            "post_peak_exponent": post_peak_exponent,
        },
    )


def write_hyetograph_csv(hyetograph: Hyetograph, output_path: Path) -> None:
    """Write canonical rainfall nodes for Overlander and validation tools."""
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("time_s", "rate_mm_per_hour"))
        for time_s, rate_m_per_s in zip(
            hyetograph.node_times_s,
            hyetograph.node_rates_m_per_s,
            strict=True,
        ):
            writer.writerow((f"{time_s:.12g}", f"{rate_m_per_s * 3.6e6:.12g}"))


def read_hyetograph_csv(
    input_path: Path,
    storm_type: str = "external_hyetograph_csv",
    metadata: Mapping[str, object] | None = None,
) -> Hyetograph:
    """Read a CSV hyetograph without changing its temporal representation.

    The preferred interchange format is FloodFM-data's canonical node CSV,
    with ``node_time_s`` and ``node_rate_m_per_s`` columns. Extra columns
    such as human-readable hours and mm/h are ignored. The legacy
    Overlander node (``time_s,rate_mm_per_hour``) and interval
    (``start_time_s,end_time_s,intensity_mm_per_hour``) formats remain
    readable for validation-case compatibility.

    Canonical nodes are passed directly to :class:`Hyetograph`: rates remain
    in m/s and the piecewise-linear interpolation is therefore identical to
    the prepared FloodFM model. No design-depth scaling occurs here.
    """
    input_path = Path(input_path)
    supplied_metadata = dict(metadata or {})
    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        field_names = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"hyetograph contains no data: {input_path}")

    canonical_columns = {"node_time_s", "node_rate_m_per_s"}
    if canonical_columns.issubset(field_names):
        try:
            node_times_s = np.asarray(
                [float(row["node_time_s"]) for row in rows], dtype=np.float64
            )
            node_rates_m_per_s = np.asarray(
                [float(row["node_rate_m_per_s"]) for row in rows], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid canonical hyetograph values: {input_path}"
            ) from error
        return Hyetograph(
            node_times_s=node_times_s,
            node_rates_m_per_s=node_rates_m_per_s,
            storm_type=storm_type,
            metadata={
                **supplied_metadata,
                "source": str(input_path),
                "csv_format": "canonical_nodes_m_per_s",
            },
        )
    if field_names == ["time_s", "rate_mm_per_hour"]:
        return Hyetograph(
            node_times_s=np.asarray([float(row["time_s"]) for row in rows]),
            node_rates_m_per_s=np.asarray([float(row["rate_mm_per_hour"]) / 3.6e6 for row in rows]),
            storm_type=storm_type,
            metadata=supplied_metadata,
        )
    if field_names == ["start_time_s", "end_time_s", "intensity_mm_per_hour"]:
        starts_s = np.asarray([float(row["start_time_s"]) for row in rows])
        ends_s = np.asarray([float(row["end_time_s"]) for row in rows])
        rates = np.asarray([float(row["intensity_mm_per_hour"]) / 3.6e6 for row in rows])
        if starts_s[0] != 0.0 or not np.allclose(starts_s[1:], ends_s[:-1], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"hyetograph intervals are not contiguous: {input_path}")
        return Hyetograph(
            node_times_s=np.concatenate((starts_s[:1], ends_s)),
            node_rates_m_per_s=np.concatenate((rates, rates[-1:])),
            storm_type=storm_type,
            metadata={
                **supplied_metadata,
                "interpolation": "legacy_piecewise_constant",
            },
        )
    raise ValueError(f"unrecognized hyetograph columns: {input_path}")


def write_lisflood_rainfall(hyetograph: Hyetograph, output_path: Path) -> None:
    """Write canonical piecewise-linear rainfall nodes in LISFLOOD-FP format.

    LISFLOOD-FP holds the final entry's rate beyond the last node, so a
    zero-rate termination entry is appended one second after the storm end —
    without it a non-zero terminal rate keeps raining for the rest of the
    simulation.
    """
    lines = ["Overlander rainfall hyetograph", f"{hyetograph.node_times_s.size + 1} seconds"]
    lines.extend(
        f"{rate_m_per_s * 3.6e6:.12g} {time_s:.12g}"
        for time_s, rate_m_per_s in zip(
            hyetograph.node_times_s,
            hyetograph.node_rates_m_per_s,
            strict=True,
        )
    )
    lines.append(f"0.0 {hyetograph.duration_s + 1.0:.12g}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_storm_inputs(duration_s: float, total_depth_m: float) -> None:
    if min(duration_s, total_depth_m) <= 0.0:
        raise ValueError("storm duration and depth must be positive")
