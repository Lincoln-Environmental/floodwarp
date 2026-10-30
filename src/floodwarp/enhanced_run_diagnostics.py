"""Opt-in read-only accepted-step observations for long catchment validation.

No hydraulic state or accounting is written. Counters remain on the device;
readback occurs only at caller-requested diagnostic checkpoint boundaries.
"""
import numpy as np
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.enhanced_kernels import hydraulic_step_enabled
from floodwarp.local_inertial_kernels import DL_DT, DL_TIME


@wp.kernel
def component_transport(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
    offsets: wp.array(dtype=int), adjacency: wp.array(dtype=int),
    proposed: wp.array(dtype=WP_FLOAT), accepted: wp.array(dtype=WP_FLOAT),
    alpha: wp.array(dtype=wp.float64), statistics: wp.array2d(dtype=wp.float64)):
    if not hydraulic_step_enabled(control, status):
        return
    c = wp.tid()
    if alpha[c] < wp.float64(1.0):
        statistics[c,0] += wp.float64(1.0)
    demand = wp.float64(0.0)
    actual = wp.float64(0.0)
    for k in range(offsets[c], offsets[c+1]):
        entry = adjacency[k]
        f = wp.abs(entry)-1
        sign = wp.float64(1.0)
        if entry < 0:
            sign = wp.float64(-1.0)
        demand += wp.max(wp.float64(0.0), sign*wp.float64(proposed[f]))
        actual += wp.max(wp.float64(0.0), sign*wp.float64(accepted[f]))
    statistics[c,1] += control[DL_DT]*demand
    statistics[c,2] += control[DL_DT]*actual


@wp.kernel
def accepted_step(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
    totals: wp.array(dtype=wp.float64), summary: wp.array(dtype=wp.float64),
    histogram: wp.array(dtype=int), samples: wp.array2d(dtype=wp.float64), sample_s: wp.float64, origin_s: wp.float64):
    if not hydraulic_step_enabled(control, status):
        return
    dt = control[DL_DT]
    end = control[DL_TIME]+dt
    rate = (totals[4]-summary[0])/dt
    summary[0] = totals[4]
    summary[1] += wp.float64(1.0)
    summary[2] = wp.min(summary[2], dt)
    summary[3] = wp.max(summary[3], dt)
    if rate > summary[4]:
        summary[4] = rate
        summary[5] = end
    bin = wp.clamp(int(wp.floor(wp.log2(dt)*wp.float64(16.)))+384,0,histogram.shape[0]-1)
    histogram[bin] += 1
    row = int(wp.floor((end-origin_s)/sample_s+wp.float64(1.e-8)))
    previous = int(wp.floor((control[DL_TIME]-origin_s)/sample_s+wp.float64(1.e-8)))
    if row > previous and row < samples.shape[0]:
        samples[row,0] = end
        samples[row,1] = rate
        samples[row,2] = totals[4]
        samples[row,3] = totals[3]
        samples[row,4] = dt


def enable(*, solver, duration_s, sample_s):
    if not np.isfinite(duration_s) or not np.isfinite(sample_s) or duration_s <= 0 or sample_s <= 0:
        raise ValueError('Positive finite duration and sample interval required')
    solver._ec_run_summary = wp.zeros(6, dtype=wp.float64, device=solver.device)
    initial = np.zeros(6)
    initial[0] = solver.cumulative_totals.numpy()[4]
    initial[2] = np.inf
    solver._ec_run_summary.assign(initial)
    solver._ec_run_histogram = wp.zeros(512, dtype=int, device=solver.device)
    solver._ec_run_samples = wp.zeros((int(np.ceil(duration_s/sample_s))+1,5), dtype=wp.float64, device=solver.device)
    solver._ec_run_components = wp.zeros((solver._component_count,3), dtype=wp.float64, device=solver.device)
    solver._ec_run_sample_s = float(sample_s)
    solver._ec_run_origin_s = float(solver.simulation_time)


def read(*, solver):
    summary = solver._ec_run_summary.numpy()
    components = solver._ec_run_components.numpy()
    histogram = solver._ec_run_histogram.numpy()
    ids = np.flatnonzero(components[:,0])
    top = ids[np.argsort(components[ids,0])[-10:][::-1]]
    return dict(accepted_steps=int(summary[1]),minimum_dt_s=float(summary[2]),maximum_dt_s=float(summary[3]),
        boundary_peak_m3_s=float(summary[4]),boundary_peak_time_s=float(summary[5]),
        limited_component_steps=int(components[:,0].sum()),limited_components=int(len(ids)),
        proposed_outward_volume_m3=float(components[:,1].sum()),accepted_outward_volume_m3=float(components[:,2].sum()),
        most_limited_components=[dict(component=int(c),limited_steps=int(components[c,0])) for c in top],
        histogram_counts=histogram.tolist(),histogram_log2_left_edges=((np.arange(512)-384)/16).tolist())
