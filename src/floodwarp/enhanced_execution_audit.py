"""Opt-in GPU accumulated timestep attribution; never writes hydraulic state."""
import numpy as np
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp.enhanced_kernels import hydraulic_step_enabled
from floodwarp.local_inertial_kernels import DL_DT, DL_TIME, DL_GROWTH_REFERENCE


@wp.kernel
def reset(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
          scratch: wp.array(dtype=wp.float64)):
    if not hydraulic_step_enabled(control, status):
        return
    scratch[0] = wp.float64(1.e100)
    scratch[1] = wp.float64(1.e100)
    scratch[2] = wp.float64(1.e100)
    scratch[3] = control[DL_GROWTH_REFERENCE]


@wp.kernel
def before(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
           limits: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=WP_FLOAT),
           eta: wp.array(dtype=wp.float64), area: wp.array(dtype=wp.float64),
           offsets: wp.array(dtype=int), adjacency: wp.array(dtype=int), q: wp.array(dtype=WP_FLOAT),
           scratch: wp.array(dtype=wp.float64), saved: wp.array2d(dtype=wp.float64)):
    if not hydraulic_step_enabled(control, status):
        return
    c = wp.tid()
    wp.atomic_min(scratch, 0, limits[c,0])
    wp.atomic_min(scratch, 1, limits[c,1])
    net = wp.float64(0.)
    for k in range(offsets[c], offsets[c+1]):
        entry = adjacency[k]
        sign = wp.float64(1.)
        if entry < 0:
            sign = wp.float64(-1.)
        net -= sign*wp.float64(q[wp.abs(entry)-1])
    saved[c,0] = wp.float64(volume[c])
    saved[c,1] = eta[c]
    saved[c,2] = area[c]
    saved[c,3] = net


@wp.kernel
def after(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
          limits: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=WP_FLOAT),
          eta: wp.array(dtype=wp.float64), alpha: wp.array(dtype=wp.float64),
          saved: wp.array2d(dtype=wp.float64), scratch: wp.array(dtype=wp.float64),
          stats: wp.array2d(dtype=wp.float64), relative_band: wp.float64):
    if not hydraulic_step_enabled(control, status):
        return
    c = wp.tid()
    bound = wp.min(scratch[0], scratch[1])
    local = wp.min(limits[c,0], limits[c,1])
    if local <= bound*(wp.float64(1.)+relative_band):
        stats[c,1] += wp.float64(1.)
    if local <= bound*(wp.float64(1.)+wp.float64(1.e-10)):
        stats[c,0] += wp.float64(1.)
        stats[c,8] = saved[c,0]
        stats[c,9] = saved[c,2]
        stats[c,10] = limits[c,2]
        stats[c,11] = saved[c,3]
        stats[c,12] = limits[c,3]
        stats[c,13] = limits[c,0]
        stats[c,14] = limits[c,1]
        wp.atomic_min(scratch, 2, wp.float64(c))
    stats[c,2] += wp.float64(alpha[c] < wp.float64(1.))
    stats[c,3] += control[DL_DT]*limits[c,2]
    stats[c,4] += wp.abs(wp.float64(volume[c])-saved[c,0])
    stats[c,5] += control[DL_DT]*limits[c,3]
    stats[c,6] = wp.max(stats[c,6], wp.abs(eta[c]-saved[c,1]))
    stats[c,7] = wp.max(stats[c,7], wp.abs(wp.float64(volume[c])-saved[c,0]))


@wp.kernel
def summarize(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
              scratch: wp.array(dtype=wp.float64), intervals: wp.array2d(dtype=wp.float64),
              histogram: wp.array2d(dtype=int), interval_s: wp.float64, maximum: wp.float64,
              growth: wp.float64, end_time: wp.float64):
    if not hydraulic_step_enabled(control, status):
        return
    row = wp.min(int(control[DL_TIME]/interval_s), intervals.shape[0]-1)
    dt = control[DL_DT]
    mechanism = int(0)
    bound = wp.min(scratch[0], scratch[1])
    if scratch[1] < scratch[0]:
        mechanism = 1
    if dt < bound*(wp.float64(1.)-wp.float64(1.e-10)):
        mechanism = 2
        if growth >= wp.float64(1.) and scratch[3] > wp.float64(0.) and dt < maximum:
            mechanism = 3
        if control[DL_TIME]+dt >= end_time-wp.float64(1.e-9):
            mechanism = 4
    intervals[row,mechanism] += wp.float64(1.)
    intervals[row,5] += dt
    intervals[row,6] = wp.min(intervals[row,6], dt)
    intervals[row,7] = wp.max(intervals[row,7], dt)
    intervals[row,8] += scratch[0]
    intervals[row,9] += scratch[1]
    bin = wp.clamp(int(wp.floor(wp.log2(dt)*wp.float64(16.)))+384,0,511)
    histogram[row,bin] += 1


@wp.kernel
def connections_before(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
                       q: wp.array(dtype=WP_FLOAT),saved: wp.array2d(dtype=wp.float64)):
    if hydraulic_step_enabled(control,status):
        f=wp.tid()
        saved[f,0]=wp.float64(q[f])


@wp.kernel
def connections_after(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
                      q: wp.array(dtype=WP_FLOAT),proposed: wp.array(dtype=WP_FLOAT),
                      d: wp.array(dtype=wp.float64),r: wp.array(dtype=wp.float64),pressure: wp.array(dtype=wp.float64),
                      saved: wp.array2d(dtype=wp.float64),stats: wp.array2d(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):
        return
    f=wp.tid()
    stats[f,0]=wp.max(stats[f,0],wp.abs(wp.float64(q[f])-saved[f,0]))
    if saved[f,4]>wp.float64(0.):
        stats[f,1]=wp.max(stats[f,1],wp.abs(pressure[f]-saved[f,3]))
        if d[f]>wp.float64(0.) and saved[f,1]>wp.float64(0.):
            stats[f,2]=wp.max(stats[f,2],wp.abs(d[f]-saved[f,1])/wp.max(d[f],saved[f,1]))
            stats[f,3]=wp.max(stats[f,3],wp.abs(r[f]-saved[f,2])/wp.max(wp.max(r[f],saved[f,2]),wp.float64(1.e-100)))
        elif d[f]!=saved[f,1]:
            stats[f,4]+=wp.float64(1.)
    if proposed[f]!=q[f]:
        stats[f,5]+=wp.float64(1.)
    stats[f,6]+=control[DL_DT]*wp.abs(wp.float64(proposed[f])-wp.float64(q[f]))
    stats[f,7]+=control[DL_DT]*wp.abs(wp.float64(proposed[f]))
    stats[f,8]+=control[DL_DT]*wp.abs(wp.float64(q[f]))
    saved[f,1]=d[f]
    saved[f,2]=r[f]
    saved[f,3]=pressure[f]
    saved[f,4]+=wp.float64(1.)


def enable(*, solver, duration_s, interval_s, relative_band):
    if duration_s <= 0 or interval_s <= 0 or relative_band < 0:
        raise ValueError('Positive duration/interval and nonnegative band required')
    solver.enable_timestep_diagnostics()
    count = int(np.ceil(duration_s/interval_s))
    solver._ec_audit_scratch = wp.zeros(4, dtype=wp.float64, device=solver.device)
    solver._ec_audit_saved = wp.zeros((solver._component_count,4), dtype=wp.float64, device=solver.device)
    solver._ec_audit_stats = wp.zeros((solver._component_count,15), dtype=wp.float64, device=solver.device)
    values = np.zeros((count,10))
    values[:,6] = np.inf
    solver._ec_audit_intervals = wp.array(values, dtype=wp.float64, device=solver.device)
    solver._ec_audit_histogram = wp.zeros((count,512), dtype=int, device=solver.device)
    solver._ec_audit_interval_s = interval_s
    solver._ec_audit_band = relative_band
    solver._ec_audit_connection_saved=wp.zeros((solver._connection_count,5),dtype=wp.float64,device=solver.device)
    solver._ec_audit_connections=wp.zeros((solver._connection_count,9),dtype=wp.float64,device=solver.device)


def enqueue_before(*, solver, control):
    wp.launch(kernel=reset, dim=1, inputs=[control,solver._ec_status,solver._ec_audit_scratch], device=solver.device)
    wp.launch(kernel=before, dim=solver._component_count, inputs=[control,solver._ec_status,
        solver._ec_timestep_limits,solver._ec_volume,solver._ec_eta,solver._ec_area,
        solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_q,
        solver._ec_audit_scratch,solver._ec_audit_saved], device=solver.device)
    wp.launch(kernel=connections_before,dim=solver._connection_count,inputs=[control,solver._ec_status,
        solver._ec_q,solver._ec_audit_connection_saved],device=solver.device)


def enqueue_after(*, solver, control, end_time):
    wp.launch(kernel=after, dim=solver._component_count, inputs=[control,solver._ec_status,
        solver._ec_timestep_limits,solver._ec_volume,solver._ec_eta,solver._ec_alpha,
        solver._ec_audit_saved,solver._ec_audit_scratch,solver._ec_audit_stats,solver._ec_audit_band], device=solver.device)
    wp.launch(kernel=summarize, dim=1, inputs=[control,solver._ec_status,solver._ec_audit_scratch,
        solver._ec_audit_intervals,solver._ec_audit_histogram,solver._ec_audit_interval_s,
        float(solver.maximum_dt),float(solver.adaptive_dt_growth_factor or 0.),float(end_time)], device=solver.device)
    wp.launch(kernel=connections_after,dim=solver._connection_count,inputs=[control,solver._ec_status,
        solver._ec_q,solver._ec_proposed,solver._ec_d,solver._ec_r,solver._ec_forcing,
        solver._ec_audit_connection_saved,solver._ec_audit_connections],device=solver.device)


def read(*, solver):
    stats = solver._ec_audit_stats.numpy()
    indices = np.argsort(stats[:,0])[::-1][:20]
    cells = solver._enhanced_geometry.state_cell
    connections=solver._ec_audit_connections.numpy()
    return dict(interval_columns=['wave','throughflow','user_cap','growth','endpoint','simulated_s','dt_min','dt_max','wave_sum','throughflow_sum'],
        intervals=solver._ec_audit_intervals.numpy().tolist(),
        histogram=solver._ec_audit_histogram.numpy().tolist(),
        component_columns=['controlling_steps','comparable_steps','limited_steps','gross_volume','absolute_storage_change','positive_supply',
            'max_stage_change','max_volume_change','last_control_volume','last_control_area','last_control_gross_q',
            'last_control_net_q','last_control_source','last_control_wave_dt','last_control_flow_dt'],
        component_count=solver._component_count,controlled_components=int(np.count_nonzero(stats[:,0])),
        comparable_components=int(np.count_nonzero(stats[:,1])),
        mean_comparable_components_per_step=float(stats[:,1].sum()/max(1.,solver._ec_audit_intervals.numpy()[:,:5].sum())),
        connection_columns=['maximum_Q_change','maximum_pressure_change','maximum_relative_D_change','maximum_relative_R_change',
            'activation_changes','limited_steps','rejected_transfer_volume','proposed_transfer_volume','accepted_transfer_volume'],
        connection_maxima=np.max(connections,axis=0).tolist(),connection_totals=np.sum(connections,axis=0).tolist(),
        top_components=[dict(component=int(c),x_m=float((cells[c]%solver.nx+.5)*float(solver.dx)),
            y_m=float((cells[c]//solver.nx+.5)*float(solver.dy)),values=stats[c].tolist()) for c in indices])
