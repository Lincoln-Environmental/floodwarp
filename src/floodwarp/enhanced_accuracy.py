"""Opt-in transactional temporal-error control for component hydraulics.

Accept the two half steps without Richardson extrapolation, preserving positive
independent storage and equal/opposite connection transfers. A maximum norm
tests every component and connection, including initially stationary momentum.
The embedded variant tests an exponential-midpoint discharge/storage defect
after one update. Pressure-wave CFL is checked at source and endpoint states.
"""
import numpy as np
import warp as wp

from floodwarp.config import WP_FLOAT
from floodwarp import enhanced_kernels as kernels
from floodwarp.enhanced_momentum import frozen_discharge
from floodwarp import enhanced_transaction
from floodwarp.enhanced_defect import frozen_certificates
from floodwarp.local_inertial_kernels import (
    DL_SIZE, DL_DT, DL_TIME, DL_ERROR, DL_DONE, DL_GROWTH_REFERENCE,
    device_loop_stage_fill_kernel,
    device_loop_accumulate_face_volume_qx_kernel,
    device_loop_accumulate_face_volume_qy_kernel,
)

# Internal device control slots.
TARGET = 0
DT = 1
ERROR = 2
ATTEMPTS = 3
REJECTED = 4
ACCEPTED = 5
UPDATES = 6
MAX_ERROR = 7
WAVE_ERROR = 8
STAGE_ERROR = 9
Q_ERROR = 10
COEFFICIENT_ERROR = 11
TRANSFER_ERROR = 12
WAVE_BOUND = 13
MAX_ACCEPTED_WAVE = 14
WAVE_REJECTIONS = 15
PROPOSED = 16
CLIPPED = 17


@wp.func
def maximum(state: wp.array(dtype=wp.float64), slot: int, value: wp.float64):
    if value>state[slot]:
        wp.atomic_max(state,slot,value)


@wp.func
def minimum(state: wp.array(dtype=wp.float64), slot: int, value: wp.float64):
    if value<state[slot]:
        wp.atomic_min(state,slot,value)


@wp.kernel
def initialize(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
               state: wp.array(dtype=wp.float64), condition: wp.array(dtype=int), maximum: wp.float64):
    condition[0] = 0
    if control[DL_DONE] == wp.float64(0.) and control[DL_ERROR] == wp.float64(0.) and status[0] == 0:
        condition[0] = 1
        if state[TARGET] <= wp.float64(0.):
            state[TARGET] = maximum
        state[ATTEMPTS] = wp.float64(0.)


@wp.kernel
def prepare(control: wp.array(dtype=wp.float64), trial: wp.array(dtype=wp.float64),
            state: wp.array(dtype=wp.float64), part: int, growth: wp.float64):
    for i in range(DL_SIZE):
        trial[i] = control[i]
    trial[DL_ERROR] = wp.float64(0.)
    trial[DL_DONE] = wp.float64(0.)
    trial[DL_GROWTH_REFERENCE] = wp.float64(0.)
    trial[DL_DT] = state[TARGET]
    if part == 0:
        if growth>=wp.float64(1.) and control[DL_GROWTH_REFERENCE]>wp.float64(0.):
            trial[DL_DT]=wp.min(trial[DL_DT],control[DL_GROWTH_REFERENCE]*growth)
        state[PROPOSED]=trial[DL_DT]
        state[ERROR] = wp.float64(0.)
        state[WAVE_ERROR] = wp.float64(0.)
        state[STAGE_ERROR] = wp.float64(0.)
        state[Q_ERROR] = wp.float64(0.)
        state[COEFFICIENT_ERROR] = wp.float64(0.)
        state[TRANSFER_ERROR] = wp.float64(0.)
        state[WAVE_BOUND] = wp.float64(1.e100)
    else:
        trial[DL_DT] = state[DT]*wp.float64(.5)
        if part == 2:
            trial[DL_TIME] += state[DT]*wp.float64(.5)


@wp.kernel
def remember(trial: wp.array(dtype=wp.float64), state: wp.array(dtype=wp.float64)):
    state[DT] = trial[DL_DT]
    state[CLIPPED]=wp.float64(0.)
    if state[DT]<state[PROPOSED]*(wp.float64(1.)-wp.float64(1.e-12)):
        state[CLIPPED]=wp.float64(1.)


@wp.kernel
def trial_failure(status: wp.array(dtype=int),probe_status: wp.array(dtype=int),state: wp.array(dtype=wp.float64)):
    if status[0]!=0 or probe_status[0]!=0:
        maximum(state,ERROR,wp.float64(1.e10))


@wp.kernel
def wave(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
         offsets: wp.array(dtype=int), adjacency: wp.array(dtype=int),
         d: wp.array(dtype=wp.float64), area: wp.array(dtype=wp.float64),
         state: wp.array(dtype=wp.float64), gravity: wp.float64, cfl: wp.float64):
    if not kernels.hydraulic_step_enabled(control,status):
        return
    c = wp.tid()
    degree = wp.float64(0.)
    for k in range(offsets[c],offsets[c+1]):
        inertia = d[wp.abs(adjacency[k])-1]
        if inertia > wp.float64(0.):
            degree += wp.float64(1.)/inertia
    if degree > wp.float64(0.):
        bound = cfl*wp.sqrt(wp.float64(2.)*area[c]/(gravity*degree))
        minimum(state,WAVE_BOUND,bound)
        ratio = control[DL_DT]/bound
        maximum(state,WAVE_ERROR,ratio)
        # Fold violations into the rejection test. CFL itself is unchanged.
        if ratio > wp.float64(1.):
            maximum(state,ERROR,ratio*ratio)


@wp.kernel
def component_error(volume: wp.array(dtype=WP_FLOAT), eta: wp.array(dtype=wp.float64),
                    area: wp.array(dtype=wp.float64), full_volume: wp.array(dtype=WP_FLOAT),
                    full_eta: wp.array(dtype=wp.float64), full_area: wp.array(dtype=wp.float64),
                    initial_eta: wp.array(dtype=wp.float64), state: wp.array(dtype=wp.float64),
                    stage_atol: wp.float64, rtol: wp.float64):
    c = wp.tid()
    # Relative scaling uses stage excursion, never absolute elevation/datum.
    scale = stage_atol+rtol*wp.max(wp.abs(eta[c]-initial_eta[c]),wp.abs(full_eta[c]-initial_eta[c]))
    error = wp.abs(eta[c]-full_eta[c])/scale
    storage_scale = scale*wp.max(wp.max(area[c],full_area[c]),wp.float64(1.e-30))
    error = wp.max(error,wp.abs(wp.float64(volume[c])-wp.float64(full_volume[c]))/storage_scale)
    maximum(state,STAGE_ERROR,error)
    maximum(state,ERROR,error)


@wp.kernel
def connection_error(q: wp.array(dtype=WP_FLOAT), full_q: wp.array(dtype=WP_FLOAT),
                     d: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64),
                     full_d: wp.array(dtype=wp.float64), full_r: wp.array(dtype=wp.float64),
                     state: wp.array(dtype=wp.float64), q_atol: wp.float64,
                     rtol: wp.float64, coefficient_rtol: wp.float64):
    f = wp.tid()
    magnitude = wp.max(wp.abs(wp.float64(q[f])),wp.abs(wp.float64(full_q[f])))
    error = wp.abs(wp.float64(q[f])-wp.float64(full_q[f]))/(q_atol+rtol*magnitude)
    maximum(state,Q_ERROR,error)
    maximum(state,ERROR,error)
    # Near activation, discharge's absolute tolerance prevents relative
    # coefficient singularities from forcing zero-length steps.
    weight = magnitude/(q_atol+magnitude)
    de = wp.abs(d[f]-full_d[f])/wp.max(wp.max(wp.abs(d[f]),wp.abs(full_d[f])),wp.float64(1.e-100))
    re = wp.abs(r[f]-full_r[f])/wp.max(wp.max(wp.abs(r[f]),wp.abs(full_r[f])),wp.float64(1.e-100))
    ce = weight*wp.max(de,re)/coefficient_rtol
    maximum(state,COEFFICIENT_ERROR,ce)
    # A table/activation coefficient jump is not a truncation error: its
    # hydraulic effect is tested through Q and independent storage instead.


@wp.kernel
def transfer_error(control: wp.array(dtype=wp.float64), proposed: wp.array(dtype=WP_FLOAT),
                   accepted: wp.array(dtype=WP_FLOAT), transfer: wp.array(dtype=wp.float64),
                   full_transfer: wp.array(dtype=wp.float64), state: wp.array(dtype=wp.float64),
                   q_atol: wp.float64, rtol: wp.float64, part: int):
    f = wp.tid()
    rejected = control[DL_DT]*wp.abs(wp.float64(proposed[f])-wp.float64(accepted[f]))
    if part == 0:
        full_transfer[f] = rejected
        transfer[f] = wp.float64(0.)
    else:
        transfer[f] += rejected
        if part == 2:
            scale = state[DT]*(q_atol+rtol*wp.abs(wp.float64(accepted[f])))
            error = wp.abs(transfer[f]-full_transfer[f])/scale
            maximum(state,TRANSFER_ERROR,error)
            maximum(state,ERROR,error)


@wp.kernel
def decide(control: wp.array(dtype=wp.float64), trial: wp.array(dtype=wp.float64),
           status: wp.array(dtype=int), state: wp.array(dtype=wp.float64),
           condition: wp.array(dtype=int), minimum: wp.float64, maximum: wp.float64,
           safety: wp.float64, max_attempts: int,wave_proposal_fraction: wp.float64):
    if trial[DL_DONE] != wp.float64(0.):
        for i in range(DL_SIZE):
            control[i] = trial[i]
        condition[0] = 0
        return
    state[ATTEMPTS] += wp.float64(1.)
    state[UPDATES] += wp.float64(3.)
    error = state[ERROR]
    if status[0] != 0 or not wp.isfinite(error):
        error = wp.float64(1.e10)
    factor = wp.clamp(safety/wp.sqrt(wp.max(error,wp.float64(1.e-16))),wp.float64(.2),wp.float64(2.))
    state[TARGET] = wp.min(maximum,state[DT]*factor)
    state[TARGET] = wp.min(state[TARGET],state[WAVE_BOUND]*wave_proposal_fraction)
    if error > wp.float64(1.):
        state[REJECTED] += wp.float64(1.)
        if state[WAVE_ERROR]>wp.float64(1.):state[WAVE_REJECTIONS]+=wp.float64(1.)
        if state[ATTEMPTS] >= wp.float64(max_attempts) or state[TARGET] < minimum:
            condition[0] = 0
            control[DL_ERROR] = wp.float64(4.)
    else:
        if state[DT]<minimum and state[CLIPPED]==wp.float64(0.):
            condition[0]=0
            control[DL_ERROR]=wp.float64(2.)
            return
        state[ACCEPTED] += wp.float64(1.)
        state[MAX_ERROR] = wp.max(state[MAX_ERROR],error)
        state[MAX_ACCEPTED_WAVE] = wp.max(state[MAX_ACCEPTED_WAVE],state[WAVE_ERROR])
        previous_reference=control[DL_GROWTH_REFERENCE]
        for i in range(DL_SIZE):
            control[i] = trial[i]
        control[DL_DT] = state[DT]
        control[DL_GROWTH_REFERENCE] = state[DT]
        if state[CLIPPED]!=wp.float64(0.):
            # Scheduled closure must not seed a tiny timestep in the next
            # output interval or lower the user's ordinary growth reference.
            control[DL_GROWTH_REFERENCE]=previous_reference
            state[TARGET]=wp.min(maximum,wp.min(state[PROPOSED],state[WAVE_BOUND]*wave_proposal_fraction))
        condition[0] = 0


@wp.kernel
def flag_failure(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    if control[DL_ERROR] != wp.float64(0.):
        status[0] = 6


@wp.kernel
def branch_flags(control: wp.array(dtype=wp.float64),state: wp.array(dtype=wp.float64),
                 condition: wp.array(dtype=int),failure: int):
    condition[0]=0
    if failure==1:
        if control[DL_ERROR]!=wp.float64(0.):condition[0]=1
    elif state[ATTEMPTS]>wp.float64(0.):condition[0]=1


@wp.kernel
def effective_initial_q(q: wp.array(dtype=WP_FLOAT),eta: wp.array(dtype=wp.float64),heads: wp.array(dtype=wp.float64),
        endpoints: wp.array2d(dtype=int),sills: wp.array(dtype=wp.float64),out: wp.array(dtype=WP_FLOAT),dry: wp.float64):
    f=wp.tid();donor=eta[endpoints[f,0]]
    if q[f]<WP_FLOAT(0.):
        donor=heads[f]
        if endpoints[f,1]>=0:donor=eta[endpoints[f,1]]
    out[f]=q[f]
    if donor<=sills[f]+dry:out[f]=WP_FLOAT(0.)


@wp.kernel
def embedded_connections(control: wp.array(dtype=wp.float64),q: wp.array(dtype=WP_FLOAT),
        initial_q: wp.array(dtype=WP_FLOAT),proposed: wp.array(dtype=WP_FLOAT),kind: wp.array(dtype=int),
        endpoints: wp.array2d(dtype=int),offsets: wp.array(dtype=int),source_area: wp.array(dtype=wp.float64),
        volume: wp.array(dtype=WP_FLOAT),area: wp.array(dtype=wp.float64),
        d: wp.array(dtype=wp.float64),r: wp.array(dtype=wp.float64),
        pressure: wp.array(dtype=wp.float64),start_d: wp.array(dtype=wp.float64),
        start_r: wp.array(dtype=wp.float64),start_pressure: wp.array(dtype=wp.float64),
        mid_q: wp.array(dtype=WP_FLOAT),mid_radius: wp.array(dtype=wp.float64),counts: wp.array2d(dtype=int),
        state: wp.array(dtype=wp.float64),gravity: wp.float64,q_atol: wp.float64,rtol: wp.float64,
        coefficient_rtol: wp.float64,stage_atol: wp.float64,bounded: int):
    f=wp.tid()
    corrected=wp.float64(q[f]);mid=wp.float64(q[f])
    magnitude=wp.max(wp.abs(wp.float64(q[f])),wp.abs(wp.float64(initial_q[f])))
    fast=int(0);radius=wp.float64(0.);error=wp.float64(0.)
    if d[f]>wp.float64(0.) and start_d[f]>wp.float64(0.):
        a=gravity*wp.float64(.5)*(pressure[f]/d[f]+start_pressure[f]/start_d[f])
        b=gravity*wp.float64(.5)*(r[f]/d[f]+start_r[f]/start_d[f])
        if bounded==1 and proposed[f]==q[f] and kind[f]!=2 and (kind[f]<6 or kind[f]>9):
            certificate=frozen_certificates(wp.float64(initial_q[f]),wp.float64(q[f]),
                gravity*start_pressure[f]/start_d[f],gravity*start_r[f]/start_d[f],
                gravity*pressure[f]/d[f],gravity*r[f]/d[f],control[DL_DT])
            roundoff=wp.float64(16.)*wp.float64(kernels.STATE_EPSILON)*(magnitude+control[DL_DT]*(wp.abs(gravity*start_pressure[f]/start_d[f])+wp.abs(gravity*pressure[f]/d[f])))
            certificate[0]+=roundoff
            certificate[2]+=roundoff
            ca=endpoints[f,0];cb=endpoints[f,1]
            budget=wp.min(stage_atol*wp.min(source_area[ca],area[ca]),wp.float64(volume[ca]))/wp.float64(offsets[ca+1]-offsets[ca])
            if cb>=0:budget=wp.min(budget,wp.min(stage_atol*wp.min(source_area[cb],area[cb]),wp.float64(volume[cb]))/wp.float64(offsets[cb+1]-offsets[cb]))
            # Certify a small error envelope relative to each endpoint's
            # allocated absolute storage budget; otherwise use exact probes.
            if certificate[0]<=wp.float64(.25)*(q_atol+rtol*magnitude) and control[DL_DT]*certificate[2]<=wp.float64(.25)*budget:
                fast=1
                mid=certificate[1];radius=certificate[2]
                error=certificate[0]/(q_atol+rtol*magnitude)
        if fast==0:
            corrected=frozen_discharge(wp.float64(initial_q[f]),a,b,control[DL_DT])
            mid=frozen_discharge(wp.float64(initial_q[f]),a,b,control[DL_DT]*wp.float64(.5))
        weight=magnitude/(q_atol+magnitude)
        de=wp.abs(d[f]-start_d[f])/wp.max(d[f],start_d[f])
        re=wp.abs(r[f]-start_r[f])/wp.max(wp.max(r[f],start_r[f]),wp.float64(1.e-100))
        ce=weight*wp.max(de,re)/coefficient_rtol
        maximum(state,COEFFICIENT_ERROR,ce)
    elif d[f]!=start_d[f]:
        maximum(state,COEFFICIENT_ERROR,wp.float64(1.)/coefficient_rtol)
        # Integrate the changed activation's pressure/drag for the trial
        # interval. This defect vanishes with dt even at a coefficient jump.
        if d[f]>wp.float64(0.):
            corrected=frozen_discharge(wp.float64(initial_q[f]),gravity*pressure[f]/d[f],gravity*r[f]/d[f],control[DL_DT])
            mid=frozen_discharge(wp.float64(initial_q[f]),gravity*pressure[f]/d[f],gravity*r[f]/d[f],control[DL_DT]*wp.float64(.5))
        else:
            corrected=wp.float64(0.);mid=wp.float64(0.)
    if fast==0:error=wp.abs(corrected-wp.float64(q[f]))/(q_atol+rtol*wp.max(wp.abs(corrected),magnitude))
    maximum(state,Q_ERROR,error)
    maximum(state,ERROR,error)
    mid_q[f]=WP_FLOAT(mid)
    mid_radius[f]=radius
    counts[f,fast]+=1


@wp.kernel
def embedded_components(control: wp.array(dtype=wp.float64),volume: wp.array(dtype=WP_FLOAT),
        eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),initial_eta: wp.array(dtype=wp.float64),
        source_area: wp.array(dtype=wp.float64),
        storage_offsets: wp.array(dtype=int),storage_rows: wp.array2d(dtype=wp.float64),
        offsets: wp.array(dtype=int),adjacency: wp.array(dtype=int),q: wp.array(dtype=WP_FLOAT),
        mid_q: wp.array(dtype=WP_FLOAT),mid_radius: wp.array(dtype=wp.float64),proposed: wp.array(dtype=WP_FLOAT),
        state: wp.array(dtype=wp.float64),stage_atol: wp.float64,rtol: wp.float64):
    c=wp.tid();defect=wp.float64(0.);rejected=wp.float64(0.);uncertainty=wp.float64(0.)
    for k in range(offsets[c],offsets[c+1]):
        entry=adjacency[k];sign=wp.float64(1.)
        if entry<0:sign=wp.float64(-1.)
        f=wp.abs(entry)-1
        defect-=sign*(wp.float64(mid_q[f])-wp.float64(q[f]))*control[DL_DT]
        uncertainty+=mid_radius[f]*control[DL_DT]
        rejected+=wp.abs(wp.float64(proposed[f])-wp.float64(q[f]))*control[DL_DT]
    scale=stage_atol*source_area[c]+rtol*wp.abs(eta[c]-initial_eta[c])*area[c]
    error=(wp.abs(defect)+uncertainty)/scale
    stage_scale=stage_atol+rtol*wp.abs(eta[c]-initial_eta[c])
    for side in range(2):
        signed=wp.float64(-1.)
        if side==1:signed=wp.float64(1.)
        corrected_volume=wp.max(wp.float64(0.),wp.float64(volume[c])+defect+signed*uncertainty)
        k=kernels.storage_index(corrected_volume,storage_offsets[c],storage_offsets[c+1],storage_rows)
        corrected_eta=storage_rows[k,0]+(corrected_volume-storage_rows[k,1])/storage_rows[k,2]
        error=wp.max(error,wp.abs(corrected_eta-eta[c])/stage_scale)
    # A materially donor-limited trial requires a shorter step, rather than
    # estimating truncation error across the nonsmooth clipping operation.
    te=rejected/scale
    maximum(state,TRANSFER_ERROR,te)
    maximum(state,STAGE_ERROR,error)
    maximum(state,ERROR,wp.max(error,te))
    if wp.float64(volume[c])+defect-uncertainty<wp.float64(0.):
        maximum(state,ERROR,wp.float64(2.))


def configure(*, solver, stage_atol_m, discharge_atol_m3_s, relative_tolerance,
              coefficient_relative_tolerance, safety, max_attempts, estimator, defect_evaluation,wave_proposal_fraction):
    values = (stage_atol_m,discharge_atol_m3_s,relative_tolerance,coefficient_relative_tolerance,safety)
    if not all(np.isfinite(values)) or min(values)<=0 or safety>=1 or max_attempts<1:
        raise ValueError('Positive finite tolerances, safety < 1 and positive retry count required')
    if not np.isfinite(wave_proposal_fraction) or not 0<wave_proposal_fraction<=1:
        raise ValueError('Wave proposal fraction must be positive and no greater than one')
    if solver.structures or solver._kinetic_transport:
        raise ValueError('Adaptive component stepping currently requires discharge transport without structures or external hooks')
    if estimator not in ('doubling','embedded'):
        raise ValueError('estimator must be doubling or embedded')
    if defect_evaluation not in ('analytic','bounded'):
        raise ValueError('defect_evaluation must be analytic or bounded')
    if estimator=='embedded' and (not solver._exact_momentum or solver.apply_sources_after_momentum or solver.infiltration_method not in (0,2)):
        raise ValueError('Embedded estimator requires analytic momentum, leading sources and constant zero infiltration')
    if estimator=='embedded' and np.any(solver.infiltration_capacity.numpy()!=0):
        raise ValueError('Embedded estimator currently requires zero infiltration')
    solver._accuracy_options = dict(stage_atol_m=stage_atol_m,discharge_atol_m3_s=discharge_atol_m3_s,
        relative_tolerance=relative_tolerance,coefficient_relative_tolerance=coefficient_relative_tolerance,
        safety=safety,max_attempts=max_attempts,estimator=estimator,defect_evaluation=defect_evaluation,
        wave_proposal_fraction=wave_proposal_fraction)
    if not hasattr(solver,'_accuracy_execution'):solver._accuracy_execution='optimized'
    solver._component_timestep = 'adaptive'
    solver._ensure_provisional_buffers()
    allocate(solver=solver)


def allocate(*, solver):
    # Exactly the fields written by sources, coefficients, momentum,
    # continuity, output reconstruction and half-step face diagnostics.
    names = ('_ec_volume','_ec_eta','_ec_area','_ec_soil','_ec_cap','_ec_applied',
        '_ec_source_ledger','_ec_source_sums','_ec_q','_ec_previous_d','_ec_proposed',
        '_ec_maximum_eta','_ec_alpha','_ec_alpha_free','_ec_d','_ec_r','_ec_forcing',
        '_ec_heads','_ec_kind','_ec_scalar_interval','_ec_status','cumulative_totals',
        'boundary_stage_qx','boundary_stage_qy',
        'free_outflow_diagnostics',
        'water_depth','qx','qy','qx_next','qy_next','cumulative_infiltration',
        'infiltration_cap_remaining_m','applied_rain','applied_other','applied_infil',
        'max_water_depth_m','_diagnostic_max_depth_m','cumulative_net_qx_volume_m3',
        'cumulative_net_qy_volume_m3','cumulative_abs_qx_volume_m3','cumulative_abs_qy_volume_m3',
        '_diagnostic_peak_qx_rows','_diagnostic_peak_qy_rows')
    current={name:getattr(solver,name) for name in names if isinstance(getattr(solver,name,None),wp.array)}
    if hasattr(solver,'_accuracy_saved'):
        bound=getattr(solver,'_accuracy_bound_fields',{})
        if current.keys()==bound.keys() and all(value is bound[name] for name,value in current.items()):return
        del solver._accuracy_saved
    solver._accuracy_saved = {name:wp.empty_like(getattr(solver,name)) for name in names
        if isinstance(getattr(solver,name,None),wp.array)}
    for name in ('d','r','forcing'):
        if getattr(solver,'_accuracy_start_'+name,None) is None or getattr(solver,'_accuracy_start_'+name).shape!=getattr(solver,'_ec_'+name).shape:
            setattr(solver,'_accuracy_start_'+name,wp.empty_like(getattr(solver,'_ec_'+name)))
    if not hasattr(solver,'_accuracy_mid_q') or solver._accuracy_mid_q.shape!=solver._ec_q.shape:
        solver._accuracy_mid_q=wp.zeros_like(solver._ec_q)
        solver._accuracy_effective_q=wp.zeros_like(solver._ec_q)
        solver._accuracy_mid_radius=wp.zeros(solver._connection_count,dtype=wp.float64,device=solver.device)
        solver._accuracy_defect_counts=wp.zeros((solver._connection_count,2),dtype=int,device=solver.device)
    for name in ('_ec_volume','_ec_eta','_ec_area','_ec_q','_ec_d','_ec_r'):
        if not hasattr(solver,'_accuracy_full'+name) or getattr(solver,'_accuracy_full'+name).shape!=getattr(solver,name).shape:
            setattr(solver,'_accuracy_full'+name,wp.empty_like(getattr(solver,name)))
    if not hasattr(solver,'_accuracy_state'):
        solver._accuracy_state = wp.zeros(18,dtype=wp.float64,device=solver.device)
        solver._accuracy_condition = wp.zeros(1,dtype=int,device=solver.device)
        solver._accuracy_control = wp.zeros(DL_SIZE,dtype=wp.float64,device=solver.device)
        solver._accuracy_selected_control = wp.zeros_like(solver._accuracy_control)
        solver._accuracy_transfer = wp.zeros(solver._connection_count,dtype=wp.float64,device=solver.device)
        solver._accuracy_full_transfer = wp.zeros_like(solver._accuracy_transfer)
        solver._accuracy_failure_condition=wp.zeros(1,dtype=int,device=solver.device)
        solver._accuracy_retry_condition=wp.zeros(1,dtype=int,device=solver.device)
    if solver._accuracy_transfer.shape!=solver._ec_q.shape:
        solver._accuracy_transfer=wp.zeros(solver._connection_count,dtype=wp.float64,device=solver.device)
        solver._accuracy_full_transfer=wp.zeros_like(solver._accuracy_transfer)
    solver._accuracy_transactions=enhanced_transaction.prepare(solver=solver,saved=solver._accuracy_saved)
    solver._accuracy_bound_fields={name:getattr(solver,name) for name in solver._accuracy_saved}
    solver._accuracy_layout_epoch=getattr(solver,'_accuracy_layout_epoch',0)+1
    wp.load_module(module=initialize.module,device=solver.device)


def rebind(*,solver):
    """Recreate transaction views after a fresh-solver checkpoint restoration."""
    if hasattr(solver,'_accuracy_saved'):del solver._accuracy_saved
    if hasattr(solver,'_accuracy_transactions'):del solver._accuracy_transactions
    allocate(solver=solver)
    if hasattr(solver,'_embedded_private'):
        from floodwarp import enhanced_embedded_execution
        enhanced_embedded_execution.rebind(solver=solver)


def validate(*, solver):
    """Recheck mutable public options at each diagnostic/run boundary."""
    options=solver._accuracy_options
    if solver.structures or solver._kinetic_transport:
        raise ValueError('Adaptive stepping cannot transact external hooks or structures')
    if options['estimator']=='embedded':
        if not solver._exact_momentum or solver.apply_sources_after_momentum or solver.infiltration_method not in (0,2):
            raise ValueError('Embedded estimator requires analytic momentum and leading zero-infiltration sources')
        if solver.infiltration_method==0 and np.any(solver.infiltration_capacity.numpy()!=0):
            raise ValueError('Embedded estimator requires zero infiltration; use doubling for soil processes')


def copy_state(*, solver, restore):
    enhanced_transaction.copy(groups=solver._accuracy_transactions,restore=restore,device=solver.device)


def substep(*, solver, control, options, part):
    trial = solver._accuracy_control
    state = solver._accuracy_state
    wp.launch(kernel=prepare,dim=1,inputs=[control,trial,state,part,float(solver.adaptive_dt_growth_factor or 0.)],device=solver.device)
    # Reuse the production forcing and endpoint sampler for every trial.
    wp.launch(kernel=kernels.device_finalize,dim=1,inputs=[trial,float(options['end_time']),0.,
        float(solver.maximum_dt),0.,0,solver.cfl_number,solver.dx,solver.gravity,solver.dry_depth,
        options['hyeto_times'],options['hyeto_rates'],options['use_hyetograph'],float(options['storm_duration']),
        int(not solver.apply_sources_after_momentum),options['stage_times_qx'],options['stage_values_qx'],options['use_stage_qx'],
        options['stage_times_qy'],options['stage_values_qy'],options['use_stage_qy'],options['checkpoint_times'],
        options['checkpoint_count'] if part==0 else 0],device=solver.device)
    if part==0:
        wp.launch(kernel=remember,dim=1,inputs=[trial,state],device=solver.device)
        wp.copy(solver._accuracy_selected_control,trial)
    # Reuse actual-source wave probing before committing sources to the trial.
    solver._ec_trial_status.zero_()
    solver._component_sources(control=trial,use_hyetograph=options['use_hyetograph'],
        provisional=2 if solver.apply_sources_after_momentum else 1)
    solver._component_coefficients(control=trial,eta=solver._ec_trial_eta,status=solver._ec_status,
        use_stage_qx=options['use_stage_qx'],use_stage_qy=options['use_stage_qy'])
    if part==0:
        for name in ('d','r','forcing'):
            wp.copy(getattr(solver,'_accuracy_start_'+name),getattr(solver,'_ec_'+name))
        wp.launch(kernel=effective_initial_q,dim=solver._connection_count,inputs=[solver._ec_q,
            solver._ec_trial_eta,solver._ec_heads,solver._ec_endpoints,solver._ec_sill,
            solver._accuracy_effective_q,float(solver.dry_depth)],device=solver.device)
    wp.launch(kernel=wave,dim=solver._component_count,inputs=[trial,solver._ec_status,
        solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_d,solver._ec_trial_area,
        state,float(solver.gravity),float(solver.cfl_number)],device=solver.device)
    if options['use_stage_qx'] or options['use_stage_qy']:
        wp.launch(kernel=device_loop_stage_fill_kernel,dim=(solver.ny+1,solver.nx+1),inputs=[
            solver.boundary_type_qx,solver.boundary_stage_qx,solver.boundary_type_qy,solver.boundary_stage_qy,
            trial,options['use_stage_qx'],options['use_stage_qy']],device=solver.device)
    solver._enqueue_component_hydraulics(control=trial,use_hyetograph=options['use_hyetograph'],
        defer_outputs=True,require_faces=True,coefficients_ready=True,
        collect_free_outflow_diagnostics=getattr(solver,'_accuracy_collect_free_outflow_diagnostics',False))
    wp.launch(kernel=transfer_error,dim=solver._connection_count,inputs=[trial,solver._ec_proposed,solver._ec_q,
        solver._accuracy_transfer,solver._accuracy_full_transfer,state,solver._accuracy_options['discharge_atol_m3_s'],
        solver._accuracy_options['relative_tolerance'],part],device=solver.device)
    if solver.track_cumulative_face_volumes:
        for axis,dim,width in (('x',(solver.ny,solver.nx+1),solver.dy),('y',(solver.ny+1,solver.nx),solver.dx)):
            kernel = device_loop_accumulate_face_volume_qx_kernel if axis=='x' else device_loop_accumulate_face_volume_qy_kernel
            wp.launch(kernel=kernel,dim=dim,inputs=[getattr(solver,'q'+axis),getattr(solver,'cumulative_net_q'+axis+'_volume_m3'),
                getattr(solver,'cumulative_abs_q'+axis+'_volume_m3') if solver.track_absolute_face_volumes else solver._diagnostic_absolute_placeholder,
                getattr(solver,'_diagnostic_peak_q'+axis+'_rows'),width,trial,int(solver.track_absolute_face_volumes)],device=solver.device)
    solver._component_coefficients(control=trial,use_stage_qx=options['use_stage_qx'],use_stage_qy=options['use_stage_qy'])
    wp.launch(kernel=wave,dim=solver._component_count,inputs=[trial,solver._ec_status,
        solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_d,solver._ec_area,
        state,float(solver.gravity),float(solver.cfl_number)],device=solver.device)
    wp.launch(kernel=trial_failure,dim=1,inputs=[solver._ec_status,solver._ec_trial_status,state],device=solver.device)


def attempt(*, solver, control):
    options = solver._ec_trial_options
    if solver._accuracy_options['estimator']=='embedded':
        wp.launch(kernel=branch_flags,dim=1,inputs=[control,solver._accuracy_state,solver._accuracy_retry_condition,0],device=solver.device)
        wp.capture_if(solver._accuracy_retry_condition,on_true=copy_state,solver=solver,restore=True)
    else:
        copy_state(solver=solver,restore=True)
    substep(solver=solver,control=control,options=options,part=0)
    tuning=solver._accuracy_options
    if tuning['estimator']=='embedded':
        wp.launch(kernel=embedded_connections,dim=solver._connection_count,inputs=[solver._accuracy_control,
            solver._ec_q,solver._accuracy_effective_q,solver._ec_proposed,solver._ec_kind,solver._ec_endpoints,
            solver._ec_adjacency_offsets,solver._ec_source_area,solver._ec_volume,solver._ec_area,solver._ec_d,solver._ec_r,solver._ec_forcing,
            solver._accuracy_start_d,solver._accuracy_start_r,solver._accuracy_start_forcing,solver._accuracy_mid_q,
            solver._accuracy_mid_radius,solver._accuracy_defect_counts,solver._accuracy_state,
            float(solver.gravity),tuning['discharge_atol_m3_s'],tuning['relative_tolerance'],
            tuning['coefficient_relative_tolerance'],tuning['stage_atol_m'],int(tuning['defect_evaluation']=='bounded')],device=solver.device)
        wp.launch(kernel=embedded_components,dim=solver._component_count,inputs=[solver._accuracy_control,
            solver._ec_volume,solver._ec_eta,solver._ec_area,solver._accuracy_saved['_ec_eta'],solver._ec_source_area,
            solver._ec_storage_offsets,solver._ec_storage_rows,
            solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_q,solver._accuracy_mid_q,solver._accuracy_mid_radius,solver._ec_proposed,
            solver._accuracy_state,tuning['stage_atol_m'],tuning['relative_tolerance']],device=solver.device)
        wp.launch(kernel=decide,dim=1,inputs=[control,solver._accuracy_selected_control,solver._ec_status,solver._accuracy_state,
            solver._accuracy_condition,float(solver.minimum_dt),float(solver.maximum_dt),tuning['safety'],tuning['max_attempts'],tuning['wave_proposal_fraction']],device=solver.device)
        return
    for name in ('_ec_volume','_ec_eta','_ec_area','_ec_q','_ec_d','_ec_r'):
        wp.copy(getattr(solver,'_accuracy_full'+name),getattr(solver,name))
    copy_state(solver=solver,restore=True)
    substep(solver=solver,control=control,options=options,part=1)
    substep(solver=solver,control=control,options=options,part=2)
    tuning = solver._accuracy_options
    wp.launch(kernel=component_error,dim=solver._component_count,inputs=[solver._ec_volume,solver._ec_eta,solver._ec_area,
        solver._accuracy_full_ec_volume,solver._accuracy_full_ec_eta,solver._accuracy_full_ec_area,
        solver._accuracy_saved['_ec_eta'],solver._accuracy_state,tuning['stage_atol_m'],tuning['relative_tolerance']],device=solver.device)
    wp.launch(kernel=connection_error,dim=solver._connection_count,inputs=[solver._ec_q,solver._accuracy_full_ec_q,
        solver._ec_d,solver._ec_r,solver._accuracy_full_ec_d,solver._accuracy_full_ec_r,solver._accuracy_state,
        tuning['discharge_atol_m3_s'],tuning['relative_tolerance'],tuning['coefficient_relative_tolerance']],device=solver.device)
    wp.launch(kernel=decide,dim=1,inputs=[control,solver._accuracy_selected_control,solver._ec_status,solver._accuracy_state,
        solver._accuracy_condition,float(solver.minimum_dt),float(solver.maximum_dt),tuning['safety'],tuning['max_attempts'],tuning['wave_proposal_fraction']],device=solver.device)


def prepare_graph(*, solver, control, options):
    solver._ensure_provisional_buffers()
    allocate(solver=solver)
    solver._ec_trial_options = options
    if solver._accuracy_options['estimator']=='embedded' and getattr(solver,'_accuracy_execution','reference')=='optimized':
        from floodwarp import enhanced_embedded_execution
        enhanced_embedded_execution.allocate(solver=solver)
    if solver.device.is_cuda:
        with wp.ScopedDevice(solver.device):
            if not wp.is_conditional_graph_supported():
                raise ValueError('Adaptive timestep requires CUDA conditional graphs')


def enqueue(*, solver, control):
    if solver._accuracy_options['estimator']=='embedded' and getattr(solver,'_accuracy_execution','reference')=='optimized':
        from floodwarp import enhanced_embedded_execution
        enhanced_embedded_execution.enqueue(solver=solver,control=control)
        return
    copy_state(solver=solver,restore=False)
    wp.launch(kernel=initialize,dim=1,inputs=[control,solver._ec_status,solver._accuracy_state,
        solver._accuracy_condition,float(solver.maximum_dt)],device=solver.device)
    if solver.device.is_cuda:
        if solver.device.is_capturing:
            wp.capture_while(solver._accuracy_condition,while_body=attempt,solver=solver,control=control)
            enqueue_failure_rollback(solver=solver,control=control)
        else:
            with wp.ScopedCapture(device=solver.device,force_module_load=True) as loop:
                wp.capture_while(solver._accuracy_condition,while_body=attempt,solver=solver,control=control)
                enqueue_failure_rollback(solver=solver,control=control)
            wp.capture_launch(loop.graph)
    else:
        while solver._accuracy_condition.numpy()[0]:
            attempt(solver=solver,control=control)
        enqueue_failure_rollback(solver=solver,control=control)


def failure_rollback(*,solver,control):
    copy_state(solver=solver,restore=True)
    wp.launch(kernel=flag_failure,dim=1,inputs=[control,solver._ec_status],device=solver.device)


def enqueue_failure_rollback(*,solver,control):
    wp.launch(kernel=branch_flags,dim=1,inputs=[control,solver._accuracy_state,solver._accuracy_failure_condition,1],device=solver.device)
    wp.capture_if(solver._accuracy_failure_condition,on_true=failure_rollback,solver=solver,control=control)


def statistics(*, solver):
    v = solver._accuracy_state.numpy()
    embedded=solver._accuracy_options['estimator']=='embedded'
    counts=solver._accuracy_defect_counts.numpy().astype(np.int64).sum(axis=0)
    return dict(estimator=solver._accuracy_options['estimator'],defect_evaluation=solver._accuracy_options['defect_evaluation'],
        analytic_connection_probes=int(counts[0]),certified_connection_probes=int(counts[1]),accepted_macrosteps=int(v[ACCEPTED]),
        accepted_hydraulic_updates=int((1 if embedded else 2)*v[ACCEPTED]),
        provisional_hydraulic_updates=int(v[UPDATES]/(3 if embedded else 1)),rejected_attempts=int(v[REJECTED]),
        maximum_accepted_normalized_error=float(v[MAX_ERROR]),last_normalized_error=float(v[ERROR]),
        last_wave_ratio=float(v[WAVE_ERROR]),last_stage_error=float(v[STAGE_ERROR]),last_q_error=float(v[Q_ERROR]),
        last_coefficient_error=float(v[COEFFICIENT_ERROR]),last_transfer_error=float(v[TRANSFER_ERROR]),
        maximum_accepted_wave_ratio=float(v[MAX_ACCEPTED_WAVE]),wave_rejections=int(v[WAVE_REJECTIONS]))
