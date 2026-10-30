"""Private embedded trials and guarded, acceptance-only state/accounting commits.

Host bindings change only while emitting CPU/GPU commands. CUDA graphs retain
fixed array arguments. The reference controller remains in enhanced_accuracy.
"""
from contextlib import contextmanager
import warp as wp
from floodwarp import enhanced_accuracy as reference
from floodwarp import enhanced_kernels as kernels
from floodwarp import enhanced_transaction
from floodwarp import enhanced_embedded_coefficients
from floodwarp import enhanced_embedded_defect
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import DL_ERROR, DL_DONE, DL_DT

# Actual trial writers. Outputs/ledgers/maxima outside this list are deferred.
PRIVATE_FIELDS=('_ec_volume','_ec_eta','_ec_area','_ec_soil','_ec_cap','_ec_q','_ec_previous_d',
    '_ec_applied','_ec_proposed','_ec_alpha','_ec_alpha_free','_ec_d','_ec_r','_ec_forcing',
    '_ec_heads','_ec_kind','_ec_scalar_interval','_ec_status','_ec_source_ledger',
    'boundary_stage_qx','boundary_stage_qy')

# Reuse the exact existing forcing sampler and accept/retry decision arithmetic.
sample_forcing=wp.func(kernels.device_finalize.func)
decide_control=wp.func(reference.decide.func)


def allocate(*,solver):
    epoch=getattr(solver,'_accuracy_layout_epoch',0)
    if hasattr(solver,'_embedded_private') and getattr(solver,'_embedded_layout_epoch',-1)==epoch:return
    private={}
    for name in PRIVATE_FIELDS:
        value=getattr(solver,name);private_name='_embedded_trial'+name
        array=getattr(solver,private_name,None)
        shape=(solver._component_count,4) if name=='_ec_source_ledger' else value.shape
        if array is None or array.shape!=shape or array.dtype!=value.dtype:
            array=wp.empty(shape,dtype=value.dtype,device=solver.device)
            setattr(solver,private_name,array)
            if name=='_ec_scalar_interval':array.fill_(-1)
            if name=='_ec_heads':wp.copy(array,value)
        private[name]=array
    solver._embedded_private=private;solver._embedded_layout_epoch=epoch
    solver._embedded_boundaries=enhanced_transaction.prepare(solver=solver,
        saved={name:private[name] for name in ('boundary_stage_qx','boundary_stage_qy')})
    if not hasattr(solver,'_embedded_accept'):solver._embedded_accept=wp.zeros(1,dtype=int,device=solver.device)
    if not hasattr(solver,'_embedded_commit_control'):solver._embedded_commit_control=wp.zeros_like(solver._accuracy_control)
    if not hasattr(solver,'_embedded_coefficients_valid'):
        solver._embedded_coefficients_valid=wp.zeros(1,dtype=int,device=solver.device)
        solver._embedded_coefficient_eta=wp.zeros(solver._component_count,dtype=wp.float64,device=solver.device)
    if not hasattr(solver,'_embedded_reuse_counts') or solver._embedded_reuse_counts.shape[0]!=solver._connection_count:
        solver._embedded_reuse_counts=wp.zeros((solver._connection_count,2),dtype=int,device=solver.device)
    values=getattr(solver,'_ec_coefficient_values',solver._ec_rows)
    indices=getattr(solver,'_ec_coefficient_stage_index',solver._ec_adjacency)
    wp.overload(enhanced_embedded_coefficients.source_coefficients,{'values':wp.array2d(dtype=values.dtype),'indices':wp.array(dtype=indices.dtype)})
    wp.load_module(module=enhanced_embedded_coefficients.source_coefficients.module,device=solver.device)
    wp.load_module(module=decide_accept.module,device=solver.device)
    wp.load_module(module=enhanced_embedded_defect.connections.module,device=solver.device)


def rebind(*,solver):
    solver._embedded_layout_epoch=-1
    allocate(solver=solver)


@contextmanager
def private_bindings(*,solver):
    accepted={name:getattr(solver,name) for name in solver._embedded_private}
    dynamic=solver._ec_trial_options['use_stage_qx'] or solver._ec_trial_options['use_stage_qy']
    for name,array in solver._embedded_private.items():
        if name in ('boundary_stage_qx','boundary_stage_qy') and not dynamic:continue
        setattr(solver,name,array)
    try:yield accepted
    finally:
        for name,array in accepted.items():setattr(solver,name,array)


@wp.kernel(enable_backward=False)
def decide_accept(control: wp.array(dtype=wp.float64),selected: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
    state: wp.array(dtype=wp.float64),condition: wp.array(dtype=int),minimum: wp.float64,maximum: wp.float64,
    safety: wp.float64,max_attempts: int,wave_fraction: wp.float64,trial: wp.array(dtype=wp.float64),
    flag: wp.array(dtype=int),commit_control: wp.array(dtype=wp.float64),accepted_status: wp.array(dtype=int)):
    decide_control(control,selected,status,state,condition,minimum,maximum,safety,max_attempts,wave_fraction)
    flag[0]=int(condition[0]==0 and control[DL_ERROR]==wp.float64(0.) and trial[DL_DONE]==wp.float64(0.))
    for k in range(reference.DL_SIZE):commit_control[k]=trial[k]
    if flag[0]==0:commit_control[DL_DONE]=wp.float64(1.)
    if control[DL_ERROR]!=wp.float64(0.):accepted_status[0]=6


def substep(*,solver,control,accepted):
    options=solver._ec_trial_options;trial=solver._accuracy_control;state=solver._accuracy_state
    wp.launch(kernel=prepare_selected,dim=1,inputs=[control,trial,solver._accuracy_selected_control,state,
        float(solver.adaptive_dt_growth_factor or 0.),float(options['end_time']),float(solver.maximum_dt),
        solver.cfl_number,solver.dx,solver.gravity,solver.dry_depth,options['hyeto_times'],options['hyeto_rates'],options['use_hyetograph'],
        float(options['storm_duration']),options['stage_times_qx'],options['stage_values_qx'],options['use_stage_qx'],
        options['stage_times_qy'],options['stage_values_qy'],options['use_stage_qy'],options['checkpoint_times'],options['checkpoint_count'],solver._ec_status],device=solver.device)
    wp.launch(kernel=source_private,dim=solver._component_count,inputs=[accepted['_ec_volume'],accepted['_ec_eta'],accepted['_ec_area'],
        accepted['_ec_soil'],accepted['_ec_cap'],solver._ec_volume,solver._ec_eta,solver._ec_area,solver._ec_soil,solver._ec_cap,
        solver._ec_applied,solver._ec_source_ledger,solver._ec_state_cell,solver.nx,solver._ec_source_area,
        solver._ec_storage_offsets,solver._ec_storage_rows,solver.rainfall_rate,solver.other_source_rate,solver.additional_source_rate,
        solver.rainfall_spatial_multiplier,solver.infiltration_exclusion_mask,solver.infiltration_cap_active,trial,solver._ec_status,
        solver._ec_maximum_eta,float(solver._enhanced_geometry.maximum_stage),options['use_hyetograph']],device=solver.device)
    with start_coefficients(solver=solver):
        enhanced_embedded_coefficients.enqueue(solver=solver,control=trial,accepted=accepted)
        wp.launch(kernel=reference.wave,dim=solver._component_count,block_dim=128,inputs=[trial,solver._ec_status,
            solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_d,solver._ec_area,state,float(solver.gravity),float(solver.cfl_number)],device=solver.device)
        wp.launch(kernel=momentum_initial,dim=solver._connection_count,block_dim=getattr(solver,'_accuracy_connection_block_dim',128),
            inputs=[solver._ec_d,solver._ec_r,solver._ec_forcing,accepted['_ec_q'],solver._ec_previous_d,solver._ec_proposed,
                solver._accuracy_effective_q,trial,float(solver.gravity),solver._ec_status,solver._ec_endpoints,solver._ec_sill,
                solver._ec_eta,solver._ec_heads,solver._ec_kind,float(solver.dry_depth)],device=solver.device)
    safety=1.000000000000002 if WP_FLOAT==wp.float64 else 1.000001
    wp.launch(kernel=kernels.demands,dim=solver._component_count,inputs=[solver._ec_volume,solver._ec_adjacency_offsets,solver._ec_adjacency,
        solver._ec_proposed,solver._ec_alpha,trial,safety,solver._ec_status,solver._ec_kind,solver._ec_alpha_free,solver.free_outflow_limiter_mode],device=solver.device)
    wp.launch(kernel=limit_private,dim=solver._connection_count,inputs=[solver._ec_endpoints,solver._ec_proposed,solver._ec_alpha,
        solver._ec_q,trial,solver._ec_status,solver._ec_kind,solver._ec_alpha_free],device=solver.device)
    wp.launch(kernel=kernels.continuity,dim=solver._component_count,inputs=[solver._ec_volume,solver._ec_adjacency_offsets,solver._ec_adjacency,
        solver._ec_q,trial,solver._ec_status,1,solver._ec_storage_offsets,solver._ec_storage_rows,solver._ec_eta,solver._ec_area,
        solver._ec_maximum_eta,float(solver._enhanced_geometry.maximum_stage),0],device=solver.device)
    if options['use_stage_qx'] or options['use_stage_qy']:
        wp.launch(kernel=reference.device_loop_stage_fill_kernel,dim=(solver.ny+1,solver.nx+1),inputs=[solver.boundary_type_qx,
            solver.boundary_stage_qx,solver.boundary_type_qy,solver.boundary_stage_qy,trial,options['use_stage_qx'],options['use_stage_qy']],device=solver.device)
    solver._component_coefficients(control=trial,use_stage_qx=options['use_stage_qx'],use_stage_qy=options['use_stage_qy'])


def attempt(*,solver,control):
    options=solver._ec_trial_options
    if options['use_stage_qx'] or options['use_stage_qy']:
        enhanced_transaction.copy(groups=solver._embedded_boundaries,restore=False,device=solver.device)
    with private_bindings(solver=solver) as accepted:
        substep(solver=solver,control=control,accepted=accepted)
        tuning=solver._accuracy_options
        wp.launch(kernel=enhanced_embedded_defect.connections,dim=solver._connection_count,
            block_dim=getattr(solver,'_accuracy_connection_block_dim',128),inputs=[solver._accuracy_control,
            solver._ec_q,solver._accuracy_effective_q,solver._ec_proposed,solver._ec_kind,solver._ec_endpoints,
            solver._ec_adjacency_offsets,solver._ec_source_area,solver._ec_volume,solver._ec_area,solver._ec_d,solver._ec_r,solver._ec_forcing,
            solver._accuracy_start_d,solver._accuracy_start_r,solver._accuracy_start_forcing,solver._accuracy_mid_q,
            solver._accuracy_mid_radius,solver._accuracy_defect_counts,solver._accuracy_state,
            float(solver.gravity),tuning['discharge_atol_m3_s'],tuning['relative_tolerance'],
            tuning['coefficient_relative_tolerance'],tuning['stage_atol_m'],int(tuning['defect_evaluation']=='bounded')],device=solver.device)
        wp.launch(kernel=component_wave_error,dim=solver._component_count,block_dim=128,inputs=[solver._accuracy_control,solver._ec_status,
            solver._ec_volume,solver._ec_eta,solver._ec_area,accepted['_ec_eta'],solver._ec_source_area,solver._ec_storage_offsets,
            solver._ec_storage_rows,solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_q,solver._accuracy_mid_q,
            solver._accuracy_mid_radius,solver._ec_proposed,solver._ec_d,solver._accuracy_state,tuning['stage_atol_m'],
            tuning['relative_tolerance'],float(solver.gravity),float(solver.cfl_number)],device=solver.device)
        wp.launch(kernel=decide_accept,dim=1,inputs=[control,solver._accuracy_selected_control,solver._ec_status,
            solver._accuracy_state,solver._accuracy_condition,float(solver.minimum_dt),float(solver.maximum_dt),
            tuning['safety'],tuning['max_attempts'],tuning['wave_proposal_fraction'],solver._accuracy_control,solver._embedded_accept,
            solver._embedded_commit_control,accepted['_ec_status']],device=solver.device)


def commit(*,solver):
    commit_fused(solver=solver)
    options=solver._ec_trial_options
    if options['use_stage_qx'] or options['use_stage_qy']:wp.capture_if(solver._embedded_accept,on_true=commit_boundaries,solver=solver)
    guarded_reconstruction(solver=solver,faces=solver.track_cumulative_face_volumes or options['checkpoint_count']>0)
    trial=solver._embedded_commit_control
    if getattr(solver,'_accuracy_collect_free_outflow_diagnostics',False):
        wp.launch(kernel=kernels.free_outflow_diagnostics,dim=solver.ny*solver.nx,inputs=[solver._ec_cell_component_offsets,
            solver._ec_adjacency_offsets,solver._ec_adjacency,solver._ec_kind,solver._ec_proposed,solver._ec_q,trial,
            solver._ec_status,solver.free_outflow_diagnostics],device=solver.device)
    if solver.track_cumulative_face_volumes:
        for axis,dim,width in (('x',(solver.ny,solver.nx+1),solver.dy),('y',(solver.ny+1,solver.nx),solver.dx)):
            kernel=reference.device_loop_accumulate_face_volume_qx_kernel if axis=='x' else reference.device_loop_accumulate_face_volume_qy_kernel
            wp.launch(kernel=kernel,dim=dim,inputs=[getattr(solver,'q'+axis),getattr(solver,'cumulative_net_q'+axis+'_volume_m3'),
                getattr(solver,'cumulative_abs_q'+axis+'_volume_m3') if solver.track_absolute_face_volumes else solver._diagnostic_absolute_placeholder,
                getattr(solver,'_diagnostic_peak_q'+axis+'_rows'),width,trial,int(solver.track_absolute_face_volumes)],device=solver.device)


def enqueue(*,solver,control):
    allocate(solver=solver)
    wp.launch(kernel=initialize_private,dim=1,inputs=[control,solver._ec_status,solver._accuracy_state,solver._accuracy_condition,
        float(solver.maximum_dt),solver._embedded_accept,solver._embedded_commit_control],device=solver.device)
    if solver.device.is_cuda:
        if solver.device.is_capturing:wp.capture_while(solver._accuracy_condition,while_body=attempt,solver=solver,control=control)
        else:
            with wp.ScopedCapture(device=solver.device,force_module_load=True) as loop:
                wp.capture_while(solver._accuracy_condition,while_body=attempt,solver=solver,control=control)
            wp.capture_launch(loop.graph)
    else:
        while solver._accuracy_condition.numpy()[0]:attempt(solver=solver,control=control)
    commit(solver=solver)


@wp.kernel(enable_backward=False)
def prepare_selected(control: wp.array(dtype=wp.float64),trial: wp.array(dtype=wp.float64),
    selected: wp.array(dtype=wp.float64),state: wp.array(dtype=wp.float64),growth: wp.float64,
    end_time: wp.float64,maximum_dt: wp.float64,cfl: WP_FLOAT,dx: WP_FLOAT,gravity: WP_FLOAT,dry: WP_FLOAT,
    hyeto_times: wp.array(dtype=wp.float64),hyeto_rates: wp.array(dtype=wp.float64),use_hyetograph: int,
    storm_duration: wp.float64,stage_times_qx: wp.array(dtype=wp.float64),stage_values_qx: wp.array(dtype=wp.float64),
    use_stage_qx: int,stage_times_qy: wp.array(dtype=wp.float64),stage_values_qy: wp.array(dtype=wp.float64),use_stage_qy: int,
    checkpoint_times: wp.array(dtype=wp.float64),checkpoint_count: int,status: wp.array(dtype=int)):
    status[0]=0
    for i in range(reference.DL_SIZE):trial[i]=control[i]
    trial[DL_ERROR]=wp.float64(0.);trial[DL_DONE]=wp.float64(0.)
    trial[reference.DL_GROWTH_REFERENCE]=wp.float64(0.)
    trial[DL_DT]=state[reference.TARGET]
    if growth>=wp.float64(1.) and control[reference.DL_GROWTH_REFERENCE]>wp.float64(0.):
        trial[DL_DT]=wp.min(trial[DL_DT],control[reference.DL_GROWTH_REFERENCE]*growth)
    state[reference.PROPOSED]=trial[DL_DT]
    state[reference.ERROR]=wp.float64(0.);state[reference.WAVE_ERROR]=wp.float64(0.)
    state[reference.STAGE_ERROR]=wp.float64(0.);state[reference.Q_ERROR]=wp.float64(0.)
    state[reference.COEFFICIENT_ERROR]=wp.float64(0.);state[reference.TRANSFER_ERROR]=wp.float64(0.)
    state[reference.WAVE_BOUND]=wp.float64(1.e100)
    sample_forcing(trial,end_time,wp.float64(0.),maximum_dt,wp.float64(0.),0,cfl,dx,gravity,dry,
        hyeto_times,hyeto_rates,use_hyetograph,storm_duration,1,
        stage_times_qx,stage_values_qx,use_stage_qx,stage_times_qy,stage_values_qy,use_stage_qy,checkpoint_times,checkpoint_count)
    state[reference.DT]=trial[DL_DT];state[reference.CLIPPED]=wp.float64(0.)
    if state[reference.DT]<state[reference.PROPOSED]*(wp.float64(1.)-wp.float64(1.e-12)):state[reference.CLIPPED]=wp.float64(1.)
    for i in range(reference.DL_SIZE):selected[i]=trial[i]


@wp.kernel(enable_backward=False)
def momentum_initial(d: wp.array(dtype=wp.float64),r: wp.array(dtype=wp.float64),forcing: wp.array(dtype=wp.float64),
    q: wp.array(dtype=WP_FLOAT),previous_d: wp.array(dtype=wp.float64),proposed: wp.array(dtype=WP_FLOAT),
    effective: wp.array(dtype=WP_FLOAT),control: wp.array(dtype=wp.float64),gravity: wp.float64,status: wp.array(dtype=int),
    endpoints: wp.array2d(dtype=int),sills: wp.array(dtype=wp.float64),eta: wp.array(dtype=wp.float64),
    heads: wp.array(dtype=wp.float64),kind: wp.array(dtype=int),dry: wp.float64):
    if not kernels.hydraulic_step_enabled(control,status):return
    f=wp.tid();donor=eta[endpoints[f,0]]
    if q[f]<WP_FLOAT(0.):
        donor=heads[f]
        if endpoints[f,1]>=0:donor=eta[endpoints[f,1]]
    effective[f]=q[f]
    if donor<=sills[f]+dry:effective[f]=WP_FLOAT(0.)
    free=int(endpoints[f,1]<0 and (kind[f]==2 or (kind[f]>=6 and kind[f]<=9)))
    proposed[f]=kernels.connection_discharge(d[f],r[f],forcing[f],q[f],wp.float64(0.),control[DL_DT],gravity,
        0,donor,sills[f],dry,free,1)
    if not wp.isfinite(proposed[f]):wp.atomic_max(status,0,4)
    previous_d[f]=d[f]


@wp.kernel(enable_backward=False)
def component_wave_error(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
        volume: wp.array(dtype=WP_FLOAT),eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),
        initial_eta: wp.array(dtype=wp.float64),source_area: wp.array(dtype=wp.float64),
        storage_offsets: wp.array(dtype=int),storage_rows: wp.array2d(dtype=wp.float64),
        offsets: wp.array(dtype=int),adjacency: wp.array(dtype=int),q: wp.array(dtype=WP_FLOAT),
        mid_q: wp.array(dtype=WP_FLOAT),mid_radius: wp.array(dtype=wp.float64),proposed: wp.array(dtype=WP_FLOAT),
        d: wp.array(dtype=wp.float64),state: wp.array(dtype=wp.float64),stage_atol: wp.float64,rtol: wp.float64,
        gravity: wp.float64,cfl: wp.float64):
    c=wp.tid();defect=wp.float64(0.);rejected=wp.float64(0.);uncertainty=wp.float64(0.);degree=wp.float64(0.)
    for k in range(offsets[c],offsets[c+1]):
        entry=adjacency[k];sign=wp.float64(1.)
        if entry<0:sign=wp.float64(-1.)
        f=wp.abs(entry)-1
        defect-=sign*(wp.float64(mid_q[f])-wp.float64(q[f]))*control[DL_DT]
        uncertainty+=mid_radius[f]*control[DL_DT]
        rejected+=wp.abs(wp.float64(proposed[f])-wp.float64(q[f]))*control[DL_DT]
        if d[f]>wp.float64(0.):degree+=wp.float64(1.)/d[f]
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
    te=rejected/scale
    reference.maximum(state,reference.TRANSFER_ERROR,te)
    reference.maximum(state,reference.STAGE_ERROR,error)
    reference.maximum(state,reference.ERROR,wp.max(error,te))
    if wp.float64(volume[c])+defect-uncertainty<wp.float64(0.):reference.maximum(state,reference.ERROR,wp.float64(2.))
    if kernels.hydraulic_step_enabled(control,status) and degree>wp.float64(0.):
        bound=cfl*wp.sqrt(wp.float64(2.)*area[c]/(gravity*degree))
        reference.minimum(state,reference.WAVE_BOUND,bound)
        ratio=control[DL_DT]/bound
        reference.maximum(state,reference.WAVE_ERROR,ratio)
        if ratio>wp.float64(1.):reference.maximum(state,reference.ERROR,ratio*ratio)


@contextmanager
def start_coefficients(*,solver):
    arrays={name:getattr(solver,'_ec_'+name) for name in ('d','r','forcing')}
    for name in arrays:setattr(solver,'_ec_'+name,getattr(solver,'_accuracy_start_'+name))
    try:yield
    finally:
        for name,array in arrays.items():setattr(solver,'_ec_'+name,array)


@wp.kernel(enable_backward=False)
def source_private(accepted_volume: wp.array(dtype=WP_FLOAT),accepted_eta: wp.array(dtype=wp.float64),
    accepted_area: wp.array(dtype=wp.float64),accepted_soil: wp.array(dtype=WP_FLOAT),accepted_cap: wp.array(dtype=WP_FLOAT),
    volume: wp.array(dtype=WP_FLOAT),eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),
    soil: wp.array(dtype=WP_FLOAT),cap: wp.array(dtype=WP_FLOAT),applied: wp.array2d(dtype=WP_FLOAT),
    ledger: wp.array2d(dtype=wp.float64),state_cell: wp.array(dtype=int),nx: int,source_area: wp.array(dtype=wp.float64),
    offsets: wp.array(dtype=int),storage: wp.array2d(dtype=wp.float64),rain: wp.array2d(dtype=WP_FLOAT),
    other: wp.array2d(dtype=WP_FLOAT),additional_source: wp.array2d(dtype=WP_FLOAT),multiplier: wp.array2d(dtype=WP_FLOAT),
    excluded: wp.array2d(dtype=int),cap_active: int,control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
    maximum: wp.array(dtype=wp.float64),maximum_stage: wp.float64,use_hyetograph: int):
    if not kernels.hydraulic_step_enabled(control,status):return
    c=wp.tid();cell=state_cell[c];j=cell//nx;i=cell%nx;plan=source_area[c];v=wp.float64(accepted_volume[c])
    raw=rain[j,i]
    if use_hyetograph==1:raw=WP_FLOAT(control[kernels.DL_RAIN_RATE])*multiplier[j,i]
    # This is the exact zero-capacity specialization of _apply_source_state,
    # not an alternative integration rule. Preserve the depth/volume roundtrip
    # and the source addition order even when all source rates vanish.
    dt=WP_FLOAT(control[DL_DT])
    rd=wp.max(WP_FLOAT(0.),raw)*dt
    od=wp.max(WP_FLOAT(0.),other[j,i])*dt
    gd=wp.max(WP_FLOAT(0.),additional_source[j,i])*dt
    supply=WP_FLOAT(v/plan)+rd+od+gd
    volume[c]=WP_FLOAT(wp.float64(supply)*plan)
    eta[c]=accepted_eta[c];area[c]=accepted_area[c]
    soil[c]=accepted_soil[c];cap[c]=accepted_cap[c]
    if cap_active==1 and excluded[j,i]!=1:cap[c]=wp.max(WP_FLOAT(0.),accepted_cap[c])
    applied[c,0]=rd;applied[c,1]=od;applied[c,2]=WP_FLOAT(0.)
    ledger[c,0]=wp.float64(rd)*plan;ledger[c,1]=wp.float64(0.)
    ledger[c,2]=wp.float64(od)*plan;ledger[c,3]=wp.float64(gd)*plan
    if wp.float64(volume[c])!=v:
        kernels.update_component_stage(c,volume,offsets,storage,eta,area,maximum,maximum_stage,0,status,control)


@wp.kernel(enable_backward=False)
def limit_private(endpoints: wp.array2d(dtype=int),proposed: wp.array(dtype=WP_FLOAT),alpha: wp.array(dtype=wp.float64),
    q: wp.array(dtype=WP_FLOAT),control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
    kind: wp.array(dtype=int),alpha_free: wp.array(dtype=wp.float64)):
    if not kernels.hydraulic_step_enabled(control,status):return
    f=wp.tid();donor=endpoints[f,0]
    if proposed[f]<WP_FLOAT(0.):donor=endpoints[f,1]
    value=wp.float64(proposed[f])
    if donor>=0:
        if kind[f]==2:value*=alpha_free[donor]
        else:value*=alpha[donor]
    q[f]=WP_FLOAT(value)


@wp.kernel(enable_backward=False)
def accept_components(v: wp.array(dtype=WP_FLOAT),eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),
    soil: wp.array(dtype=WP_FLOAT),cap: wp.array(dtype=WP_FLOAT),applied: wp.array2d(dtype=WP_FLOAT),
    alpha: wp.array(dtype=wp.float64),alpha_free: wp.array(dtype=wp.float64),ledger: wp.array2d(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),total: wp.array(dtype=wp.float64),coefficient_eta: wp.array(dtype=wp.float64),tv: wp.array(dtype=WP_FLOAT),te: wp.array(dtype=wp.float64),ta: wp.array(dtype=wp.float64),
    ts: wp.array(dtype=WP_FLOAT),tc: wp.array(dtype=WP_FLOAT),tap: wp.array2d(dtype=WP_FLOAT),
    talpha: wp.array(dtype=wp.float64),talpha_free: wp.array(dtype=wp.float64),delta: wp.array2d(dtype=wp.float64),track: int,deferred: int,flag: wp.array(dtype=int)):
    if flag[0]==0:return
    c=wp.tid();v[c]=tv[c];eta[c]=te[c];area[c]=ta[c];soil[c]=ts[c];cap[c]=tc[c]
    coefficient_eta[c]=te[c]
    for k in range(3):applied[c,k]=tap[c,k]
    alpha[c]=talpha[c];alpha_free[c]=talpha_free[c]
    if track==1:maximum[c]=wp.max(maximum[c],te[c])
    for k in range(4):
        if deferred==1:ledger[c,k]+=delta[c,k]
        elif delta[c,k]!=wp.float64(0.):
            destination=k
            if k==3:destination=5
            wp.atomic_add(total,destination,delta[c,k])


@wp.kernel(enable_backward=False)
def accept_connections(q: wp.array(dtype=WP_FLOAT),previous: wp.array(dtype=wp.float64),proposed: wp.array(dtype=WP_FLOAT),
    d: wp.array(dtype=wp.float64),r: wp.array(dtype=wp.float64),forcing: wp.array(dtype=wp.float64),
    heads: wp.array(dtype=wp.float64),kind: wp.array(dtype=int),cache: wp.array2d(dtype=int),
    tq: wp.array(dtype=WP_FLOAT),tp: wp.array(dtype=wp.float64),tproposed: wp.array(dtype=WP_FLOAT),
    td: wp.array(dtype=wp.float64),tr: wp.array(dtype=wp.float64),tf: wp.array(dtype=wp.float64),
    th: wp.array(dtype=wp.float64),tk: wp.array(dtype=int),tcache: wp.array2d(dtype=int),
    endpoints: wp.array2d(dtype=int),control: wp.array(dtype=wp.float64),total: wp.array(dtype=wp.float64),valid: wp.array(dtype=int),flag: wp.array(dtype=int)):
    if flag[0]==0:return
    f=wp.tid();q[f]=tq[f];previous[f]=tp[f];proposed[f]=tproposed[f]
    if f==0:valid[0]=1
    d[f]=td[f];r[f]=tr[f];forcing[f]=tf[f];heads[f]=th[f];kind[f]=tk[f]
    if f<cache.shape[0]:
        for k in range(4):cache[f,k]=tcache[f,k]
    if endpoints[f,1]<0:
        amount=control[DL_DT]*wp.float64(tq[f])
        if amount>=wp.float64(0.):wp.atomic_add(total,4,amount)
        else:wp.atomic_add(total,3,-amount)


def commit_fused(*,solver):
    p=solver._embedded_private
    wp.launch(kernel=accept_components,dim=solver._component_count,inputs=[solver._ec_volume,solver._ec_eta,solver._ec_area,
        solver._ec_soil,solver._ec_cap,solver._ec_applied,solver._ec_alpha,solver._ec_alpha_free,solver._ec_source_ledger,
        solver._ec_maximum_eta,solver.cumulative_totals,solver._embedded_coefficient_eta,p['_ec_volume'],p['_ec_eta'],p['_ec_area'],p['_ec_soil'],p['_ec_cap'],p['_ec_applied'],
        p['_ec_alpha'],p['_ec_alpha_free'],p['_ec_source_ledger'],int(solver.track_max_water_depth),int(solver.use_reduced_mass_accounting),solver._embedded_accept],device=solver.device)
    wp.launch(kernel=accept_connections,dim=solver._connection_count,inputs=[solver._ec_q,solver._ec_previous_d,solver._ec_proposed,
        solver._ec_d,solver._ec_r,solver._ec_forcing,solver._ec_heads,solver._ec_kind,solver._ec_scalar_interval,
        p['_ec_q'],p['_ec_previous_d'],p['_ec_proposed'],p['_ec_d'],p['_ec_r'],p['_ec_forcing'],p['_ec_heads'],p['_ec_kind'],p['_ec_scalar_interval'],
        solver._ec_endpoints,solver._accuracy_control,solver.cumulative_totals,solver._embedded_coefficients_valid,solver._embedded_accept],device=solver.device)


@wp.kernel(enable_backward=False)
def initialize_private(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),state: wp.array(dtype=wp.float64),
    condition: wp.array(dtype=int),maximum: wp.float64,flag: wp.array(dtype=int),commit_control: wp.array(dtype=wp.float64)):
    flag[0]=0;condition[0]=0
    commit_control[DL_DONE]=wp.float64(1.)
    if control[DL_DONE]==wp.float64(0.) and control[DL_ERROR]==wp.float64(0.) and status[0]==0:
        condition[0]=1
        if state[reference.TARGET]<=wp.float64(0.):state[reference.TARGET]=maximum
        state[reference.ATTEMPTS]=wp.float64(0.)


@wp.kernel(enable_backward=False)
def coarse_accepted(flag: wp.array(dtype=int),offsets: wp.array(dtype=int),volume: wp.array(dtype=WP_FLOAT),
    cell_area: wp.float64,depth: wp.array2d(dtype=WP_FLOAT),active: wp.array2d(dtype=int),
    maximum: wp.array2d(dtype=WP_FLOAT),diagnostic: wp.array2d(dtype=WP_FLOAT),track: int,track_diagnostic: int):
    if flag[0]==0:return
    j,i=wp.tid();cell=j*depth.shape[1]+i;v=wp.float64(0.)
    for c in range(offsets[cell],offsets[cell+1]):v+=wp.float64(volume[c])
    depth[j,i]=WP_FLOAT(v/cell_area)
    if active[j,i]==1:
        if track==1:maximum[j,i]=wp.max(maximum[j,i],depth[j,i])
        if track_diagnostic==1:diagnostic[j,i]=wp.max(diagnostic[j,i],depth[j,i])


@wp.kernel(enable_backward=False)
def faces_accepted(flag: wp.array(dtype=int),offsets: wp.array(dtype=int),members: wp.array(dtype=int),
    signs: wp.array(dtype=int),q: wp.array(dtype=WP_FLOAT),width: wp.float64,nx: int,x_count: int,
    qx: wp.array2d(dtype=WP_FLOAT),qy: wp.array2d(dtype=WP_FLOAT)):
    if flag[0]==0:return
    face=wp.tid();value=wp.float64(0.)
    for k in range(offsets[face],offsets[face+1]):
        f=members[k];value+=wp.float64(signs[f])*wp.float64(q[f])
    if face<x_count:qx[face//(nx+1),face%(nx+1)]=WP_FLOAT(value/width)
    else:
        f=face-x_count;qy[f//nx,f%nx]=WP_FLOAT(value/width)


def commit_boundaries(*,solver):
    enhanced_transaction.copy(groups=solver._embedded_boundaries,restore=True,device=solver.device)


def guarded_reconstruction(*,solver,faces):
    wp.launch(kernel=coarse_accepted,dim=(solver.ny,solver.nx),inputs=[solver._embedded_accept,solver._ec_cell_component_offsets,
        solver._ec_volume,float(solver.dx)*float(solver.dy),solver.water_depth,solver.active_mask,
        solver.max_water_depth_m if solver.track_max_water_depth else solver.water_depth,
        solver._diagnostic_max_depth_m if solver.track_cumulative_face_volumes and solver.track_diagnostic_max_depth else solver.water_depth,
        int(solver.track_max_water_depth),int(solver.track_cumulative_face_volumes and solver.track_diagnostic_max_depth)],device=solver.device)
    if faces:
        wp.launch(kernel=faces_accepted,dim=solver.ny*(solver.nx+1)+(solver.ny+1)*solver.nx,inputs=[solver._embedded_accept,
            solver._ec_face_offsets,solver._ec_face_members,solver._ec_sign,solver._ec_q,float(solver.dx),solver.nx,
            solver.ny*(solver.nx+1),solver.qx,solver.qy],device=solver.device)
