"""Exact source coefficient reuse when both endpoint evaluation states match."""
from typing import Any
import warp as wp
from floodwarp import enhanced_kernels as kernels
from floodwarp.config import WP_FLOAT


@wp.kernel(enable_backward=False)
def source_coefficients(endpoints: wp.array2d(dtype=int), halves: wp.array2d(dtype=int),
                 sills: wp.array(dtype=wp.float64), offsets: wp.array(dtype=int),
                 rows: wp.array2d(dtype=wp.float64), eta: wp.array(dtype=wp.float64),
                 heads: wp.array(dtype=wp.float64), kind: wp.array(dtype=int),
                 dry_depth: wp.float64, d: wp.array(dtype=wp.float64),
                 r: wp.array(dtype=wp.float64), forcing: wp.array(dtype=wp.float64),
                 maximum_stage: wp.float64, status: wp.array(dtype=int),
                 values: wp.array2d(dtype=Any), column: int,
                 indices: wp.array(dtype=Any), ranges: wp.array2d(dtype=int),
                 control: wp.array(dtype=wp.float64),tail_start: wp.float64,tail_reference: wp.float64,
                 tail_halves: wp.array2d(dtype=int),tail_offsets: wp.array(dtype=int),
                 tail_rows: wp.array2d(dtype=wp.float64), interval_cache: wp.array2d(dtype=int), inertia_only: int,
                 faces: wp.array(dtype=int), nx: int, x_count: int,
                 type_x: wp.array2d(dtype=int),type_y: wp.array2d(dtype=int),
                 stage_x: wp.array2d(dtype=WP_FLOAT),stage_y: wp.array2d(dtype=WP_FLOAT),
                 override: wp.array(dtype=int),override_head: wp.array(dtype=wp.float64),use_override: int,
                 use_stage_x: int,use_stage_y: int,
                 accepted_d: wp.array(dtype=wp.float64),accepted_r: wp.array(dtype=wp.float64),
                 accepted_forcing: wp.array(dtype=wp.float64),accepted_heads: wp.array(dtype=wp.float64),
                 accepted_kind: wp.array(dtype=int),coefficient_eta: wp.array(dtype=wp.float64),
                 valid: wp.array(dtype=int),counts: wp.array2d(dtype=int)):
    f=wp.tid()
    kernels.boundary_heads(f,faces,endpoints,sills,nx,x_count,type_x,type_y,stage_x,stage_y,
        heads,kind,override,override_head,use_override,control,use_stage_x,use_stage_y)
    if not kernels.hydraulic_step_enabled(control,status):return
    a=endpoints[f,0];b=endpoints[f,1];other=heads[f]
    if b>=0:other=eta[b]
    unchanged=valid[0]==1 and eta[a]==coefficient_eta[a]
    if b>=0:unchanged=unchanged and eta[b]==coefficient_eta[b]
    else:unchanged=unchanged and heads[f]==accepted_heads[f] and kind[f]==accepted_kind[f]
    if unchanged:
        d[f]=accepted_d[f];r[f]=accepted_r[f];forcing[f]=accepted_forcing[f]
        counts[f,0]+=1
        return
    counts[f,1]+=1
    d[f]=wp.float64(0.0);r[f]=wp.float64(0.0);forcing[f]=wp.float64(0.0)
    if b<0 and (kind[f]==0 or kind[f]==1):return
    stage=wp.max(eta[a],other)
    if not wp.isfinite(stage):wp.atomic_max(status,0,4)
    elif stage>maximum_stage:wp.atomic_max(status,0,1)
    if stage-sills[f]<=dry_depth:return
    # CFL reads only inertia. Default scalar encoding can skip resistance
    # interpolation and h^(10/3), which are recomputed after accepted sources.
    if inertia_only==1 and column<0 and stage<=tail_start:
        left=halves[f,0];right=halves[f,1]
        first_d=kernels.cached_scalar_arm(stage,ranges[left,0],ranges[left,1],rows,indices,values,interval_cache,f,0)/(stage-sills[f])
        second_d=kernels.cached_scalar_arm(stage,ranges[right,0],ranges[right,1],rows,indices,values,interval_cache,f,1)/(stage-sills[f])
        d[f]=first_d+second_d
        return
    first=wp.vec2d(0.0);second=wp.vec2d(0.0)
    if stage>tail_start:
        first=kernels.tail_arm(stage,sills[f],tail_halves[f,0],tail_reference,tail_offsets,tail_rows)
        second=kernels.tail_arm(stage,sills[f],tail_halves[f,1],tail_reference,tail_offsets,tail_rows)
    elif column<0:
        left=halves[f,0];right=halves[f,1];head=stage-sills[f]
        first[0]=kernels.cached_scalar_arm(stage,ranges[left,0],ranges[left,1],rows,indices,values,interval_cache,f,0)/head
        first[1]=kernels.cached_scalar_arm(stage,ranges[left,2],ranges[left,3],rows,indices,values,interval_cache,f,2)/wp.pow(head,wp.float64(10.0)/wp.float64(3.0))
        second[0]=kernels.cached_scalar_arm(stage,ranges[right,0],ranges[right,1],rows,indices,values,interval_cache,f,1)/head
        second[1]=kernels.cached_scalar_arm(stage,ranges[right,2],ranges[right,3],rows,indices,values,interval_cache,f,3)/wp.pow(head,wp.float64(10.0)/wp.float64(3.0))
    else:
        first=kernels.arm(stage,sills[f],halves[f,0],offsets,rows,values,column,indices,ranges)
        second=kernels.arm(stage,sills[f],halves[f,1],offsets,rows,values,column,indices,ranges)
    d[f]=first[0]+second[0];r[f]=first[1]+second[1];forcing[f]=eta[a]-other



def enqueue(*,solver,control,accepted):
    g=solver._enhanced_geometry;options=solver._ec_trial_options
    wp.launch(kernel=source_coefficients,block_dim=128,dim=solver._connection_count,inputs=[solver._ec_endpoints,
        solver._ec_halves,solver._ec_sill,solver._ec_offsets,solver._ec_rows,solver._ec_eta,solver._ec_heads,
        solver._ec_kind,wp.float64(float(solver.dry_depth)),solver._ec_d,solver._ec_r,solver._ec_forcing,
        wp.float64(g.maximum_stage),solver._ec_status,getattr(solver,'_ec_coefficient_values',solver._ec_rows),
        -1 if g.coefficient_ranges is not None else (0 if g.coefficient_values is not None else 2),
        getattr(solver,'_ec_coefficient_stage_index',solver._ec_adjacency),
        getattr(solver,'_ec_coefficient_ranges',solver._ec_halves),control,wp.float64(g.tail_start),wp.float64(g.tail_reference),
        getattr(solver,'_ec_tail_halves',solver._ec_halves),getattr(solver,'_ec_tail_offsets',solver._ec_offsets),
        getattr(solver,'_ec_tail_rows',solver._ec_storage_rows),solver._ec_scalar_interval,0,
        solver._ec_face,solver.nx,solver.ny*(solver.nx+1),solver.boundary_type_qx,solver.boundary_type_qy,
        solver.boundary_stage_qx,solver.boundary_stage_qy,solver._ec_boundary_override,solver._ec_boundary_head_values,
        solver._ec_has_boundary_override,int(options['use_stage_qx']),int(options['use_stage_qy']),
        accepted['_ec_d'],accepted['_ec_r'],accepted['_ec_forcing'],accepted['_ec_heads'],accepted['_ec_kind'],
        solver._embedded_coefficient_eta,solver._embedded_coefficients_valid,solver._embedded_reuse_counts],device=solver.device)
