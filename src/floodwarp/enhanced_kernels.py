"""Component finite volumes using production source semantics and reduced geometry.

Q is total connection discharge [m3/s]; component V is volume [m3].
Native raster traversal is confined to preprocessing and requested output.
"""
from typing import Any
import warp as wp
from floodwarp.config import WP_FLOAT
from floodwarp.enhanced_momentum import frozen_discharge

STATE_EPSILON = 2.0**-23 if WP_FLOAT==wp.float32 else 2.0**-52
UNKNOWN_SOURCE_STAGE = wp.constant(float("inf"))
from floodwarp.local_inertial_kernels import (
    _apply_source_state, _device_timestep_finalize_from_cfl, _device_commit_clock,
    DL_TIME, DL_DT, DL_ERROR, DL_RAIN_RATE, DL_COUNTER, DL_DONE, DL_STAGE_QX, DL_STAGE_QY,
    DL_CHECKPOINT_INDEX, DL_RECORD_CHECKPOINT, DL_GROWTH_REFERENCE,
    LTS_CTRL_BASE_DT,
    _lts_finalize_schedule_from_dt, _device_piecewise_linear_interp,
)


@wp.func
def hydraulic_step_enabled(control: wp.array(dtype=wp.float64), status: wp.array(dtype=int)):
    return control[DL_DONE]==wp.float64(0.0) and control[DL_ERROR]==wp.float64(0.0) and status[0]==0


@wp.func
def scalar_interval(stage: wp.float64, begin: int, end: int,
        rows: wp.array2d(dtype=wp.float64),indices: wp.array(dtype=Any)):
    lo=begin;hi=end
    while lo<hi:
        mid=(lo+hi)//2
        if rows[int(indices[mid]),0]<=stage:lo=mid+1
        else:hi=mid
    return wp.max(begin,wp.min(end-2,lo-1))


@wp.func
def scalar_segment(stage: wp.float64,k: int,rows: wp.array2d(dtype=wp.float64),
        indices: wp.array(dtype=Any),values: wp.array2d(dtype=Any)):
    left=rows[int(indices[k]),0];right=rows[int(indices[k+1]),0]
    fraction=wp.clamp((stage-left)/(right-left),wp.float64(0.0),wp.float64(1.0))
    a=wp.float64(values[k,0]);b=wp.float64(values[k+1,0])
    return a+fraction*(b-a)


@wp.func
def scalar_arm(stage: wp.float64,begin: int,end: int,rows: wp.array2d(dtype=wp.float64),
        indices: wp.array(dtype=Any),values: wp.array2d(dtype=Any)):
    result=wp.float64(values[begin,0])
    if end-begin>1:
        k=scalar_interval(stage,begin,end,rows,indices)
        result=scalar_segment(stage,k,rows,indices,values)
    return result


@wp.func
def cached_scalar_arm(stage: wp.float64,begin: int,end: int,rows: wp.array2d(dtype=wp.float64),
        indices: wp.array(dtype=Any),values: wp.array2d(dtype=Any),
        cache: wp.array2d(dtype=int),connection: int,slot: int):
    # Cache an interval, never its coefficient. Validate against the current
    # range and stage on every use; arbitrary stage changes remain exact.
    # Interior knots belong to their right interval, matching upper_bound.
    result=wp.float64(values[begin,0])
    if end-begin>1:
        k=cache[connection,slot]
        valid=k>=begin and k<end-1
        if valid:
            left=rows[int(indices[k]),0];right=rows[int(indices[k+1]),0]
            valid=(stage>=left or k==begin) and (stage<right or k==end-2)
        if not valid:
            k=scalar_interval(stage,begin,end,rows,indices)
            cache[connection,slot]=k
        result=scalar_segment(stage,k,rows,indices,values)
    return result


@wp.func
def arm(stage: wp.float64, sill: wp.float64, port: int,
        offsets: wp.array(dtype=int), rows: wp.array2d(dtype=wp.float64),
        values: wp.array2d(dtype=Any),column: int,
        indices: wp.array(dtype=Any), ranges: wp.array2d(dtype=int)):
    result = wp.vec2d(0.0)
    if stage > sill:
        if column<0:
            ds=scalar_arm(stage,ranges[port,0],ranges[port,1],rows,indices,values)
            rs=scalar_arm(stage,ranges[port,2],ranges[port,3],rows,indices,values)
            h=stage-sill
            result[0]=ds/h
            result[1]=rs/wp.pow(h,(wp.float64(10.0)/wp.float64(3.0)))
            return result
        lo = offsets[port]
        hi = offsets[port+1]
        end = hi
        while lo < hi:
            mid = (lo+hi)//2
            if rows[mid, 0] <= stage:
                lo = mid+1
            else:
                hi = mid
        k = wp.max(offsets[port], wp.min(end-1, lo-1))
        t = wp.clamp((stage-rows[k, 0])/(rows[k, 1]-rows[k, 0]), wp.float64(0.0), wp.float64(1.0))
        h = stage-sill
        da=wp.float64(values[k,column]);db=wp.float64(values[k,column+1])
        ra=wp.float64(values[k,column+2]);rb=wp.float64(values[k,column+3])
        result[0] = (da+t*(db-da))/h
        result[1] = (ra+t*(rb-ra))/wp.pow(h, (wp.float64(10.0)/wp.float64(3.0)))
    return result


@wp.func
def storage_index(volume: wp.float64, begin: int, end: int,
                  rows: wp.array2d(dtype=wp.float64)):
    lo=begin;hi=end
    while lo<hi:
        mid=(lo+hi)//2
        if rows[mid,1]<=volume:lo=mid+1
        else:hi=mid
    return wp.max(begin,wp.min(end-1,lo-1))


@wp.func
def source_stage_upper(c: int, volume: wp.array(dtype=WP_FLOAT),
        source_area: wp.array(dtype=wp.float64), storage_offsets: wp.array(dtype=int),
        rows: wp.array2d(dtype=wp.float64),state_cell: wp.array(dtype=int),nx: int,
        rain: wp.array2d(dtype=WP_FLOAT),other: wp.array2d(dtype=WP_FLOAT),
        additional_source: wp.array2d(dtype=WP_FLOAT),multiplier: wp.array2d(dtype=WP_FLOAT),
        uniform_rain_bound: wp.float64,use_uniform_rain: int,cap: wp.float64):
    # Nonnegative supply bounds the accepted source volume: infiltration can
    # only remove water. gamma_12 covers at most twelve state-precision roundings
    # on each positive supply path, including depth/volume round trips and FMA.
    cell=state_cell[c];j=cell//nx;i=cell%nx;plan=source_area[c]
    raw=rain[j,i]
    if use_uniform_rain==1:raw=WP_FLOAT(uniform_rain_bound)*multiplier[j,i]
    rate=(wp.float64(wp.max(raw,WP_FLOAT(0.0)))+
        wp.float64(wp.max(other[j,i],WP_FLOAT(0.0)))+
        wp.float64(wp.max(additional_source[j,i],WP_FLOAT(0.0))))
    depth=wp.float64(volume[c])/plan
    added_depth=rate*cap
    minimum_normal=wp.float64(2.2250738585072014e-308)
    if wp.static(WP_FLOAT==wp.float32):minimum_normal=wp.float64(1.1754943508222875e-38)
    if (depth>wp.float64(0.0) and depth<minimum_normal) or (added_depth>wp.float64(0.0) and added_depth<minimum_normal):
        return wp.float64(UNKNOWN_SOURCE_STAGE)  # No relative-error proof in the subnormal range.
    gamma=wp.float64(wp.static(12.0*STATE_EPSILON/(1.0-12.0*STATE_EPSILON)))
    upper=(wp.float64(volume[c])+added_depth*plan)*(wp.float64(1.0)+gamma)
    if not wp.isfinite(upper):return wp.float64(UNKNOWN_SOURCE_STAGE)
    k=storage_index(upper,storage_offsets[c],storage_offsets[c+1],rows)
    stage=rows[k,0]+(upper-rows[k,1])/rows[k,2]
    # Enclose FP64 stage-inversion rounding, including knot-side roundoff.
    operations=wp.float64(4*(storage_offsets[c+1]-storage_offsets[c])+8)
    error=operations*wp.float64(2.220446049250313e-16)
    error=error/(wp.float64(1.0)-error)
    magnitude=wp.max(wp.float64(1.0),wp.max(wp.abs(stage),wp.abs(rows[storage_offsets[c],0])))
    return stage+error*magnitude


@wp.kernel
def upper_stages(volume: wp.array(dtype=WP_FLOAT),source_area: wp.array(dtype=wp.float64),
        offsets: wp.array(dtype=int),rows: wp.array2d(dtype=wp.float64),
        state_cell: wp.array(dtype=int),nx: int,rain: wp.array2d(dtype=WP_FLOAT),
        other: wp.array2d(dtype=WP_FLOAT),additional_source: wp.array2d(dtype=WP_FLOAT),
        multiplier: wp.array2d(dtype=WP_FLOAT),rain_bound: wp.array(dtype=wp.float64),
        use_uniform: int,control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
        result: wp.array(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):return
    c=wp.tid()
    result[c]=source_stage_upper(c,volume,source_area,offsets,rows,state_cell,nx,rain,
        other,additional_source,multiplier,rain_bound[0],use_uniform,control[3])


@wp.func
def tail_arm(stage: wp.float64,sill: wp.float64,port: int,reference: wp.float64,
    offsets: wp.array(dtype=int),rows: wp.array2d(dtype=wp.float64)):
    result=wp.vec2d(0.0)
    if port<0:return result
    x=wp.float64(1.0)/(stage-reference);begin=offsets[port];end=offsets[port+1]
    lo=begin;hi=end
    while lo<hi:
        mid=(lo+hi)//2
        if rows[mid,0]<=x:lo=mid+1
        else:hi=mid
    k=wp.max(begin,wp.min(end-2,lo-1))
    fraction=(x-rows[k,0])/(rows[k+1,0]-rows[k,0]);head=stage-sill
    result[0]=(rows[k,1]+fraction*(rows[k+1,1]-rows[k,1]))/head
    result[1]=(rows[k,2]+fraction*(rows[k+1,2]-rows[k,2]))/wp.pow(head,wp.float64(10.0)/wp.float64(3.0))
    return result


@wp.func
def update_component_stage(c: int, volume: wp.array(dtype=WP_FLOAT), offsets: wp.array(dtype=int),
           rows: wp.array2d(dtype=wp.float64), eta: wp.array(dtype=wp.float64),
           area: wp.array(dtype=wp.float64), maximum: wp.array(dtype=wp.float64),
           maximum_stage: wp.float64, track: int, status: wp.array(dtype=int),
           control: wp.array(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):return
    v=wp.float64(volume[c]);begin=offsets[c]
    k=storage_index(v,begin,offsets[c+1],rows)
    stage=rows[k,0]+(v-rows[k,1])/rows[k,2]
    eta[c]=stage;area[c]=rows[k,2]
    if k>begin and v==rows[k,1]:area[c]=rows[k-1,2]
    if track==1:maximum[c]=wp.max(maximum[c],stage)
    if not wp.isfinite(stage) or not wp.isfinite(v):wp.atomic_max(status,0,4)
    elif v<wp.float64(0.0):wp.atomic_max(status,0,2)
    elif stage>maximum_stage:wp.atomic_max(status,0,1)


@wp.kernel
def stages(volume: wp.array(dtype=WP_FLOAT), offsets: wp.array(dtype=int),
           rows: wp.array2d(dtype=wp.float64), eta: wp.array(dtype=wp.float64),
           area: wp.array(dtype=wp.float64), maximum: wp.array(dtype=wp.float64),
           maximum_stage: wp.float64, track: int, status: wp.array(dtype=int),
           control: wp.array(dtype=wp.float64)):
    update_component_stage(wp.tid(),volume,offsets,rows,eta,area,maximum,maximum_stage,track,status,control)


@wp.func
def boundary_heads(f: int, faces: wp.array(dtype=int), endpoints: wp.array2d(dtype=int),
                   sills: wp.array(dtype=wp.float64), nx: int, x_count: int,
                   type_x: wp.array2d(dtype=int), type_y: wp.array2d(dtype=int),
                   stage_x: wp.array2d(dtype=WP_FLOAT), stage_y: wp.array2d(dtype=WP_FLOAT),
                   heads: wp.array(dtype=wp.float64), kind: wp.array(dtype=int),
                   override: wp.array(dtype=int),override_head: wp.array(dtype=wp.float64),use_override: int,
                   control: wp.array(dtype=wp.float64),use_stage_x: int,use_stage_y: int):
    kind[f]=0
    if endpoints[f,1]<0:
        face=faces[f];code=int(1);head=sills[f]
        if face<x_count:
            j=face//(nx+1);i=face%(nx+1);code=type_x[j,i]
            if code==3 or code==5 or code==10:head=wp.float64(stage_x[j,i])
            if code==3 and use_stage_x==1:head=wp.float64(WP_FLOAT(control[DL_STAGE_QX]))
        else:
            face-=x_count;j=face//nx;i=face%nx;code=type_y[j,i]
            if code==3 or code==4 or code==11:head=wp.float64(stage_y[j,i])
            if code==3 and use_stage_y==1:head=wp.float64(WP_FLOAT(control[DL_STAGE_QY]))
        if use_override==1 and (code==3 or code==4 or code==5):
            if override[f]==1:head=override_head[f]
        if use_override==1:
            if override[f]==2:
                code=3
                head=override_head[f]
        heads[f]=head;kind[f]=code


@wp.kernel
def coefficients(endpoints: wp.array2d(dtype=int), halves: wp.array2d(dtype=int),
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
                 use_stage_x: int,use_stage_y: int):
    f=wp.tid()
    boundary_heads(f,faces,endpoints,sills,nx,x_count,type_x,type_y,stage_x,stage_y,
        heads,kind,override,override_head,use_override,control,use_stage_x,use_stage_y)
    if not hydraulic_step_enabled(control,status):return
    a=endpoints[f,0];b=endpoints[f,1];other=heads[f]
    if b>=0:other=eta[b]
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
        first_d=cached_scalar_arm(stage,ranges[left,0],ranges[left,1],rows,indices,values,interval_cache,f,0)/(stage-sills[f])
        second_d=cached_scalar_arm(stage,ranges[right,0],ranges[right,1],rows,indices,values,interval_cache,f,1)/(stage-sills[f])
        d[f]=first_d+second_d
        return
    first=wp.vec2d(0.0);second=wp.vec2d(0.0)
    if stage>tail_start:
        first=tail_arm(stage,sills[f],tail_halves[f,0],tail_reference,tail_offsets,tail_rows)
        second=tail_arm(stage,sills[f],tail_halves[f,1],tail_reference,tail_offsets,tail_rows)
    elif column<0:
        left=halves[f,0];right=halves[f,1];head=stage-sills[f]
        first[0]=cached_scalar_arm(stage,ranges[left,0],ranges[left,1],rows,indices,values,interval_cache,f,0)/head
        first[1]=cached_scalar_arm(stage,ranges[left,2],ranges[left,3],rows,indices,values,interval_cache,f,2)/wp.pow(head,wp.float64(10.0)/wp.float64(3.0))
        second[0]=cached_scalar_arm(stage,ranges[right,0],ranges[right,1],rows,indices,values,interval_cache,f,1)/head
        second[1]=cached_scalar_arm(stage,ranges[right,2],ranges[right,3],rows,indices,values,interval_cache,f,3)/wp.pow(head,wp.float64(10.0)/wp.float64(3.0))
    else:
        first=arm(stage,sills[f],halves[f,0],offsets,rows,values,column,indices,ranges)
        second=arm(stage,sills[f],halves[f,1],offsets,rows,values,column,indices,ranges)
    d[f]=first[0]+second[0];r[f]=first[1]+second[1];forcing[f]=eta[a]-other


@wp.kernel
def sources(volume: wp.array(dtype=WP_FLOAT), state_cell: wp.array(dtype=int), nx: int,
            source_area: wp.array(dtype=wp.float64), offsets: wp.array(dtype=int),
            rows: wp.array2d(dtype=wp.float64), rain: wp.array2d(dtype=WP_FLOAT),
            other: wp.array2d(dtype=WP_FLOAT), additional_source: wp.array2d(dtype=WP_FLOAT),
            capacity: wp.array2d(dtype=WP_FLOAT), f0: wp.array2d(dtype=WP_FLOAT),
            fd: wp.array2d(dtype=WP_FLOAT), inf_k: wp.array2d(dtype=WP_FLOAT),
            ks: wp.array2d(dtype=WP_FLOAT), psi: wp.array2d(dtype=WP_FLOAT),
            deficit: wp.array2d(dtype=WP_FLOAT), cumulative: wp.array(dtype=WP_FLOAT),
            remaining: wp.array(dtype=WP_FLOAT), excluded: wp.array2d(dtype=int),
            method: int, source_mode: int, cap_active: int, dry: WP_FLOAT,
            control: wp.array(dtype=wp.float64), total: wp.array(dtype=wp.float64),
            applied: wp.array2d(dtype=WP_FLOAT),status: wp.array(dtype=int),
            rain_multiplier: wp.array2d(dtype=WP_FLOAT),use_uniform_rain: int,
            adjacency_offsets: wp.array(dtype=int),adjacency: wp.array(dtype=int),q: wp.array(dtype=WP_FLOAT),
            committed: int, deferred: int, source_ledger: wp.array2d(dtype=wp.float64),
            refresh_stage: int, eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),
            maximum: wp.array(dtype=wp.float64),maximum_stage: wp.float64,
            provisional: int,trial_volume: wp.array(dtype=WP_FLOAT)):
    if not hydraulic_step_enabled(control,status):return
    c=wp.tid();cell=state_cell[c];j=cell//nx;i=cell%nx
    if provisional==2:
        trial_volume[c]=volume[c]
        update_component_stage(c,trial_volume,offsets,rows,eta,area,maximum,maximum_stage,0,status,control)
        return
    plan=source_area[c];v=wp.float64(volume[c]);fraction=WP_FLOAT(0.0);outward=wp.float64(0.0)
    if committed==1:
        for k in range(adjacency_offsets[c],adjacency_offsets[c+1]):
            entry=adjacency[k];sign=wp.float64(1.0)
            if entry<0:sign=wp.float64(-1.0)
            outward+=wp.max(wp.float64(0.0),sign*wp.float64(q[wp.abs(entry)-1]))
    ponded_value=wp.max(wp.float64(0.0),v-control[DL_DT]*outward)
    if ponded_value>wp.float64(0.0):
        k=storage_index(ponded_value,offsets[c],offsets[c+1],rows)
        if k>offsets[c] and ponded_value==rows[k,1]:k-=1
        fraction=WP_FLOAT(rows[k,2]/plan)
    raw_rain=rain[j,i]
    if use_uniform_rain==1:raw_rain=WP_FLOAT(control[DL_RAIN_RATE])*rain_multiplier[j,i]
    h,F,limit,rain_depth,other_depth,gw_depth,uptake=_apply_source_state(
        WP_FLOAT(v/plan),raw_rain,other[j,i],additional_source[j,i],capacity[j,i],
        f0[j,i],fd[j,i],inf_k[j,i],ks[j,i],psi[j,i],deficit[j,i],cumulative[c],
        method,1,fraction,WP_FLOAT(control[1]),source_mode,dry,WP_FLOAT(control[DL_DT]*outward/plan),committed,
        excluded[j,i],cap_active,remaining[c])
    # Bound the accepted source sink in the same arithmetic as the source
    # supply. Split rain/ponding uptake can exceed their rounded sum by an ulp;
    # cap the flux and its soil/cap ledger before committing, not stored water
    # after transport. This prevents tiny negative dry-state volumes.
    supply=WP_FLOAT(v/plan)+rain_depth+other_depth+gw_depth
    uptake=wp.min(uptake,supply)
    h=supply-uptake
    accepted_volume=WP_FLOAT(wp.float64(h)*plan)
    uptake_volume=wp.float64(uptake)*plan
    if committed==1:
        # Retain the actual volume representation while reserving accepted
        # transport. A depth round-trip may otherwise move V below dt*Q.
        supplied_volume=v+(wp.float64(rain_depth)+wp.float64(other_depth)+wp.float64(gw_depth))*plan
        # One relative state ulp also covers the subsequent state cast and
        # fused multiply-add in continuity at an exactly exhausted donor.
        reserved=control[DL_DT]*outward*(wp.float64(1.0)+wp.float64(STATE_EPSILON))
        # Limit the actual sink in conservative V units. Depth-space ulp
        # corrections underflow during complete recession (e.g. V~1e-315 m3)
        # and cannot reliably prevent a negative subnormal residual.
        uptake_volume=wp.min(uptake_volume,wp.max(wp.float64(0.0),supplied_volume-reserved))
        accepted_volume=WP_FLOAT(supplied_volume-uptake_volume)
        uptake=WP_FLOAT(uptake_volume/plan)
    if method==1 or method==3:F=wp.max(cumulative[c],WP_FLOAT(0.0))+uptake
    if cap_active==1 and excluded[j,i]!=1:
        limit=wp.max(WP_FLOAT(0.0),remaining[c]-uptake)
    if provisional==1:
        trial_volume[c]=accepted_volume
        update_component_stage(c,trial_volume,offsets,rows,eta,area,maximum,maximum_stage,0,status,control)
        return
    volume[c]=accepted_volume;cumulative[c]=F;remaining[c]=limit
    applied[c,0]=rain_depth;applied[c,1]=other_depth;applied[c,2]=uptake
    rain_volume=wp.float64(rain_depth)*plan
    other_volume=wp.float64(other_depth)*plan
    additional_source_volume=wp.float64(gw_depth)*plan
    if deferred==1:
        # One owner per component: no contended domain-wide atomics. Reduce
        # accepted FP64 volumes at every existing device-loop observation.
        source_ledger[c,0]+=rain_volume
        source_ledger[c,1]+=uptake_volume
        source_ledger[c,2]+=other_volume
        source_ledger[c,3]+=additional_source_volume
    else:
        if rain_volume!=wp.float64(0.0):wp.atomic_add(total,0,rain_volume)
        if uptake_volume!=wp.float64(0.0):wp.atomic_add(total,1,uptake_volume)
        if other_volume!=wp.float64(0.0):wp.atomic_add(total,2,other_volume)
        if additional_source_volume!=wp.float64(0.0):wp.atomic_add(total,5,additional_source_volume)
    # Initialization/prior continuity already inverted this exact volume.
    # Rain completely absorbed by infiltration changes soil, not eta/area.
    if refresh_stage==1 and wp.float64(accepted_volume)!=v:
        update_component_stage(c,volume,offsets,rows,eta,area,maximum,maximum_stage,0,status,control)


@wp.kernel
def book_source_totals(sums: wp.array2d(dtype=wp.float64),total: wp.array(dtype=wp.float64)):
    index=wp.tid();destination=index
    if index==3:destination=5
    total[destination]+=sums[0,index]


@wp.kernel
def timestep(offsets: wp.array(dtype=int), adjacent: wp.array(dtype=int),
             storage_offsets: wp.array(dtype=int), storage: wp.array2d(dtype=wp.float64),
             d: wp.array(dtype=wp.float64), q: wp.array(dtype=WP_FLOAT),
             eta: wp.array(dtype=wp.float64), area: wp.array(dtype=wp.float64),
             state_cell: wp.array(dtype=int), nx: int,
             source_area: wp.array(dtype=wp.float64), rain: wp.array2d(dtype=WP_FLOAT),
             other: wp.array2d(dtype=WP_FLOAT), additional_source: wp.array2d(dtype=WP_FLOAT),
             gravity: wp.float64, cfl: wp.float64, fraction: wp.float64,
             dry: wp.float64, control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
             rain_multiplier: wp.array2d(dtype=WP_FLOAT),use_uniform_rain: int,
             local_dt: wp.array(dtype=wp.float64),wave_only: int,track_limits: int,
             limits: wp.array2d(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):return
    c=wp.tid();degree=wp.float64(0.0);demand=wp.float64(0.0)
    for k in range(offsets[c],offsets[c+1]):
        f=wp.abs(adjacent[k])-1
        if d[f]>wp.float64(0.0):degree+=wp.float64(1.0)/d[f]
        demand+=wp.abs(wp.float64(q[f]))
    dt=control[3]
    wave=wp.float64(1.e100)
    if degree>wp.float64(0.0):wave=cfl*wp.sqrt(wp.float64(2.0)*area[c]/(gravity*degree))
    dt=wp.min(dt,wave)
    cell=state_cell[c];j=cell//nx;i=cell%nx
    raw_rain=rain[j,i]
    if use_uniform_rain==1:raw_rain=WP_FLOAT(control[DL_RAIN_RATE])*rain_multiplier[j,i]
    rate=demand+(wp.float64(wp.max(raw_rain,WP_FLOAT(0.0)))+
        wp.float64(wp.max(other[j,i],WP_FLOAT(0.0)))+
        wp.float64(wp.max(additional_source[j,i],WP_FLOAT(0.0))))*source_area[c]
    change=wp.float64(1.e100)
    if rate>wp.float64(0.0):
        depth=wp.max(eta[c]-storage[storage_offsets[c],0],dry)
        change=fraction*depth*area[c]/rate
        if wave_only==0:dt=wp.min(dt,change)
    if track_limits==1:
        limits[c,0]=wave;limits[c,1]=change;limits[c,2]=demand
        limits[c,3]=rate-demand;limits[c,4]=eta[c];limits[c,5]=area[c]
    wp.atomic_min(control,1,dt)
    local_dt[c]=dt


@wp.func
def connection_discharge(d: wp.float64,r: wp.float64,forcing: wp.float64,q: WP_FLOAT,
    previous_d: wp.float64,dt: wp.float64,gravity: wp.float64,kinetic: int,
    donor_stage: wp.float64,sill: wp.float64,dry: wp.float64,free: int,exact: int):
    value=wp.float64(0.0)
    if d>wp.float64(0.0):
        old=wp.float64(q)
        if kinetic==1 and previous_d>wp.float64(0.0):old*=wp.sqrt(previous_d/d)
        if donor_stage<=sill+dry:old=wp.float64(0.0)
        if exact==1:value=frozen_discharge(old,gravity*forcing/d,gravity*r/d,dt)
        else:value=(d*old+gravity*dt*forcing)/(d+gravity*dt*r*wp.abs(old))
        if free==1:value=wp.max(wp.float64(0.0),value)
    return WP_FLOAT(value)


@wp.kernel
def momentum(d: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64),
             forcing: wp.array(dtype=wp.float64), q: wp.array(dtype=WP_FLOAT),
             previous_d: wp.array(dtype=wp.float64), proposed: wp.array(dtype=WP_FLOAT),
             control: wp.array(dtype=wp.float64), gravity: wp.float64, status: wp.array(dtype=int),
             kinetic: int, endpoints: wp.array2d(dtype=int), sills: wp.array(dtype=wp.float64),
             eta: wp.array(dtype=wp.float64), heads: wp.array(dtype=wp.float64),
             kind: wp.array(dtype=int), dry: wp.float64,exact: int):
    f=wp.tid()
    if not hydraulic_step_enabled(control,status):return
    donor=eta[endpoints[f,0]]
    if q[f]<WP_FLOAT(0.0):
        donor=heads[f]
        if endpoints[f,1]>=0:donor=eta[endpoints[f,1]]
    free=int(endpoints[f,1]<0 and (kind[f]==2 or (kind[f]>=6 and kind[f]<=9)))
    proposed[f]=connection_discharge(d[f],r[f],forcing[f],q[f],previous_d[f],control[1],gravity,
        kinetic,donor,sills[f],dry,free,exact)
    if exact==1 and not wp.isfinite(proposed[f]):wp.atomic_max(status,0,4)
    previous_d[f]=d[f]


@wp.kernel
def demands(volume: wp.array(dtype=WP_FLOAT), offsets: wp.array(dtype=int),
            adjacent: wp.array(dtype=int), proposed: wp.array(dtype=WP_FLOAT),
            alpha: wp.array(dtype=wp.float64), control: wp.array(dtype=wp.float64), safety: wp.float64,
            status: wp.array(dtype=int),kind: wp.array(dtype=int),
            alpha_free: wp.array(dtype=wp.float64),mode: int):
    if not hydraulic_step_enabled(control,status):return
    c=wp.tid();demand=wp.float64(0.0);free=wp.float64(0.0);other=wp.float64(0.0)
    for k in range(offsets[c],offsets[c+1]):
        entry=adjacent[k];sign=wp.float64(1.0)
        if entry<0:sign=wp.float64(-1.0)
        f=wp.abs(entry)-1
        outward=wp.max(sign*wp.float64(proposed[f]),wp.float64(0.0));demand+=outward
        if kind[f]==2:free+=outward
        else:other+=outward
    available=wp.float64(volume[c]);alpha[c]=wp.float64(1.0)
    if control[1]*demand>available:alpha[c]=available/(control[1]*demand*safety)
    alpha_free[c]=alpha[c]
    if mode==1:
        alpha_free[c]=wp.float64(1.0)
        if control[1]*free>available:alpha_free[c]=available/(control[1]*free*safety)
        leftover=wp.max(wp.float64(0.0),available-control[1]*free*alpha_free[c])
        alpha[c]=wp.float64(1.0)
        if control[1]*other>leftover:alpha[c]=leftover/(control[1]*other*safety)
    elif mode==2:alpha_free[c]=wp.float64(1.0)


@wp.kernel
def limit(endpoints: wp.array2d(dtype=int), proposed: wp.array(dtype=WP_FLOAT),
          alpha: wp.array(dtype=wp.float64), q: wp.array(dtype=WP_FLOAT),
          control: wp.array(dtype=wp.float64), total: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
          kind: wp.array(dtype=int),alpha_free: wp.array(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):return
    f=wp.tid();donor=endpoints[f,0]
    if proposed[f]<WP_FLOAT(0.0):donor=endpoints[f,1]
    value=wp.float64(proposed[f])
    if donor>=0:
        if kind[f]==2:value*=alpha_free[donor]
        else:value*=alpha[donor]
    q[f]=WP_FLOAT(value)
    if endpoints[f,1]<0:
        amount=control[1]*wp.float64(q[f])
        if amount>=wp.float64(0.0):wp.atomic_add(total,4,amount)
        else:wp.atomic_add(total,3,-amount)


@wp.kernel
def continuity(volume: wp.array(dtype=WP_FLOAT), offsets: wp.array(dtype=int),
               adjacent: wp.array(dtype=int), q: wp.array(dtype=WP_FLOAT),
               control: wp.array(dtype=wp.float64), status: wp.array(dtype=int),
               refresh_stage: int,storage_offsets: wp.array(dtype=int),storage: wp.array2d(dtype=wp.float64),
               eta: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),
               maximum: wp.array(dtype=wp.float64),maximum_stage: wp.float64,track: int):
    if not hydraulic_step_enabled(control,status):return
    c=wp.tid();net=wp.float64(0.0)
    for k in range(offsets[c],offsets[c+1]):
        entry=adjacent[k];sign=wp.float64(1.0)
        if entry<0:sign=wp.float64(-1.0)
        net-=sign*wp.float64(q[wp.abs(entry)-1])
    old_volume=volume[c]
    value=wp.float64(old_volume)+control[1]*net
    if value<wp.float64(0.0):wp.atomic_max(status,0,2)
    if not wp.isfinite(value):wp.atomic_max(status,0,4)
    volume[c]=WP_FLOAT(value)
    if refresh_stage>0:
        # After-momentum sources have changed V without refreshing eta; that
        # path must invert the final storage even when transport net is zero.
        if refresh_stage==2 or volume[c]!=old_volume:
            update_component_stage(c,volume,storage_offsets,storage,eta,area,maximum,maximum_stage,track,status,control)
        elif track==1:
            maximum[c]=wp.max(maximum[c],eta[c])


@wp.kernel
def coarse_state(offsets: wp.array(dtype=int), volume: wp.array(dtype=WP_FLOAT),
                 source_area: wp.array(dtype=wp.float64), cumulative: wp.array(dtype=WP_FLOAT),
                 remaining: wp.array(dtype=WP_FLOAT), applied: wp.array2d(dtype=WP_FLOAT),
                 cell_area: wp.float64, depth: wp.array2d(dtype=WP_FLOAT),
                 soil: wp.array2d(dtype=WP_FLOAT), cap: wp.array2d(dtype=WP_FLOAT),
                 rain: wp.array2d(dtype=wp.float64), other: wp.array2d(dtype=wp.float64),
                 infil: wp.array2d(dtype=wp.float64), full_fields: int,
                 active: wp.array2d(dtype=int),maximum: wp.array2d(dtype=WP_FLOAT),
                 diagnostic_maximum: wp.array2d(dtype=WP_FLOAT),track_maximum: int,track_diagnostic: int):
    j,i=wp.tid();cell=j*depth.shape[1]+i;v=wp.float64(0.0);f=wp.float64(0.0)
    limit=wp.float64(0.0);r=wp.float64(0.0);o=wp.float64(0.0);u=wp.float64(0.0)
    for c in range(offsets[cell],offsets[cell+1]):
        v+=wp.float64(volume[c])
        if full_fields==1:
            plan=source_area[c];f+=wp.float64(cumulative[c])*plan
            limit+=wp.float64(remaining[c])*plan;r+=wp.float64(applied[c,0])*plan
            o+=wp.float64(applied[c,1])*plan;u+=wp.float64(applied[c,2])*plan
    depth[j,i]=WP_FLOAT(v/cell_area)
    if active[j,i]==1:
        if track_maximum==1:maximum[j,i]=wp.max(maximum[j,i],depth[j,i])
        if track_diagnostic==1:diagnostic_maximum[j,i]=wp.max(diagnostic_maximum[j,i],depth[j,i])
    if full_fields==0:return
    soil[j,i]=WP_FLOAT(f/cell_area)
    cap[j,i]=WP_FLOAT(limit/cell_area);rain[j,i]=r/cell_area
    other[j,i]=o/cell_area;infil[j,i]=u/cell_area


@wp.kernel
def coarse_flux(offsets: wp.array(dtype=int), members: wp.array(dtype=int), signs: wp.array(dtype=int),
                q: wp.array(dtype=WP_FLOAT), width: wp.float64, nx: int, x_count: int,
                qx: wp.array2d(dtype=WP_FLOAT), qy: wp.array2d(dtype=WP_FLOAT)):
    face=wp.tid();value=wp.float64(0.0)
    for k in range(offsets[face],offsets[face+1]):
        f=members[k];value+=wp.float64(signs[f])*wp.float64(q[f])
    if face<x_count:qx[face//(nx+1),face%(nx+1)]=WP_FLOAT(value/width)
    else:
        f=face-x_count;qy[f//nx,f%nx]=WP_FLOAT(value/width)


@wp.kernel
def set_control(control: wp.array(dtype=wp.float64), dt: wp.float64):
    control[1]=dt;control[3]=dt


@wp.kernel
def device_finalize(
    control: wp.array(dtype=wp.float64),
    end_time: wp.float64,
    minimum_dt: wp.float64,
    maximum_dt: wp.float64,
    adaptive_dt_growth_factor: wp.float64,
    use_acc_timestep: int,
    cfl_number: WP_FLOAT,
    dx: WP_FLOAT,
    gravity: WP_FLOAT,
    dry_depth: WP_FLOAT,
    hyeto_times: wp.array(dtype=wp.float64),
    hyeto_rates: wp.array(dtype=wp.float64),
    use_hyetograph: int,
    storm_duration: wp.float64,
    sample_at_half_step: int,
    stage_times_qx: wp.array(dtype=wp.float64),
    stage_values_qx: wp.array(dtype=wp.float64),
    use_stage_qx: int,
    stage_times_qy: wp.array(dtype=wp.float64),
    stage_values_qy: wp.array(dtype=wp.float64),
    use_stage_qy: int,
    checkpoint_times: wp.array(dtype=wp.float64),
    checkpoint_count: int,
):
    """Production controller, shared with component hydraulics."""
    _device_timestep_finalize_from_cfl(
        control, control[DL_DT], end_time, minimum_dt, maximum_dt, adaptive_dt_growth_factor, use_acc_timestep, cfl_number, dx, gravity, dry_depth, hyeto_times, hyeto_rates, use_hyetograph, storm_duration, sample_at_half_step, stage_times_qx, stage_values_qx, use_stage_qx, stage_times_qy, stage_values_qy, use_stage_qy, checkpoint_times, checkpoint_count
    )


@wp.kernel
def prepare_device_timestep(control: wp.array(dtype=wp.float64), maximum_dt: wp.float64,
        times: wp.array(dtype=wp.float64),rates: wp.array(dtype=wp.float64),use_hyetograph: int,
        rain_bound: wp.array(dtype=wp.float64),growth: wp.float64,storm_duration: wp.float64):
    if control[DL_DONE]!=wp.float64(0.0) or control[DL_ERROR]!=wp.float64(0.0):return
    cap=maximum_dt
    if growth>=wp.float64(1.0) and control[DL_GROWTH_REFERENCE]>wp.float64(0.0):
        cap=wp.min(cap,control[DL_GROWTH_REFERENCE]*growth)
    control[1]=cap;control[3]=cap
    if use_hyetograph==1 and control[DL_TIME]>=storm_duration:
        rain_bound[0]=wp.float64(0.0)
    elif use_hyetograph==1:
        # Include all knots in the fixed forecast interval, not just endpoints:
        # maximum_dt may cross a hyetograph node before final time clipping.
        first=_device_piecewise_linear_interp(times,rates,control[DL_TIME],control)
        last=_device_piecewise_linear_interp(times,rates,control[DL_TIME]+cap,control)
        bound=wp.max(first,last)
        lo=int(0);hi=times.shape[0]
        while lo<hi:
            mid=(lo+hi)//2
            if times[mid]<=control[DL_TIME]:lo=mid+1
            else:hi=mid
        while lo<times.shape[0]:
            if times[lo]>=control[DL_TIME]+cap:break
            bound=wp.max(bound,rates[lo]);lo+=1
        rain_bound[0]=bound*(wp.float64(1.0)+wp.float64(8.881784197001252e-16))


@wp.kernel
def checkpoint_fields(depth: wp.array2d(dtype=WP_FLOAT),qx: wp.array2d(dtype=WP_FLOAT),
    offsets: wp.array(dtype=int),volume: wp.array(dtype=WP_FLOAT),
    control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
    outlet_rows: wp.array(dtype=int),field_indices: wp.array(dtype=int),
    outlet_out: wp.array2d(dtype=wp.float64),volume_out: wp.array(dtype=wp.float64),
    depth_out: wp.array3d(dtype=WP_FLOAT),checkpoint_count: int):
    if not hydraulic_step_enabled(control,status):return
    index=int(control[DL_CHECKPOINT_INDEX])
    if control[DL_RECORD_CHECKPOINT]==wp.float64(0.0) or index>=checkpoint_count:return
    j,i=wp.tid();cell=j*depth.shape[1]+i;total=wp.float64(0.0)
    for c in range(offsets[cell],offsets[cell+1]):total+=wp.float64(volume[c])
    wp.atomic_add(volume_out,index,total)
    if i==depth.shape[1]-1 and outlet_rows[j]==1:
        outlet_out[index,j]=wp.float64(wp.max(qx[j,depth.shape[1]],WP_FLOAT(0.0)))
    field_index=field_indices[index]
    if field_index>=0:depth_out[field_index,j,i]=depth[j,i]


@wp.kernel
def finish_device_step(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
    history: wp.array(dtype=wp.float64),history_capacity: int,record_history: int,
    checkpoint_times_out: wp.array(dtype=wp.float64),checkpoint_count: int):
    if status[0]!=0:
        control[DL_ERROR]=wp.float64(4.0);control[DL_DONE]=wp.float64(1.0)
        return
    if control[DL_ERROR]!=wp.float64(0.0):
        control[DL_DONE]=wp.float64(1.0)
        return
    if control[DL_DONE]!=wp.float64(0.0):return
    _device_commit_clock(control,history,history_capacity,record_history)
    index=int(control[DL_CHECKPOINT_INDEX])
    if control[DL_RECORD_CHECKPOINT]!=wp.float64(0.0) and index<checkpoint_count:
        checkpoint_times_out[index]=control[DL_TIME]


@wp.kernel
def free_outflow_diagnostics(cell_offsets: wp.array(dtype=int),offsets: wp.array(dtype=int),
    adjacent: wp.array(dtype=int),kind: wp.array(dtype=int),proposed: wp.array(dtype=WP_FLOAT),
    accepted: wp.array(dtype=WP_FLOAT),control: wp.array(dtype=wp.float64),
    status: wp.array(dtype=int),diagnostics: wp.array(dtype=wp.float64)):
    if not hydraulic_step_enabled(control,status):return
    cell=wp.tid();raw=wp.float64(0.0);limited=wp.float64(0.0)
    for c in range(cell_offsets[cell],cell_offsets[cell+1]):
        for k in range(offsets[c],offsets[c+1]):
            entry=adjacent[k];f=wp.abs(entry)-1
            if entry>0 and kind[f]==2:
                raw+=wp.max(wp.float64(0.0),wp.float64(proposed[f]))
                limited+=wp.max(wp.float64(0.0),wp.float64(accepted[f]))
    if raw>wp.float64(0.0):
        factor=limited/raw
        wp.atomic_add(diagnostics,0,control[DL_DT]*raw)
        wp.atomic_add(diagnostics,1,control[DL_DT]*limited)
        wp.atomic_add(diagnostics,2,wp.float64(1.0))
        wp.atomic_add(diagnostics,3,factor)
        if limited<raw:wp.atomic_add(diagnostics,4,wp.float64(1.0))
        wp.atomic_min(diagnostics,5,factor)


@wp.kernel
def structure_endpoints(components: wp.array(dtype=int),volume: wp.array(dtype=WP_FLOAT),
    stage: wp.array(dtype=wp.float64),endpoint_stage: wp.array(dtype=WP_FLOAT),
    endpoint_storage: wp.array(dtype=WP_FLOAT),requests: wp.array(dtype=WP_FLOAT)):
    k=wp.tid();c=components[k]
    endpoint_stage[k]=WP_FLOAT(stage[c]);endpoint_storage[k]=volume[c];requests[k]=WP_FLOAT(0.0)


@wp.kernel
def structure_demand(components: wp.array(dtype=int),volume: wp.array(dtype=WP_FLOAT),
    offsets: wp.array(dtype=int),adjacency: wp.array(dtype=int),request: wp.array(dtype=WP_FLOAT),
    safety: wp.float64,alpha: wp.array(dtype=wp.float64)):
    k=wp.tid();outward=wp.float64(0.0)
    for index in range(offsets[k],offsets[k+1]):
        entry=adjacency[index];sign=wp.float64(1.0)
        if entry<0:sign=wp.float64(-1.0)
        outward+=wp.max(wp.float64(0.0),sign*wp.float64(request[wp.abs(entry)-1]))
    available=wp.float64(volume[components[k]]);alpha[k]=wp.float64(1.0)
    if outward>available:alpha[k]=available/(outward*safety)


@wp.kernel
def structure_accept(slot_a: wp.array(dtype=int),slot_b: wp.array(dtype=int),
    request: wp.array(dtype=WP_FLOAT),alpha: wp.array(dtype=wp.float64),
    positive: wp.array(dtype=WP_FLOAT),negative: wp.array(dtype=WP_FLOAT),net: wp.array(dtype=WP_FLOAT),
    max_q: wp.array(dtype=WP_FLOAT),q: wp.array(dtype=WP_FLOAT),hits: wp.array(dtype=int),
    actual: wp.array(dtype=WP_FLOAT)):
    s=wp.tid();value=request[s];donor=slot_a[s]
    if value<WP_FLOAT(0.0):donor=slot_b[s]
    accepted=WP_FLOAT(wp.float64(value)*alpha[donor]);actual[s]=accepted
    # Preserve traditional diagnostic semantics: instantaneous Q and maximum
    # Q are the hydraulic request; transferred volumes use the donor-limited Q.
    if value>WP_FLOAT(0.0):positive[s]+=accepted
    elif value<WP_FLOAT(0.0):negative[s]+=accepted
    net[s]+=accepted
    if value!=WP_FLOAT(0.0):max_q[s]=wp.max(max_q[s],wp.abs(q[s]))
    if wp.abs(accepted)+WP_FLOAT(1.0e-12)<wp.abs(value):hits[s]+=1


@wp.kernel
def structure_continuity(components: wp.array(dtype=int),volume: wp.array(dtype=WP_FLOAT),
    offsets: wp.array(dtype=int),adjacency: wp.array(dtype=int),actual: wp.array(dtype=WP_FLOAT),
    status: wp.array(dtype=int)):
    k=wp.tid();change=wp.float64(0.0)
    for index in range(offsets[k],offsets[k+1]):
        entry=adjacency[index];sign=wp.float64(1.0)
        if entry<0:sign=wp.float64(-1.0)
        change-=sign*wp.float64(actual[wp.abs(entry)-1])
    c=components[k];value=wp.float64(volume[c])+change
    if value<wp.float64(0.0):wp.atomic_max(status,0,2)
    if not wp.isfinite(value):wp.atomic_max(status,0,4)
    volume[c]=WP_FLOAT(value)


@wp.kernel
def project_hook_flux(offsets: wp.array(dtype=int),members: wp.array(dtype=int),signs: wp.array(dtype=int),
    endpoints: wp.array2d(dtype=int),d: wp.array(dtype=wp.float64),q: wp.array(dtype=WP_FLOAT),
    qx: wp.array2d(dtype=WP_FLOAT),qy: wp.array2d(dtype=WP_FLOAT),nx: int,x_count: int,
    width: wp.float64,post_limiter: int,control: wp.array(dtype=wp.float64),
    totals: wp.array(dtype=wp.float64),status: wp.array(dtype=int)):
    """Lift a coarse hook's actual change into existing active contact groups.

    Pre-limiter changes minimize incremental kinetic energy. Post-limiter
    hooks may only rescale the accepted signed group vector towards zero,
    preserving its component donors and their availability. Unchanged coarse
    output is a strict no-op, including its FP32 aggregation roundoff.
    """
    if not hydraulic_step_enabled(control,status):return
    face=wp.tid();requested=WP_FLOAT(0.0)
    if face<x_count:requested=qx[face//(nx+1),face%(nx+1)]
    else:
        k=face-x_count;requested=qy[k//nx,k%nx]
    current=wp.float64(0.0);weight=wp.float64(0.0)
    for index in range(offsets[face],offsets[face+1]):
        f=members[index];current+=wp.float64(signs[f])*wp.float64(q[f])
        if d[f]>wp.float64(0.0):weight+=wp.float64(1.0)/d[f]
    displayed=WP_FLOAT(current/width)
    if requested==displayed:return
    if not wp.isfinite(requested) or weight<=wp.float64(0.0):
        wp.atomic_max(status,0,5);return
    target=wp.float64(requested)*width;factor=wp.float64(1.0)
    if post_limiter==1:
        if displayed==WP_FLOAT(0.0):wp.atomic_max(status,0,5);return
        factor=wp.float64(requested)/wp.float64(displayed)
        if factor<wp.float64(0.0) or factor>wp.float64(1.0):
            wp.atomic_max(status,0,5);return
    for index in range(offsets[face],offsets[face+1]):
        f=members[index];old=wp.float64(q[f]);value=old
        if post_limiter==1:value=old*factor
        elif d[f]>wp.float64(0.0):value+=wp.float64(signs[f])*(target-current)/(d[f]*weight)
        q[f]=WP_FLOAT(value)
        if post_limiter==1 and endpoints[f,1]<0:
            # The ordinary limiter booked old accepted Q. Replace that ledger
            # by the hook's accepted value before continuity commits volume.
            wp.atomic_add(totals,4,control[DL_DT]*(wp.max(wp.float64(q[f]),wp.float64(0.0))-wp.max(old,wp.float64(0.0))))
            wp.atomic_add(totals,3,control[DL_DT]*(wp.max(-wp.float64(q[f]),wp.float64(0.0))-wp.max(-old,wp.float64(0.0))))


@wp.kernel
def lts_cell_dt(offsets: wp.array(dtype=int),local_dt: wp.array(dtype=wp.float64),
    cell_dt: wp.array2d(dtype=WP_FLOAT),minimum: wp.array(dtype=WP_FLOAT),cap: wp.float64):
    j,i=wp.tid();cell=j*cell_dt.shape[1]+i;value=cap
    for c in range(offsets[cell],offsets[cell+1]):value=wp.min(value,local_dt[c])
    cell_dt[j,i]=WP_FLOAT(value)
    wp.atomic_min(minimum,0,WP_FLOAT(value))


@wp.kernel
def lts_finalize(control: wp.array(dtype=wp.float64),cap: wp.float64,minimum: wp.float64,
    levels: int,schedule: wp.array(dtype=wp.float64)):
    _lts_finalize_schedule_from_dt(control[DL_DT],cap,minimum,levels,schedule)


@wp.kernel
def lts_face_dt(cell_dt: wp.array2d(dtype=WP_FLOAT),type_x: wp.array2d(dtype=int),
    type_y: wp.array2d(dtype=int),dt_x: wp.array2d(dtype=WP_FLOAT),dt_y: wp.array2d(dtype=WP_FLOAT),
    cap: WP_FLOAT):
    j,i=wp.tid();ny=cell_dt.shape[0];nx=cell_dt.shape[1]
    if j<ny:
        value=cap
        if i>0:value=wp.min(value,cell_dt[j,i-1])
        if i<nx:value=wp.min(value,cell_dt[j,i])
        dt_x[j,i]=value
    if i<nx:
        value=cap
        if j>0:value=wp.min(value,cell_dt[j-1,i])
        if j<ny:value=wp.min(value,cell_dt[j,i])
        dt_y[j,i]=value


@wp.kernel
def lts_connection_levels(endpoints: wp.array2d(dtype=int),state_cell: wp.array(dtype=int),
    cell_levels: wp.array2d(dtype=int),levels: wp.array(dtype=int)):
    f=wp.tid();a=state_cell[endpoints[f,0]];nx=cell_levels.shape[1]
    level=cell_levels[a//nx,a%nx]
    if endpoints[f,1]>=0:
        b=state_cell[endpoints[f,1]];level=wp.min(level,cell_levels[b//nx,b%nx])
    levels[f]=level


@wp.kernel
def lts_momentum(d: wp.array(dtype=wp.float64),r: wp.array(dtype=wp.float64),
    forcing: wp.array(dtype=wp.float64),q: wp.array(dtype=WP_FLOAT),previous_d: wp.array(dtype=wp.float64),
    levels: wp.array(dtype=int),control: wp.array(dtype=wp.float64),gravity: wp.float64,status: wp.array(dtype=int),
    kinetic: int,endpoints: wp.array2d(dtype=int),sills: wp.array(dtype=wp.float64),
    eta: wp.array(dtype=wp.float64),heads: wp.array(dtype=wp.float64),kind: wp.array(dtype=int),dry: wp.float64,
    updates: wp.array(dtype=int),exact: int):
    if not hydraulic_step_enabled(control,status):return
    f=wp.tid();period=1<<levels[f]
    if int(control[DL_COUNTER])%period!=0:return
    donor=eta[endpoints[f,0]]
    if q[f]<WP_FLOAT(0.0):
        donor=heads[f]
        if endpoints[f,1]>=0:donor=eta[endpoints[f,1]]
    free=int(endpoints[f,1]<0 and (kind[f]==2 or (kind[f]>=6 and kind[f]<=9)))
    q[f]=connection_discharge(d[f],r[f],forcing[f],q[f],previous_d[f],
        control[DL_DT]*wp.float64(period),gravity,kinetic,donor,sills[f],dry,free,exact)
    previous_d[f]=d[f];wp.atomic_add(updates,0,1)


@wp.kernel
def lts_forcing(control: wp.array(dtype=wp.float64),dt: wp.float64,rain: wp.float64):
    control[DL_DT]=dt;control[DL_RAIN_RATE]=rain


@wp.kernel
def lts_accumulate_sources(applied: wp.array2d(dtype=WP_FLOAT),accumulated: wp.array2d(dtype=WP_FLOAT)):
    c,k=wp.tid();accumulated[c,k]+=applied[c,k]


@wp.kernel
def lts_next_tick(control: wp.array(dtype=wp.float64)):
    control[DL_COUNTER]+=wp.float64(1.0)


@wp.kernel
def lts_graph_prepare(control: wp.array(dtype=wp.float64),schedule: wp.array(dtype=wp.float64),
    forcing: wp.array(dtype=wp.float64)):
    dt=schedule[LTS_CTRL_BASE_DT]
    control[DL_DT]=dt
    control[DL_RAIN_RATE]=forcing[0]+forcing[1]*(control[DL_COUNTER]+wp.float64(.5))*dt


@wp.kernel
def lts_graph_finish(control: wp.array(dtype=wp.float64),tick: wp.array(dtype=int),
    condition: wp.array(dtype=int),status: wp.array(dtype=int)):
    tick[0]=int(control[DL_COUNTER]);condition[0]-=1
    if status[0]!=0:condition[0]=0


@wp.kernel
def coarse_predictor_volume(offsets: wp.array(dtype=int),source_area: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=WP_FLOAT),coarse: wp.array2d(dtype=WP_FLOAT),cell_area: wp.float64,
    result: wp.array(dtype=WP_FLOAT),status: wp.array(dtype=int)):
    j,i=wp.tid();cell=j*coarse.shape[1]+i
    if offsets[cell]==offsets[cell+1]:return
    requested=wp.float64(coarse[j,i])*cell_area;current=wp.float64(0.0)
    if not wp.isfinite(requested) or requested<wp.float64(0.0):
        wp.atomic_max(status,0,4);return
    for c in range(offsets[cell],offsets[cell+1]):current+=wp.float64(volume[c])
    for c in range(offsets[cell],offsets[cell+1]):
        predicted=wp.float64(volume[c])
        if requested>=current:predicted+=(requested-current)*source_area[c]/cell_area
        elif current>wp.float64(0.0):predicted*=requested/current
        result[c]=WP_FLOAT(predicted)
