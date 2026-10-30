"""Component-owned hydraulic storage and certified coefficient tables.

Production-owned Phase 8/9 preprocessing; equations and defaults are retained.
No dependency on the research scripts under tools is permitted here.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import hashlib
import numpy as np
from scipy.ndimage import label
from floodwarp.enhanced_reduction import (
    PortGroup, reduce_cell, boundary_nodes, raster_edges, wet_components,
    port_coefficients, transport_star,
)

class UnsupportedGeometryError(ValueError):
    """Geometry needs hydraulic states or table bounds outside this prototype."""



class StageRangeError(ValueError):
    """Hydraulic state exceeds the preprocessed stage range."""



class NumericalFailure(ValueError):
    """Nonfinite or negative hydraulic state, not unsupported terrain."""



class TimestepCollapse(NumericalFailure):
    """Hydraulic timestep is too small for meaningful forward progress."""



@dataclass
class DynamicGeometry:
    shape: tuple
    factor: int
    resolution: float
    maximum_stage: float
    formulation: str
    endpoints: np.ndarray
    halves: np.ndarray
    sill: np.ndarray
    width: np.ndarray
    external_side: np.ndarray
    port_sill: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray
    storage_offsets: np.ndarray
    storage_rows: np.ndarray
    adjacency_offsets: np.ndarray
    adjacency: np.ndarray
    preprocess_s: float
    diagnostics: dict
    coefficient_values: np.ndarray | None = None
    source_area: np.ndarray | None = None
    pixel_owner: np.ndarray | None = None
    state_cell: np.ndarray | None = None
    cell_component_offsets: np.ndarray | None = None
    coefficient_stage_index: np.ndarray | None = None
    coefficient_ranges: np.ndarray | None = None
    connection_keys: np.ndarray | None = None
    tail_offsets: np.ndarray | None = None
    tail_rows: np.ndarray | None = None
    tail_halves: np.ndarray | None = None
    tail_reference: float = 0.
    tail_start: float = np.inf

    def memory_bytes(self):
        return {name: value.nbytes for name, value in vars(self).items()
                if isinstance(value, np.ndarray)}



def native_input_identity(*, bed_m, factor, resolution_m, manning_n, active_mask,
                          native_active_mask=None, native_north_outlets=None, native_partition=None):
    """Bind reusable hydraulic coefficients to their exact physical inputs."""
    bed=np.ascontiguousarray(bed_m,dtype=np.float64)
    mask=np.ascontiguousarray(active_mask,dtype=np.uint8)
    roughness=np.broadcast_to(np.asarray(manning_n,dtype=np.float64),bed.shape)
    selected=mask.repeat(factor,axis=0).repeat(factor,axis=1).astype(bool)
    digest=hashlib.sha256()
    for value in (np.asarray(bed.shape,dtype=np.int64),np.asarray([factor],dtype=np.int64),
                  np.asarray([resolution_m],dtype=np.float64),bed,mask,np.where(selected,roughness,1.)):
        digest.update(np.ascontiguousarray(value).tobytes())
    for name,value in (("native_active",native_active_mask),("native_north_outlets",native_north_outlets),
                       ("native_partition",native_partition)):
        if value is not None:
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value,dtype=np.int32).tobytes())
    return digest.hexdigest()



def hydraulic_face_masks(*, shape, open_sides, face_open_x=None, face_open_y=None):
    """Static open/closed port topology, independent of stage boundary values.

    False interfaces are absent from the reduced operator, not merely given a
    zero runtime flux. Returned arrays own their data. Without explicit masks
    the historical open_sides contract is unchanged.
    """
    ny,nx=shape
    masks=[]
    for value,expected in ((face_open_x,(ny,nx+1)),(face_open_y,(ny+1,nx))):
        if value is None:mask=np.ones(expected,dtype=bool)
        else:
            mask=np.asarray(value)
            if mask.shape!=expected or not np.all(np.isin(mask,(0,1))):
                raise ValueError('Binary staggered hydraulic face masks must match the coarse grid')
            mask=mask.astype(bool,copy=True)
        masks.append(mask)
    x,y=masks
    if 0 not in open_sides:x[:,0]=False
    if 1 not in open_sides:x[:,-1]=False
    if 2 not in open_sides:y[0,:]=False
    if 3 not in open_sides:y[-1,:]=False
    return x,y


def face_is_open(*, cell, side, shape, face_open_x, face_open_y):
    j,i=divmod(cell,shape[1])
    if side in (0,1):return bool(face_open_x[j,i+int(side==1)])
    return bool(face_open_y[j+int(side==3),i])


def face_topology_identity(*, face_open_x, face_open_y):
    digest=hashlib.sha256()
    for values in (face_open_x,face_open_y):
        digest.update(np.ascontiguousarray(values,dtype=np.uint8).tobytes())
    return digest.hexdigest()

def storage_curve(*, patch, pixel_area):
    levels, counts = np.unique(patch, return_counts=True)
    area = np.cumsum(counts)*pixel_area
    volume = np.r_[0., np.cumsum(np.diff(levels)*area[:-1])]
    return np.column_stack((levels, volume, area))



def storage_stage(*, volume, curve):
    index = max(0, int(np.searchsorted(curve[:, 1], volume, side='right')-1))
    eta = float(curve[index, 0]+(volume-curve[index, 1])/curve[index, 2])
    area = float(curve[index, 2])
    if index > 0 and volume == curve[index, 1]:
        area = float(curve[index-1, 2])
    return eta, area



def compact_storage(*, geometry, maximum_knots):
    """Adaptive monotone V(eta) chords; exact ports and coefficient tables remain.

    Each curve retains dry/full endpoints and exact connection spill volumes.
    Greedy refinement uses
    inverse-stage error at every original elevation. The secant area is the
    derivative of the approximated storage, not an independently fitted area.
    """
    if not isinstance(maximum_knots,int) or maximum_knots<2:
        raise ValueError('storage knot budget must be an integer >=2')
    started=time.perf_counter();rows=[];offsets=[0];errors=[]
    for cell in range(len(geometry.storage_offsets)-1):
        exact=geometry.storage_rows[geometry.storage_offsets[cell]:geometry.storage_offsets[cell+1]]
        if len(exact)<=maximum_knots:
            rows.extend(exact);offsets.append(len(rows))
            errors.append(dict(maximum_stage_error_m=0.,rms_stage_error_m=0.,
                maximum_volume_error_m3=0.,rms_volume_error_m3=0.))
            continue
        connections=np.abs(geometry.adjacency[geometry.adjacency_offsets[cell]:geometry.adjacency_offsets[cell+1]])-1
        critical=np.unique(geometry.sill[connections])
        critical=critical[(critical>exact[0,0])&(critical<exact[-1,0])]
        if len(critical):
            levels=np.unique(np.r_[exact[:,0],critical])
            indices=np.maximum(0,np.searchsorted(exact[:,0],levels,side='right')-1)
            volumes=exact[indices,1]+(levels-exact[indices,0])*exact[indices,2]
            exact=np.column_stack((levels,volumes,exact[indices,2]))
        keep={0,len(exact)-1,*np.searchsorted(exact[:,0],critical).tolist()}
        if len(keep)>maximum_knots:
            raise ValueError('storage budget smaller than mandatory exact connectivity knots')
        while len(keep)<min(maximum_knots,len(exact)):
            ids=np.array(sorted(keep))
            predicted=np.interp(exact[:,1],exact[ids,1],exact[ids,0])
            error=np.abs(predicted-exact[:,0]);error[ids]=-1
            keep.add(int(np.argmax(error)))
        ids=np.array(sorted(keep));curve=exact[ids].copy()
        if len(curve)>1:
            curve[:-1,2]=np.diff(curve[:,1])/np.diff(curve[:,0])
        stage_error=np.interp(exact[:,1],curve[:,1],curve[:,0])-exact[:,0]
        volume_error=np.interp(exact[:,0],curve[:,0],curve[:,1])-exact[:,1]
        errors.append(dict(maximum_stage_error_m=float(np.max(abs(stage_error))),
            rms_stage_error_m=float(np.sqrt(np.mean(stage_error**2))),
            maximum_volume_error_m3=float(np.max(abs(volume_error))),
            rms_volume_error_m3=float(np.sqrt(np.mean(volume_error**2)))))
        rows.extend(curve);offsets.append(len(rows))
    return replace(geometry,storage_rows=np.asarray(rows).reshape(-1,3),
        storage_offsets=np.asarray(offsets,dtype=np.int32),diagnostics=dict(geometry.diagnostics,
        storage_knot_budget=maximum_knots,storage_compression_errors=errors,
        storage_compression_s=time.perf_counter()-started))



def table_value(*, stage, sill, rows,coefficient_values=None):
    if stage <= sill or not len(rows):
        return 0., 0.
    index = max(0, int(np.searchsorted(rows[:, 0], stage, side='right')-1))
    row = rows[index]
    fraction = np.clip((stage-row[0])/(row[1]-row[0]), 0., 1.)
    values=row[2:] if coefficient_values is None else coefficient_values[index].astype(np.float64)
    scaled_d = values[0]+fraction*(values[1]-values[0])
    scaled_r = values[2]+fraction*(values[3]-values[2])
    depth = stage-sill
    return float(scaled_d/depth), float(scaled_r/depth**(10/3))



def pack_geometry(*, geometry, upward_fp32=True,share_tables=True):
    """Exact stage/storage identities; optional upward-rounded coefficient values."""
    if geometry.coefficient_values is not None:
        raise ValueError('geometry is already packed')
    original_bytes=sum(geometry.memory_bytes().values());rows=[];offsets=[0];sills=[]
    mapping=[];seen={}
    for p,sill in enumerate(geometry.port_sill):
        table=geometry.rows[geometry.offsets[p]:geometry.offsets[p+1]]
        key=np.float64(sill).tobytes()+table.tobytes()
        if not share_tables or key not in seen:
            seen[key]=len(sills);sills.append(sill);rows.extend(table)
            offsets.append(len(rows))
        mapping.append(seen[key])
    rows=np.asarray(rows,dtype=np.float64).reshape(-1,6)
    halves=geometry.halves.copy();mask=halves>=0
    halves[mask]=np.asarray(mapping)[halves[mask]]
    values=rows[:,2:].copy()
    if upward_fp32:
        packed=values.astype(np.float32)
        low=packed.astype(np.float64)<values
        packed[low]=np.nextafter(packed[low],np.float32(np.inf))
        if not np.all(np.isfinite(packed)) or np.any(packed<=0):
            raise ValueError('coefficient values outside positive finite FP32 range')
        relative=(packed.astype(np.float64)-values)/values
        if np.any(relative<0) or np.max(relative,initial=0.)>np.finfo(np.float32).eps:
            raise ValueError('coefficient rounding exceeds one relative FP32 ulp')
        values=packed
    else:
        relative=np.zeros_like(values)
    result=replace(geometry,halves=halves,port_sill=np.asarray(sills),offsets=np.asarray(offsets,dtype=np.int32),
        rows=rows[:,:2].copy(),coefficient_values=values,diagnostics=dict(geometry.diagnostics,
        coefficient_value_dtype=str(values.dtype),shared_table_count=len(sills),original_table_count=len(mapping),
        maximum_coefficient_relative_increase=float(np.max(relative,initial=0.)),unpacked_static_bytes=original_bytes))
    result.diagnostics['packed_static_bytes']=sum(result.memory_bytes().values())
    return result



def coefficient_chords(*, table, relative_tolerance, critical_levels, merge_jumps=False):
    """Compile the retained endpoint-certified greedy chord algorithm."""
    from floodwarp.enhanced_preprocessing import coefficient_chords as compiled
    return compiled(np.asarray(table, dtype=np.float64), relative_tolerance,
        np.asarray(sorted(critical_levels), dtype=np.float64), merge_jumps)


def _coefficient_chords_reference(*, table, relative_tolerance, critical_levels, merge_jumps=False):
    """Bound each positive piecewise-linear coefficient at BOTH interval limits.

    Columns are low/high stage followed by one or more left/right value pairs.
    A ratio of positive affine functions is monotone on each old interval;
    checking both endpoints also bounds discontinuous old functions. Gaps and
    declared critical levels always remain separate. Small jumps can be encoded
    by a continuous chord only when the same relative bound holds on both sides.
    """
    rows=[];maximum_error=0.;start=0
    while start<len(table):
        end=start
        for candidate in range(start+1,len(table)):
            left,right=table[candidate-1],table[candidate]
            if left[1]!=right[0] or right[0] in critical_levels:
                break
            if not merge_jumps and not np.allclose(left[3::2],right[2::2],rtol=1e-12,atol=0.):
                break
            block=table[start:candidate+1]
            levels=block[:,:2].ravel();fraction=(levels-table[start,0])/(table[candidate,1]-table[start,0])
            actual=np.column_stack([block[:,c:c+2].ravel() for c in range(2,table.shape[1],2)])
            first=table[start,2::2];last=table[candidate,3::2]
            predicted=first[None,:]+fraction[:,None]*(last-first)[None,:]
            if np.max(abs(predicted-actual)/actual)>relative_tolerance:break
            end=candidate
        block=table[start:end+1]
        first=table[start];last=table[end]
        row=np.r_[first[0],last[1],np.column_stack((first[2::2],last[3::2])).ravel()]
        levels=block[:,:2].ravel();fraction=(levels-row[0])/(row[1]-row[0])
        actual=np.column_stack([block[:,c:c+2].ravel() for c in range(2,table.shape[1],2)])
        predicted=row[2::2][None,:]+fraction[:,None]*(row[3::2]-row[2::2])[None,:]
        maximum_error=max(maximum_error,float(np.max(abs(predicted-actual)/actual)))
        rows.append(row);start=end+1
    return np.asarray(rows).reshape(-1,table.shape[1]),maximum_error



def compact_coefficients(*, geometry, relative_tolerance):
    """Phase 7 interval encoding; its global event/continuity policy is retained."""
    if geometry.coefficient_values is not None or not 0<relative_tolerance<1:
        raise ValueError('unpacked coefficients and tolerance in (0,1) required')
    started=time.perf_counter();rows=[];offsets=[0];maximum_error=0.
    critical=set(geometry.port_sill.tolist())
    for port in range(len(geometry.offsets)-1):
        table=geometry.rows[geometry.offsets[port]:geometry.offsets[port+1]]
        result,error=coefficient_chords(table=table,relative_tolerance=relative_tolerance,critical_levels=critical)
        rows.extend(result);maximum_error=max(maximum_error,error);offsets.append(len(rows))
    return replace(geometry,rows=np.asarray(rows).reshape(-1,6),offsets=np.asarray(offsets,np.int32),
        diagnostics=dict(geometry.diagnostics,coefficient_compression_s=time.perf_counter()-started,
            original_coefficient_rows=len(geometry.rows),compact_coefficient_rows=len(rows),
            certified_table_relative_error=maximum_error,coefficient_compression_tolerance=relative_tolerance))



def energy_normalized_diagonal(*, mass):
    """Positive diagonal majorant after unit-diagonal kinetic normalization."""
    scale=np.sqrt(np.diag(mass))
    if np.any(scale<=0) or not np.all(np.isfinite(scale)):
        raise NumericalFailure('positive finite native kinetic diagonal required')
    return scale*(np.abs(mass)@(1./scale))



def contact_runs(*, sills, grouping):
    """Stable runs with a connected wet subset at every stage.

    A local-maximum plateau separates valleys; retain that plateau as its own
    group. Monotone or single-valley faces need one group. This partition is
    reflection invariant, and never joins separated wet runs within a group.
    """
    changes = np.r_[0, np.flatnonzero(np.diff(sills) != 0)+1, len(sills)]
    if grouping == 'sill_runs':
        return list(zip(changes[:-1], changes[1:]))
    cuts = [0, len(sills)]
    for start, stop in zip(changes[:-1], changes[1:]):
        if start > 0 and stop < len(sills) and sills[start] > max(sills[start-1], sills[stop]):
            cuts.extend((int(start), int(stop)))
    cuts = np.unique(cuts)
    return list(zip(cuts[:-1], cuts[1:]))



def append_stage_tables(*, levels, ports, port_sills, evaluate, port_rows,
                        opening_depth_m, tolerance, maximum_rows):
    maximum_error = 0.
    for low, high in zip(levels[:-1], levels[1:]):
        active_ids = [p for p in ports if port_sills[p] <= low]
        if not active_ids:
            continue
        epsilon = min(opening_depth_m, (high-low)/4)
        # Symmetric terrain can produce adjacent representable critical levels.
        # Sampling at the dry sill then omits newly opening ports entirely.
        opening = max(float(low+epsilon), float(np.nextafter(low, high)))
        left = evaluate(stage=opening)
        right = evaluate(stage=float(high))
        for p in active_ids:
            port_rows[p].append((float(low), opening, left[p][0], left[p][0], left[p][1], left[p][1]))
        if opening == high:
            continue
        pending = [(opening, float(high), left, right, 0)]
        while pending:
            a, b, va, vb, depth = pending.pop()
            midpoint = (a+b)/2
            if a < midpoint < b:
                vm = evaluate(stage=midpoint)
                error = max(float(np.max(np.abs((va[p]+vb[p])/2-vm[p])/vm[p])) for p in active_ids)
            else:
                # Adjacent FP64 endpoints have no representable interior
                # stage: interpolation is exact at every possible query.
                # Comparing their mean coefficient to an endpoint cannot
                # measure an interior error and cannot be refined further.
                error = 0.
            if error > tolerance:
                if depth >= 28:
                    raise NumericalFailure(f'stage interpolation failed tolerance in interval {(a,b)}')
                pending.extend(((midpoint, b, vm, vb, depth+1), (a, midpoint, va, vm, depth+1)))
            else:
                maximum_error = max(maximum_error, error)
                for p in active_ids:
                    port_rows[p].append((a, b, va[p][0], vb[p][0], va[p][1], vb[p][1]))
                    if len(port_rows[p]) > maximum_rows:
                        raise UnsupportedGeometryError('stage table exceeds declared per-port memory bound')
    return maximum_error



def scaled_tail_coefficients(*, reciprocal_head, reference, patch, owner_mask,
    entries, contacts, port_sills, contact_roughness, manning_n,resolution_m,factor,formulation):
    """Exact scale-covariant native reduction at finite head or its limit.

    For H=eta-reference and x=1/H, set eta'=1, z'=x(z-reference).
    Conductance scales as H^(5/3), inertance as H^-1 and resistance
    as H^(-10/3); the normalized native lift is unchanged. x=0 is the
    fully wetted infinite-head limit, computed without large elevations.
    Persistent owner masks keep other components excluded at all x.
    """
    x=reciprocal_head
    native=np.where(owner_mask,(patch-reference)*x,2.)
    nodes=tuple(e[2] for e in entries)
    labels=wet_components(bed=native,stage_m=1.).ravel()
    groups=tuple(PortGroup(side=e[1],component=int(labels[n[0]]),segments=np.arange(len(n)))
        for e,n in zip(entries,nodes))
    cs=tuple((contacts[e[0]]-reference)*x for e in entries)
    cn=tuple(contact_roughness[e[0]] for e in entries) if contact_roughness is not None else None
    flat=(np.ndim(manning_n)==0 and np.all(owner_mask) and np.all(patch==patch.flat[0])
        and all(len(e[2])==factor and np.all(contacts[e[0]]==patch.flat[0]) for e in entries))
    if flat:
        depth=1.-(patch.flat[0]-reference)*x
        d=np.full(len(entries),.5/depth)
        r=np.full(len(entries),manning_n**2/(2*factor*resolution_m*depth**(10/3)))
        return {e[0]:np.array([di*(1.-(port_sills[e[0]]-reference)*x),
            ri*(1.-(port_sills[e[0]]-reference)*x)**(10/3)]) for e,di,ri in zip(entries,d,r)}
    reduction=reduce_cell(bed_m=native,stage_m=1.,resolution_m=resolution_m,manning_n=manning_n,
        port_groups=groups,contact_sills_m=cs,contact_nodes=nodes,contact_manning_n=cn,compute_conductance=False,
        momentum_source_weights=np.maximum(1.-native,0.)**(5/3))
    d,r=port_coefficients(reduction=reduction,formulation='matrix_lumped')
    if formulation=='transport_star':d,_=transport_star(reduction=reduction)
    capacity=(np.array([resolution_m*np.sum((1.-c)**(5/3))/manning_n for c in cs]) if cn is None
        else np.array([resolution_m*np.sum((1.-c)**(5/3)/n) for c,n in zip(cs,cn)]))
    magnitude=np.abs(reduction.flux_basis);weighted=magnitude@capacity
    r=(magnitude.T@(reduction.edge_resistance*weighted**2))/capacity**2
    return {e[0]:np.array([di*(1.-(port_sills[e[0]]-reference)*x),
        ri*(1.-(port_sills[e[0]]-reference)*x)**(10/3)]) for e,di,ri in zip(entries,d,r)}


def append_stage_tail(*, ports,evaluate,reference,start,tolerance,maximum_rows,tables):
    """Adaptive positive normalized coefficients on compact x in [0,1/H].

    Error is checked at quarter/mid/three-quarter points, as an interpolation
    diagnostic (not an analytic error certificate for the nonlinear reduction).
    There are no topology changes in this wholly wet range.
    """
    end=1./(start-reference);cache={}
    def sample(x):
        if x not in cache:cache[x]=evaluate(reciprocal_head=x)
        return cache[x]
    intervals=[];pending=[(0.,end,sample(0.),sample(end),0)];worst=0.
    while pending:
        a,b,va,vb,depth=pending.pop();error=0.
        for fraction in (.25,.5,.75):
            exact=sample(a+fraction*(b-a))
            for p in ports:
                error=max(error,float(np.max(abs(va[p]+fraction*(vb[p]-va[p])-exact[p])/exact[p])))
        if error>tolerance:
            if depth>=28:raise NumericalFailure('Reciprocal-head tail interpolation did not converge')
            middle=(a+b)/2;vm=sample(middle)
            pending.extend(((middle,b,vm,vb,depth+1),(a,middle,va,vm,depth+1)))
        else:
            intervals.append((a,b,va,vb));worst=max(worst,error)
        if len(intervals)+len(pending)>maximum_rows:
            raise UnsupportedGeometryError('Unbounded coefficient tail exceeds per-port resource bound')
    intervals.sort(key=tail_interval_start)
    for p in ports:
        knots=[(intervals[0][0],*intervals[0][2][p])]
        knots.extend((b,*vb[p]) for a,b,va,vb in intervals)
        tables[p]=np.asarray(knots)
    return worst


def tail_interval_start(interval):
    return interval[0]


def attach_stage_tail(*,geometry,tables,reference,start,error):
    offsets=[0];rows=[];seen={};mapping=[]
    for table in tables:
        key=table.tobytes()
        if key not in seen:
            seen[key]=len(offsets)-1;rows.extend(table);offsets.append(len(rows))
        mapping.append(seen[key])
    halves=geometry.halves.copy();valid=halves>=0
    halves[valid]=np.asarray(mapping,dtype=np.int32)[halves[valid]]
    return replace(geometry,maximum_stage=np.inf,tail_start=start,tail_reference=reference,
        tail_offsets=np.asarray(offsets,dtype=np.int32),tail_rows=np.asarray(rows).reshape(-1,3),tail_halves=halves,
        diagnostics=dict(geometry.diagnostics,unbounded_stage=True,
            maximum_tail_sample_relative_error=error,tail_table_count=len(seen)))


def build_dynamic_geometry(*, bed_m, factor, resolution_m, manning_n,
                           maximum_stage_m, open_sides, formulation,
                           interpolation_tolerance, maximum_ports_per_cell,
                           maximum_rows_per_port, opening_depth_m=1e-5, contact_grouping='connected_runs',
                           inertia_weighting='uniform', face_open_x=None, face_open_y=None,
                           unbounded_stage=False, preprocessing_workers=1):
    started = time.perf_counter()
    bed = np.asarray(bed_m, dtype=float)
    if (bed.ndim != 2 or not bed.size or not np.all(np.isfinite(bed))
            or not isinstance(factor, int) or factor < 1 or any(n % factor for n in bed.shape)):
        raise ValueError('finite native raster divisible by positive integer factor required')
    if (not np.isfinite(resolution_m) or resolution_m <= 0 or not np.isfinite(manning_n)
            or manning_n <= 0 or not np.isfinite(maximum_stage_m) or maximum_stage_m <= bed.min()):
        raise ValueError('positive finite resolution/roughness and valid maximum stage required')
    if unbounded_stage and (maximum_stage_m<=bed.max() or inertia_weighting!='uniform'):
        raise ValueError('Unbounded stage tables require all terrain wet at the tail start and uniform inertia weighting')
    if formulation not in ('matrix_lumped', 'transport_star'):
        raise ValueError('select matrix_lumped or transport_star')
    if contact_grouping not in ('sill_runs', 'connected_runs'):
        raise ValueError('select sill_runs or connected_runs contact grouping')
    if inertia_weighting not in ('uniform','energy_normalized'):
        raise ValueError('select uniform or energy_normalized kinetic lumping')
    if (not 0 < interpolation_tolerance < 1 or maximum_ports_per_cell < 1
            or maximum_rows_per_port < 1 or len(set(open_sides)) != len(open_sides)
            or any(s not in range(4) for s in open_sides)
            or not np.isfinite(opening_depth_m) or opening_depth_m <= 0):
        raise ValueError('invalid compact table configuration')
    shape = tuple(n//factor for n in bed.shape)
    face_open_x,face_open_y=hydraulic_face_masks(shape=shape,open_sides=open_sides,
        face_open_x=face_open_x,face_open_y=face_open_y)
    keys=[]
    patches = bed.reshape(shape[0], factor, shape[1], factor).transpose(0, 2, 1, 3).reshape(-1, factor, factor)
    native_sides = boundary_nodes(shape=(factor, factor))
    # Diagnose every topology interval, including interior pools without ports.
    invalid = []
    for cell, patch in enumerate(patches):
        for level in np.unique(patch[patch < maximum_stage_m]):
            if wet_components(bed=patch, stage_m=float(np.nextafter(level, np.inf))).max() > 0:
                invalid.append((cell, float(level)))
                break
    if invalid:
        raise UnsupportedGeometryError(f'disconnected internal storage in {len(invalid)}/{len(patches)} cells; '
                         f'first cell/elevation {invalid[0]}; additional water levels required')
    cell_ports = [[] for _ in patches]
    endpoints, halves, sills, widths, exterior = [], [], [], [], []
    port_sills, port_contacts, ghosts = [], [], []
    # Store one stable connection for each common-sill run on an interface.
    for cell, patch in enumerate(patches):
        j, i = divmod(cell, shape[1])
        for side, (dj, di) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            nj, ni = j+dj, i+di
            internal = 0 <= nj < shape[0] and 0 <= ni < shape[1]
            if internal and side not in (1, 3):
                continue
            if not internal and side not in open_sides:
                continue
            if not face_is_open(cell=cell,side=side,shape=shape,
                    face_open_x=face_open_x,face_open_y=face_open_y):continue
            other = nj*shape[1]+ni if internal else -1
            opposite = (1, 0, 3, 2)[side]
            sill = patch.ravel()[native_sides[side]].copy()
            if internal:
                sill = np.maximum(sill, patches[other].ravel()[native_sides[opposite]])
            for begin, end in contact_runs(sills=sill, grouping=contact_grouping):
                minimum = float(sill[begin:end].min())
                if minimum < maximum_stage_m:
                    connection = len(endpoints)
                    segments = np.arange(begin, end)
                    first = len(port_sills)
                    port_sills.append(minimum)
                    port_contacts.append(sill[begin:end].copy())
                    cell_ports[cell].append((first, side, segments, connection))
                    second = -1
                    if internal:
                        second = len(port_sills)
                        port_sills.append(minimum)
                        port_contacts.append(sill[begin:end].copy())
                        cell_ports[other].append((second, opposite, segments, connection))
                    elif contact_grouping == 'connected_runs':
                        second = len(port_sills)
                        port_sills.append(minimum)
                        port_contacts.append(sill[begin:end].copy())
                        ghosts.append(second)
                    nodes=native_sides[side][segments]
                    first_native=(j*factor+nodes[0]//factor)*bed.shape[1]+i*factor+nodes[0]%factor
                    last_native=(j*factor+nodes[-1]//factor)*bed.shape[1]+i*factor+nodes[-1]%factor
                    shift=(1 if side==1 else bed.shape[1]) if internal else 0
                    keys.append((first_native,first_native+shift if internal else -1,
                        last_native,last_native+shift if internal else -1,side))
                    endpoints.append((cell, other)); halves.append((first, second))
                    sills.append(minimum); widths.append((end-begin)*resolution_m)
                    exterior.append(-1 if internal else side)
    counts = [len(p) for p in cell_ports]
    if max(counts, default=0) > maximum_ports_per_cell:
        raise UnsupportedGeometryError(f'port count {max(counts)} exceeds bounded limit {maximum_ports_per_cell}')
    port_rows = [[] for _ in port_sills]
    tail_tables=[np.empty((0,3)) for _ in port_sills];tail_error=0.;tail_reference=float(bed.min())
    max_error = 0.
    backend_diagnostics = {}
    from floodwarp.enhanced_native_sparse import merge_diagnostics
    from floodwarp.enhanced_native_compiled import warm_native_kernel
    warm_native_kernel()
    jobs = connected_table_jobs(patches=patches, cell_ports=cell_ports,
        port_contacts=port_contacts, port_sills=port_sills, native_sides=native_sides,
        maximum_stage_m=maximum_stage_m, manning_n=manning_n, resolution_m=resolution_m,
        factor=factor, formulation=formulation, opening_depth_m=opening_depth_m,
        interpolation_tolerance=interpolation_tolerance, maximum_rows_per_port=maximum_rows_per_port,
        unbounded_stage=unbounded_stage, tail_reference=tail_reference, inertia_weighting=inertia_weighting)
    pool = None
    if preprocessing_workers > 1:
        import multiprocessing
        pool = multiprocessing.get_context('spawn').Pool(processes=preprocessing_workers)
        results = pool.imap(component_stage_tables, jobs, chunksize=4)
    else:
        results = map(component_stage_tables, jobs)
    try:
        for ports, result in zip((p for p in cell_ports if p), results, strict=True):
            tables, tails, local_error, local_tail_error, local_diagnostics = result
            merge_diagnostics(destination=backend_diagnostics, source=local_diagnostics)
            for port, table, tail in zip(ports, tables, tails, strict=True):
                port_rows[port[0]] = table.tolist()
                tail_tables[port[0]] = tail
            max_error = max(max_error, local_error)
            tail_error = max(tail_error, local_tail_error)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    for port in ghosts:
        contacts = port_contacts[port]
        levels = np.unique(np.r_[contacts[contacts < maximum_stage_m], maximum_stage_m])

        def ghost_coefficients(stage):
            depths = np.maximum(stage-contacts, 0.)
            area = resolution_m*np.sum(depths)
            capacity = resolution_m*np.sum(depths**(5/3))/manning_n
            half_length = factor*resolution_m/2
            h = stage-port_sills[port]
            return {port: np.array([half_length*h/area, half_length*h**(10/3)/capacity**2])}

        max_error = max(max_error, append_stage_tables(levels=levels, ports=[port], port_sills=port_sills,
            evaluate=ghost_coefficients, port_rows=port_rows, opening_depth_m=opening_depth_m,
            tolerance=interpolation_tolerance, maximum_rows=maximum_rows_per_port))
        if unbounded_stage:
            def ghost_tail(reciprocal_head):
                h=1.-(contacts-tail_reference)*reciprocal_head
                depth=1.-(port_sills[port]-tail_reference)*reciprocal_head
                area=resolution_m*h.sum();capacity=resolution_m*np.sum(h**(5/3))/manning_n
                length=factor*resolution_m/2
                return {port:np.array([length*depth/area,length*depth**(10/3)/capacity**2])}
            tail_error=max(tail_error,append_stage_tail(ports=[port],evaluate=ghost_tail,
                reference=tail_reference,start=maximum_stage_m,tolerance=interpolation_tolerance/4,
                maximum_rows=maximum_rows_per_port,tables=tail_tables))
    offsets, rows = [0], []
    for entries in port_rows:
        entries.sort()
        rows.extend(entries); offsets.append(len(rows))
    storage_offsets, storage_rows = [0], []
    for patch in patches:
        curve = storage_curve(patch=patch, pixel_area=resolution_m**2)
        storage_rows.extend(curve); storage_offsets.append(len(storage_rows))
    adjacent = [[] for _ in patches]
    for connection, (a, b) in enumerate(endpoints):
        adjacent[a].append(connection+1)
        if b >= 0:
            adjacent[b].append(-connection-1)
    adjacency_offsets, adjacency = [0], []
    for entries in adjacent:
        adjacency.extend(entries); adjacency_offsets.append(len(adjacency))
    coefficient_rows = np.asarray(rows).reshape(-1, 6)
    if not np.all(np.isfinite(coefficient_rows)) or np.any(coefficient_rows[:, 2:] <= 0):
        raise NumericalFailure('stage tables require finite positive inertia and resistance')
    result=DynamicGeometry(shape=shape, factor=factor, resolution=resolution_m,
        maximum_stage=maximum_stage_m, formulation=formulation,
        connection_keys=np.asarray(keys,dtype=np.int64).reshape(-1,5),
        endpoints=np.asarray(endpoints, dtype=np.int32).reshape(-1, 2),
        halves=np.asarray(halves, dtype=np.int32).reshape(-1, 2), sill=np.asarray(sills),
        width=np.asarray(widths), external_side=np.asarray(exterior, dtype=np.int32),
        port_sill=np.asarray(port_sills), offsets=np.asarray(offsets, dtype=np.int32),
        rows=coefficient_rows, storage_offsets=np.asarray(storage_offsets, dtype=np.int32),
        storage_rows=np.asarray(storage_rows).reshape(-1, 3),
        adjacency_offsets=np.asarray(adjacency_offsets, dtype=np.int32), adjacency=np.asarray(adjacency, dtype=np.int32),
        preprocess_s=time.perf_counter()-started, diagnostics=dict(face_topology_identity=face_topology_identity(
            face_open_x=face_open_x,face_open_y=face_open_y),maximum_ports=max(counts, default=0),
        mean_ports=float(np.mean(counts)), maximum_interpolation_midpoint_error=max_error,
        maximum_rows_per_port=max(np.diff(offsets), default=0), disconnected_cells=0,
        opening_depth_m=opening_depth_m, contact_grouping=contact_grouping,inertia_weighting=inertia_weighting,
        unbounded_stage=unbounded_stage, reduction_backends=backend_diagnostics))
    if unbounded_stage:
        result=attach_stage_tail(geometry=result,tables=tail_tables,reference=tail_reference,
            start=maximum_stage_m,error=tail_error)
    return result



def valley_ownership(*, bed):
    """Identical persistent minimum/plateau ownership in compiled raster order."""
    from floodwarp.enhanced_preprocessing import valley_labels
    bed = np.asarray(bed, dtype=float)
    if bed.ndim != 2 or not bed.size or not np.all(np.isfinite(bed)):
        raise ValueError('finite nonempty two-dimensional terrain required')
    return valley_labels(bed)


def _valley_ownership_reference(*, bed):
    """Grow equal-height D4 plateaus from lower valleys in ascending elevation.

    A new local-minimum plateau creates one persistent state. At a saddle the
    plateau belongs to its lowest-seed neighbour (native index breaks exact
    ties). Other valleys retain their owners and meet across real native faces.
    All pixels are retained. The ridge ownership tie is a geometric approximation,
    not an extra hydraulic connection; rotation sensitivity must be measured.
    """
    bed = np.asarray(bed, dtype=float)
    if bed.ndim != 2 or not bed.size or not np.all(np.isfinite(bed)):
        raise ValueError('finite nonempty two-dimensional terrain required')
    owners = np.full(bed.shape, -1, dtype=np.int32)
    seeds = []
    edges = raster_edges(shape=bed.shape)
    for elevation in np.unique(bed):
        plateaus, count = label(bed == elevation)
        for plateau in range(1, count + 1):
            mask = plateaus == plateau
            neighbours = []
            for a, b in (edges, edges[::-1]):
                selected = mask.ravel()[a] & (owners.ravel()[b] >= 0)
                neighbours.extend(owners.ravel()[b[selected]].tolist())
            if neighbours:
                candidates = sorted(set(neighbours), key=seeds.__getitem__)
                owner = candidates[0]
            else:
                owner = len(seeds)
                seeds.append((float(elevation), int(np.flatnonzero(mask)[0])))
            owners[mask] = owner
    return owners



def component_volumes(*, geometry, native_depth):
    depth = np.asarray(native_depth, dtype=float)
    if (geometry.pixel_owner is None or depth.shape != geometry.pixel_owner.shape
            or not np.all(np.isfinite(depth)) or np.any(depth < 0)):
        raise ValueError('nonnegative native depth on the ownership raster required')
    valid=geometry.pixel_owner>=0
    if np.any(depth[~valid]!=0):raise ValueError('nonzero water on inactive native pixels')
    return np.bincount(geometry.pixel_owner[valid], weights=depth[valid]*geometry.resolution**2,
                       minlength=len(geometry.storage_offsets)-1)



def component_depth(*, geometry, volume, bed):
    stages = np.array([storage_stage(volume=v, curve=geometry.storage_rows[
        geometry.storage_offsets[c]:geometry.storage_offsets[c+1]])[0]
        for c, v in enumerate(volume)])
    result=np.zeros_like(bed,dtype=float);valid=geometry.pixel_owner>=0
    result[valid]=np.maximum(stages[geometry.pixel_owner[valid]]-bed[valid],0.)
    return result



def component_stage_tables(parameters):
    """Independent native preprocessing job; deterministic ordered results.

    Inputs contain only one component patch and its native contacts. No GPU
    objects, global geometry, hydraulic state or runtime traversal is involved.
    """
    original_patch=parameters['patch']
    owner_mask=parameters['owner_mask']
    maximum_stage_m=parameters['maximum_stage_m']
    patch=original_patch.copy();patch[~owner_mask]=maximum_stage_m+1.
    entries=parameters['entries'];contacts=parameters['contacts'];port_sills=parameters['port_sills']
    contact_roughness=parameters['contact_roughness'];manning_n=parameters['manning_n']
    spatial_roughness=np.ndim(manning_n)!=0
    inertia_weighting=parameters.get('inertia_weighting','uniform')
    resolution_m=parameters['resolution_m'];factor=parameters['factor'];formulation=parameters['formulation']
    opening_depth_m=parameters['opening_depth_m'];interpolation_tolerance=parameters['interpolation_tolerance']
    maximum_rows_per_port=parameters['maximum_rows_per_port'];unbounded_stage=parameters['unbounded_stage']
    tail_reference=parameters['tail_reference']
    rows_per_port=[[] for _ in entries]
    tail_tables=[np.empty((0,3)) for _ in entries];tail_error=0.;error=0.
    levels = np.unique(np.r_[patch[patch <= maximum_stage_m],
        np.concatenate([contacts[p] for p, _, _ in entries]), maximum_stage_m])
    cache = {}
    backend_diagnostics = {}
    from floodwarp.enhanced_native_compiled import prepare_component, evaluate_component
    prepared = prepare_component(patch=patch, owner_mask=owner_mask, entries=entries, contacts=contacts,
        port_sills=port_sills, manning_n=manning_n, contact_roughness=contact_roughness,
        resolution=resolution_m, factor=factor, formulation=formulation, inertia_weighting=inertia_weighting) if parameters.get('_compiled_native',True) else None

    def evaluate(stage):
        if stage in cache: return cache[stage]
        active = [e for e in entries if stage > port_sills[e[0]]]
        # This is the existing exact flat closure, evaluated before the
        # native solve whose coefficients it would overwrite entirely.
        if not spatial_roughness and np.all(owner_mask) and np.all(patch == patch.flat[0]) and all(
                len(e[2]) == factor and np.all(contacts[e[0]] == patch.flat[0]) for e in active):
            depth = stage-patch.flat[0]
            d = np.full(len(active), .5/depth)
            r = np.full(len(active), manning_n**2/(2*factor*resolution_m*depth**(10/3)))
            values = {e[0]: np.array([di*(stage-port_sills[e[0]]), ri*(stage-port_sills[e[0]])**(10/3)])
                      for e, di, ri in zip(active, d, r)}
            backend_diagnostics['analytical_flat_stages'] = backend_diagnostics.get('analytical_flat_stages', 0) + 1
            cache[stage] = values
            return values
        if prepared is not None:
            values = evaluate_component(prepared=prepared, stage=stage, diagnostics=backend_diagnostics)
            if values is not None:
                cache[stage] = values
                return values
        backend_diagnostics['oracle_finite_stages'] = backend_diagnostics.get('oracle_finite_stages', 0) + 1
        nodes = tuple(e[2][contacts[e[0]] < stage] for e in active)
        labels = wet_components(bed=patch, stage_m=stage).ravel()
        groups = tuple(PortGroup(side=e[1], component=int(labels[n[0]]), segments=np.arange(len(n)))
                       for e, n in zip(active, nodes))
        cs = tuple(contacts[e[0]][contacts[e[0]] < stage] for e in active)
        cn=tuple(contact_roughness[e[0]][contacts[e[0]]<stage] for e in active) if spatial_roughness else None
        reduction = reduce_cell(bed_m=patch, stage_m=stage, resolution_m=resolution_m, manning_n=manning_n,
            port_groups=groups, contact_sills_m=cs, contact_nodes=nodes, contact_manning_n=cn, compute_conductance=False,
            momentum_source_weights=np.maximum(stage-patch, 0.)**(5/3), _wet_labels=labels.reshape(patch.shape))
        d, r = port_coefficients(reduction=reduction, formulation='matrix_lumped')
        star_applied = False
        if formulation == 'transport_star':
            d, star_info = transport_star(reduction=reduction)
            star_applied = star_info['applied']
        if inertia_weighting == 'energy_normalized' and not star_applied:
            d = energy_normalized_diagonal(mass=reduction.inertance)
        capacity = (np.array([resolution_m*np.sum((stage-c)**(5/3))/manning_n for c in cs]) if not spatial_roughness
                    else np.array([resolution_m*np.sum((stage-c)**(5/3)/n) for c,n in zip(cs,cn)]))
        magnitude = np.abs(reduction.flux_basis)
        weighted = magnitude @ capacity
        r = (magnitude.T @ (reduction.edge_resistance*weighted**2))/capacity**2
        values = {e[0]: np.array([di*(stage-port_sills[e[0]]), ri*(stage-port_sills[e[0]])**(10/3)])
                  for e, di, ri in zip(active, d, r)}
        cache[stage] = values
        return values

    error = max(error, append_stage_tables(levels=levels, ports=[e[0] for e in entries],
        port_sills=port_sills, evaluate=evaluate, port_rows=rows_per_port,
        opening_depth_m=opening_depth_m, tolerance=interpolation_tolerance, maximum_rows=maximum_rows_per_port))
    if unbounded_stage:
        original_patch=parameters['patch']
        flat_tail = (not spatial_roughness and np.all(owner_mask)
            and np.all(original_patch == original_patch.flat[0])
            and all(len(e[2]) == factor and np.all(contacts[e[0]] == original_patch.flat[0]) for e in entries))
        def evaluate_tail(reciprocal_head):
            if prepared is not None and original_patch.size > 64 and not flat_tail:
                values = evaluate_component(prepared=prepared, stage=1., reciprocal=reciprocal_head,
                    reference=tail_reference, tail=True, diagnostics=backend_diagnostics)
                if values is not None:
                    return values
            backend_diagnostics['oracle_tail_parity_stages'] = backend_diagnostics.get('oracle_tail_parity_stages', 0) + 1
            # Preserve exact FP64 tail sharing, including endpoint byte keys.
            # Last-bit changes here can change table IDs even when every
            # physical coefficient agrees to double-precision roundoff.
            return scaled_tail_coefficients(reciprocal_head=reciprocal_head,reference=tail_reference,
                patch=original_patch,owner_mask=owner_mask,entries=entries,contacts=contacts,
                port_sills=port_sills,contact_roughness=contact_roughness if spatial_roughness else None,
                manning_n=manning_n,resolution_m=resolution_m,factor=factor,formulation=formulation)
        tail_error=max(tail_error,append_stage_tail(ports=[e[0] for e in entries],evaluate=evaluate_tail,
            reference=tail_reference,start=maximum_stage_m,tolerance=interpolation_tolerance/4,
            maximum_rows=maximum_rows_per_port,tables=tail_tables))
    result = ([np.asarray(sorted(table)).reshape(-1,6) for table in rows_per_port], tail_tables, error, tail_error)
    return (*result, backend_diagnostics) if parameters.get('_return_diagnostics', False) else result


def component_table_jobs(*,ports,patches,state_cell,state_masks,contacts,port_sills,
        contact_roughness,native_n,spatial_roughness,manning_n,shape,factor,resolution_m,
        maximum_stage_m,formulation,opening_depth_m,interpolation_tolerance,maximum_rows_per_port,
        unbounded_stage,tail_reference):
    """Stream small independent jobs instead of copying the complete terrain."""
    for state,entries in enumerate(ports):
        if not entries:continue
        j,i=divmod(state_cell[state],shape[1])
        roughness=native_n[j*factor:(j+1)*factor,i*factor:(i+1)*factor] if spatial_roughness else manning_n
        yield dict(patch=patches[state_cell[state]],owner_mask=state_masks[state],
            entries=[(index,e[1],e[2]) for index,e in enumerate(entries)],
            contacts=tuple(contacts[e[0]] for e in entries),
            port_sills=np.array([port_sills[e[0]] for e in entries]),
            contact_roughness=tuple(contact_roughness[e[0]] for e in entries) if spatial_roughness else None,
            manning_n=roughness,resolution_m=resolution_m,factor=factor,maximum_stage_m=maximum_stage_m,
            formulation=formulation,opening_depth_m=opening_depth_m,interpolation_tolerance=interpolation_tolerance,
            maximum_rows_per_port=maximum_rows_per_port,unbounded_stage=unbounded_stage,
            tail_reference=tail_reference,_return_diagnostics=True)


def connected_table_jobs(*, patches, cell_ports, port_contacts, port_sills, native_sides,
        maximum_stage_m, manning_n, resolution_m, factor, formulation, opening_depth_m,
        interpolation_tolerance, maximum_rows_per_port, unbounded_stage, tail_reference, inertia_weighting):
    """Connected and owned domains share the same coefficient evaluator."""
    for patch, ports in zip(patches, cell_ports, strict=True):
        if ports:
            yield dict(patch=patch, owner_mask=np.ones_like(patch, dtype=bool),
                entries=[(index, p[1], native_sides[p[1]][p[2]]) for index, p in enumerate(ports)],
                contacts=tuple(port_contacts[p[0]] for p in ports),
                port_sills=np.array([port_sills[p[0]] for p in ports]), contact_roughness=None,
                manning_n=manning_n, resolution_m=resolution_m, factor=factor,
                maximum_stage_m=maximum_stage_m, formulation=formulation, opening_depth_m=opening_depth_m,
                interpolation_tolerance=interpolation_tolerance, maximum_rows_per_port=maximum_rows_per_port,
                unbounded_stage=unbounded_stage, tail_reference=tail_reference, inertia_weighting=inertia_weighting,
                _return_diagnostics=True)


def native_component_ownership(parameters):
    """Independent static patch job; invalid pixels never join native valleys."""
    patch,valid,partition=parameters
    if not np.any(valid):return np.full(patch.shape,-1,dtype=np.int32)
    local=valley_ownership(bed=np.where(valid,patch,float(patch.max()+1.)))
    tags=np.zeros_like(local) if partition is None else partition
    result=np.full_like(local,-1);next_owner=0
    for valley,tag in np.unique(np.column_stack((local[valid],tags[valid])),axis=0):
        connected,count=label(valid & (local==valley) & (tags==tag))
        for component in range(1,count+1):
            result[connected==component]=next_owner;next_owner+=1
    return result


def build_component_geometry(*, bed_m, factor, resolution_m, manning_n,
                             maximum_stage_m, open_sides, formulation,
                             interpolation_tolerance, maximum_ports_per_cell,
                             maximum_rows_per_port, opening_depth_m=1e-5,
                             contact_grouping='connected_runs', inertia_weighting='uniform', active_mask=None,
                             face_open_x=None, face_open_y=None,unbounded_stage=False,preprocessing_workers=1,
                             native_active_mask=None, native_north_outlets=None, native_partition=None):
    """Reduce each persistent valley with the existing B/star native flux lift.

    Shared interfaces are split by owner pair, orientation, spatial run and wet
    contact topology. Internal sill links have two delta/2 contact half-edges,
    exactly as native adjacent pixel momenta. All disconnected components run;
    explicit port/table resource bounds still reject with the offending state.
    """
    if not isinstance(preprocessing_workers,int) or isinstance(preprocessing_workers,bool) or preprocessing_workers<1:
        raise ValueError('preprocessing_workers must be a positive integer')
    started = time.perf_counter()
    bed = np.asarray(bed_m, dtype=float)
    if (bed.ndim != 2 or not bed.size or not np.all(np.isfinite(bed))
            or not isinstance(factor, int) or factor < 1 or any(n % factor for n in bed.shape)):
        raise ValueError('finite native raster divisible by positive integer factor required')
    if (not np.isfinite(maximum_stage_m) or maximum_stage_m <= bed.max()
            or not np.isfinite(resolution_m) or resolution_m <= 0
            or not np.all(np.isfinite(manning_n)) or np.any(np.asarray(manning_n) <= 0)):
        raise ValueError('positive finite resolution/roughness and maximum stage above all terrain required')
    if (formulation not in ('matrix_lumped', 'transport_star')
            or contact_grouping not in ('connected_runs', 'sill_runs') or inertia_weighting != 'uniform'
            or not 0 < interpolation_tolerance < 1 or not np.isfinite(opening_depth_m)
            or opening_depth_m <= 0 or maximum_ports_per_cell < 1 or maximum_rows_per_port < 1
            or len(set(open_sides)) != len(open_sides) or any(s not in range(4) for s in open_sides)):
        raise ValueError('invalid component geometry configuration')
    shape = tuple(n//factor for n in bed.shape)
    face_open_x,face_open_y=hydraulic_face_masks(shape=shape,open_sides=open_sides,
        face_open_x=face_open_x,face_open_y=face_open_y)
    keys=[]
    patches = bed.reshape(shape[0], factor, shape[1], factor).transpose(0, 2, 1, 3).reshape(-1, factor, factor)
    active=np.ones(shape,dtype=bool) if active_mask is None else np.asarray(active_mask)
    if active.shape!=shape or not np.all(np.isin(active,(0,1))):
        raise ValueError('binary active mask must match the coarse geometry')
    active=active.astype(bool)
    native_valid=active.repeat(factor,0).repeat(factor,1)
    if native_active_mask is not None:
        given=np.asarray(native_active_mask)
        if given.shape!=bed.shape or not np.all(np.isin(given,(0,1))):
            raise ValueError('Native active mask must be binary and match terrain')
        if np.any(given & ~native_valid):raise ValueError('Native footprint extends outside active coarse cells')
        native_valid=given.astype(bool)
        occupied=native_valid.reshape(shape[0],factor,shape[1],factor).any(axis=(1,3))
        if not np.array_equal(occupied,active):raise ValueError('Coarse activity must equal native occupancy')
    north=None
    if native_north_outlets is not None:
        north=np.asarray(native_north_outlets,dtype=bool)
        if north.shape!=bed.shape or np.any(north & ~native_valid):
            raise ValueError('Native north contacts must belong to the active footprint')
        neighbour=np.zeros_like(native_valid);neighbour[:-1]=native_valid[1:]
        if np.any(north & neighbour):raise ValueError('Native coastal contact must face inactive terrain')
    partition=None if native_partition is None else np.asarray(native_partition)
    if partition is not None and (partition.shape!=bed.shape or partition.dtype.kind not in 'iu'):
        raise ValueError('Native partition requires an integer raster matching terrain')
    identity=native_input_identity(bed_m=bed,factor=factor,resolution_m=resolution_m,
        manning_n=manning_n,active_mask=active,native_active_mask=native_active_mask,
        native_north_outlets=native_north_outlets,native_partition=native_partition)

    spatial_roughness=np.ndim(manning_n)!=0
    native_n=np.asarray(manning_n,dtype=float)
    if spatial_roughness and native_n.shape!=bed.shape:
        raise ValueError('native Manning must match native terrain shape')
    if spatial_roughness:
        selected=native_n[active.repeat(factor,axis=0).repeat(factor,axis=1)]
        if len(selected) and np.all(selected==selected[0]):
            manning_n=float(selected[0]);spatial_roughness=False
    owner = np.full_like(bed, -1, dtype=np.int32)
    state_cell, state_masks, offsets = [], [], [0]
    ownership_pool=None
    if native_active_mask is not None or partition is not None:
        valid_patches=native_valid.reshape(shape[0],factor,shape[1],factor).transpose(0,2,1,3).reshape(-1,factor,factor)
        tag_patches=(None if partition is None else partition.reshape(shape[0],factor,shape[1],factor).transpose(0,2,1,3).reshape(-1,factor,factor))
        jobs=((patch,valid_patches[cell],None if tag_patches is None else tag_patches[cell]) for cell,patch in enumerate(patches))
        if preprocessing_workers>1:
            import multiprocessing
            ownership_pool=multiprocessing.get_context('spawn').Pool(processes=preprocessing_workers)
            local_owners=ownership_pool.imap(native_component_ownership,jobs,chunksize=32)
        else:local_owners=map(native_component_ownership,jobs)
    else:local_owners=(valley_ownership(bed=patch) if active.ravel()[cell] else None for cell,patch in enumerate(patches))
    for cell, (patch,local) in enumerate(zip(patches,local_owners,strict=True)):
        if not active.ravel()[cell]:
            offsets.append(len(state_cell))
            continue
        j, i = divmod(cell, shape[1])
        base = len(state_cell)
        owner[j*factor:(j+1)*factor, i*factor:(i+1)*factor] = np.where(local>=0,local+base,-1)
        for component in range(local.max()+1):
            state_cell.append(cell); state_masks.append(local == component)
        offsets.append(len(state_cell))
        if ownership_pool is not None and (cell+1)%10000==0:
            print(f'Enhanced native ownership {cell+1}/{len(patches)}; wall {time.perf_counter()-started:.1f} s',flush=True)
    if ownership_pool is not None:ownership_pool.close();ownership_pool.join()
    source_area = np.bincount(owner[owner>=0], minlength=len(state_cell))*resolution_m**2
    build_parameters=dict(maximum_stage_m=maximum_stage_m,formulation=formulation,
        interpolation_tolerance=interpolation_tolerance,maximum_ports_per_cell=maximum_ports_per_cell,
        maximum_rows_per_port=maximum_rows_per_port,opening_depth_m=opening_depth_m,
        contact_grouping=contact_grouping,inertia_weighting=inertia_weighting,unbounded_stage=unbounded_stage)

    # The accepted Phase 7 builder is retained exactly for a wholly connected domain.
    if len(state_cell) == len(patches) and np.all(active) and not spatial_roughness and native_active_mask is None and north is None and partition is None:
        result = build_dynamic_geometry(bed_m=bed, factor=factor, resolution_m=resolution_m,
            manning_n=manning_n, maximum_stage_m=maximum_stage_m, open_sides=open_sides,
            formulation=formulation, interpolation_tolerance=interpolation_tolerance,
            maximum_ports_per_cell=maximum_ports_per_cell, maximum_rows_per_port=maximum_rows_per_port,
            opening_depth_m=opening_depth_m, contact_grouping=contact_grouping,
            face_open_x=face_open_x,face_open_y=face_open_y,unbounded_stage=unbounded_stage,
            preprocessing_workers=preprocessing_workers)
        return replace(result, source_area=source_area, pixel_owner=owner, state_cell=np.array(state_cell, np.int32),
            cell_component_offsets=np.array(offsets, np.int32), preprocess_s=time.perf_counter()-started,
            diagnostics=dict(result.diagnostics, storage_states=len(state_cell), connected_domain_identity=True, native_input_identity=identity,build_parameters=build_parameters))
    # Each record contains native centre -> interface half-edges; no ghost state.
    native_width = bed.shape[1]
    from floodwarp.enhanced_topology import native_contact_groups
    group_keys,group_starts,group_stops,native_a,native_b,native_sill=native_contact_groups(
        owner=owner,bed=bed,state_cell=state_cell,shape=shape,open_sides=open_sides,
        face_x=face_open_x,face_y=face_open_y,north=north)
    ports = [[] for _ in state_cell]
    endpoints, halves, sills, widths, exterior = [], [], [], [], []
    port_sills, contacts, ghosts, contact_roughness = [], [], [], []
    opposite = (1, 0, 3, 2)
    for (sa, sb, side, line),group_start,group_stop in zip(group_keys,group_starts,group_stops):
        sa,sb,side,line=int(sa),int(sb),int(side),int(line)
        ids = native_a[group_start:group_stop]
        stride = native_width if side in (0, 1) else 1
        cuts = np.r_[0, np.flatnonzero(np.diff(ids) != stride)+1, len(ids)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            levels = native_sill[group_start+start:group_start+stop]
            for begin, end in contact_runs(sills=levels, grouping=contact_grouping):
                left=group_start+start+begin;right=group_start+start+end
                a,b=native_a[left:right],native_b[left:right]
                contact = native_sill[left:right]
                f = len(endpoints); arms = []
                for state, direction, column in ((sa, side, 0), (sb, opposite[side], 1)):
                    p = len(port_sills); port_sills.append(float(contact.min())); contacts.append(contact)
                    if spatial_roughness:
                        contact_roughness.append(np.where(b<0,native_n.ravel()[a],
                            np.maximum(native_n.ravel()[a],native_n.ravel()[b])))
                    arms.append(p)
                    if state < 0:
                        ghosts.append(p)
                    else:
                        cell = state_cell[state]; j, i = divmod(cell, shape[1])
                        native_nodes=a if column==0 else b
                        nodes = (((native_nodes//native_width)-j*factor)*factor+
                                          (native_nodes % native_width)-i*factor).astype(np.int32)
                        ports[state].append((p, direction, nodes))
                keys.append((a[0],b[0],a[-1],b[-1],side))
                endpoints.append((sa, sb)); halves.append(arms); sills.append(float(contact.min()))
                widths.append(len(a)*resolution_m); exterior.append(side if sb < 0 else -1)
    counts = [len(p) for p in ports]
    if max(counts, default=0) > maximum_ports_per_cell:
        raise UnsupportedGeometryError(f'component port resource bound: {max(counts)} > {maximum_ports_per_cell}; no states discarded')
    rows_per_port = [[] for _ in port_sills]
    tail_tables=[np.empty((0,3)) for _ in port_sills];tail_error=0.;tail_reference=float(bed.min())
    error = 0.
    jobs=component_table_jobs(ports=ports,patches=patches,state_cell=state_cell,state_masks=state_masks,
        contacts=contacts,port_sills=port_sills,contact_roughness=contact_roughness,native_n=native_n,
        spatial_roughness=spatial_roughness,manning_n=manning_n,shape=shape,factor=factor,resolution_m=resolution_m,
        maximum_stage_m=maximum_stage_m,formulation=formulation,opening_depth_m=opening_depth_m,
        interpolation_tolerance=interpolation_tolerance,maximum_rows_per_port=maximum_rows_per_port,
        unbounded_stage=unbounded_stage,tail_reference=tail_reference)
    pool=None
    backend_diagnostics = {}
    from floodwarp.enhanced_native_sparse import merge_diagnostics
    if any(ports):
        from floodwarp.enhanced_native_compiled import warm_native_kernel
        warm_native_kernel()
    if preprocessing_workers>1:
        import multiprocessing
        pool=multiprocessing.get_context('spawn').Pool(processes=preprocessing_workers)
        results=pool.imap(component_stage_tables,jobs,chunksize=4)
    else:results=map(component_stage_tables,jobs)
    try:
        for table_index,(entries,result) in enumerate(zip((p for p in ports if p),results,strict=True)):
            tables,tails,local_error,local_tail_error,local_diagnostics=result
            merge_diagnostics(destination=backend_diagnostics, source=local_diagnostics)
            for entry,table,tail in zip(entries,tables,tails,strict=True):
                rows_per_port[entry[0]]=table;tail_tables[entry[0]]=tail
            error=max(error,local_error);tail_error=max(tail_error,local_tail_error)
            if native_active_mask is not None and (table_index+1)%1000==0:
                print(f'Enhanced native tables {table_index+1}/{sum(bool(p) for p in ports)}; wall {time.perf_counter()-started:.1f} s',flush=True)
    finally:
        if pool is not None:pool.terminate();pool.join()
    for p in ghosts:
        levels = np.unique(np.r_[contacts[p], maximum_stage_m])

        def evaluate_ghost(stage):
            h = np.maximum(stage-contacts[p], 0.)
            area = resolution_m*h.sum()
            capacity = (resolution_m*np.sum(h**(5/3))/manning_n if not spatial_roughness else
                        resolution_m*np.sum(h**(5/3)/contact_roughness[p]))
            length = factor*resolution_m/2; depth = stage-port_sills[p]
            return {p: np.array([length*depth/area, length*depth**(10/3)/capacity**2])}

        error = max(error, append_stage_tables(levels=levels, ports=[p], port_sills=port_sills,
            evaluate=evaluate_ghost, port_rows=rows_per_port, opening_depth_m=opening_depth_m,
            tolerance=interpolation_tolerance, maximum_rows=maximum_rows_per_port))
        if unbounded_stage:
            def ghost_tail(reciprocal_head):
                h=1.-(contacts[p]-tail_reference)*reciprocal_head
                depth=1.-(port_sills[p]-tail_reference)*reciprocal_head
                area=resolution_m*h.sum()
                capacity=(resolution_m*np.sum(h**(5/3))/manning_n if not spatial_roughness else
                    resolution_m*np.sum(h**(5/3)/contact_roughness[p]))
                length=factor*resolution_m/2
                return {p:np.array([length*depth/area,length*depth**(10/3)/capacity**2])}
            tail_error=max(tail_error,append_stage_tail(ports=[p],evaluate=ghost_tail,
                reference=tail_reference,start=maximum_stage_m,tolerance=interpolation_tolerance/4,
                maximum_rows=maximum_rows_per_port,tables=tail_tables))
    po=np.r_[0,np.cumsum([len(table) for table in rows_per_port])].astype(np.int32)
    rows=np.concatenate([np.asarray(sorted(table) if isinstance(table,list) else table).reshape(-1,6)
        for table in rows_per_port]) if rows_per_port else np.empty((0,6))
    del rows_per_port
    storage, so = [], [0]
    for state, cell in enumerate(state_cell):
        storage.extend(storage_curve(patch=patches[cell][state_masks[state]], pixel_area=resolution_m**2))
        so.append(len(storage))
    adjacent = [[] for _ in state_cell]
    for f, (a, b) in enumerate(endpoints):
        adjacent[a].append(f+1)
        if b >= 0: adjacent[b].append(-f-1)
    adj, ao = [], [0]
    for entries in adjacent: adj.extend(entries); ao.append(len(adj))
    rows = np.asarray(rows).reshape(-1, 6)
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 2:] <= 0):
        raise NumericalFailure('component reduction requires positive finite coefficients')
    result=DynamicGeometry(shape=shape, factor=factor, resolution=resolution_m, maximum_stage=maximum_stage_m,
        formulation=formulation, connection_keys=np.asarray(keys,dtype=np.int64).reshape(-1,5), endpoints=np.asarray(endpoints, np.int32).reshape(-1, 2),
        halves=np.asarray(halves, np.int32).reshape(-1, 2), sill=np.asarray(sills), width=np.asarray(widths),
        external_side=np.asarray(exterior, np.int32), port_sill=np.asarray(port_sills), offsets=np.asarray(po, np.int32),
        rows=rows, storage_offsets=np.asarray(so, np.int32), storage_rows=np.asarray(storage).reshape(-1, 3),
        adjacency_offsets=np.asarray(ao, np.int32), adjacency=np.asarray(adj, np.int32),
        preprocess_s=time.perf_counter()-started, source_area=source_area, pixel_owner=owner,
        state_cell=np.asarray(state_cell, np.int32), cell_component_offsets=np.asarray(offsets, np.int32),
        diagnostics=dict(reduction_backends=backend_diagnostics, face_topology_identity=face_topology_identity(
            face_open_x=face_open_x,face_open_y=face_open_y),native_input_identity=identity,build_parameters=build_parameters, storage_states=len(state_cell), coarse_cells=len(patches), maximum_components=int(np.diff(offsets).max()),
            maximum_ports=max(counts, default=0), maximum_interpolation_midpoint_error=error,
            internal_links=sum(b >= 0 and state_cell[a] == state_cell[b] for a, b in endpoints),
            connected_domain_identity=False, opening_depth_m=opening_depth_m, ownership='persistent_minimum_plateaus',
            native_boundary_contacts=int(north.sum()) if north is not None else 0,
            native_active_pixels=int(native_valid.sum()),native_partitioned=partition is not None))
    if unbounded_stage:
        result=attach_stage_tail(geometry=result,tables=tail_tables,reference=tail_reference,
            start=maximum_stage_m,error=tail_error)
    return result



def port_critical_levels(*, geometry):
    """Activation is exact and separate from the scaled coefficient function.

    Other contacts opening can change an already active port's coefficients.
    Chords may span those changes only when BOTH one-sided coefficient limits
    satisfy the tolerance. They cannot activate the other contact: its exact
    sill and unchanged independent storage still control that operation.
    """
    return [{float(s)} for s in geometry.port_sill]



def scalar_nodes(*, table):
    """Encode exact one-sided knots, gaps and jumps without Python row loops."""
    from floodwarp.enhanced_preprocessing import scalar_nodes as compiled
    return compiled(np.asarray(table, dtype=np.float64))


def _scalar_nodes_reference(*, table):
    """Exact knot encoding, including duplicate nodes for jumps and gap plateaus."""
    levels=[];values=[]
    for low,high,left,right in table:
        if levels and low>levels[-1]:
            levels.append(low);values.append(values[-1])
        if not levels or low!=levels[-1] or left!=values[-1]:
            levels.append(low);values.append(left)
        levels.append(high);values.append(right)
    return np.asarray(levels),np.asarray(values)



def encode_port_coefficients(parameters):
    """Retain the exact chord certificate and upward FP32 rounding order."""
    from floodwarp.enhanced_preprocessing import encode_port
    original, critical, relative_tolerance = parameters
    return encode_port(np.asarray(original, dtype=np.float64),
        np.asarray(sorted(critical), dtype=np.float64),
        0. if relative_tolerance is None else relative_tolerance, relative_tolerance is not None)


def _encode_port_coefficients_reference(parameters):
    """The established scalar encoder for one port; independent offline work."""
    original,critical,relative_tolerance=parameters
    error=0.;maximum_rounding=0.;old_nodes=0
    encoded=[]
    for columns in ((0,1,2,3),(0,1,4,5)):
        table=original[:,columns]
        if relative_tolerance is not None:
            table,bound=coefficient_chords(table=table,relative_tolerance=relative_tolerance,
                critical_levels=critical,merge_jumps=True)
            error=max(error,bound)
        # Round BEFORE sharing endpoints, exactly as the Phase 8 FP32 encoder.
        packed=table[:,2:].astype(np.float32)
        low=packed.astype(np.float64)<table[:,2:]
        packed[low]=np.nextafter(packed[low],np.float32(np.inf))
        if not np.all(np.isfinite(packed)) or np.any(packed<=0):
            raise ValueError('coefficient outside positive finite FP32 range')
        rounding=(packed.astype(np.float64)-table[:,2:])/table[:,2:]
        maximum_rounding=max(maximum_rounding,float(np.max(rounding,initial=0.)))
        if np.any(rounding<0) or maximum_rounding>np.finfo(np.float32).eps:
            raise ValueError('coefficient rounding exceeds the declared FP32 error bound')
        table=table.copy();table[:,2:]=packed
        levels,values=scalar_nodes(table=table)
        old_nodes+=len(levels)
        if len(values) and np.all(values==values[0]):
            # Its stage is irrelevant: activation uses the physical connection sill.
            levels=np.array([0.]);values=values[:1]
        values=values.astype(np.float32)
        encoded.append((levels,values))
    return encoded,error,maximum_rounding,old_nodes


def pack_coefficients(*, geometry, relative_tolerance, preprocessing_workers=1):
    """Encode with a certified coefficient tolerance; None preserves interval data.

    Scalar tables are shared only when their encoded stage/value bytes match.
    A constant scalar is stored once and evaluated directly. Opening elevations
    remain exact in geometry.sill; stored component volumes/curves are untouched.
    No row/table count cap or geometric eligibility restriction is added.
    """
    if geometry.coefficient_values is not None or geometry.diagnostics.get('coefficient_compression_tolerance') is not None:
        raise ValueError('compact encoding requires original uncompressed, unpacked tables')
    if relative_tolerance is not None and (not np.isfinite(relative_tolerance) or not 0<relative_tolerance<1):
        raise ValueError('coefficient tolerance must be None or finite in (0,1)')
    started=time.perf_counter();critical=port_critical_levels(geometry=geometry)
    tables=[];ranges=[];seen={};error=0.;maximum_rounding=0.;old_nodes=0
    jobs=((geometry.rows[geometry.offsets[port]:geometry.offsets[port+1]],critical[port],relative_tolerance)
        for port in range(len(geometry.port_sill)))
    workers=None
    if preprocessing_workers>1:
        from concurrent.futures import ThreadPoolExecutor
        from itertools import chain
        from floodwarp.enhanced_preprocessing import encode_port
        encode_port(np.empty((0,6)),np.empty(0),0.,False)
        workers=ThreadPoolExecutor(max_workers=preprocessing_workers)
        batches=iter_port_batches(jobs=jobs)
        encoded_ports=chain.from_iterable(workers.map(encode_port_batch,batches))
    else:encoded_ports=map(encode_port_coefficients,jobs)
    try:
        for encoded,local_error,local_rounding,node_count in encoded_ports:
            error=max(error,local_error);maximum_rounding=max(maximum_rounding,local_rounding)
            old_nodes+=node_count;record=[]
            for levels,values in encoded:
                key=levels.tobytes()+values.tobytes()
                if key not in seen:
                    seen[key]=len(tables);tables.append((levels,values))
                record.append(seen[key])
            ranges.append(record)
    finally:
        if workers is not None:workers.shutdown(wait=True)
    pool=np.unique(np.concatenate([t[0] for t in tables])) if tables else np.empty(0)
    index_type=np.uint16 if len(pool)<=np.iinfo(np.uint16).max+1 else np.int32
    sizes=np.asarray([len(table[0]) for table in tables],dtype=np.int64)
    ends=np.cumsum(sizes);starts=np.r_[0,ends[:-1]] if len(ends) else np.empty(0,np.int64)
    bounds=np.column_stack((starts,ends))
    all_levels=np.concatenate([table[0] for table in tables]) if tables else np.empty(0)
    indices=np.searchsorted(pool,all_levels).astype(index_type)
    values=np.concatenate([table[1] for table in tables]) if tables else np.empty(0,np.float32)
    paired=[];pair_ids={};mapping=[];sills=[]
    for port,pair in enumerate(ranges):
        key=tuple(pair)
        if key not in pair_ids:
            pair_ids[key]=len(paired);a,b=pair
            paired.append([*bounds[a],*bounds[b]]);sills.append(geometry.port_sill[port])
        mapping.append(pair_ids[key])
    halves=geometry.halves.copy();valid=halves>=0
    halves[valid]=np.asarray(mapping,np.int32)[halves[valid]]
    ranges=np.asarray(paired,np.int32).reshape(-1,4)
    values=np.asarray(values,np.float32).reshape(-1,1)
    return replace(geometry,halves=halves,port_sill=np.asarray(sills),offsets=np.array([0],np.int32),rows=pool.reshape(-1,1),
        coefficient_stage_index=np.asarray(indices,dtype=index_type),coefficient_ranges=ranges,
        coefficient_values=values,diagnostics=dict(geometry.diagnostics,
            coefficient_encoding='scalar_knots',coefficient_encoding_s=time.perf_counter()-started,
            coefficient_compression_tolerance=relative_tolerance,certified_table_relative_error=error,
            maximum_coefficient_relative_increase=maximum_rounding,
            encoded_error_bound=(1+error)*(1+maximum_rounding)-1,
            exact_stage_pool_count=len(pool),scalar_table_count=len(tables),original_scalar_table_count=2*len(geometry.port_sill),shared_port_table_count=len(ranges),
            scalar_nodes_before_sharing=old_nodes,scalar_nodes=len(values),stage_index_dtype=np.dtype(index_type).name))


def iter_port_batches(*, jobs):
    from itertools import islice
    iterator=iter(jobs)
    while True:
        batch=list(islice(iterator,256))
        if not batch:return
        yield batch


def encode_port_batch(parameters):
    return [encode_port_coefficients(job) for job in parameters]



def scalar_table_value(*, stage, levels, indices, values, begin, end):
    if end-begin==1:
        return float(values[begin,0])
    knots=levels[indices[begin:end],0]
    index=min(max(0,int(np.searchsorted(knots,stage,side='right')-1)),end-begin-2)
    k=begin+index
    fraction=np.clip((stage-levels[indices[k],0])/(levels[indices[k+1],0]-levels[indices[k],0]),0.,1.)
    a,b=float(values[k,0]),float(values[k+1,0])
    return float(a+fraction*(b-a))
