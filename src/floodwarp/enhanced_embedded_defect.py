"""Committed embedded probes with an exact inactive-connection exit."""
import warp as wp
from floodwarp import enhanced_accuracy as reference
from floodwarp import enhanced_kernels as kernels
from floodwarp.enhanced_defect import frozen_certificates
from floodwarp.enhanced_momentum import frozen_discharge
from floodwarp.config import WP_FLOAT
from floodwarp.local_inertial_kernels import DL_DT


@wp.kernel(enable_backward=False)
def connections(control: wp.array(dtype=wp.float64),q: wp.array(dtype=WP_FLOAT),
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
    if control[reference.DL_DONE]!=wp.float64(0.) or control[reference.DL_ERROR]!=wp.float64(0.):return
    if d[f]==wp.float64(0.) and start_d[f]==wp.float64(0.) and q[f]==WP_FLOAT(0.):
        mid_q[f]=WP_FLOAT(0.);mid_radius[f]=wp.float64(0.)
        counts[f,0]+=1
        return
    corrected=wp.float64(q[f]);mid=wp.float64(q[f])
    magnitude=wp.max(wp.abs(wp.float64(q[f])),wp.abs(wp.float64(initial_q[f])))
    fast=int(0);radius=wp.float64(0.);error=wp.float64(0.)
    if d[f]>wp.float64(0.) and start_d[f]>wp.float64(0.):
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
            a=gravity*wp.float64(.5)*(pressure[f]/d[f]+start_pressure[f]/start_d[f])
            b=gravity*wp.float64(.5)*(r[f]/d[f]+start_r[f]/start_d[f])
            corrected=frozen_discharge(wp.float64(initial_q[f]),a,b,control[DL_DT])
            mid=frozen_discharge(wp.float64(initial_q[f]),a,b,control[DL_DT]*wp.float64(.5))
        if magnitude>wp.float64(0.) and (d[f]!=start_d[f] or r[f]!=start_r[f]):
            weight=magnitude/(q_atol+magnitude)
            de=wp.abs(d[f]-start_d[f])/wp.max(d[f],start_d[f])
            re=wp.abs(r[f]-start_r[f])/wp.max(wp.max(r[f],start_r[f]),wp.float64(1.e-100))
            ce=weight*wp.max(de,re)/coefficient_rtol
            reference.maximum(state,reference.COEFFICIENT_ERROR,ce)
    elif d[f]!=start_d[f]:
        reference.maximum(state,reference.COEFFICIENT_ERROR,wp.float64(1.)/coefficient_rtol)
        # Integrate the changed activation's pressure/drag for the trial
        # interval. This defect vanishes with dt even at a coefficient jump.
        if d[f]>wp.float64(0.):
            corrected=frozen_discharge(wp.float64(initial_q[f]),gravity*pressure[f]/d[f],gravity*r[f]/d[f],control[DL_DT])
            mid=frozen_discharge(wp.float64(initial_q[f]),gravity*pressure[f]/d[f],gravity*r[f]/d[f],control[DL_DT]*wp.float64(.5))
        else:
            corrected=wp.float64(0.);mid=wp.float64(0.)
    if fast==0:error=wp.abs(corrected-wp.float64(q[f]))/(q_atol+rtol*wp.max(wp.abs(corrected),magnitude))
    reference.maximum(state,reference.Q_ERROR,error)
    reference.maximum(state,reference.ERROR,error)
    mid_q[f]=WP_FLOAT(mid)
    mid_radius[f]=radius
    counts[f,fast]+=1
