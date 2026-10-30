"""Transactional, source-consistent wave CFL for enhanced global stepping.

The retry state is device-resident. Rejected trials never advance the accepted
clock, storage, soil state, momentum or source ledgers. No fractional-volume
criterion is used. The iteration bound detects failure; it does not accept an
unconverged or clipped step.
"""
import warp as wp
from floodwarp.local_inertial_kernels import (
    DL_SIZE, DL_TIME, DL_DT, DL_ERROR, DL_DONE, DL_GROWTH_REFERENCE, DL_RECORD_CHECKPOINT,
)

CAP = 0
CANDIDATE = 1
BOUND = 2
ATTEMPTS = 3
CHECKS = 4
REJECTIONS = 5
MAX_COURANT = 6
MAX_RETRIES = 64


@wp.kernel
def initialize(control: wp.array(dtype=wp.float64),status: wp.array(dtype=int),
               state: wp.array(dtype=wp.float64),condition: wp.array(dtype=int),cap: wp.float64):
    condition[0]=0
    if control[DL_DONE]==wp.float64(0.0) and control[DL_ERROR]==wp.float64(0.0) and status[0]==0:
        condition[0]=1
        state[CAP]=cap
        state[ATTEMPTS]=wp.float64(0.0)


@wp.kernel
def prepare(control: wp.array(dtype=wp.float64),trial: wp.array(dtype=wp.float64),
            status: wp.array(dtype=int),state: wp.array(dtype=wp.float64)):
    for i in range(DL_SIZE):trial[i]=control[i]
    trial[DL_DT]=state[CAP]
    status[0]=0


@wp.kernel
def remember(trial: wp.array(dtype=wp.float64),state: wp.array(dtype=wp.float64)):
    state[CANDIDATE]=trial[DL_DT]
    state[BOUND]=wp.float64(1.e100)


@wp.kernel
def wave_bound(offsets: wp.array(dtype=int),adjacent: wp.array(dtype=int),
               d: wp.array(dtype=wp.float64),area: wp.array(dtype=wp.float64),eta: wp.array(dtype=wp.float64),
               gravity: wp.float64,cfl: wp.float64,state: wp.array(dtype=wp.float64),
               status: wp.array(dtype=int),local: wp.array(dtype=wp.float64),
               track: int,limits: wp.array2d(dtype=wp.float64)):
    if status[0]!=0:return
    c=wp.tid();degree=wp.float64(0.0)
    for k in range(offsets[c],offsets[c+1]):
        inertia=d[wp.abs(adjacent[k])-1]
        if inertia>wp.float64(0.0):degree+=wp.float64(1.0)/inertia
    bound=wp.float64(1.e100)
    if degree>wp.float64(0.0):bound=cfl*wp.sqrt(wp.float64(2.0)*area[c]/(gravity*degree))
    wp.atomic_min(state,BOUND,bound)
    local[c]=wp.min(state[CANDIDATE],bound)
    if track==1:
        limits[c,0]=bound;limits[c,1]=wp.float64(1.e100)
        limits[c,2]=wp.float64(wp.nan);limits[c,3]=wp.float64(wp.nan)
        limits[c,4]=eta[c];limits[c,5]=area[c]


@wp.kernel
def decide(control: wp.array(dtype=wp.float64),trial: wp.array(dtype=wp.float64),
           accepted_status: wp.array(dtype=int),trial_status: wp.array(dtype=int),
           state: wp.array(dtype=wp.float64),condition: wp.array(dtype=int),
           minimum: wp.float64,maximum: wp.float64,growth: wp.float64,end: wp.float64,cfl: wp.float64):
    state[CHECKS]+=wp.float64(1.0);state[ATTEMPTS]+=wp.float64(1.0)
    candidate=state[CANDIDATE];bound=state[BOUND]
    if trial_status[0]!=0 or trial[DL_ERROR]!=wp.float64(0.0):
        accepted_status[0]=trial_status[0]
        if accepted_status[0]==0:accepted_status[0]=6
        condition[0]=0
        control[DL_ERROR]=wp.float64(4.0)
    elif trial[DL_DONE]!=wp.float64(0.0):
        for i in range(DL_SIZE):control[i]=trial[i]
        condition[0]=0
    elif candidate<=wp.float64(0.0) or not wp.isfinite(candidate) or bound<=wp.float64(0.0):
        accepted_status[0]=6;control[DL_ERROR]=wp.float64(4.0);condition[0]=0
    elif bound<candidate:
        state[CAP]=bound
        state[REJECTIONS]+=wp.float64(1.0)
        if state[ATTEMPTS]>=wp.float64(MAX_RETRIES):
            accepted_status[0]=6;control[DL_ERROR]=wp.float64(4.0);condition[0]=0
    else:
        uncapped=wp.min(maximum,wp.min(state[CAP],bound))
        if growth>=wp.float64(1.0) and control[DL_GROWTH_REFERENCE]>wp.float64(0.0):
            uncapped=wp.min(uncapped,control[DL_GROWTH_REFERENCE]*growth)
        for i in range(DL_SIZE):control[i]=trial[i]
        remaining=end-control[DL_TIME]
        if trial[DL_RECORD_CHECKPOINT]!=wp.float64(0.0):remaining=wp.min(remaining,candidate)
        if uncapped<minimum and remaining>=minimum:
            control[DL_ERROR]=wp.float64(2.0)
        state[MAX_COURANT]=wp.max(state[MAX_COURANT],cfl*candidate/bound)
        condition[0]=0


@wp.kernel
def host_clock(control: wp.array(dtype=wp.float64),time: wp.float64,reference: wp.float64):
    control[DL_TIME]=time;control[DL_GROWTH_REFERENCE]=reference
    control[DL_DONE]=wp.float64(0.0);control[DL_ERROR]=wp.float64(0.0)
