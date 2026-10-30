"""Conservative certificates that avoid unnecessary exponential trial solves.

For q'=a-b*q*abs(q), b>=0, the frozen trajectory is monotone and the drag is
monotone dissipative. For midpoint coefficient changes da,db, its disturbance
is bounded over the trajectory by the maximum of |da-db*q*abs(q)| at its two
endpoints. Thus the endpoint difference is at most dt*B, and the midpoint
solution lies within |q1-q0|/2 + dt*B/2 of (q0+q1)/2. No small-stiffness
assumption, Taylor cancellation, or relaxation-time approximation is used.
"""
import warp as wp


@wp.func
def frozen_certificates(q0: wp.float64,q1: wp.float64,a0: wp.float64,b0: wp.float64,
                        a1: wp.float64,b1: wp.float64,dt: wp.float64):
    da=wp.float64(.5)*(a1-a0)
    db=wp.float64(.5)*(b1-b0)
    bound=wp.max(wp.abs(da-db*q0*wp.abs(q0)),wp.abs(da-db*q1*wp.abs(q1)))
    q_error=dt*bound
    midpoint=wp.float64(.5)*(q0+q1)
    radius=wp.float64(.5)*(wp.abs(q1-q0)+q_error)
    return wp.vec3d(q_error,midpoint,radius)
