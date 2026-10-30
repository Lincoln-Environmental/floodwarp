"""Exact local-inertial momentum for frozen pressure and Manning coefficients."""
import warp as wp


@wp.func
def frozen_discharge(q: wp.float64,a: wp.float64,b: wp.float64,dt: wp.float64):
    """Solve Q'=a-b Q|Q|, b>=0, without iteration or a global solve.

    Turn the forcing positive. Opposing flow follows a tangent branch until
    its exact zero-crossing, then a hyperbolic branch. Reciprocal/convex forms
    avoid overflow in the strongly super/sub-equilibrium positive branches.
    """
    if dt==wp.float64(0.0):return q
    if b==wp.float64(0.0):return q+a*dt
    if a==wp.float64(0.0):return q/(wp.float64(1.0)+b*dt*wp.abs(q))
    sign=wp.float64(1.0)
    if a<wp.float64(0.0):sign=wp.float64(-1.0)
    force=wp.abs(a);u=sign*q
    root_force=wp.sqrt(force);root_drag=wp.sqrt(b)
    equilibrium=root_force/root_drag;rate=root_force*root_drag
    remaining=dt
    if u<wp.float64(0.0):
        ratio=(-u)*(root_drag/root_force)
        if not wp.isfinite(root_drag/root_force):ratio=((-u)*root_drag)/root_force
        crossing=(-u)/force
        if ratio>wp.float64(0.0):crossing=wp.atan(ratio)/rate
        if dt<crossing:
            x=rate*dt;h=dt
            if x!=wp.float64(0.0):h=wp.tan(x)/rate
            return sign*((u+force*h)/(wp.float64(1.0)-b*h*u))
        remaining=dt-crossing;u=wp.float64(0.0)
    x=rate*remaining
    t=wp.tanh(x)
    if t==wp.float64(1.0):return sign*equilibrium
    if wp.isfinite(equilibrium) and x!=wp.float64(0.0):
        if u>=equilibrium:
            z=equilibrium/u
            return sign*equilibrium*((wp.float64(1.0)+z*t)/(z+t))
        z=u/equilibrium
        weight=(wp.float64(1.0)+z)*t/(wp.float64(1.0)+z*t)
        return sign*(u+(equilibrium-u)*weight)
    h=remaining
    if x!=wp.float64(0.0):h=t/rate
    return sign*((u+force*h)/(wp.float64(1.0)+b*h*u))
