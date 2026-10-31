# Experimental static block FV1 qualification budgets

These budgets are specified before evaluating static-mesh results. They are
independent of the exact-execution equivalence controls. All fine geometry and
source history remain immutable/persistent at their original resolution.

The first mesh has two levels, regular blocks, spatial ratio 2:1, a fixed leaf
partition and one global timestep. No temporal subcycling is permitted. Each
physical volume evolves on exactly one leaf. Coarse–fine transport uses the
same single accepted fine subface flux on both sides, and a shared coarse donor
budget across all of its subfaces.

Conservation residual must be <= 1e-8 times max(1 m3, initial plus supplied
water). Depth must remain finite and nonnegative. Representable lake equilibrium
must retain depth and momentum to 1e-12 SI absolute error. Restriction and
prolongation must preserve area-integrated water and momentum to FP64 roundoff.

Against dense FloodWarp, require simultaneous wet-union depth RMSE <= 0.001 m
or relative L2 <= 0.02, and maximum-depth RMSE <= 0.001 m or relative L2 <= 0.02.
Require maximum-event inundation IoU >= 0.95 or domain mismatch <= 0.0005 at
the retained 0.001 m diagnostic threshold. This threshold never deletes water.
For hydrographs, require NRMSE <= 0.05, peak-discharge relative error <= 0.02,
and peak-time difference <= one sampled output interval, independently.
Cumulative export, infiltration and storage each require error <= 5 m3 plus
2% of the larger reference quantity. Record source-aggregation error separately.

Geometry vetoes must protect physical-mask interfaces, prescribed openings,
explicit protected features and non-affine/connectivity-changing terrain.
Source/roughness heterogeneity also vetoes initial coarse blocks. Mixed wet/dry
blocks are ineligible. Free-surface variation is assessed separately from depth
variation, so an affine-bed stationary lake can be coarsened.

For the initial representation, a coarse wet cell uses a common free surface
and velocity over its immutable fine bed. Fine source calculations receive
these reconstructed nonnegative depths; accepted conservative source increments
are restricted back to the leaf. Dry coarse cells require flat geometry. An
unrepresentable mixed wet/dry coarse state must reject the numerical step rather
than silently redistribute water through a sill. Static controls must demonstrate
these restrictions before considering changing topology.

A useful static candidate must pass integrity/equilibrium controls and show a
measured accuracy/runtime improvement. Fewer leaves alone is not a speedup.
Profile projection, restriction, subface aggregation and source costs separately.
If the initial execution is slower, preserve the failure and do not enable
dynamic adaptation merely to call the stage complete.
