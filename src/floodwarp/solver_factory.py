"""Public selection of the frozen standalone hydraulic methods."""
from floodwarp.local_inertial_solver import LocalInertialSolver
from floodwarp.full_swe_solver import FullSWESolver


def create_overland_solver(*, solver_method="fv1", ny, nx, dx, dy,
                           bed_elevation, manning_n, active_mask, device="cpu",
                           validation_profile=None, explicit_minimum_dt=1.e-3,
                           explicit_maximum_dt=60., adaptive_dt_growth_factor=1.5,
                           hydraulic_geometry_mode="coarse", subgrid_bed_elevation=None,
                           subgrid_resolution_m=None, structures=(), **options):
    common = dict(ny=ny, nx=nx, dx=dx, dy=dy, bed_elevation=bed_elevation,
                  manning_n=manning_n, active_mask=active_mask, device=device,
                  minimum_dt=explicit_minimum_dt, maximum_dt=explicit_maximum_dt)
    if solver_method in ("fv1", "fv2", "fv1_muscl_euler", "fv1_hancock"):
        if hydraulic_geometry_mode != "coarse" or subgrid_bed_elevation is not None or subgrid_resolution_m is not None:
            raise ValueError(f"{solver_method.upper()} supports coarse geometry without SGS")
        if validation_profile is not None:
            raise ValueError(f"{solver_method.upper()} does not use local-inertial validation profiles")
        if solver_method != "fv1":
            from floodwarp.fv2_solver import FV2Solver, HancockSolver, MUSCLEulerSolver
            methods = {"fv2": FV2Solver, "fv1_muscl_euler": MUSCLEulerSolver, "fv1_hancock": HancockSolver}
            return methods[solver_method](**common, structures=structures, **options)
        return FullSWESolver(**common, structures=structures, **options)
    if solver_method in ("explicit", "li"):
        return LocalInertialSolver(**common, validation_profile=validation_profile,
            adaptive_dt_growth_factor=adaptive_dt_growth_factor,
            hydraulic_geometry_mode=hydraulic_geometry_mode,
            subgrid_bed_elevation=subgrid_bed_elevation,
            subgrid_resolution_m=subgrid_resolution_m, structures=structures, **options)
    raise ValueError("solver_method must be 'li', 'explicit' or one of 'fv1', 'fv2', 'fv1_muscl_euler', 'fv1_hancock'")
