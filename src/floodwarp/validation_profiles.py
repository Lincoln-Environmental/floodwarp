"""Named, opt-in numerical compatibility profiles for external validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalInertialValidationProfile:
    """Numerical controls used to configure the local-inertial solver."""

    name: str
    dry_depth: float
    source_mode: int
    manning_face_mode: int
    use_two_dimensional_friction: bool
    free_outflow_mode: int
    free_outflow_limiter_mode: int
    use_lisflood_acc_timestep: bool
    apply_sources_after_momentum: bool
    require_cell_depth_above_dry: bool
    include_boundary_transverse_friction: bool = False


DEFAULT_PROFILE = LocalInertialValidationProfile(
    name="default",
    dry_depth=1.0e-4,
    source_mode=0,
    manning_face_mode=0,
    use_two_dimensional_friction=False,
    free_outflow_mode=0,
    free_outflow_limiter_mode=0,
    use_lisflood_acc_timestep=False,
    apply_sources_after_momentum=False,
    require_cell_depth_above_dry=False,
)


# Exact numerical-compatibility profile retained for targeted regression tests.
# This includes the LISFLOOD-FP ACC global depth-based timestep rule.
LISFLOOD_FP_ACC_PROFILE = LocalInertialValidationProfile(
    name="lisflood_fp_acc",
    dry_depth=1.0e-3,
    source_mode=1,
    manning_face_mode=1,
    use_two_dimensional_friction=True,
    free_outflow_mode=1,
    free_outflow_limiter_mode=0,
    use_lisflood_acc_timestep=True,
    apply_sources_after_momentum=True,
    require_cell_depth_above_dry=True,
)


# Parity profile for normal validation. The hydraulic/source discretisation is
# kept comparable with LISFLOOD-FP ACC, but Overlander is deliberately free to
# select its own face-based adaptive timestep. This is the profile to use when
# LISFLOOD-FP is acting as an independent reference solution rather than as a
# timestep specification.
LISFLOOD_FP_ACC_PARITY_PROFILE = LocalInertialValidationProfile(
    name="lisflood_fp_acc_parity",
    dry_depth=1.0e-3,
    source_mode=1,
    manning_face_mode=1,
    use_two_dimensional_friction=True,
    free_outflow_mode=1,
    free_outflow_limiter_mode=0,
    use_lisflood_acc_timestep=False,
    apply_sources_after_momentum=True,
    require_cell_depth_above_dry=True,
)


PROFILES = {
    DEFAULT_PROFILE.name: DEFAULT_PROFILE,
    LISFLOOD_FP_ACC_PROFILE.name: LISFLOOD_FP_ACC_PROFILE,
    LISFLOOD_FP_ACC_PARITY_PROFILE.name: LISFLOOD_FP_ACC_PARITY_PROFILE,
}


# Research candidate only. Historical records above remain unchanged. Mode 2
# averages n² over the two equal half-cell resistance lengths; it is not a
# universal effective roughness for varying depth or parallel flow.
PHYSICAL_CANDIDATE_PROFILE = LocalInertialValidationProfile(
    name="physical_candidate",
    dry_depth=1.0e-4,
    source_mode=0,
    manning_face_mode=2,
    use_two_dimensional_friction=True,
    free_outflow_mode=0,
    free_outflow_limiter_mode=0,
    use_lisflood_acc_timestep=False,
    apply_sources_after_momentum=False,
    require_cell_depth_above_dry=False,
    include_boundary_transverse_friction=True,
)

PROFILES[PHYSICAL_CANDIDATE_PROFILE.name] = PHYSICAL_CANDIDATE_PROFILE


def resolve_validation_profile(profile_name: str | LocalInertialValidationProfile | None) -> LocalInertialValidationProfile:
    """Return a named solver profile."""
    if isinstance(profile_name, LocalInertialValidationProfile):
        if profile_name.name in PROFILES and profile_name != PROFILES[profile_name.name]:
            raise ValueError("An experimental profile must have its own name")
        return profile_name
    resolved_name = DEFAULT_PROFILE.name if profile_name is None else profile_name
    if resolved_name not in PROFILES:
        valid_names = tuple(PROFILES)
        raise ValueError(
            f"unknown validation profile {profile_name!r}; expected one of {valid_names}"
        )
    return PROFILES[resolved_name]
