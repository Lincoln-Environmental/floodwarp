"""Explicit diagnostic profile controls; never modifies registered profiles."""
from dataclasses import replace
from pathlib import Path

from floodwarp.validation_profiles import PHYSICAL_CANDIDATE_PROFILE
from validation.li_replay import replay


if __name__ == "__main__":
    fixtures = Path('validation/fixtures')
    output = Path('outputs/li_review_v1/attribution_v2')
    default_output = Path('outputs/li_review_v1/shipped_defaults_v2')
    device = 'cpu'
    cases = ('acc_steep_10m_0mmh_dt1', 'acc_heterogeneous_10m_1mmh_dt0.5', 'acc_complex_5m_0mmh_dt1')
    variants = (
        dict(name='arithmetic_resistance', manning_face_mode=1),
        dict(name='no_boundary_transverse', include_boundary_transverse_friction=False),
        dict(name='acc_activation', require_cell_depth_above_dry=True),
        dict(name='sources_after', apply_sources_after_momentum=True),
        dict(name='acc_source_availability', source_mode=1),
        dict(name='acc_source_and_order', source_mode=1, apply_sources_after_momentum=True),
    )
    closures = ('local_or_prescribed_slope',)
    history_capacity = 200000
    profiles = tuple(replace(PHYSICAL_CANDIDATE_PROFILE, **variant) for variant in variants)
    replay(namespace='floodwarp', fixtures=fixtures, output=output, device=device, profiles=profiles,
           closures=closures, history_capacity=history_capacity, case_ids=cases)
    replay(namespace='floodwarp', fixtures=fixtures, output=default_output, device=device, profiles=(None,),
           closures=(None,), history_capacity=history_capacity, shipped_defaults=True)
