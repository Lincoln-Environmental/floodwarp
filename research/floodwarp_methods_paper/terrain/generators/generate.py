"""Seeded variations of the eight audited continuous synthetic landform equations."""

import hashlib
import numpy as np

from validation.hancock_long_storm_review.landforms import physical_fields


def parameters(*, design, template, family, seed, initial):
    index=design["landforms"].index(family)
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
    grade,incision,_=template["landforms"][family]
    return dict(grade=float(grade*rng.uniform(*design["slope_scale"])),incision_m=float(incision*rng.uniform(*design["incision_scale"])),phase_seed=int(rng.integers(2**31)),initial=initial)


def fields(*, design, template, family, seed, initial, dx, dem):
    params=parameters(design=design,template=template,family=family,seed=seed,initial=initial)
    length,width=design["dimensions_m"]
    arrays=physical_fields(landform=family,spec=(params["grade"],params["incision_m"],initial),seed=params["phase_seed"],length=length,width=width,dx=dx,terrain_dx=dem)
    return arrays,params


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
