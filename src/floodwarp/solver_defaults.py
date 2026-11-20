"""Numerical selection for new models and for replaying saved options."""

DEFAULT_SOLVER_METHOD = "fv1_hancock"
DEFAULT_HANCOCK_LIMITER = "adaptive"
DEFAULT_FULL_SWE_FLUX = "contact_hll"


def canonical_solver_method(*, method):
    return "fv1_hancock" if method == "hancock" else method


def saved_solver_options(*, options):
    """Resolve historical implicit FV1 and pre-promotion implicit Hancock MC."""
    restored = dict(options)
    restored["solver_method"] = canonical_solver_method(method=restored.get("solver_method", "fv1"))
    if options.get("solver_method") == "fv1_hancock":
        restored.setdefault("limiter", "mc")
    return restored
