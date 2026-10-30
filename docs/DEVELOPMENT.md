# Development

Work on dev. main receives ordinary validated merges/fast-forwards. This
repository has independent history; do not change the original Overlander
research checkout as part of FloodWarp maintenance.

Use a fresh Python 3.14 environment with `pip install '.[test,raster,plot]'`.
The numerical core pins the source baseline's NumPy 2.3.5, SciPy 1.16.3,
Numba 0.67.0 and Warp 1.17.0. Do not upgrade them as housekeeping.

Maintained entry points, from the repository root:

- `python -m examples.run_synthetic`: small LI/FV1 installation smoke test.
- `python -m examples.run_purau`: 40×40 real-terrain CPU rainfall demonstration.
- `python -m examples.run_direct`: user-editable GeoTIFF paths and controls.
- `python -m validation.compare`: bounded independent reference comparisons.
- `python -m validation.sensitivity`: assess completed comparison outputs.
- `python -m pytest -q`: retained numerical and direct-input regressions.
- `OVERLANDER_TEST_FULL_SWE_CUDA=1 python -m pytest -q`: opt-in extra FV1 CUDA controls; historical test flag retained.

Python scripts use named reusable functions, pathlib paths, explicit keyword
arguments and adjustable controls inside `if __name__ == "__main__":`.
No argparse, path injection or other-repository imports are needed.
Numerical library modules are preserved baseline code; avoid incidental
formatting/refactoring that obscures operator provenance.

The standalone precision environment names are FLOODWARP_PRECISION and
FLOODWARP_FLOAT. Existing OVERLANDER_* tuning flags remain for reproducible
kernel/test controls; they do not imply a dependency on another installation.
CPU getters may return views: copy arrays before retaining sampled output.

## Numerical constraints

Preserve physical LI defaults, FV1 contact_hll/velocity_extrapolation, FP64,
CFL bounds, wet/dry tolerances, reference arrays and restart identity guards.
Source-compatible profile choices are explicit. Never replace operators with
LISFLOOD code or tune roughness to improve comparison scores. Treat native
comparisons as independent implementations, not physical truth.

Both SGS modes remain experimental with visible warnings and their historical
physics. Do not apply physical-LI friction/boundary updates to SGS as cleanup.
Inclined SGS, semi-implicit and routing research are excluded. No campaigns,
training, external-model coupling or prepared-data workflows belong here.

Record simultaneous fields separately from maximum-depth envelopes. Never
sum cellwise event maxima as storage. Report failed gates and missing coverage.
Keep this candidate private until the software redistribution licence and
remaining release blockers are resolved.
