---
name: Bug report
about: Reproducible installation, input, interface or numerical issue
title: ''
labels: bug
assignees: ''
---

**What happened and what did you expect?**

**Reproduction**
Provide the smallest script or UI steps, exact command, duration and output
interval. Synthetic shareable inputs are preferred. Do not include confidential
data, access tokens or credentials.

**Environment**
FloodWarp version/commit and installation source (wheel/source); Python and OS;
CPU/CUDA device and driver; `python -m pip check` result.

**Model configuration**
Solver method, precision, sparse/dense mode, graph setting, boundary closures,
roughness, infiltration and rainfall units/interpolation. Include DEM CRS,
transform, shape, resolution, datum and active-cell coverage for input issues.

**Evidence**
Attach the concise error, relevant run.log excerpt, run_config.json with private
paths removed, sampled diagnostics and whether a restart checkpoint survives.
For numerical differences distinguish final depth, event maxima, extent,
aggregate boundary flow and mass residual. State the comparison source and
physical tolerances. Completion alone does not establish calibration.
