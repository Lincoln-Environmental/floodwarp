"""Subgrid hydraulic geometry: host preprocessing for the experimental mode.

The subgrid mode keeps the coarse cell as the computational control volume
but represents its storage and face hydraulics from the native-resolution
user-supplied DEM (or a synthetic-terrain ``subgrid_dem``
product).  The solver state ``water_depth`` remains the conservative
storage-equivalent depth ``s = V / (dx*dy)``; this module precomputes the
static geometry needed to derive, per step, the physical quantities the
coarse mode reads directly off ``bed_elevation + water_depth``:

* per-cell hypsometry (sorted native elevations and the storage breakpoints
  of the piecewise-linear storage curve) for the deterministic
  storage-to-stage inversion ``eta = S^-1(s)``;
* per-face native segment elevations: each internal active-active segment
  uses the maximum of the two adjoining native cells. At an active/inactive
  mask edge, only the active cell's native edge defines the outflow sill.
  Segment minima drive wet/dry detection and CFL wave speeds;
* the integer subgrid factor.

Mathematical definition (see AGENTS.md / docs for the full model)
-----------------------------------------------------------------
For one coarse cell with native elevations z_k (k = 0..K-1, K = factor**2)
and water-surface elevation eta::

    s(eta) = (1/K) * sum_k max(eta - z_k, 0)

With z sorted ascending and cumulative sums C_m = sum_{k<=m} z_k, the
storage breakpoints are ``s_bp[m] = (m * z_m - C_{m-1}) / K`` (the storage
at eta = z_m).  The inverse walks the piecewise-linear segments: on the
segment above z_m there are m+1 wet subcells so ds/deta = (m+1)/K, and
above the top breakpoint ds/deta = 1.  A flat subgrid (all z_k equal to
the coarse bed) collapses the inversion to ``eta = z + s`` and the face
quantities below to the coarse model exactly.

For a face with segment elevations z_{f,k} and face water surface
eta_f = max(eta_L, eta_R) (the existing upwind/max-stage convention)::

    h_A = (1/F) sum_k max(eta_f - z_{f,k}, 0)      (area depth, preserves
                                                    wetted cross-section)
    M   = (1/F) sum_k max(eta_f - z_{f,k}, 0)^(5/3)
    n_eff = n_f * h_A^(5/3) / M   (M > 0)

so the steady coarse-face discharge q = M sqrt(Sf) / n_f is the
width-normalised sum of the native-segment Manning conveyances.  When a
native roughness raster is supplied, each segment k carries its own n_k and
the reduction generalises without changing the convention::

    K   = (1/F) sum_k max(eta_f - z_{f,k}, 0)^(5/3) / n_k
    n_eff = h_A^(5/3) / K   (K > 0)

which is exactly the uniform-roughness form above when every n_k = n_f
(K = M / n_f).  Dry/blocked segments have h_k = 0 and contribute nothing
through either reduction.  The
maximum local face depth ``max(eta_f - min_k z_{f,k}, 0)`` — not h_A —
drives wet/dry detection and CFL wave speeds so a narrow deep channel is
neither dried out nor underestimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HYDRAULIC_GEOMETRY_MODES = ("coarse", "subgrid", "enhanced", "traditional")


def resolve_hydraulic_geometry_mode(mode: str) -> str:
    """Validate a hydraulic-geometry mode name.

    :param mode: ``"coarse"`` for ordinary hydraulics, ``"enhanced"`` (also
        ``"subgrid"``) for component SGS, or ``"traditional"`` for the original
        common-stage SGS. Validation retains the supplied spelling.
    :return: The validated mode string.
    :raises ValueError: On any unsupported mode (never silently falls back).
    """
    if mode not in HYDRAULIC_GEOMETRY_MODES:
        raise ValueError(
            "hydraulic_geometry_mode must be one of "
            f"{HYDRAULIC_GEOMETRY_MODES}, got {mode!r}"
        )
    return mode


def resolve_subgrid_factor(
    ny: int,
    nx: int,
    native_shape: tuple[int, int],
) -> int:
    """Validate native alignment and return the integer subgrid factor.

    :param ny: Coarse-grid row count (solver orientation).
    :param nx: Coarse-grid column count.
    :param native_shape: ``(rows, cols)`` of the native-resolution raster in
        the same orientation.
    :return: The integer factor with ``native_shape == (ny*factor, nx*factor)``.
    :raises ValueError: On non-integer ratios or mismatched dimensions.
    """
    if ny < 1 or nx < 1:
        raise ValueError("coarse-grid dimensions must be positive")
    native_rows, native_cols = (int(value) for value in native_shape)
    factor_y = native_rows / ny
    factor_x = native_cols / nx
    if (
        abs(factor_y - round(factor_y)) > 1e-9
        or abs(factor_x - round(factor_x)) > 1e-9
        or int(round(factor_y)) != int(round(factor_x))
    ):
        raise ValueError(
            f"subgrid terrain shape {native_shape} must be an exact common "
            f"integer multiple of the model grid {(ny, nx)} in both axes "
            f"(got {factor_y} x {factor_x}); non-integer ratios are not "
            "supported"
        )
    factor = int(round(factor_y))
    if native_rows != ny * factor or native_cols != nx * factor:
        raise ValueError(
            f"subgrid terrain shape {native_shape} is inconsistent with the "
            f"model grid {(ny, nx)} at factor {factor}"
        )
    if factor < 1:
        raise ValueError(f"subgrid factor must be at least 1, got {factor}")
    return factor


@dataclass(frozen=True)
class SubgridGeometry:
    """Static subgrid geometry precomputed on the host with NumPy.

    Array orientations match the solver convention (row 0 = southernmost
    row); the caller must supply the native raster already converted.  All
    arrays are float64 host arrays ready for ``wp.array`` upload.
    """

    ny: int
    nx: int
    factor: int
    subgrid_resolution_m: float

    #: (ny, nx, K) sorted-ascending native elevations per coarse cell.
    cell_z: np.ndarray
    #: (ny, nx, K) storage-equivalent depth at eta = cell_z[..., m].
    cell_storage_breakpoints: np.ndarray
    #: (ny, nx+1, factor) x-face native segment elevations.
    face_z_x: np.ndarray
    #: (ny, nx+1) x-face minimum segment elevation.
    face_z_min_x: np.ndarray
    #: (ny+1, nx, factor) y-face native segment elevations.
    face_z_y: np.ndarray
    #: (ny+1, nx) y-face minimum segment elevation.
    face_z_min_y: np.ndarray
    #: (ny, nx+1, factor) x-face native segment Manning values, or None when
    #: no native roughness was supplied (the coarse face value drives n_eff).
    face_n_x: np.ndarray | None = None
    #: (ny+1, nx, factor) y-face native segment Manning values, or None.
    face_n_y: np.ndarray | None = None

    @property
    def subgrid_cell_count(self) -> int:
        """Native subcells K per coarse cell (``factor**2``)."""
        return int(self.factor) * int(self.factor)

    def memory_bytes(self) -> dict[str, int]:
        """Approximate host/device footprint of the precomputed arrays."""
        sizes = {
            "cell_z": self.cell_z.nbytes,
            "cell_storage_breakpoints": self.cell_storage_breakpoints.nbytes,
            "face_z_x": self.face_z_x.nbytes,
            "face_z_min_x": self.face_z_min_x.nbytes,
            "face_z_y": self.face_z_y.nbytes,
            "face_z_min_y": self.face_z_min_y.nbytes,
        }
        if self.face_n_x is not None:
            sizes["face_n_x"] = self.face_n_x.nbytes
        if self.face_n_y is not None:
            sizes["face_n_y"] = self.face_n_y.nbytes
        return {"total": sum(sizes.values()), **sizes}


def build_subgrid_geometry(
    ny: int,
    nx: int,
    dx: float,
    subgrid_bed_elevation: np.ndarray,
    subgrid_resolution_m: float,
    subgrid_manning_n: np.ndarray | None = None,
    manning_face_mode: int = 0,
    active_mask: np.ndarray | None = None,
    dy: float | None = None,
) -> SubgridGeometry:
    """Precompute cell hypsometry and face segments from the native raster.

    :param ny: Coarse-grid row count.
    :param nx: Coarse-grid column count.
    :param dx: Coarse cell size (metres); must be an exact integer multiple
        of ``subgrid_resolution_m``.
    :param subgrid_bed_elevation: Native-resolution raster ``(ny*f, nx*f)``
        in solver orientation (row 0 = southernmost row), fully finite.
    :param dy: Optional coarse y size; must equal dx. Omission means square cells.
    :param subgrid_resolution_m: Native square pixel size (metres).
    :param active_mask: Optional coarse active-cell mask. At an active/inactive
        face, only the active cell's native edge defines the hydraulic sill.
    :param subgrid_manning_n: Optional native-resolution Manning raster with
        the same shape as ``subgrid_bed_elevation``. When supplied, native
        face roughness segments are built on exactly the same face layout as
        the face-elevation segments and drive the variable-roughness
        conveyance reduction.
    :param manning_face_mode: Coarse face-roughness convention reused
        segment-by-segment when combining the two native segments of an
        internal face: ``1`` arithmetic mean, otherwise maximum.
    :return: The immutable :class:`SubgridGeometry`.
    :raises ValueError: On non-integer ratios, malformed dimensions,
        non-finite native terrain, or invalid native roughness.
    """
    coarse_dx = float(dx)
    coarse_dy = coarse_dx if dy is None else float(dy)
    if not np.isfinite(coarse_dx) or coarse_dx <= 0.0:
        raise ValueError("subgrid dx must be finite and positive")
    if not np.isfinite(coarse_dy) or coarse_dy <= 0.0:
        raise ValueError("subgrid dy must be finite and positive")
    if coarse_dx != coarse_dy:
        raise ValueError("subgrid mode requires square coarse cells (dx == dy)")
    resolution = float(subgrid_resolution_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("subgrid_resolution_m must be finite and positive")
    factor = coarse_dx / resolution
    if not np.isfinite(factor) or factor < 1.0 or abs(factor - round(factor)) > 1e-9:
        raise ValueError(
            "subgrid mode requires dx to be an exact integer multiple of "
            f"subgrid_resolution_m (got {float(dx)} / {resolution} = {factor})"
        )
    native = np.asarray(subgrid_bed_elevation, dtype=np.float64)
    if native.ndim != 2:
        raise ValueError(
            f"subgrid_bed_elevation must be two-dimensional, got {native.shape}"
        )
    declared_factor = int(round(factor))
    factor = resolve_subgrid_factor(ny=ny, nx=nx, native_shape=native.shape)
    if factor != declared_factor:
        raise ValueError(
            f"native raster shape {native.shape} implies factor {factor}, but "
            f"dx={coarse_dx}, dy={coarse_dy} and subgrid_resolution_m={resolution} "
            f"require factor {declared_factor} and shape "
            f"{(ny * declared_factor, nx * declared_factor)}"
        )
    active = None
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (ny, nx):
            raise ValueError("active_mask shape must match the coarse grid")
    if not np.all(np.isfinite(native)):
        raise ValueError(
            "subgrid_bed_elevation must be finite everywhere; resolve native "
            "gaps before constructing a subgrid solver"
        )

    # ---- per-cell hypsometry ------------------------------------------------
    subgrid_cell_count = factor * factor
    cell_z = (
        native.reshape(ny, factor, nx, factor)
        .transpose(0, 2, 1, 3)
        .reshape(ny, nx, subgrid_cell_count)
        .copy()  # Sorting must never alias or reorder the caller's DEM.
    )
    cell_z.sort(axis=2)
    cumulative = np.cumsum(cell_z, axis=2)
    previous_cumulative = np.concatenate(
        [np.zeros((ny, nx, 1)), cumulative[:, :, :-1]], axis=2
    )
    segment_index = np.arange(subgrid_cell_count, dtype=np.float64)
    storage_breakpoints = (
        segment_index * cell_z - previous_cumulative
    ) / subgrid_cell_count

    # ---- x faces (normal in x, segments along y) ---------------------------
    # Internal x face (j, i) sits between native columns i*f-1 and i*f; each
    # native-width segment takes the maximum of the two adjoining cells.
    left = native[:, np.arange(1, nx) * factor - 1]
    right = native[:, np.arange(1, nx) * factor]
    left_x = left.reshape(ny, factor, nx - 1).transpose(0, 2, 1)
    right_x = right.reshape(ny, factor, nx - 1).transpose(0, 2, 1)
    internal_x = np.maximum(left_x, right_x)
    if active is not None:
        internal_x = np.where(
            (active[:, :-1] & ~active[:, 1:])[..., None], left_x,
            np.where((~active[:, :-1] & active[:, 1:])[..., None], right_x, internal_x),
        )
    west = native[:, 0].reshape(ny, factor)[:, np.newaxis, :]
    east = native[:, -1].reshape(ny, factor)[:, np.newaxis, :]
    face_z_x = np.concatenate([west, internal_x, east], axis=1)

    # ---- y faces (normal in y, segments along x) ---------------------------
    # Internal y face (j, i) sits between native rows j*f-1 (top of coarse
    # row j-1) and j*f (bottom of coarse row j).  A native row is laid out
    # column-major in coarse cells (column = i*f + c), so each row reshapes
    # directly to (nx, factor) segments.
    below = native[np.arange(1, ny) * factor - 1, :]
    above = native[np.arange(1, ny) * factor, :]
    below_y = below.reshape(ny - 1, nx, factor)
    above_y = above.reshape(ny - 1, nx, factor)
    internal_y = np.maximum(below_y, above_y)
    if active is not None:
        internal_y = np.where(
            (active[:-1, :] & ~active[1:, :])[..., None], below_y,
            np.where((~active[:-1, :] & active[1:, :])[..., None], above_y, internal_y),
        )
    south = native[0, :].reshape(nx, factor)[np.newaxis, :, :]
    north = native[-1, :].reshape(nx, factor)[np.newaxis, :, :]
    face_z_y = np.concatenate([south, internal_y, north], axis=0)

    # ---- native face roughness (optional) ----------------------------------
    # Built on exactly the same native face layout as the face-elevation
    # segments above. Internal faces combine the two adjoining native
    # segments segment-by-segment with the solver's existing coarse
    # manning_face_mode convention (mode 1: arithmetic mean, otherwise
    # maximum); envelope faces take the native edge values of the adjacent
    # active cell. No new roughness averaging convention is introduced.
    face_n_x = None
    face_n_y = None
    if subgrid_manning_n is not None:
        native_n = np.asarray(subgrid_manning_n, dtype=np.float64)
        if native_n.shape != native.shape:
            raise ValueError(
                f"subgrid_manning_n shape {native_n.shape} must match "
                f"subgrid_bed_elevation shape {native.shape}"
            )
        if not np.all(np.isfinite(native_n)) or np.any(native_n <= 0.0):
            raise ValueError(
                "subgrid_manning_n must be finite and strictly positive"
            )

        def combine_face_manning(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            if int(manning_face_mode) == 1:
                return 0.5 * (left + right)
            return np.maximum(left, right)

        left_n = native_n[:, np.arange(1, nx) * factor - 1]
        right_n = native_n[:, np.arange(1, nx) * factor]
        left_n_x = left_n.reshape(ny, factor, nx - 1).transpose(0, 2, 1)
        right_n_x = right_n.reshape(ny, factor, nx - 1).transpose(0, 2, 1)
        internal_n_x = combine_face_manning(left_n_x, right_n_x)
        if active is not None:
            internal_n_x = np.where(
                (active[:, :-1] & ~active[:, 1:])[..., None], left_n_x,
                np.where((~active[:, :-1] & active[:, 1:])[..., None], right_n_x, internal_n_x),
            )
        west_n = native_n[:, 0].reshape(ny, factor)[:, np.newaxis, :]
        east_n = native_n[:, -1].reshape(ny, factor)[:, np.newaxis, :]
        face_n_x = np.concatenate([west_n, internal_n_x, east_n], axis=1)

        below_n = native_n[np.arange(1, ny) * factor - 1, :]
        above_n = native_n[np.arange(1, ny) * factor, :]
        below_n_y = below_n.reshape(ny - 1, nx, factor)
        above_n_y = above_n.reshape(ny - 1, nx, factor)
        internal_n_y = combine_face_manning(below_n_y, above_n_y)
        if active is not None:
            internal_n_y = np.where(
                (active[:-1, :] & ~active[1:, :])[..., None], below_n_y,
                np.where((~active[:-1, :] & active[1:, :])[..., None], above_n_y, internal_n_y),
            )
        south_n = native_n[0, :].reshape(nx, factor)[np.newaxis, :, :]
        north_n = native_n[-1, :].reshape(nx, factor)[np.newaxis, :, :]
        face_n_y = np.concatenate([south_n, internal_n_y, north_n], axis=0)

    return SubgridGeometry(
        ny=ny,
        nx=nx,
        factor=factor,
        subgrid_resolution_m=resolution,
        cell_z=np.ascontiguousarray(cell_z),
        cell_storage_breakpoints=np.ascontiguousarray(storage_breakpoints),
        face_z_x=np.ascontiguousarray(face_z_x),
        face_z_min_x=np.ascontiguousarray(face_z_x.min(axis=2)),
        face_z_y=np.ascontiguousarray(face_z_y),
        face_z_min_y=np.ascontiguousarray(face_z_y.min(axis=2)),
        face_n_x=(
            None if face_n_x is None else np.ascontiguousarray(face_n_x)
        ),
        face_n_y=(
            None if face_n_y is None else np.ascontiguousarray(face_n_y)
        ),
    )


def storage_from_stage(
    geometry: SubgridGeometry,
    stage: np.ndarray,
) -> np.ndarray:
    """Forward hypsometry ``s(eta)`` per coarse cell (host reference).

    Used by tests and by :meth:`set_water_surface_elevation`-style entry
    points; the device kernels carry their own copy of the inversion.

    :param geometry: Precomputed subgrid geometry.
    :param stage: Water-surface elevations ``(ny, nx)``.
    :return: Storage-equivalent depths ``(ny, nx)`` (``V / (dx*dy)``).
    """
    eta = np.asarray(stage, dtype=np.float64)
    if eta.shape != (geometry.ny, geometry.nx):
        raise ValueError(f"stage must have shape {(geometry.ny, geometry.nx)}")
    depth = np.maximum(eta[:, :, np.newaxis] - geometry.cell_z, 0.0)
    return depth.mean(axis=2)


def stage_from_storage(
    geometry: SubgridGeometry,
    storage: np.ndarray,
) -> np.ndarray:
    """Inverse hypsometry ``eta = S^-1(s)`` per coarse cell (host reference).

    Mirrors the device inversion exactly: scan the piecewise-linear segments
    in ascending storage, then fall through to the unit-slope top branch.

    :param geometry: Precomputed subgrid geometry.
    :param storage: Storage-equivalent depths ``(ny, nx)``.
    :return: Water-surface elevations ``(ny, nx)``.
    """
    s = np.asarray(storage, dtype=np.float64)
    if s.shape != (geometry.ny, geometry.nx):
        raise ValueError(f"storage must have shape {(geometry.ny, geometry.nx)}")
    cell_z = geometry.cell_z
    breakpoints = geometry.cell_storage_breakpoints
    subgrid_cell_count = geometry.subgrid_cell_count
    if subgrid_cell_count == 1:
        # The native-resolution control has no interior hypsometric knots.
        return cell_z[:, :, 0] + s
    qualifies = s[:, :, np.newaxis] <= breakpoints[:, :, 1:]
    any_hit = qualifies.any(axis=2)
    first = np.argmax(qualifies, axis=2)  # first qualifying segment per cell
    rows = np.arange(geometry.ny)[:, np.newaxis]
    cols = np.arange(geometry.nx)[np.newaxis, :]
    candidate = cell_z[rows, cols, first] + (
        (s - breakpoints[rows, cols, first])
        * (subgrid_cell_count / (first + 1))
    )
    top = cell_z[:, :, -1] + (s - breakpoints[:, :, -1])
    return np.where(any_hit, candidate, top)


def face_hydraulics(
    geometry: SubgridGeometry,
    face_water_surface_x: np.ndarray,
    face_water_surface_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Face area depth, conveyance moment, and maximum local depth (host).

    :param geometry: Precomputed subgrid geometry.
    :param face_water_surface_x: ``eta_f`` per x face ``(ny, nx+1)``.
    :param face_water_surface_y: ``eta_f`` per y face ``(ny+1, nx)``.
    :return: ``(h_A_x, h_A_y, max_local_x, max_local_y)``; the conveyance
        moment ``M`` follows from ``h_k ** (5/3)`` means and is returned via
        the diagnostic dict :meth:`face_conveyance_moment` when needed.
    """
    depth_x = np.maximum(
        np.asarray(face_water_surface_x)[:, :, np.newaxis] - geometry.face_z_x,
        0.0,
    )
    depth_y = np.maximum(
        np.asarray(face_water_surface_y)[:, :, np.newaxis] - geometry.face_z_y,
        0.0,
    )
    h_a_x = depth_x.mean(axis=2)
    h_a_y = depth_y.mean(axis=2)
    max_local_x = depth_x.max(axis=2)
    max_local_y = depth_y.max(axis=2)
    return h_a_x, h_a_y, max_local_x, max_local_y


def face_conveyance_moment(
    geometry: SubgridGeometry,
    face_water_surface_x: np.ndarray,
    face_water_surface_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Width-normalised ``sum_k w_k h_k^(5/3)`` per coarse face (host)."""
    depth_x = np.maximum(
        np.asarray(face_water_surface_x)[:, :, np.newaxis] - geometry.face_z_x,
        0.0,
    )
    depth_y = np.maximum(
        np.asarray(face_water_surface_y)[:, :, np.newaxis] - geometry.face_z_y,
        0.0,
    )
    moment_x = np.power(depth_x, 5.0 / 3.0).mean(axis=2)
    moment_y = np.power(depth_y, 5.0 / 3.0).mean(axis=2)
    return moment_x, moment_y


def face_conveyance_conductance(
    geometry: SubgridGeometry,
    face_water_surface_x: np.ndarray,
    face_water_surface_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Width-normalised ``sum_k h_k^(5/3) / n_k`` per coarse face (host).

    Variable-roughness counterpart of :func:`face_conveyance_moment`; the
    caller must have built the geometry with ``subgrid_manning_n``.

    :raises ValueError: If the geometry carries no native face roughness.
    """
    if geometry.face_n_x is None or geometry.face_n_y is None:
        raise ValueError(
            "face_conveyance_conductance requires geometry built with "
            "subgrid_manning_n"
        )
    depth_x = np.maximum(
        np.asarray(face_water_surface_x)[:, :, np.newaxis] - geometry.face_z_x,
        0.0,
    )
    depth_y = np.maximum(
        np.asarray(face_water_surface_y)[:, :, np.newaxis] - geometry.face_z_y,
        0.0,
    )
    conductance_x = (
        np.power(depth_x, 5.0 / 3.0) / geometry.face_n_x
    ).mean(axis=2)
    conductance_y = (
        np.power(depth_y, 5.0 / 3.0) / geometry.face_n_y
    ).mean(axis=2)
    return conductance_x, conductance_y


def cell_diagnostics(
    geometry: SubgridGeometry,
    stage: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derived per-cell subgrid state (host reference for tests/output).

    :return: Dict with ``wet_fraction``, ``maximum_local_depth``, and
        ``mean_wet_depth`` arrays of shape ``(ny, nx)``.
    """
    eta = np.asarray(stage, dtype=np.float64)
    submerged = eta[:, :, np.newaxis] > geometry.cell_z
    wet_fraction = submerged.mean(axis=2)
    maximum_local_depth = np.maximum(
        eta - geometry.cell_z[:, :, 0], 0.0
    )
    mean_wet_depth = np.where(
        wet_fraction > 0.0,
        np.maximum(eta[:, :, np.newaxis] - geometry.cell_z, 0.0).sum(axis=2)
        / np.maximum(wet_fraction * geometry.subgrid_cell_count, 1.0),
        0.0,
    )
    return {
        "wet_fraction": wet_fraction,
        "maximum_local_depth": maximum_local_depth,
        "mean_wet_depth": mean_wet_depth,
    }
