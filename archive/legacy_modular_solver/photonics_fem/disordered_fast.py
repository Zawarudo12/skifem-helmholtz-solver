from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import grad

from .pml_generic import stretch_1d

@dataclass(frozen=True)
class DisorderedConfig:
    wavelength: float

    width: float = 7.0
    xmin: float = -3.5
    xmax: float = 3.5

    scatter_ymin: float = -3.5
    scatter_ymax: float = 3.5

    physical_ymin: float = -4.0
    physical_ymax: float = 4.0

    pml_low: float = 1.0
    pml_high: float = 1.0
    pml_order: int = 3
    pml_sigma_max: float = 8.0

    n_background: complex = 1.0 + 0.0j
    n_disk: complex = 3.0 + 0.0j

    @property
    def k0(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @property
    def ymin(self) -> float:
        return self.physical_ymin - self.pml_low

    @property
    def ymax(self) -> float:
        return self.physical_ymax + self.pml_high

    @property
    def top_monitor_y(self) -> float:

        return 0.5 * (
            self.scatter_ymax
            + self.physical_ymax
        )

    @property
    def bottom_monitor_y(self) -> float:

        return 0.5 * (
            self.scatter_ymin
            + self.physical_ymin
        )

@dataclass
class SolveTimings:
    mesh_and_basis: float
    periodic_setup: float
    matrix_assembly: float
    source_assembly: float
    linear_solve: float
    reconstruction: float
    total: float

@dataclass
class DisorderedSolveResult:
    cfg: DisorderedConfig
    mesh: MeshTri
    basis: Basis
    u_scattered: np.ndarray
    full_dofs: int
    reduced_dofs: int
    free_reduced_dofs: int
    timings: SolveTimings

def _require_named_groups(
    mesh: MeshTri,
) -> None:

    required_subdomains = {
        "bottom_pml",
        "bottom_air",
        "scatter_background",
        "top_air",
        "top_pml",
        "disks",
    }

    required_boundaries = {
        "left",
        "right",
        "top",
        "bottom",
    }

    actual_subdomains = set(
        mesh.subdomains.keys()
        if mesh.subdomains is not None
        else []
    )

    actual_boundaries = set(
        mesh.boundaries.keys()
        if mesh.boundaries is not None
        else []
    )

    missing_subdomains = (
        required_subdomains
        - actual_subdomains
    )

    missing_boundaries = (
        required_boundaries
        - actual_boundaries
    )

    if missing_subdomains or missing_boundaries:
        raise RuntimeError(
            "Mesh physical groups were not loaded as expected.\n"
            f"Available subdomains: {sorted(actual_subdomains)}\n"
            f"Available boundaries: {sorted(actual_boundaries)}\n"
            f"Missing subdomains: {sorted(missing_subdomains)}\n"
            f"Missing boundaries: {sorted(missing_boundaries)}"
        )

def load_disordered_mesh(
    mesh_file: str | Path,
) -> MeshTri:

    mesh = MeshTri.load(
        str(mesh_file)
    )

    _require_named_groups(
        mesh
    )

    return mesh

def build_normal_incidence_periodic_projection(
    basis: Basis,
    *,
    atol: float = 2e-7,
):
    """
    Identify right-boundary P2 DOFs with the matching left-boundary DOFs.

    Normal incidence => Bloch phase = 1.

    This performs periodicity only.  No mirror/symmetry reduction is used.
    """

    left = np.asarray(
        basis.get_dofs(
            "left"
        ).flatten(),
        dtype=int,
    )

    right = np.asarray(
        basis.get_dofs(
            "right"
        ).flatten(),
        dtype=int,
    )

    if left.size != right.size:
        raise RuntimeError(
            f"Periodic DOF count mismatch: "
            f"left={left.size}, right={right.size}"
        )

    doflocs = basis.doflocs

    left_order = np.argsort(
        doflocs[
            1,
            left,
        ],
        kind="stable",
    )

    right_order = np.argsort(
        doflocs[
            1,
            right,
        ],
        kind="stable",
    )

    left = left[
        left_order
    ]

    right = right[
        right_order
    ]

    y_left = doflocs[
        1,
        left,
    ]

    y_right = doflocs[
        1,
        right,
    ]

    max_y_mismatch = float(
        np.max(
            np.abs(
                y_left
                - y_right
            )
        )
    )

    if max_y_mismatch > atol:
        raise RuntimeError(
            "Periodic left/right P2 DOFs do not match in y. "
            f"max mismatch = {max_y_mismatch:.3e} um"
        )

    N = basis.N

    is_slave = np.zeros(
        N,
        dtype=bool,
    )

    is_slave[
        right
    ] = True

    masters_and_interior = np.flatnonzero(
        ~is_slave
    )

    full_to_reduced = np.full(
        N,
        -1,
        dtype=int,
    )

    full_to_reduced[
        masters_and_interior
    ] = np.arange(
        masters_and_interior.size,
        dtype=int,
    )

    full_to_reduced[
        right
    ] = full_to_reduced[
        left
    ]

    if np.any(
        full_to_reduced < 0
    ):
        raise RuntimeError(
            "Periodic projection construction left unmapped DOFs."
        )

    rows = np.arange(
        N,
        dtype=int,
    )

    cols = full_to_reduced

    data = np.ones(
        N,
        dtype=np.complex128,
    )

    P = coo_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            N,
            masters_and_interior.size,
        ),
    ).tocsr()

    return (
        P,
        full_to_reduced,
        left,
        right,
        max_y_mismatch,
    )

def _stretch_y(
    y,
    cfg: DisorderedConfig,
):
    return stretch_1d(
        coordinate=y,
        physical_min=cfg.physical_ymin,
        physical_max=cfg.physical_ymax,
        pml_low=cfg.pml_low,
        pml_high=cfg.pml_high,
        sigma_max=cfg.pml_sigma_max,
        order=cfg.pml_order,
    )

def assemble_static_matrices(
    basis: Basis,
    cfg: DisorderedConfig,
):
    """
    TE/Ez operator:

        A(k0) = K - k0^2 M

    with y-directed PML stretch sy.
    """

    @BilinearForm(
        dtype=np.complex128
    )
    def stiffness(
        u,
        v,
        w,
    ):
        sy = _stretch_y(
            w.x[1],
            cfg,
        )

        gu = grad(u)
        gv = grad(v)

        return (
            sy
            * gu[0]
            * gv[0]

            +

            (1.0 / sy)
            * gu[1]
            * gv[1]
        )

    K = asm(
        stiffness,
        basis,
    ).tocsr()

    M = csr_matrix(
        (
            basis.N,
            basis.N,
        ),
        dtype=np.complex128,
    )

    region_index = {
        "bottom_pml": cfg.n_background,
        "bottom_air": cfg.n_background,
        "scatter_background": cfg.n_background,
        "top_air": cfg.n_background,
        "top_pml": cfg.n_background,
        "disks": cfg.n_disk,
    }

    for region, n in region_index.items():

        subbasis = basis.with_elements(
            basis.mesh.subdomains[
                region
            ]
        )

        eps_r = n ** 2

        @BilinearForm(
            dtype=np.complex128
        )
        def mass(
            u,
            v,
            w,
        ):
            sy = _stretch_y(
                w.x[1],
                cfg,
            )

            return (
                eps_r
                * sy
                * u
                * v
            )

        M = (
            M
            + asm(
                mass,
                subbasis,
            )
        )

    return (
        K.tocsr(),
        M.tocsr(),
    )

def assemble_scattered_source(
    basis: Basis,
    cfg: DisorderedConfig,
) -> np.ndarray:
    """
    Scattered-field source for a downward plane wave.

    Time convention: exp(-i omega t)
    Downward (-y) incident field: exp(-i k y)

    Only the n=3 disks differ from the n=1 background.
    """

    disk_basis = basis.with_elements(
        basis.mesh.subdomains[
            "disks"
        ]
    )

    contrast = (
        cfg.n_disk ** 2
        - cfg.n_background ** 2
    )

    @LinearForm(
        dtype=np.complex128
    )
    def source(
        v,
        w,
    ):
        y = w.x[1]

        u_inc = np.exp(
            -1j
            * cfg.k0
            * cfg.n_background
            * y
        )

        return (
            cfg.k0 ** 2
            * contrast
            * u_inc
            * v
        )

    return np.asarray(
        asm(
            source,
            disk_basis,
        ),
        dtype=np.complex128,
    )

def solve_disordered(
    cfg: DisorderedConfig,
    mesh_file: str | Path,
    *,
    intorder: int = 8,
) -> DisorderedSolveResult:

    total_start = perf_counter()

    t0 = perf_counter()

    mesh = load_disordered_mesh(
        mesh_file
    )

    basis = Basis(
        mesh,
        ElementTriP2(),
        intorder=intorder,
    )

    t1 = perf_counter()

    (
        P,
        full_to_reduced,
        left_dofs,
        right_dofs,
        periodic_y_mismatch,
    ) = build_normal_incidence_periodic_projection(
        basis
    )

    PH = (
        P
        .conjugate()
        .transpose()
        .tocsr()
    )

    top_full = np.asarray(
        basis.get_dofs(
            "top"
        ).flatten(),
        dtype=int,
    )

    bottom_full = np.asarray(
        basis.get_dofs(
            "bottom"
        ).flatten(),
        dtype=int,
    )

    outer_full = np.unique(
        np.concatenate(
            [
                top_full,
                bottom_full,
            ]
        )
    )

    outer_red = np.unique(
        full_to_reduced[
            outer_full
        ]
    )

    reduced_dofs = P.shape[1]

    all_red = np.arange(
        reduced_dofs,
        dtype=int,
    )

    free_red = np.setdiff1d(
        all_red,
        outer_red,
    )

    t2 = perf_counter()

    (
        K_full,
        M_full,
    ) = assemble_static_matrices(
        basis,
        cfg,
    )

    K_red = (
        PH
        @ K_full
        @ P
    ).tocsr()

    M_red = (
        PH
        @ M_full
        @ P
    ).tocsr()

    K_free = K_red[
        free_red
    ][:, free_red].tocsc()

    M_free = M_red[
        free_red
    ][:, free_red].tocsc()

    t3 = perf_counter()

    b_full = assemble_scattered_source(
        basis,
        cfg,
    )

    b_red = np.asarray(
        PH @ b_full,
        dtype=np.complex128,
    ).reshape(-1)

    b_free = b_red[
        free_red
    ]

    t4 = perf_counter()

    A_free = (
        K_free
        - cfg.k0 ** 2
        * M_free
    ).tocsc()

    z_free = spsolve(
        A_free,
        b_free,
    )

    t5 = perf_counter()

    z = np.zeros(
        reduced_dofs,
        dtype=np.complex128,
    )

    z[
        free_red
    ] = z_free

    u_scattered = np.asarray(
        P @ z,
        dtype=np.complex128,
    ).reshape(-1)

    t6 = perf_counter()

    timings = SolveTimings(
        mesh_and_basis=t1 - t0,
        periodic_setup=t2 - t1,
        matrix_assembly=t3 - t2,
        source_assembly=t4 - t3,
        linear_solve=t5 - t4,
        reconstruction=t6 - t5,
        total=t6 - total_start,
    )

    timings.periodic_y_mismatch = periodic_y_mismatch
    timings.left_periodic_dofs = int(
        left_dofs.size
    )
    timings.right_periodic_dofs = int(
        right_dofs.size
    )

    return DisorderedSolveResult(
        cfg=cfg,
        mesh=mesh,
        basis=basis,
        u_scattered=u_scattered,
        full_dofs=basis.N,
        reduced_dofs=reduced_dofs,
        free_reduced_dofs=free_red.size,
        timings=timings,
    )

def sample_horizontal_line(
    result: DisorderedSolveResult,
    y: float,
    *,
    npoints: int = 1024,
    chunk: int = 32,
):
    """
    Midpoint samples across x = [-3.5, +3.5].
    """

    cfg = result.cfg

    dx = (
        cfg.width
        / npoints
    )

    x = (
        cfg.xmin
        +
        (
            np.arange(
                npoints,
                dtype=float,
            )
            + 0.5
        )
        * dx
    )

    values = np.empty(
        npoints,
        dtype=np.complex128,
    )

    interpolator = (
        result.basis.interpolator(
            result.u_scattered
        )
    )

    for start in range(
        0,
        npoints,
        chunk,
    ):

        stop = min(
            start + chunk,
            npoints,
        )

        xx = x[
            start:stop
        ]

        yy = np.full_like(
            xx,
            y,
        )

        values[
            start:stop
        ] = np.asarray(
            interpolator(
                np.vstack(
                    [
                        xx,
                        yy,
                    ]
                )
            ),
            dtype=np.complex128,
        )

    return (
        x,
        values,
    )

def diffraction_orders(
    result: DisorderedSolveResult,
    *,
    npoints: int = 1024,
    cutoff_rel_tol: float = 1e-9,
) -> dict:

    cfg = result.cfg

    if not np.isclose(
        cfg.n_background.imag,
        0.0,
    ):
        raise ValueError(
            "Current R/T extractor assumes a lossless background."
        )

    k_air = float(
        np.real(
            cfg.k0
            * cfg.n_background
        )
    )

    ky_inc = k_air

    x, us_top = sample_horizontal_line(
        result,
        cfg.top_monitor_y,
        npoints=npoints,
    )

    _, us_bottom = sample_horizontal_line(
        result,
        cfg.bottom_monitor_y,
        npoints=npoints,
    )

    u_inc_bottom = np.exp(
        -1j
        * cfg.k0
        * cfg.n_background
        * cfg.bottom_monitor_y
    )

    u_total_bottom = (
        us_bottom
        + u_inc_bottom
    )

    m_prop_est = int(
        np.floor(
            k_air
            * cfg.width
            / (
                2.0
                * np.pi
            )
        )
    )

    m_values = np.arange(
        -m_prop_est - 3,
        m_prop_est + 4,
        dtype=int,
    )

    scale = (
        k_air ** 2
    )

    rows = []
    cutoff_orders = []

    for m in m_values:

        kx_m = (
            2.0
            * np.pi
            * m
            / cfg.width
        )

        ky2 = (
            k_air ** 2
            - kx_m ** 2
        )

        near_cutoff = (
            abs(
                ky2
            )
            <= cutoff_rel_tol
            * scale
        )

        propagating = (
            ky2
            > cutoff_rel_tol
            * scale
        )

        if near_cutoff:
            cutoff_orders.append(
                int(m)
            )

        if propagating:
            ky_m = float(
                np.sqrt(
                    ky2
                )
            )
        else:
            ky_m = complex(
                np.sqrt(
                    ky2
                    + 0.0j
                )
            )

        phase = np.exp(
            -1j
            * kx_m
            * x
        )

        r_m = np.mean(
            us_top
            * phase
        )

        t_m = np.mean(
            u_total_bottom
            * phase
        )

        if propagating:

            factor = (
                ky_m
                / ky_inc
            )

            R_m = float(
                factor
                * abs(
                    r_m
                ) ** 2
            )

            T_m = float(
                factor
                * abs(
                    t_m
                ) ** 2
            )

        else:

            R_m = 0.0
            T_m = 0.0

        rows.append(
            {
                "m": int(m),
                "kx": kx_m,
                "ky": ky_m,
                "propagating": bool(
                    propagating
                ),
                "near_cutoff": bool(
                    near_cutoff
                ),
                "r": r_m,
                "t": t_m,
                "R": R_m,
                "T": T_m,
            }
        )

    R_total = float(
        sum(
            row["R"]
            for row in rows
        )
    )

    T_total = float(
        sum(
            row["T"]
            for row in rows
        )
    )

    return {
        "orders": rows,
        "R": R_total,
        "T": T_total,
        "R_plus_T": (
            R_total
            + T_total
        ),
        "energy_error": abs(
            R_total
            + T_total
            - 1.0
        ),
        "cutoff_orders": cutoff_orders,
        "top_monitor_y": cfg.top_monitor_y,
        "bottom_monitor_y": cfg.bottom_monitor_y,
    }
