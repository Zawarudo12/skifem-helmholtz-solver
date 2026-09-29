from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix
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

from .bloch import BlochReduction, build_bloch_projection
from .full_pml import FullPMLConfig
from .pml_generic import stretch_1d
from .solver import FEMSolution

@dataclass(frozen=True)
class Objective3Config(FullPMLConfig):
    """Fixed geometry for Objective 3.

    Geometry:
        period = width = 1 um
        slab thickness = 1 um
        circular air hole diameter = 0.5 um
    """

    hole_diameter: float = 0.50

    @property
    def hole_radius(self) -> float:
        return 0.5 * self.hole_diameter

    @property
    def hole_center_x(self) -> float:
        return 0.5 * self.width

    @property
    def hole_center_y(self) -> float:
        return 0.5 * (self.slab_y0 + self.slab_y1)

@dataclass
class Objective3SolveResult:
    scattered: FEMSolution
    reduction: BlochReduction
    reduced_dofs: int
    free_reduced_dofs: int

def build_bodyfitted_gmsh_mesh(
    filename: str | Path,
    cfg: Objective3Config,
    h_bulk: float = 0.018,
    h_hole: float = 0.008,
    refine_distance: float = 0.20,
    force: bool = False,
) -> Path:
    """Generate one fixed, body-fitted Gmsh mesh.

    The circular interface is an actual geometric boundary.
    The left/right boundary meshes are made periodic copies.

    IMPORTANT:
        This mesh is intentionally independent of wavelength.
    """

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists() and not force:
        return filename

    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "The body-fitted Objective 3 mesh requires gmsh. "
            "Install with: python -m pip install gmsh meshio"
        ) from exc

    gmsh.initialize()

    try:
        gmsh.model.add("objective3_bodyfitted")
        geo = gmsh.model.geo

        x0 = 0.0
        x1 = cfg.width

        levels = [
            -cfg.pml_top,
            0.0,
            cfg.slab_y0,
            cfg.slab_y1,
            cfg.total_height,
            cfg.total_height + cfg.pml_bottom,
        ]

        p_left = []
        p_right = []

        for y in levels:
            p_left.append(
                geo.addPoint(
                    x0,
                    y,
                    0.0,
                    h_bulk,
                )
            )

            p_right.append(
                geo.addPoint(
                    x1,
                    y,
                    0.0,
                    h_bulk,
                )
            )

        horizontal = []

        for i in range(len(levels)):
            horizontal.append(
                geo.addLine(
                    p_left[i],
                    p_right[i],
                )
            )

        left_vertical = []
        right_vertical = []

        for i in range(len(levels) - 1):
            left_vertical.append(
                geo.addLine(
                    p_left[i],
                    p_left[i + 1],
                )
            )

            right_vertical.append(
                geo.addLine(
                    p_right[i],
                    p_right[i + 1],
                )
            )

        xc = cfg.hole_center_x
        yc = cfg.hole_center_y
        r = cfg.hole_radius

        pc = geo.addPoint(
            xc,
            yc,
            0.0,
            h_hole,
        )

        pr = geo.addPoint(
            xc + r,
            yc,
            0.0,
            h_hole,
        )

        pt = geo.addPoint(
            xc,
            yc + r,
            0.0,
            h_hole,
        )

        pl = geo.addPoint(
            xc - r,
            yc,
            0.0,
            h_hole,
        )

        pb = geo.addPoint(
            xc,
            yc - r,
            0.0,
            h_hole,
        )

        circle_arcs = [
            geo.addCircleArc(pr, pc, pt),
            geo.addCircleArc(pt, pc, pl),
            geo.addCircleArc(pl, pc, pb),
            geo.addCircleArc(pb, pc, pr),
        ]

        circle_loop = geo.addCurveLoop(
            circle_arcs
        )

        surfaces: dict[str, int] = {}

        def outer_loop(i: int) -> int:
            return geo.addCurveLoop(
                [
                    horizontal[i],
                    right_vertical[i],
                    -horizontal[i + 1],
                    -left_vertical[i],
                ]
            )

        surfaces["top_pml"] = geo.addPlaneSurface(
            [outer_loop(0)]
        )

        surfaces["top_air"] = geo.addPlaneSurface(
            [outer_loop(1)]
        )

        slab_outer = outer_loop(2)

        surfaces["slab"] = geo.addPlaneSurface(
            [
                slab_outer,
                circle_loop,
            ]
        )

        surfaces["hole"] = geo.addPlaneSurface(
            [circle_loop]
        )

        surfaces["bottom_air"] = geo.addPlaneSurface(
            [outer_loop(3)]
        )

        surfaces["bottom_pml"] = geo.addPlaneSurface(
            [outer_loop(4)]
        )

        geo.synchronize()

        for name, tag in surfaces.items():
            ptag = gmsh.model.addPhysicalGroup(
                2,
                [tag],
            )

            gmsh.model.setPhysicalName(
                2,
                ptag,
                name,
            )

        for name, curves in (
            ("top", [horizontal[0]]),
            ("bottom", [horizontal[-1]]),
            ("left", left_vertical),
            ("right", right_vertical),
            ("hole_boundary", circle_arcs),
        ):
            ptag = gmsh.model.addPhysicalGroup(
                1,
                curves,
            )

            gmsh.model.setPhysicalName(
                1,
                ptag,
                name,
            )

        affine = [
            1.0, 0.0, 0.0, cfg.width,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

        gmsh.model.mesh.setPeriodic(
            1,
            right_vertical,
            left_vertical,
            affine,
        )

        distance = gmsh.model.mesh.field.add(
            "Distance"
        )

        gmsh.model.mesh.field.setNumbers(
            distance,
            "CurvesList",
            circle_arcs,
        )

        gmsh.model.mesh.field.setNumber(
            distance,
            "Sampling",
            200,
        )

        threshold = gmsh.model.mesh.field.add(
            "Threshold"
        )

        gmsh.model.mesh.field.setNumber(
            threshold,
            "InField",
            distance,
        )

        gmsh.model.mesh.field.setNumber(
            threshold,
            "SizeMin",
            h_hole,
        )

        gmsh.model.mesh.field.setNumber(
            threshold,
            "SizeMax",
            h_bulk,
        )

        gmsh.model.mesh.field.setNumber(
            threshold,
            "DistMin",
            0.0,
        )

        gmsh.model.mesh.field.setNumber(
            threshold,
            "DistMax",
            refine_distance,
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            threshold
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            h_hole,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            h_bulk,
        )

        gmsh.option.setNumber(
            "Mesh.Algorithm",
            6,
        )

        gmsh.option.setNumber(
            "Mesh.Smoothing",
            5,
        )

        gmsh.option.setNumber(
            "Mesh.MshFileVersion",
            2.2,
        )

        gmsh.option.setNumber(
            "Mesh.Binary",
            0,
        )

        gmsh.model.mesh.generate(2)

        gmsh.write(
            str(filename)
        )

    finally:
        gmsh.finalize()

    return filename

def load_bodyfitted_mesh(
    filename: str | Path,
    cfg: Objective3Config,
) -> MeshTri:
    """Load the fixed Gmsh mesh and attach robust tags."""

    mesh = MeshTri.load(
        str(filename)
    )

    scale = max(
        cfg.width,
        cfg.total_height
        + cfg.pml_top
        + cfg.pml_bottom,
        1.0,
    )

    tol = (
        1e-8 * scale
    )

    mesh = mesh.with_boundaries(
        {
            "top": lambda X: np.isclose(
                X[1],
                -cfg.pml_top,
                atol=tol,
            ),

            "bottom": lambda X: np.isclose(
                X[1],
                cfg.total_height
                + cfg.pml_bottom,
                atol=tol,
            ),

            "left": lambda X: np.isclose(
                X[0],
                0.0,
                atol=tol,
            ),

            "right": lambda X: np.isclose(
                X[0],
                cfg.width,
                atol=tol,
            ),
        }
    )

    def in_hole(X):
        r2 = (
            (X[0] - cfg.hole_center_x) ** 2
            +
            (X[1] - cfg.hole_center_y) ** 2
        )

        return (
            (X[1] > cfg.slab_y0 + tol)
            &
            (X[1] < cfg.slab_y1 - tol)
            &
            (
                r2
                < cfg.hole_radius ** 2
            )
        )

    def in_slab(X):
        r2 = (
            (X[0] - cfg.hole_center_x) ** 2
            +
            (X[1] - cfg.hole_center_y) ** 2
        )

        return (
            (X[1] > cfg.slab_y0 + tol)
            &
            (X[1] < cfg.slab_y1 - tol)
            &
            (
                r2
                > cfg.hole_radius ** 2
            )
        )

    mesh = mesh.with_subdomains(
        {
            "top_pml": lambda X: (
                X[1] < -tol
            ),

            "top_air": lambda X: (
                (X[1] > -tol)
                &
                (
                    X[1]
                    < cfg.slab_y0 - tol
                )
            ),

            "slab": in_slab,

            "hole": in_hole,

            "bottom_air": lambda X: (
                (
                    X[1]
                    > cfg.slab_y1 + tol
                )
                &
                (
                    X[1]
                    < cfg.total_height - tol
                )
            ),

            "bottom_pml": lambda X: (
                X[1]
                > cfg.total_height + tol
            ),
        }
    )

    return mesh

def _assemble_operator(
    basis: Basis,
    cfg: Objective3Config,
) -> csr_matrix:
    """TE Helmholtz operator on the body-fitted geometry."""

    A = csr_matrix(
        (basis.N, basis.N),
        dtype=np.complex128,
    )

    region_index = {
        "top_pml": cfg.n_inc,
        "top_air": cfg.n_inc,
        "slab": cfg.n_slab,
        "hole": cfg.n_inc,
        "bottom_air": cfg.n_out,
        "bottom_pml": cfg.n_out,
    }

    for region, n in region_index.items():

        subbasis = basis.with_elements(
            basis.mesh.subdomains[region]
        )

        eps_r = (
            n ** 2
        )

        @BilinearForm(
            dtype=np.complex128
        )
        def form(u, v, w):
            y = w.x[1]

            sy = stretch_1d(
                coordinate=y,

                physical_min=0.0,
                physical_max=cfg.total_height,

                pml_low=cfg.pml_top,
                pml_high=cfg.pml_bottom,

                sigma_max=cfg.pml_sigma_max,
                order=cfg.pml_order,
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

                -

                cfg.k0 ** 2
                * eps_r
                * sy
                * u
                * v
            )

        A = (
            A
            + asm(
                form,
                subbasis,
            )
        )

    return A.tocsr()

def _assemble_source(
    basis: Basis,
    cfg: Objective3Config,
) -> np.ndarray:
    """Scattered-field source.

    Only the dielectric material is different from the
    air background.  The circular hole is background air.
    """

    slab_basis = basis.with_elements(
        basis.mesh.subdomains[
            "slab"
        ]
    )

    contrast = (
        cfg.n_slab ** 2
        - cfg.n_inc ** 2
    )

    @LinearForm(
        dtype=np.complex128
    )
    def source(v, w):
        y = w.x[1]

        u_inc = np.exp(
            1j
            * cfg.k0
            * cfg.n_inc
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
            slab_basis,
        ),
        dtype=np.complex128,
    )

def solve_objective3_bodyfitted(
    cfg: Objective3Config,
    mesh_file: str | Path,
) -> Objective3SolveResult:
    """Solve Objective 3 on the fixed body-fitted mesh."""

    mesh = load_bodyfitted_mesh(
        mesh_file,
        cfg,
    )

    basis = Basis(
        mesh,
        ElementTriP2(),
        intorder=8,
    )

    A_full = _assemble_operator(
        basis,
        cfg,
    )

    b = _assemble_source(
        basis,
        cfg,
    )

    reduction = build_bloch_projection(
        basis=basis,
        period=cfg.width,
        kx=0.0,
    )

    P = reduction.P

    PH = (
        P
        .conjugate()
        .transpose()
    )

    A_red = (
        PH
        @ A_full
        @ P
    ).tocsr()

    b_red = np.asarray(
        PH @ b,
        dtype=np.complex128,
    ).reshape(-1)

    top_full = np.asarray(
        basis
        .get_dofs("top")
        .flatten(),
        dtype=int,
    )

    bottom_full = np.asarray(
        basis
        .get_dofs("bottom")
        .flatten(),
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
        reduction.full_to_reduced[
            outer_full
        ]
    )

    all_red = np.arange(
        A_red.shape[0],
        dtype=int,
    )

    free_red = np.setdiff1d(
        all_red,
        outer_red,
    )

    z = np.zeros(
        A_red.shape[0],
        dtype=np.complex128,
    )

    z[free_red] = spsolve(
        A_red[
            free_red
        ][:, free_red].tocsc(),
        b_red[
            free_red
        ],
    )

    u_scattered = np.asarray(
        P @ z,
        dtype=np.complex128,
    ).reshape(-1)

    solution = FEMSolution(
        cfg=cfg,
        order=2,
        mesh=mesh,
        basis=basis,
        u=u_scattered,
        A=A_full,
        b=b,
    )

    return Objective3SolveResult(
        scattered=solution,
        reduction=reduction,
        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )

def sample_horizontal_line(
    sol: FEMSolution,
    y: float,
    width: float,
    npoints: int = 512,
    chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-safe midpoint sampling over one period."""

    dx = (
        width
        / npoints
    )

    x = (
        np.arange(
            npoints,
            dtype=float,
        )
        + 0.5
    ) * dx

    values = np.empty(
        npoints,
        dtype=np.complex128,
    )

    interpolator = (
        sol.basis.interpolator(
            sol.u
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
                    [xx, yy]
                )
            ),
            dtype=np.complex128,
        )

    return x, values

def diffraction_orders_bodyfitted(
    result: Objective3SolveResult,
    cfg: Objective3Config,
    npoints: int = 512,
    cutoff_rel_tol: float = 1e-9,
) -> dict:
    """Rayleigh/Fourier diffraction-order decomposition.

    Normal incidence, air above and below.
    """

    sol = result.scattered

    if not np.isclose(
        cfg.n_inc,
        cfg.n_out,
    ):
        raise ValueError(
            "Current Objective 3 extractor assumes "
            "n_inc == n_out."
        )

    period = cfg.width

    k_air = float(
        np.real(
            cfg.k0
            * cfg.n_inc
        )
    )

    ky_inc = k_air

    y_top = (
        0.5
        * cfg.air_top
    )

    y_bottom = (
        cfg.slab_y1
        + 0.5
        * cfg.air_bottom
    )

    x, us_top = sample_horizontal_line(
        sol,
        y_top,
        period,
        npoints=npoints,
    )

    _, us_bottom = sample_horizontal_line(
        sol,
        y_bottom,
        period,
        npoints=npoints,
    )

    u_inc_bottom = np.exp(
        1j
        * cfg.k0
        * cfg.n_inc
        * y_bottom
    )

    u_total_bottom = (
        us_bottom
        + u_inc_bottom
    )

    m_prop_est = int(
        np.floor(
            k_air
            * period
            / (2.0 * np.pi)
        )
    )

    m_values = np.arange(
        -m_prop_est - 3,
        m_prop_est + 4,
        dtype=int,
    )

    rows = []

    cutoff_orders = []

    scale = (
        k_air ** 2
    )

    for m in m_values:

        kx_m = (
            2.0
            * np.pi
            * m
            / period
        )

        ky2 = (
            k_air ** 2
            - kx_m ** 2
        )

        near_cutoff = (
            abs(ky2)
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
                * abs(r_m) ** 2
            )

            T_m = float(
                factor
                * abs(t_m) ** 2
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

    lookup = {
        row["m"]: row
        for row in rows
    }

    symmetry_abs = []

    for m in range(
        1,
        max(
            abs(int(x))
            for x in m_values
        ) + 1,
    ):
        if (
            m in lookup
            and -m in lookup
            and lookup[m]["propagating"]
            and lookup[-m]["propagating"]
        ):
            symmetry_abs.append(
                abs(
                    lookup[m]["R"]
                    - lookup[-m]["R"]
                )
            )

            symmetry_abs.append(
                abs(
                    lookup[m]["T"]
                    - lookup[-m]["T"]
                )
            )

    symmetry_max = (
        max(symmetry_abs)
        if symmetry_abs
        else 0.0
    )

    return {
        "orders": rows,
        "R": R_total,
        "T": T_total,
        "R_plus_T": (
            R_total + T_total
        ),
        "energy_error": abs(
            R_total + T_total - 1.0
        ),
        "symmetry_abs_max": float(
            symmetry_max
        ),
        "cutoff_orders": cutoff_orders,
        "y_top": y_top,
        "y_bottom": y_bottom,
    }
