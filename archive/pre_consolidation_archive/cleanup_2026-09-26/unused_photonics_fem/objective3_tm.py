from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    LinearForm,
    asm,
)
from skfem.helpers import grad

from .bloch import build_bloch_projection
from .objective3_bodyfitted import (
    Objective3Config,
    Objective3SolveResult,
    load_bodyfitted_mesh,
)
from .pml_generic import stretch_1d
from .solver import FEMSolution


def _assemble_tm_operator(
    basis: Basis,
    cfg: Objective3Config,
) -> csr_matrix:
    """2D TM scalar operator for H_z.

    PDE:
        div((1/eps_r) grad(H_z)) + k0^2 H_z = 0

    Weak form with y-PML:
        ∫ [
            (1/eps_r) sy H_x v_x
          + (1/eps_r) (1/sy) H_y v_y
          - k0^2 sy H v
        ] dΩ
    """

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

        eps_r = n ** 2
        p = 1.0 / eps_r

        @BilinearForm(dtype=np.complex128)
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
                p * sy * gu[0] * gv[0]
                +
                p * (1.0 / sy) * gu[1] * gv[1]
                -
                cfg.k0 ** 2 * sy * u * v
            )

        A = A + asm(
            form,
            subbasis,
        )

    return A.tocsr()


def _assemble_tm_source(
    basis: Basis,
    cfg: Objective3Config,
) -> np.ndarray:
    """Scattered-field source for normal-incidence TM H_z.

    Background is air.

    A_p(H_s, v) =
        -[A_p - A_pb](H_inc, v)

    Since q=1 is unchanged, only p=1/eps changes:

        b(v) = ∫ (p_b - p_slab) grad(H_inc)·grad(v) dΩ
    """

    slab_basis = basis.with_elements(
        basis.mesh.subdomains["slab"]
    )

    p_bg = 1.0 / (cfg.n_inc ** 2)
    p_slab = 1.0 / (cfg.n_slab ** 2)

    delta = (
        p_bg
        - p_slab
    )

    ky_inc = (
        cfg.k0
        * cfg.n_inc
    )

    @LinearForm(dtype=np.complex128)
    def source(v, w):
        y = w.x[1]

        h_inc = np.exp(
            1j
            * ky_inc
            * y
        )

        # grad(H_inc) = (0, i*ky_inc*H_inc)
        grad_inc_y = (
            1j
            * ky_inc
            * h_inc
        )

        gv = grad(v)

        return (
            delta
            * grad_inc_y
            * gv[1]
        )

    return np.asarray(
        asm(
            source,
            slab_basis,
        ),
        dtype=np.complex128,
    )


def solve_objective3_tm(
    cfg: Objective3Config,
    mesh_file: str | Path,
) -> Objective3SolveResult:
    """Solve the same body-fitted Objective 3 geometry as TM H_z."""

    mesh = load_bodyfitted_mesh(
        mesh_file,
        cfg,
    )

    basis = Basis(
        mesh,
        ElementTriP2(),
        intorder=8,
    )

    A_full = _assemble_tm_operator(
        basis,
        cfg,
    )

    b = _assemble_tm_source(
        basis,
        cfg,
    )

    # Normal incidence -> periodic phase = 1.
    reduction = build_bloch_projection(
        basis=basis,
        period=cfg.width,
        kx=0.0,
    )

    P = reduction.P
    PH = P.conjugate().transpose()

    A_red = (
        PH @ A_full @ P
    ).tocsr()

    b_red = np.asarray(
        PH @ b,
        dtype=np.complex128,
    ).reshape(-1)

    top_full = np.asarray(
        basis.get_dofs("top").flatten(),
        dtype=int,
    )

    bottom_full = np.asarray(
        basis.get_dofs("bottom").flatten(),
        dtype=int,
    )

    outer_full = np.unique(
        np.concatenate(
            [top_full, bottom_full]
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

    h_scattered = np.asarray(
        P @ z,
        dtype=np.complex128,
    ).reshape(-1)

    solution = FEMSolution(
        cfg=cfg,
        order=2,
        mesh=mesh,
        basis=basis,
        u=h_scattered,
        A=A_full,
        b=b,
    )

    return Objective3SolveResult(
        scattered=solution,
        reduction=reduction,
        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )
