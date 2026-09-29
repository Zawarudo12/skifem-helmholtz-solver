from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.sparse.linalg import spsolve

from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
)

from .bloch import (
    BlochReduction,
    build_bloch_projection,
)

from .forms_oblique import (
    assemble_oblique_top_source,
)

from .forms_pml import (
    assemble_bottom_pml,
    assemble_physical_volume,
)

from .pml import (
    BottomPMLConfig,
    make_bottom_pml_mesh,
)

from .solver import FEMSolution


@dataclass
class ObliqueSolveResult:
    solution: FEMSolution
    reduction: BlochReduction

    kx: complex
    ky_inc: complex

    reduced_dofs: int
    free_reduced_dofs: int


def solve_oblique_te(
    cfg: BottomPMLConfig,
    angle_deg: float,
    h_target: float,
    order: int = 2,
) -> ObliqueSolveResult:
    """TE slab solve with PML + Bloch-periodic sides."""

    if order == 1:
        elem = ElementTriP1()

    elif order == 2:
        elem = ElementTriP2()

    else:
        raise ValueError(
            "order must be 1 or 2"
        )

    mesh = make_bottom_pml_mesh(
        cfg,
        h_target=h_target,
    )

    basis = Basis(
        mesh,
        elem,
        intorder=max(
            4,
            2 * order + 2,
        ),
    )

    # Physical domain
    K, M = assemble_physical_volume(
        basis,
        cfg,
    )

    A_physical = (
        K
        - cfg.k0**2 * M
    )

    # Bottom PML
    A_pml = assemble_bottom_pml(
        basis,
        cfg,
    )

    # Oblique top boundary/source
    B_top, b, kx, ky_inc = (
        assemble_oblique_top_source(
            basis,
            cfg,
            angle_deg,
        )
    )

    A_full = (
        A_physical
        + A_pml
        + B_top
    ).tocsr()

    # ----------------------------------------------
    # Bloch condition:
    #
    # u(x + Lambda)
    #   = exp(i kx Lambda) u(x)
    # ----------------------------------------------

    reduction = build_bloch_projection(
        basis=basis,
        period=cfg.width,
        kx=float(
            np.real(kx)
        ),
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

    # ----------------------------------------------
    # Bottom PML termination
    # ----------------------------------------------

    bottom_full = np.asarray(
        basis
        .get_dofs("bottom")
        .flatten(),
        dtype=int,
    )

    bottom_red = np.unique(
        reduction.full_to_reduced[
            bottom_full
        ]
    )

    all_red = np.arange(
        A_red.shape[0],
        dtype=int,
    )

    free_red = np.setdiff1d(
        all_red,
        bottom_red,
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

    u_full = np.asarray(
        P @ z,
        dtype=np.complex128,
    ).reshape(-1)

    solution = FEMSolution(
        cfg=cfg,
        order=order,
        mesh=mesh,
        basis=basis,
        u=u_full,
        A=A_full,
        b=b,
    )

    return ObliqueSolveResult(
        solution=solution,
        reduction=reduction,
        kx=kx,
        ky_inc=ky_inc,
        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )