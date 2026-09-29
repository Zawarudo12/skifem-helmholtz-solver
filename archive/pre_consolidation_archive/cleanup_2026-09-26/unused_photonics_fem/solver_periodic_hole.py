from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.sparse.linalg import spsolve

from skfem import (
    Basis,
    ElementTriP2,
)

from .bloch import (
    BlochReduction,
    build_bloch_projection,
)

from .full_pml import (
    make_full_pml_mesh,
)

from .periodic_hole import (
    PeriodicHoleConfig,
    assemble_periodic_hole_operator,
    assemble_periodic_hole_source,
)

from .solver import FEMSolution


@dataclass
class PeriodicHoleSolveResult:

    scattered: FEMSolution

    reduction: BlochReduction

    reduced_dofs: int
    free_reduced_dofs: int


def solve_periodic_hole(
    cfg: PeriodicHoleConfig,
    h_target: float,
) -> PeriodicHoleSolveResult:

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    mesh = make_full_pml_mesh(
        cfg,
        h_target=h_target,
    )

    # Objective uses P2.
    elem = ElementTriP2()

    basis = Basis(
        mesh,
        elem,
        intorder=8,
    )

    # --------------------------------------------------------
    # Operator + scattered-field source
    # --------------------------------------------------------

    A_full = (
        assemble_periodic_hole_operator(
            basis,
            cfg,
        )
    )

    b = (
        assemble_periodic_hole_source(
            basis,
            cfg,
        )
    )

    # --------------------------------------------------------
    # Normal incidence:
    #
    # kx = 0
    #
    # therefore Bloch phase = 1.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Outer PML walls:
    #
    # scattered field = 0
    # --------------------------------------------------------

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

    return PeriodicHoleSolveResult(
        scattered=solution,
        reduction=reduction,
        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )