from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.sparse.linalg import (
    spsolve,
)

from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
)

from .bloch import (
    BlochReduction,
    build_bloch_projection,
)

from .forms_full_pml import (
    assemble_scattered_field_source,
)

from .forms_generic_pml import (
    assemble_te_generic_pml,
)

from .full_pml import (
    FullPMLConfig,
    make_full_pml_mesh,
)

from .solver import (
    FEMSolution,
)


@dataclass
class GenericPMLSolveResult:
    scattered: FEMSolution

    reduction: BlochReduction

    kx: complex
    ky_inc: complex

    angle_deg: float

    reduced_dofs: int
    free_reduced_dofs: int


def solve_generic_pml_te(
    cfg: FullPMLConfig,
    angle_deg: float,
    h_target: float,
    order: int = 2,
) -> GenericPMLSolveResult:
    """TE scattered-field solve with generic coordinate stretching."""

    if order == 1:

        elem = ElementTriP1()

    elif order == 2:

        elem = ElementTriP2()

    else:

        raise ValueError(
            "order must be 1 or 2"
        )

    # --------------------------------------------------
    # Mesh / FEM basis
    # --------------------------------------------------

    mesh = make_full_pml_mesh(
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

    # --------------------------------------------------
    # Entire Helmholtz + PML operator in one assembly
    # --------------------------------------------------

    A_full = assemble_te_generic_pml(
        basis,
        cfg,
    )

    # --------------------------------------------------
    # Scattered-field excitation
    # --------------------------------------------------

    b, kx, ky_inc = (
        assemble_scattered_field_source(
            basis,
            cfg,
            angle_deg,
        )
    )

    # --------------------------------------------------
    # Bloch reduction
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Outer PML walls:
    #
    # scattered field = 0
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Reconstruct full FEM vector
    # --------------------------------------------------

    u_scattered = np.asarray(
        P @ z,
        dtype=np.complex128,
    ).reshape(-1)

    scattered = FEMSolution(
        cfg=cfg,
        order=order,
        mesh=mesh,
        basis=basis,
        u=u_scattered,
        A=A_full,
        b=b,
    )

    return GenericPMLSolveResult(
        scattered=scattered,
        reduction=reduction,

        kx=kx,
        ky_inc=ky_inc,

        angle_deg=angle_deg,

        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )