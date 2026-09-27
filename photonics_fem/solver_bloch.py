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

from .forms_pml import (
    assemble_bottom_pml,
    assemble_physical_volume,
    assemble_top_source,
)

from .pml import (
    BottomPMLConfig,
    make_bottom_pml_mesh,
)

from .solver import FEMSolution

@dataclass
class BlochSolveResult:
    """Full FEM solution plus Bloch reduction information."""

    solution: FEMSolution
    reduction: BlochReduction

    reduced_dofs: int
    free_reduced_dofs: int

def solve_slab_pml_bloch(
    cfg: BottomPMLConfig,
    h_target: float,
    order: int = 2,
    kx: float = 0.0,
) -> BlochSolveResult:
    """Solve slab + bottom PML + lateral Bloch constraint.

    For this validation step the optical source is still
    normal incidence, so we deliberately require kx = 0.

    Non-zero kx will be enabled in the next step when the
    incident field is upgraded consistently for oblique
    incidence.
    """

    if abs(kx) > 1e-14:
        raise ValueError(
            "Non-zero kx is intentionally disabled in this "
            "validation step. We must upgrade the incident "
            "source to oblique incidence first."
        )

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

    K, M = assemble_physical_volume(
        basis,
        cfg,
    )

    A_physical = (
        K
        - (cfg.k0**2) * M
    )

    A_pml = assemble_bottom_pml(
        basis,
        cfg,
    )

    B_top, b = assemble_top_source(
        basis,
        cfg,
    )

    A_full = (
        A_physical
        + A_pml
        + B_top
    ).tocsr()

    reduction = build_bloch_projection(
        basis=basis,
        period=cfg.width,
        kx=kx,
    )

    P = reduction.P

    PH = P.conjugate().transpose()

    A_red = (
        PH
        @ A_full
        @ P
    ).tocsr()

    b_red = np.asarray(
        PH @ b,
        dtype=np.complex128,
    ).reshape(-1)

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

    A_free = A_red[
        free_red
    ][:, free_red]

    b_free = b_red[
        free_red
    ]

    z[free_red] = spsolve(
        A_free.tocsc(),
        b_free,
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

    return BlochSolveResult(
        solution=solution,
        reduction=reduction,
        reduced_dofs=A_red.shape[0],
        free_reduced_dofs=free_red.size,
    )
