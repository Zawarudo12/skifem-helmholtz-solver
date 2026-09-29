from __future__ import annotations

import numpy as np

from scipy.sparse.linalg import spsolve

from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
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


def solve_slab_bottom_pml(
    cfg: BottomPMLConfig,
    h_target: float,
    order: int = 2,
) -> FEMSolution:
    """Solve the TE slab problem using a bottom PML."""

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

    # Ordinary Helmholtz region.
    K, M = assemble_physical_volume(
        basis,
        cfg,
    )

    A_physical = (
        K
        - (cfg.k0**2) * M
    )

    # Complex-stretched PML.
    A_pml = assemble_bottom_pml(
        basis,
        cfg,
    )

    # Top Robin radiation + incident-wave injection.
    B_top, b = assemble_top_source(
        basis,
        cfg,
    )

    A = (
        A_physical
        + A_pml
        + B_top
    ).tocsr()

    # Terminate the PML with homogeneous Dirichlet:
    #
    #     u = 0
    #
    # If the PML has done its job, essentially no field reaches here.

    bottom_dofs = (
        basis
        .get_dofs("bottom")
        .all()
    )

    all_dofs = np.arange(
        basis.N,
        dtype=int,
    )

    free_dofs = np.setdiff1d(
        all_dofs,
        bottom_dofs,
    )

    u = np.zeros(
        basis.N,
        dtype=np.complex128,
    )

    A_free = A[
        free_dofs
    ][:, free_dofs]

    b_free = b[
        free_dofs
    ]

    u[free_dofs] = spsolve(
        A_free.tocsc(),
        b_free,
    )

    return FEMSolution(
        cfg=cfg,
        order=order,
        mesh=mesh,
        basis=basis,
        u=u,
        A=A,
        b=b,
    )