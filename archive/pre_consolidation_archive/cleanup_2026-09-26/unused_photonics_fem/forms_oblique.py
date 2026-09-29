from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

from skfem import (
    Basis,
    FacetBasis,
    LinearForm,
    asm,
)

from .forms import boundary_mass
from .oblique import incident_wavevector
from .pml import BottomPMLConfig


def assemble_oblique_top_source(
    basis: Basis,
    cfg: BottomPMLConfig,
    angle_deg: float,
) -> tuple[
    csr_matrix,
    np.ndarray,
    complex,
    complex,
]:
    """Top Robin BC + oblique incident TE plane wave."""

    mesh = basis.mesh
    elem = basis.elem

    top = FacetBasis(
        mesh,
        elem,
        facets=mesh.boundaries["top"],
    )

    kx, ky = incident_wavevector(
        cfg,
        angle_deg,
    )

    # Outgoing reflected-wave condition:
    #
    # d_n u - i ky u = ...
    B_top = (
        -1j * ky
    ) * asm(
        boundary_mass,
        top,
    )

    @LinearForm(dtype=np.complex128)
    def incident(v, w):

        x = w.x[0]
        y = w.x[1]

        u_inc = np.exp(
            1j
            * (
                kx * x
                + ky * y
            )
        )

        # For the total field:
        #
        # d_n u - i ky u
        #     = -2 i ky u_inc
        return (
            -2j
            * ky
            * u_inc
            * v
        )

    b = asm(
        incident,
        top,
    )

    return (
        B_top.tocsr(),
        np.asarray(
            b,
            dtype=np.complex128,
        ),
        kx,
        ky,
    )