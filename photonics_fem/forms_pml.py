from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

from skfem import (
    Basis,
    BilinearForm,
    FacetBasis,
    LinearForm,
    asm,
)

from skfem.helpers import grad

from .forms import (
    stiffness,
    scalar_mass,
    boundary_mass,
)

from .pml import BottomPMLConfig

def assemble_physical_volume(
    basis: Basis,
    cfg: BottomPMLConfig,
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble K and M only over the non-PML physical region."""

    mesh = basis.mesh

    K = csr_matrix(
        (basis.N, basis.N),
        dtype=np.complex128,
    )

    M = csr_matrix(
        (basis.N, basis.N),
        dtype=np.complex128,
    )

    for tag, n in (
        ("top_air", cfg.n_inc),
        ("slab", cfg.n_slab),
        ("bottom_air", cfg.n_out),
    ):
        subbasis = basis.with_elements(
            mesh.subdomains[tag]
        )

        K = K + asm(
            stiffness,
            subbasis,
        )

        M = M + (n**2) * asm(
            scalar_mass,
            subbasis,
        )

    return K.tocsr(), M.tocsr()

def assemble_bottom_pml(
    basis: Basis,
    cfg: BottomPMLConfig,
) -> csr_matrix:
    """Assemble the complex-coordinate-stretched PML operator.

    y-directed coordinate stretch:

        s_y = 1 + i sigma(y)

    gives the transformed TE Helmholtz weak form:

        integral [
            s_y u_x v_x
            + (1/s_y) u_y v_y
            - k0^2 n^2 s_y u v
        ] dOmega
    """

    mesh = basis.mesh

    pml_basis = basis.with_elements(
        mesh.subdomains["bottom_pml"]
    )

    @BilinearForm(dtype=np.complex128)
    def pml_form(u, v, w):

        y = w.x[1]

        xi = (
            (y - cfg.pml_y0)
            / cfg.pml_bottom
        )

        xi = np.clip(
            xi,
            0.0,
            1.0,
        )

        sigma = (
            cfg.pml_sigma_max
            * xi**cfg.pml_order
        )

        sy = 1.0 + 1j * sigma

        gu = grad(u)
        gv = grad(v)

        gradient_term = (
            sy * gu[0] * gv[0]
            + (1.0 / sy) * gu[1] * gv[1]
        )

        mass_term = (
            (cfg.k0**2)
            * (cfg.n_out**2)
            * sy
            * u
            * v
        )

        return gradient_term - mass_term

    return asm(
        pml_form,
        pml_basis,
    ).tocsr()

def assemble_top_source(
    basis: Basis,
    cfg: BottomPMLConfig,
) -> tuple[csr_matrix, np.ndarray]:
    """Keep the validated Phase-1 Robin injection at the top only."""

    mesh = basis.mesh
    elem = basis.elem

    top = FacetBasis(
        mesh,
        elem,
        facets=mesh.boundaries["top"],
    )

    k_top = (
        cfg.k0
        * cfg.n_inc
    )

    B_top = (
        -1j * k_top
    ) * asm(
        boundary_mass,
        top,
    )

    @LinearForm(dtype=np.complex128)
    def incident(v, w):

        u_inc = np.exp(
            1j
            * k_top
            * w.x[1]
        )

        return (
            -2j
            * k_top
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
    )
