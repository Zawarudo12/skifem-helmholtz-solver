from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

from skfem import (
    Basis,
    BilinearForm,
    LinearForm,
    asm,
)

from skfem.helpers import grad

from .forms import (
    scalar_mass,
    stiffness,
)

from .full_pml import FullPMLConfig

from .oblique import (
    incident_wavevector,
)


def assemble_full_physical_volume(
    basis: Basis,
    cfg: FullPMLConfig,
) -> tuple[
    csr_matrix,
    csr_matrix,
]:
    """Ordinary TE Helmholtz operator in the non-PML physical region."""

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

        M = M + (
            n**2
        ) * asm(
            scalar_mass,
            subbasis,
        )

    return (
        K.tocsr(),
        M.tocsr(),
    )


def _assemble_one_y_pml(
    basis: Basis,
    cfg: FullPMLConfig,
    tag: str,
    interface_y: float,
    thickness: float,
    n_medium: complex,
    top: bool,
) -> csr_matrix:
    """Assemble one y-directed PML."""

    mesh = basis.mesh

    pml_basis = basis.with_elements(
        mesh.subdomains[tag]
    )

    @BilinearForm(
        dtype=np.complex128
    )
    def pml_form(u, v, w):

        y = w.x[1]

        if top:
            xi = (
                interface_y - y
            ) / thickness

        else:
            xi = (
                y - interface_y
            ) / thickness

        xi = np.clip(
            xi,
            0.0,
            1.0,
        )

        sigma = (
            cfg.pml_sigma_max
            * xi**cfg.pml_order
        )

        # Same sign works above and below because the
        # coordinate direction itself reverses when
        # travelling upward into the top PML.
        sy = (
            1.0
            + 1j * sigma
        )

        gu = grad(u)
        gv = grad(v)

        gradient_term = (
            sy
            * gu[0]
            * gv[0]
            +
            (1.0 / sy)
            * gu[1]
            * gv[1]
        )

        mass_term = (
            cfg.k0**2
            * n_medium**2
            * sy
            * u
            * v
        )

        return (
            gradient_term
            - mass_term
        )

    return asm(
        pml_form,
        pml_basis,
    ).tocsr()


def assemble_full_pml_operator(
    basis: Basis,
    cfg: FullPMLConfig,
) -> csr_matrix:
    """Assemble both top and bottom PML operators."""

    top_pml = _assemble_one_y_pml(
        basis=basis,
        cfg=cfg,
        tag="top_pml",
        interface_y=0.0,
        thickness=cfg.pml_top,
        n_medium=cfg.n_inc,
        top=True,
    )

    bottom_pml = _assemble_one_y_pml(
        basis=basis,
        cfg=cfg,
        tag="bottom_pml",
        interface_y=cfg.total_height,
        thickness=cfg.pml_bottom,
        n_medium=cfg.n_out,
        top=False,
    )

    return (
        top_pml
        + bottom_pml
    ).tocsr()


def assemble_scattered_field_source(
    basis: Basis,
    cfg: FullPMLConfig,
    angle_deg: float,
) -> tuple[
    np.ndarray,
    complex,
    complex,
]:
    """Volume source for the TE scattered-field formulation.

    Assumes the background above and below the slab is the
    same homogeneous medium.
    """

    if not np.isclose(
        cfg.n_inc,
        cfg.n_out,
    ):
        raise ValueError(
            "This scattered-field source currently "
            "requires n_inc == n_out."
        )

    kx, ky = incident_wavevector(
        cfg,
        angle_deg,
    )

    slab_basis = basis.with_elements(
        basis.mesh.subdomains[
            "slab"
        ]
    )

    eps_background = (
        cfg.n_inc**2
    )

    eps_slab = (
        cfg.n_slab**2
    )

    contrast = (
        eps_slab
        - eps_background
    )

    @LinearForm(
        dtype=np.complex128
    )
    def source(v, w):

        x = w.x[0]
        y = w.x[1]

        u_inc = np.exp(
            1j
            * (
                kx * x
                + ky * y
            )
        )

        return (
            cfg.k0**2
            * contrast
            * u_inc
            * v
        )

    b = asm(
        source,
        slab_basis,
    )

    return (
        np.asarray(
            b,
            dtype=np.complex128,
        ),
        kx,
        ky,
    )