from __future__ import annotations

import numpy as np

from scipy.sparse import csr_matrix

from skfem import (
    Basis,
    BilinearForm,
    asm,
)

from skfem.helpers import grad

from .full_pml import (
    FullPMLConfig,
)

from .pml_generic import (
    stretch_1d,
)

def assemble_te_generic_pml(
    basis: Basis,
    cfg: FullPMLConfig,
) -> csr_matrix:
    """Assemble the complete TE Helmholtz + PML operator.

    For complex coordinate stretches

        sx = d(x_tilde)/dx
        sy = d(y_tilde)/dy

    the transformed scalar TE weak form is

        integral [
            (sy/sx) ux vx
            + (sx/sy) uy vy
            - k0^2 eps_r sx sy u v
        ] dOmega.

    Currently:

        sx = 1

    because the lateral boundaries are Bloch-periodic.

    sy contains both the top and bottom PMLs.
    """

    @BilinearForm(
        dtype=np.complex128
    )
    def helmholtz_pml(u, v, w):

        x = w.x[0]
        y = w.x[1]

        sx = np.ones_like(
            x,
            dtype=np.complex128,
        )

        sy = stretch_1d(
            coordinate=y,

            physical_min=0.0,
            physical_max=cfg.total_height,

            pml_low=cfg.pml_top,
            pml_high=cfg.pml_bottom,

            sigma_max=cfg.pml_sigma_max,
            order=cfg.pml_order,
        )

        eps_r = np.full(
            y.shape,
            cfg.n_inc**2,
            dtype=np.complex128,
        )

        slab_mask = (
            (y >= cfg.slab_y0)
            & (y <= cfg.slab_y1)
        )

        bottom_mask = (
            y > cfg.slab_y1
        )

        eps_r[slab_mask] = (
            cfg.n_slab**2
        )

        eps_r[bottom_mask] = (
            cfg.n_out**2
        )

        gu = grad(u)
        gv = grad(v)

        x_term = (
            sy / sx
        ) * gu[0] * gv[0]

        y_term = (
            sx / sy
        ) * gu[1] * gv[1]

        mass_term = (
            cfg.k0**2
            * eps_r
            * sx
            * sy
            * u
            * v
        )

        return (
            x_term
            + y_term
            - mass_term
        )

    A = asm(
        helmholtz_pml,
        basis,
    )

    return A.tocsr()
