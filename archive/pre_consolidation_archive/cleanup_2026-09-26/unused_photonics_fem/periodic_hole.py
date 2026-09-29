from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skfem import (
    Basis,
    BilinearForm,
    LinearForm,
    asm,
)

from skfem.helpers import grad

from .full_pml import FullPMLConfig
from .pml_generic import stretch_1d


@dataclass(frozen=True)
class PeriodicHoleConfig(FullPMLConfig):

    hole_diameter: float = 0.50

    @property
    def hole_radius(self) -> float:
        return 0.5 * self.hole_diameter

    @property
    def hole_center_x(self) -> float:
        return 0.5 * self.width

    @property
    def hole_center_y(self) -> float:
        return 0.5 * (
            self.slab_y0
            + self.slab_y1
        )


def dielectric_mask(
    x,
    y,
    cfg: PeriodicHoleConfig,
):
    """True inside dielectric slab excluding circular air hole."""

    in_slab = (
        (y >= cfg.slab_y0)
        &
        (y <= cfg.slab_y1)
    )

    r2 = (
        (x - cfg.hole_center_x) ** 2
        +
        (y - cfg.hole_center_y) ** 2
    )

    in_hole = (
        r2
        <= cfg.hole_radius ** 2
    )

    return (
        in_slab
        &
        (~in_hole)
    )


def epsilon_r(
    x,
    y,
    cfg: PeriodicHoleConfig,
):
    """Relative permittivity distribution."""

    eps = np.full(
        np.shape(x),
        cfg.n_inc ** 2,
        dtype=np.complex128,
    )

    mask = dielectric_mask(
        x,
        y,
        cfg,
    )

    eps[mask] = (
        cfg.n_slab ** 2
    )

    return eps


def assemble_periodic_hole_operator(
    basis: Basis,
    cfg: PeriodicHoleConfig,
):
    """TE Helmholtz operator with circular air hole and y-PML."""

    @BilinearForm(
        dtype=np.complex128
    )
    def form(u, v, w):

        x = w.x[0]
        y = w.x[1]

        # No x-PML because x is periodic.
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

        eps = epsilon_r(
            x,
            y,
            cfg,
        )

        gu = grad(u)
        gv = grad(v)

        return (
            (sy / sx)
            * gu[0]
            * gv[0]

            +

            (sx / sy)
            * gu[1]
            * gv[1]

            -

            cfg.k0 ** 2
            * eps
            * sx
            * sy
            * u
            * v
        )

    return asm(
        form,
        basis,
    ).tocsr()


def assemble_periodic_hole_source(
    basis: Basis,
    cfg: PeriodicHoleConfig,
):
    """Scattered-field source for normally incident plane wave."""

    eps_background = (
        cfg.n_inc ** 2
    )

    @LinearForm(
        dtype=np.complex128
    )
    def source(v, w):

        x = w.x[0]
        y = w.x[1]

        eps = epsilon_r(
            x,
            y,
            cfg,
        )

        contrast = (
            eps
            - eps_background
        )

        # Normal-incidence downward wave:
        #
        # exp(+i k y)
        u_inc = np.exp(
            1j
            * cfg.k0
            * cfg.n_inc
            * y
        )

        return (
            cfg.k0 ** 2
            * contrast
            * u_inc
            * v
        )

    return np.asarray(
        asm(
            source,
            basis,
        ),
        dtype=np.complex128,
    )