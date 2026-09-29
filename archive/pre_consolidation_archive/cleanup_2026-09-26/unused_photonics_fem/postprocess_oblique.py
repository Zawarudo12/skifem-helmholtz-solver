from __future__ import annotations

import numpy as np

from .oblique import (
    outgoing_ky,
)

from .postprocess import (
    sample_centerline,
)

from .solver import FEMSolution


def fit_oblique_plane_waves(
    sol: FEMSolution,
    kx: complex,
) -> dict[str, complex | float]:
    """Fit forward/backward waves at fixed x for oblique incidence."""

    cfg = sol.cfg

    yt = np.linspace(
        0.12 * cfg.air_top,
        0.82 * cfg.air_top,
        100,
    )

    yb = np.linspace(
        cfg.slab_y1
        + 0.18 * cfg.air_bottom,

        cfg.total_height
        - 0.12 * cfg.air_bottom,

        100,
    )

    ut = sample_centerline(
        sol,
        yt,
    )

    ub = sample_centerline(
        sol,
        yb,
    )

    ky_top = outgoing_ky(
        cfg.k0,
        cfg.n_inc,
        kx,
    )

    ky_bottom = outgoing_ky(
        cfg.k0,
        cfg.n_out,
        kx,
    )

    Xt = np.column_stack(
        [
            np.exp(
                1j
                * ky_top
                * yt
            ),
            np.exp(
                -1j
                * ky_top
                * yt
            ),
        ]
    )

    Xb = np.column_stack(
        [
            np.exp(
                1j
                * ky_bottom
                * yb
            ),
            np.exp(
                -1j
                * ky_bottom
                * yb
            ),
        ]
    )

    a_inc, a_refl = np.linalg.lstsq(
        Xt,
        ut,
        rcond=None,
    )[0]

    a_trans, a_in_bottom = np.linalg.lstsq(
        Xb,
        ub,
        rcond=None,
    )[0]

    R = float(
        abs(
            a_refl
            / a_inc
        ) ** 2
    )

    # TE normal power flux is proportional to Re(ky).
    T = float(
        np.real(ky_bottom)
        / np.real(ky_top)
        * abs(
            a_trans
            / a_inc
        ) ** 2
    )

    return {
        "a_inc": a_inc,
        "a_refl": a_refl,
        "a_trans": a_trans,
        "a_in_bottom": a_in_bottom,

        "R": R,
        "T": T,
        "R_plus_T": R + T,

        "bottom_incoming_ratio": float(
            abs(
                a_in_bottom
                / a_inc
            )
        ),
    }