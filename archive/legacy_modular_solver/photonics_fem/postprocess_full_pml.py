from __future__ import annotations

from typing import Sequence

import numpy as np

from .oblique import (
    outgoing_ky,
)

from .solver_full_pml import (
    FullPMLSolveResult,
)

def sample_scattered_centerline(
    result: FullPMLSolveResult,
    y: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Sample the scattered FEM field at x = width / 2."""

    sol = result.scattered
    cfg = sol.cfg

    y = np.asarray(
        y,
        dtype=float,
    )

    x = np.full_like(
        y,
        0.5 * cfg.width,
    )

    pts = np.vstack(
        [x, y]
    )

    return np.asarray(
        sol.basis.interpolator(
            sol.u
        )(pts),
        dtype=np.complex128,
    )

def sample_total_centerline(
    result: FullPMLSolveResult,
    y: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Total field in the physical region.

    total = incident + scattered
    """

    sol = result.scattered
    cfg = sol.cfg

    y = np.asarray(
        y,
        dtype=float,
    )

    if np.any(y < 0.0) or np.any(
        y > cfg.total_height
    ):
        raise ValueError(
            "Total-field sampling is defined here only "
            "inside the physical region."
        )

    x = np.full_like(
        y,
        0.5 * cfg.width,
    )

    u_scattered = (
        sample_scattered_centerline(
            result,
            y,
        )
    )

    u_incident = np.exp(
        1j
        * (
            result.kx * x
            + result.ky_inc * y
        )
    )

    return (
        u_incident
        + u_scattered
    )

def fit_full_pml_rt(
    result: FullPMLSolveResult,
) -> dict[str, complex | float]:
    """Extract TE R/T from the total physical field."""

    sol = result.scattered
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

    ut = sample_total_centerline(
        result,
        yt,
    )

    ub = sample_total_centerline(
        result,
        yb,
    )

    ky_top = outgoing_ky(
        cfg.k0,
        cfg.n_inc,
        result.kx,
    )

    ky_bottom = outgoing_ky(
        cfg.k0,
        cfg.n_out,
        result.kx,
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
