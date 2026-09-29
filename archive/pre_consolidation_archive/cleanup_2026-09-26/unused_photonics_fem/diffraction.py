from __future__ import annotations

import numpy as np


def sample_horizontal_line(
    sol,
    y: float,
    width: float,
    npoints: int = 256,
    chunk: int = 32,
):
    """Memory-safe FEM sampling along one horizontal line."""

    dx = (
        width
        / npoints
    )

    # Midpoint samples avoid querying exactly on periodic edges.
    x = (
        np.arange(
            npoints,
            dtype=float,
        )
        + 0.5
    ) * dx

    values = np.empty(
        npoints,
        dtype=np.complex128,
    )

    interpolator = (
        sol.basis.interpolator(
            sol.u
        )
    )

    for start in range(
        0,
        npoints,
        chunk,
    ):

        stop = min(
            start + chunk,
            npoints,
        )

        xx = x[
            start:stop
        ]

        yy = np.full_like(
            xx,
            y,
        )

        pts = np.vstack(
            [
                xx,
                yy,
            ]
        )

        values[
            start:stop
        ] = np.asarray(
            interpolator(
                pts
            ),
            dtype=np.complex128,
        )

    return x, values


def diffraction_orders(
    result,
    cfg,
    npoints: int = 256,
):
    """Extract reflected/transmitted Rayleigh orders.

    Normal incidence.
    Same air medium above and below.
    """

    sol = result.scattered

    period = cfg.width

    k_air = (
        cfg.k0
        * np.real(cfg.n_inc)
    )

    ky_inc = k_air

    # --------------------------------------------------------
    # Sampling planes well away from slab and PML interfaces.
    # --------------------------------------------------------

    y_top = (
        0.5
        * cfg.air_top
    )

    y_bottom = (
        cfg.slab_y1
        + 0.5
        * cfg.air_bottom
    )

    # --------------------------------------------------------
    # TOP:
    #
    # scattered field only
    # =
    # reflected diffraction orders.
    # --------------------------------------------------------

    x, us_top = (
        sample_horizontal_line(
            sol,
            y=y_top,
            width=period,
            npoints=npoints,
        )
    )

    # --------------------------------------------------------
    # BOTTOM:
    #
    # total = background incident + scattered.
    # --------------------------------------------------------

    _, us_bottom = (
        sample_horizontal_line(
            sol,
            y=y_bottom,
            width=period,
            npoints=npoints,
        )
    )

    u_inc_bottom = np.exp(
        1j
        * cfg.k0
        * cfg.n_inc
        * y_bottom
    )

    u_total_bottom = (
        us_bottom
        + u_inc_bottom
    )

    # --------------------------------------------------------
    # Maximum possible propagating order.
    # Add two extra orders for diagnostics; evanescent ones
    # simply contribute zero power.
    # --------------------------------------------------------

    m_est = int(
        np.floor(
            k_air
            * period
            / (2.0 * np.pi)
        )
    )

    m_values = np.arange(
        -m_est - 2,
        m_est + 3,
        dtype=int,
    )

    rows = []

    for m in m_values:

        kx_m = (
            2.0
            * np.pi
            * m
            / period
        )

        ky2 = (
            k_air ** 2
            - kx_m ** 2
        )

        propagating = (
            ky2
            > 1e-12
        )

        if propagating:

            ky_m = float(
                np.sqrt(
                    ky2
                )
            )

        else:

            ky_m = complex(
                np.sqrt(
                    ky2
                    + 0.0j
                )
            )

        phase = np.exp(
            -1j
            * kx_m
            * x
        )

        # Fourier/Rayleigh amplitudes.
        r_m = np.mean(
            us_top
            * phase
        )

        t_m = np.mean(
            u_total_bottom
            * phase
        )

        if propagating:

            power_factor = (
                ky_m
                / ky_inc
            )

            R_m = float(
                power_factor
                * abs(r_m) ** 2
            )

            T_m = float(
                power_factor
                * abs(t_m) ** 2
            )

        else:

            R_m = 0.0
            T_m = 0.0

        rows.append(
            {
                "m": int(m),

                "kx": kx_m,

                "ky": ky_m,

                "propagating": propagating,

                "r": r_m,

                "t": t_m,

                "R": R_m,

                "T": T_m,
            }
        )

    R_total = sum(
        row["R"]
        for row in rows
    )

    T_total = sum(
        row["T"]
        for row in rows
    )

    return {
        "orders": rows,

        "R": float(R_total),

        "T": float(T_total),

        "R_plus_T": float(
            R_total
            + T_total
        ),

        "y_top": y_top,

        "y_bottom": y_bottom,
    }