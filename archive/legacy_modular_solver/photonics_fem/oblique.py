from __future__ import annotations

import numpy as np

from .config import SlabConfig

def outgoing_ky(
    k0: float,
    n: complex,
    kx: complex,
) -> complex:
    """Return the downward/outgoing y-wavenumber.

    Convention:

        exp(-i omega t)

    Downward propagation:

        exp(+i ky y)

    For passive/evanescent waves we choose Im(ky) >= 0
    so the field decays as y increases.
    """

    ky = complex(
        np.sqrt(
            (k0 * n) ** 2
            - kx**2
            + 0.0j
        )
    )

    if ky.imag < 0.0:
        ky = -ky

    elif abs(ky.imag) < 1e-14 and ky.real < 0.0:
        ky = -ky

    return ky

def incident_wavevector(
    cfg: SlabConfig,
    angle_deg: float,
) -> tuple[complex, complex]:
    """Return kx and ky in the incident medium."""

    theta = np.deg2rad(
        angle_deg
    )

    k_inc = (
        cfg.k0
        * cfg.n_inc
    )

    kx = (
        k_inc
        * np.sin(theta)
    )

    ky = outgoing_ky(
        cfg.k0,
        cfg.n_inc,
        kx,
    )

    return complex(kx), complex(ky)

def slab_rt_te_oblique(
    cfg: SlabConfig,
    angle_deg: float,
) -> tuple[
    complex,
    complex,
    float,
    float,
]:
    """Exact TE thin-film R/T at oblique incidence."""

    kx, ky0 = incident_wavevector(
        cfg,
        angle_deg,
    )

    ky1 = outgoing_ky(
        cfg.k0,
        cfg.n_slab,
        kx,
    )

    ky2 = outgoing_ky(
        cfg.k0,
        cfg.n_out,
        kx,
    )

    y0 = ky0 / cfg.k0
    y1 = ky1 / cfg.k0
    y2 = ky2 / cfg.k0

    r01 = (
        (y0 - y1)
        / (y0 + y1)
    )

    r12 = (
        (y1 - y2)
        / (y1 + y2)
    )

    t01 = (
        2.0 * y0
        / (y0 + y1)
    )

    t12 = (
        2.0 * y1
        / (y1 + y2)
    )

    delta = (
        ky1
        * cfg.slab_thickness
    )

    denominator = (
        1.0
        + r01
        * r12
        * np.exp(
            2j * delta
        )
    )

    r = (
        r01
        + r12
        * np.exp(
            2j * delta
        )
    ) / denominator

    t = (
        t01
        * t12
        * np.exp(
            1j * delta
        )
    ) / denominator

    R = float(
        abs(r) ** 2
    )

    T = float(
        np.real(y2)
        / np.real(y0)
        * abs(t) ** 2
    )

    return r, t, R, T
