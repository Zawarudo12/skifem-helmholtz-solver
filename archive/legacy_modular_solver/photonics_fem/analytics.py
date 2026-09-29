from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .config import SlabConfig

def slab_rt(cfg: SlabConfig) -> tuple[complex, complex, float, float]:
    """Air/slab/air (or n0/n1/n2) normal-incidence TE amplitudes and powers.

    Convention: exp(-i omega t), forward/downward wave exp(+i k y).
    r and t are referenced to the two slab interfaces.  For nonmagnetic
    normal incidence, T = Re(n2)/Re(n0) * |t|^2 for lossless outer media.
    """
    n0, n1, n2 = cfg.n_inc, cfg.n_slab, cfg.n_out
    delta = cfg.k0 * n1 * cfg.slab_thickness

    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    t01 = 2.0 * n0 / (n0 + n1)
    t12 = 2.0 * n1 / (n1 + n2)

    phase2 = np.exp(2j * delta)
    den = 1.0 + r01 * r12 * phase2
    r = (r01 + r12 * phase2) / den
    t = (t01 * t12 * np.exp(1j * delta)) / den

    R = float(abs(r) ** 2)
    T = float((np.real(n2) / np.real(n0)) * abs(t) ** 2)
    return r, t, R, T

def slab_field(y: ArrayLike, cfg: SlabConfig) -> NDArray[np.complex128]:
    """Exact total TE field versus depth y, with incident amplitude 1 at y=0."""
    y = np.asarray(y, dtype=float)
    r, t, _, _ = slab_rt(cfg)
    n0, n1, n2 = cfg.n_inc, cfg.n_slab, cfg.n_out
    k0 = cfg.k0
    a = cfg.slab_y0
    d = cfg.slab_thickness
    z = y - a

    A = 0.5 * ((1.0 + r) + (n0 / n1) * (1.0 - r))
    B = 0.5 * ((1.0 + r) - (n0 / n1) * (1.0 - r))
    phase_to_interface = np.exp(1j * k0 * n0 * a)

    u = np.empty_like(y, dtype=np.complex128)
    top = y <= a
    slab = (y > a) & (y < a + d)
    bottom = y >= a + d

    u[top] = phase_to_interface * (
        np.exp(1j * k0 * n0 * z[top])
        + r * np.exp(-1j * k0 * n0 * z[top])
    )
    u[slab] = phase_to_interface * (
        A * np.exp(1j * k0 * n1 * z[slab])
        + B * np.exp(-1j * k0 * n1 * z[slab])
    )
    u[bottom] = phase_to_interface * t * np.exp(
        1j * k0 * n2 * (z[bottom] - d)
    )
    return u

def fabry_perot_resonances(cfg: SlabConfig, m_values: ArrayLike) -> NDArray[np.float64]:
    """Lossless symmetric-slab resonances: lambda_m = 2 n d / m."""
    m = np.asarray(m_values, dtype=float)
    return 2.0 * float(np.real(cfg.n_slab)) * cfg.slab_thickness / m
