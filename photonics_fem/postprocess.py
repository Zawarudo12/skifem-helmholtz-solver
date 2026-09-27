from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from .analytics import slab_field
from .solver import FEMSolution


def sample_centerline(sol: FEMSolution, y: ArrayLike) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.full_like(y, 0.5 * sol.cfg.width)
    pts = np.vstack([x, y])
    return np.asarray(sol.basis.interpolator(sol.u)(pts), dtype=complex)


def relative_l2_profile_error(sol: FEMSolution, npts: int = 4001) -> float:
    """Relative L2 error of the centerline field profile."""
    y = np.linspace(0.0, sol.cfg.total_height, npts)
    uh = sample_centerline(sol, y)
    ue = slab_field(y, sol.cfg)
    num = np.trapezoid(np.abs(uh - ue) ** 2, y)
    den = np.trapezoid(np.abs(ue) ** 2, y)
    return float(np.sqrt(num / den))


def fit_plane_waves(sol: FEMSolution) -> dict[str, complex | float]:
    """Least-squares extraction of downward/upward amplitudes in homogeneous air."""
    cfg = sol.cfg
    yt = np.linspace(0.12 * cfg.air_top, 0.82 * cfg.air_top, 80)
    yb = np.linspace(cfg.slab_y1 + 0.18 * cfg.air_bottom,
                     cfg.total_height - 0.12 * cfg.air_bottom, 80)
    ut = sample_centerline(sol, yt)
    ub = sample_centerline(sol, yb)

    kt = cfg.k0 * cfg.n_inc
    kb = cfg.k0 * cfg.n_out
    Xt = np.column_stack([np.exp(1j * kt * yt), np.exp(-1j * kt * yt)])
    Xb = np.column_stack([np.exp(1j * kb * yb), np.exp(-1j * kb * yb)])
    a_inc, a_refl = np.linalg.lstsq(Xt, ut, rcond=None)[0]
    a_trans, a_in_bottom = np.linalg.lstsq(Xb, ub, rcond=None)[0]

    R = float(abs(a_refl / a_inc) ** 2)
    T = float((np.real(cfg.n_out) / np.real(cfg.n_inc)) * abs(a_trans / a_inc) ** 2)
    return {
        "a_inc": a_inc,
        "a_refl": a_refl,
        "a_trans": a_trans,
        "a_in_bottom": a_in_bottom,
        "R": R,
        "T": T,
        "R_plus_T": R + T,
        "bottom_incoming_ratio": float(abs(a_in_bottom / a_inc)),
    }
