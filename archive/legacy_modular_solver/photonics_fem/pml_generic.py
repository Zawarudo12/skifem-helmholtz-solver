from __future__ import annotations

import numpy as np

def polynomial_sigma(
    distance: np.ndarray,
    thickness: float,
    sigma_max: float,
    order: int,
) -> np.ndarray:
    """Polynomial PML attenuation profile.

    sigma = sigma_max * (distance / thickness)**order
    """

    if thickness <= 0.0:
        return np.zeros_like(
            distance,
            dtype=float,
        )

    xi = np.clip(
        distance / thickness,
        0.0,
        1.0,
    )

    return (
        sigma_max
        * xi**order
    )

def stretch_1d(
    coordinate: np.ndarray,
    physical_min: float,
    physical_max: float,
    pml_low: float,
    pml_high: float,
    sigma_max: float,
    order: int,
) -> np.ndarray:
    """Return complex coordinate stretch s = 1 + i sigma.

    The physical region is

        physical_min <= coordinate <= physical_max

    with optional PMLs on either side.
    """

    sigma = np.zeros_like(
        coordinate,
        dtype=float,
    )

    if pml_low > 0.0:

        mask = (
            coordinate
            < physical_min
        )

        distance = (
            physical_min
            - coordinate[mask]
        )

        sigma[mask] = polynomial_sigma(
            distance,
            pml_low,
            sigma_max,
            order,
        )

    if pml_high > 0.0:

        mask = (
            coordinate
            > physical_max
        )

        distance = (
            coordinate[mask]
            - physical_max
        )

        sigma[mask] = polynomial_sigma(
            distance,
            pml_high,
            sigma_max,
            order,
        )

    return (
        1.0
        + 1j * sigma
    )
