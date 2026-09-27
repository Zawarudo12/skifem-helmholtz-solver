from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SlabConfig:
    """Geometry and material parameters, using micrometres throughout.

    Coordinate convention: x is lateral and y increases downward.
    The computational domain is 0 <= x <= width, 0 <= y <= total_height.
    """

    wavelength: float = 0.633
    n_inc: complex = 1.0 + 0.0j
    n_slab: complex = 1.7 + 0.0j
    n_out: complex = 1.0 + 0.0j
    slab_thickness: float = 0.50
    air_top: float = 0.50
    air_bottom: float = 0.50
    width: float = 0.50

    @property
    def k0(self) -> float:
        import numpy as np
        return 2.0 * np.pi / self.wavelength

    @property
    def slab_y0(self) -> float:
        return self.air_top

    @property
    def slab_y1(self) -> float:
        return self.air_top + self.slab_thickness

    @property
    def total_height(self) -> float:
        return self.air_top + self.slab_thickness + self.air_bottom

    @property
    def lambda_min_material(self) -> float:
        """Approximate shortest local wavelength for lossless/weakly-lossy media."""
        nmax = max(abs(self.n_inc), abs(self.n_slab), abs(self.n_out))
        return self.wavelength / nmax

    def with_wavelength(self, wavelength: float) -> "SlabConfig":
        return replace(self, wavelength=wavelength)
