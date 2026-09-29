from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import MeshTri

from .config import SlabConfig

@dataclass(frozen=True)
class BottomPMLConfig(SlabConfig):
    """Phase-2 configuration with a bottom y-directed PML.

    The Phase-1 physical region remains unchanged:

        0 <= y <= total_height

    and the PML is appended below it:

        total_height < y <= computational_height
    """

    pml_bottom: float = 0.60
    pml_order: int = 3
    pml_sigma_max: float = 6.0

    @property
    def pml_y0(self) -> float:
        """Beginning of the bottom PML."""
        return self.total_height

    @property
    def computational_height(self) -> float:
        """Physical domain plus PML."""
        return self.total_height + self.pml_bottom

def _segment_nodes(
    a: float,
    b: float,
    h_target: float,
) -> np.ndarray:
    """Generate aligned 1D mesh coordinates."""
    n = max(
        1,
        int(np.ceil((b - a) / h_target)),
    )

    return np.linspace(
        a,
        b,
        n + 1,
    )

def make_bottom_pml_mesh(
    cfg: BottomPMLConfig,
    h_target: float,
) -> MeshTri:
    """Create a triangular mesh aligned to slab and PML interfaces."""

    nx = max(
        2,
        int(np.ceil(cfg.width / h_target)),
    )

    x = np.linspace(
        0.0,
        cfg.width,
        nx + 1,
    )

    yt = _segment_nodes(
        0.0,
        cfg.slab_y0,
        h_target,
    )

    ys = _segment_nodes(
        cfg.slab_y0,
        cfg.slab_y1,
        h_target,
    )[1:]

    yb = _segment_nodes(
        cfg.slab_y1,
        cfg.pml_y0,
        h_target,
    )[1:]

    yp = _segment_nodes(
        cfg.pml_y0,
        cfg.computational_height,
        h_target,
    )[1:]

    y = np.concatenate(
        [yt, ys, yb, yp]
    )

    mesh = MeshTri.init_tensor(x, y)

    tol = (
        100.0
        * np.finfo(float).eps
        * max(
            cfg.computational_height,
            cfg.width,
            1.0,
        )
    )

    mesh = mesh.with_boundaries(
        {
            "top": lambda X: np.isclose(
                X[1],
                0.0,
                atol=tol,
            ),
            "bottom": lambda X: np.isclose(
                X[1],
                cfg.computational_height,
                atol=tol,
            ),
            "left": lambda X: np.isclose(
                X[0],
                0.0,
                atol=tol,
            ),
            "right": lambda X: np.isclose(
                X[0],
                cfg.width,
                atol=tol,
            ),
        }
    )

    mesh = mesh.with_subdomains(
        {
            "top_air": lambda X: (
                X[1] < cfg.slab_y0 - tol
            ),
            "slab": lambda X: (
                (X[1] > cfg.slab_y0 - tol)
                & (X[1] < cfg.slab_y1 + tol)
            ),
            "bottom_air": lambda X: (
                (X[1] > cfg.slab_y1 + tol)
                & (X[1] < cfg.pml_y0 - tol)
            ),
            "bottom_pml": lambda X: (
                X[1] > cfg.pml_y0 - tol
            ),
        }
    )

    return mesh
