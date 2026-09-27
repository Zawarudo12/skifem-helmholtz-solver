from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import MeshTri

from .config import SlabConfig


@dataclass(frozen=True)
class FullPMLConfig(SlabConfig):
    """Slab configuration with PMLs above and below the physical domain."""

    pml_top: float = 0.60
    pml_bottom: float = 0.60

    pml_order: int = 3
    pml_sigma_max: float = 6.0

    @property
    def computational_ymin(self) -> float:
        return -self.pml_top

    @property
    def computational_ymax(self) -> float:
        return self.total_height + self.pml_bottom


def _segment_nodes(
    a: float,
    b: float,
    h_target: float,
) -> np.ndarray:
    n = max(
        1,
        int(np.ceil((b - a) / h_target)),
    )

    return np.linspace(
        a,
        b,
        n + 1,
    )


def make_full_pml_mesh(
    cfg: FullPMLConfig,
    h_target: float,
) -> MeshTri:
    """Create an interface-aligned mesh with top and bottom PMLs."""

    nx = max(
        2,
        int(np.ceil(cfg.width / h_target)),
    )

    x = np.linspace(
        0.0,
        cfg.width,
        nx + 1,
    )

    # Top PML: -L_top -> 0
    yp_top = _segment_nodes(
        cfg.computational_ymin,
        0.0,
        h_target,
    )

    # Physical top air: 0 -> slab_y0
    yt = _segment_nodes(
        0.0,
        cfg.slab_y0,
        h_target,
    )[1:]

    # Slab
    ys = _segment_nodes(
        cfg.slab_y0,
        cfg.slab_y1,
        h_target,
    )[1:]

    # Physical bottom air
    yb = _segment_nodes(
        cfg.slab_y1,
        cfg.total_height,
        h_target,
    )[1:]

    # Bottom PML
    yp_bottom = _segment_nodes(
        cfg.total_height,
        cfg.computational_ymax,
        h_target,
    )[1:]

    y = np.concatenate(
        [
            yp_top,
            yt,
            ys,
            yb,
            yp_bottom,
        ]
    )

    mesh = MeshTri.init_tensor(
        x,
        y,
    )

    tol = (
        100.0
        * np.finfo(float).eps
        * max(
            abs(cfg.computational_ymin),
            abs(cfg.computational_ymax),
            cfg.width,
            1.0,
        )
    )

    mesh = mesh.with_boundaries(
        {
            "top": lambda X: np.isclose(
                X[1],
                cfg.computational_ymin,
                atol=tol,
            ),
            "bottom": lambda X: np.isclose(
                X[1],
                cfg.computational_ymax,
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
            "top_pml": lambda X: (
                X[1] < -tol
            ),

            "top_air": lambda X: (
                (X[1] > -tol)
                & (X[1] < cfg.slab_y0 - tol)
            ),

            "slab": lambda X: (
                (X[1] > cfg.slab_y0 - tol)
                & (X[1] < cfg.slab_y1 + tol)
            ),

            "bottom_air": lambda X: (
                (X[1] > cfg.slab_y1 + tol)
                & (
                    X[1]
                    < cfg.total_height + tol
                )
            ),

            "bottom_pml": lambda X: (
                X[1] > cfg.total_height + tol
            ),
        }
    )

    return mesh