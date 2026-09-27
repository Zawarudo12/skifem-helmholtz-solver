from __future__ import annotations
import numpy as np
from skfem import MeshTri
from .config import SlabConfig


def _segment_nodes(a: float, b: float, h_target: float) -> np.ndarray:
    n = max(1, int(np.ceil((b - a) / h_target)))
    return np.linspace(a, b, n + 1)


def make_aligned_mesh(cfg: SlabConfig, h_target: float) -> MeshTri:
    """Structured triangular mesh whose y-grid is exactly aligned to interfaces."""
    nx = max(2, int(np.ceil(cfg.width / h_target)))
    x = np.linspace(0.0, cfg.width, nx + 1)

    yt = _segment_nodes(0.0, cfg.slab_y0, h_target)
    ys = _segment_nodes(cfg.slab_y0, cfg.slab_y1, h_target)[1:]
    yb = _segment_nodes(cfg.slab_y1, cfg.total_height, h_target)[1:]
    y = np.concatenate([yt, ys, yb])

    mesh = MeshTri.init_tensor(x, y)
    tol = 100.0 * np.finfo(float).eps * max(cfg.total_height, cfg.width, 1.0)

    mesh = mesh.with_boundaries({
        "top": lambda X: np.isclose(X[1], 0.0, atol=tol),
        "bottom": lambda X: np.isclose(X[1], cfg.total_height, atol=tol),
        "left": lambda X: np.isclose(X[0], 0.0, atol=tol),
        "right": lambda X: np.isclose(X[0], cfg.width, atol=tol),
    })
    mesh = mesh.with_subdomains({
        "top_air": lambda X: X[1] < cfg.slab_y0 - tol,
        "slab": lambda X: (X[1] > cfg.slab_y0 - tol) & (X[1] < cfg.slab_y1 + tol),
        "bottom_air": lambda X: X[1] > cfg.slab_y1 + tol,
    })
    return mesh


def max_edge_length(mesh: MeshTri) -> float:
    """Largest physical triangle edge length."""
    p, t = mesh.p, mesh.t
    a, b, c = p[:, t[0]], p[:, t[1]], p[:, t[2]]
    lengths = np.concatenate([
        np.linalg.norm(a - b, axis=0),
        np.linalg.norm(b - c, axis=0),
        np.linalg.norm(c - a, axis=0),
    ])
    return float(np.max(lengths))
