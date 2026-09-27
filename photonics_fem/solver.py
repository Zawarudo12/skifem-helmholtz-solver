from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, ElementTriP1, ElementTriP2
from .config import SlabConfig
from .forms import assemble_robin_and_source, assemble_te_volume
from .mesh import make_aligned_mesh

@dataclass
class FEMSolution:
    cfg: SlabConfig
    order: int
    mesh: Any
    basis: Basis
    u: np.ndarray
    A: csr_matrix
    b: np.ndarray

def solve_slab(cfg: SlabConfig, h_target: float, order: int = 1) -> FEMSolution:
    """Solve the Phase-1 TE slab problem using P1 or P2 triangles."""
    if order == 1:
        elem = ElementTriP1()
    elif order == 2:
        elem = ElementTriP2()
    else:
        raise ValueError("order must be 1 or 2")

    mesh = make_aligned_mesh(cfg, h_target=h_target)
    basis = Basis(mesh, elem, intorder=max(4, 2 * order + 2))

    K, M = assemble_te_volume(basis, cfg)
    B, b = assemble_robin_and_source(basis, cfg)
    A = (K - (cfg.k0 ** 2) * M + B).tocsr()

    u = spsolve(A.tocsc(), b)
    return FEMSolution(cfg=cfg, order=order, mesh=mesh, basis=basis, u=u, A=A, b=b)
