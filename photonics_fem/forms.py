from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from skfem import Basis, BilinearForm, FacetBasis, LinearForm, asm
from skfem.helpers import dot, grad
from .config import SlabConfig


@BilinearForm(dtype=np.complex128)
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm(dtype=np.complex128)
def scalar_mass(u, v, w):
    return u * v


@BilinearForm(dtype=np.complex128)
def boundary_mass(u, v, w):
    return u * v


def assemble_te_volume(basis: Basis, cfg: SlabConfig) -> tuple[csr_matrix, csr_matrix]:
    """Assemble K and epsilon-weighted M for TE(E_z): p=1, q=epsilon_r=n^2."""
    mesh, elem = basis.mesh, basis.elem
    K = asm(stiffness, basis).tocsr()

    M = None
    for tag, n in (
        ("top_air", cfg.n_inc),
        ("slab", cfg.n_slab),
        ("bottom_air", cfg.n_out),
    ):
        subbasis = basis.with_elements(mesh.subdomains[tag])
        block = (n ** 2) * asm(scalar_mass, subbasis)
        M = block if M is None else M + block
    return K, M.tocsr()


def assemble_robin_and_source(
    basis: Basis, cfg: SlabConfig
) -> tuple[csr_matrix, np.ndarray]:
    """Robin radiation at top/bottom plus incoming plane-wave injection at top.

    In outer air, p=1.  With exp(-i wt), outgoing waves satisfy
        d_n u - i k u = 0.
    The total-field top BC is
        d_n u - i k u = -2 i k u_inc.
    """
    mesh, elem = basis.mesh, basis.elem
    top = FacetBasis(mesh, elem, facets=mesh.boundaries["top"])
    bottom = FacetBasis(mesh, elem, facets=mesh.boundaries["bottom"])

    k_top = cfg.k0 * cfg.n_inc
    k_bottom = cfg.k0 * cfg.n_out
    B = (-1j * k_top) * asm(boundary_mass, top)
    B = B + (-1j * k_bottom) * asm(boundary_mass, bottom)

    @LinearForm(dtype=np.complex128)
    def incident(v, w):
        # y_top = 0 => u_inc=exp(i k y)=1 on the boundary.
        return (-2j * k_top * np.exp(1j * k_top * w.x[1])) * v

    b = asm(incident, top)
    return B.tocsr(), np.asarray(b, dtype=np.complex128)
