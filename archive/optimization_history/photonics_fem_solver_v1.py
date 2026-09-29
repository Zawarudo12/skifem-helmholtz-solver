from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, FacetBasis, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad

@dataclass(frozen=True)
class SlabConfig:
    """Geometry and material parameters, using micrometres throughout.

    Coordinate convention: x is lateral and y increases downward.
    The computational domain is 0 <= x <= width, 0 <= y <= total_height.
    """
    wavelength: float = 0.633
    n_inc: complex = 1.0 + 0j
    n_slab: complex = 1.7 + 0j
    n_out: complex = 1.0 + 0j
    slab_thickness: float = 0.5
    air_top: float = 0.5
    air_bottom: float = 0.5
    width: float = 0.5

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

    def with_wavelength(self, wavelength: float) -> 'SlabConfig':
        return replace(self, wavelength=wavelength)

def slab_rt(cfg: SlabConfig) -> tuple[complex, complex, float, float]:
    """Air/slab/air (or n0/n1/n2) normal-incidence TE amplitudes and powers.

    Convention: exp(-i omega t), forward/downward wave exp(+i k y).
    r and t are referenced to the two slab interfaces.  For nonmagnetic
    normal incidence, T = Re(n2)/Re(n0) * |t|^2 for lossless outer media.
    """
    n0, n1, n2 = (cfg.n_inc, cfg.n_slab, cfg.n_out)
    delta = cfg.k0 * n1 * cfg.slab_thickness
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    t01 = 2.0 * n0 / (n0 + n1)
    t12 = 2.0 * n1 / (n1 + n2)
    phase2 = np.exp(2j * delta)
    den = 1.0 + r01 * r12 * phase2
    r = (r01 + r12 * phase2) / den
    t = t01 * t12 * np.exp(1j * delta) / den
    R = float(abs(r) ** 2)
    T = float(np.real(n2) / np.real(n0) * abs(t) ** 2)
    return (r, t, R, T)

def slab_field(y: ArrayLike, cfg: SlabConfig) -> NDArray[np.complex128]:
    """Exact total TE field versus depth y, with incident amplitude 1 at y=0."""
    y = np.asarray(y, dtype=float)
    r, t, _, _ = slab_rt(cfg)
    n0, n1, n2 = (cfg.n_inc, cfg.n_slab, cfg.n_out)
    k0 = cfg.k0
    a = cfg.slab_y0
    d = cfg.slab_thickness
    z = y - a
    A = 0.5 * (1.0 + r + n0 / n1 * (1.0 - r))
    B = 0.5 * (1.0 + r - n0 / n1 * (1.0 - r))
    phase_to_interface = np.exp(1j * k0 * n0 * a)
    u = np.empty_like(y, dtype=np.complex128)
    top = y <= a
    slab = (y > a) & (y < a + d)
    bottom = y >= a + d
    u[top] = phase_to_interface * (np.exp(1j * k0 * n0 * z[top]) + r * np.exp(-1j * k0 * n0 * z[top]))
    u[slab] = phase_to_interface * (A * np.exp(1j * k0 * n1 * z[slab]) + B * np.exp(-1j * k0 * n1 * z[slab]))
    u[bottom] = phase_to_interface * t * np.exp(1j * k0 * n2 * (z[bottom] - d))
    return u

def fabry_perot_resonances(cfg: SlabConfig, m_values: ArrayLike) -> NDArray[np.float64]:
    """Lossless symmetric-slab resonances: lambda_m = 2 n d / m."""
    m = np.asarray(m_values, dtype=float)
    return 2.0 * float(np.real(cfg.n_slab)) * cfg.slab_thickness / m

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
    mesh, elem = (basis.mesh, basis.elem)
    K = asm(stiffness, basis).tocsr()
    M = None
    for tag, n in (('top_air', cfg.n_inc), ('slab', cfg.n_slab), ('bottom_air', cfg.n_out)):
        subbasis = basis.with_elements(mesh.subdomains[tag])
        block = n ** 2 * asm(scalar_mass, subbasis)
        M = block if M is None else M + block
    return (K, M.tocsr())

def assemble_robin_and_source(basis: Basis, cfg: SlabConfig) -> tuple[csr_matrix, np.ndarray]:
    """Robin radiation at top/bottom plus incoming plane-wave injection at top.

    In outer air, p=1.  With exp(-i wt), outgoing waves satisfy
        d_n u - i k u = 0.
    The total-field top BC is
        d_n u - i k u = -2 i k u_inc.
    """
    mesh, elem = (basis.mesh, basis.elem)
    top = FacetBasis(mesh, elem, facets=mesh.boundaries['top'])
    bottom = FacetBasis(mesh, elem, facets=mesh.boundaries['bottom'])
    k_top = cfg.k0 * cfg.n_inc
    k_bottom = cfg.k0 * cfg.n_out
    B = -1j * k_top * asm(boundary_mass, top)
    B = B + -1j * k_bottom * asm(boundary_mass, bottom)

    @LinearForm(dtype=np.complex128)
    def incident(v, w):
        return -2j * k_top * np.exp(1j * k_top * w.x[1]) * v
    b = asm(incident, top)
    return (B.tocsr(), np.asarray(b, dtype=np.complex128))

def _segment_nodes_mesh(a: float, b: float, h_target: float) -> np.ndarray:
    n = max(1, int(np.ceil((b - a) / h_target)))
    return np.linspace(a, b, n + 1)

def make_aligned_mesh(cfg: SlabConfig, h_target: float) -> MeshTri:
    """Structured triangular mesh whose y-grid is exactly aligned to interfaces."""
    nx = max(2, int(np.ceil(cfg.width / h_target)))
    x = np.linspace(0.0, cfg.width, nx + 1)
    yt = _segment_nodes_mesh(0.0, cfg.slab_y0, h_target)
    ys = _segment_nodes_mesh(cfg.slab_y0, cfg.slab_y1, h_target)[1:]
    yb = _segment_nodes_mesh(cfg.slab_y1, cfg.total_height, h_target)[1:]
    y = np.concatenate([yt, ys, yb])
    mesh = MeshTri.init_tensor(x, y)
    tol = 100.0 * np.finfo(float).eps * max(cfg.total_height, cfg.width, 1.0)
    mesh = mesh.with_boundaries({'top': lambda X: np.isclose(X[1], 0.0, atol=tol), 'bottom': lambda X: np.isclose(X[1], cfg.total_height, atol=tol), 'left': lambda X: np.isclose(X[0], 0.0, atol=tol), 'right': lambda X: np.isclose(X[0], cfg.width, atol=tol)})
    mesh = mesh.with_subdomains({'top_air': lambda X: X[1] < cfg.slab_y0 - tol, 'slab': lambda X: (X[1] > cfg.slab_y0 - tol) & (X[1] < cfg.slab_y1 + tol), 'bottom_air': lambda X: X[1] > cfg.slab_y1 + tol})
    return mesh

def max_edge_length(mesh: MeshTri) -> float:
    """Largest physical triangle edge length."""
    p, t = (mesh.p, mesh.t)
    a, b, c = (p[:, t[0]], p[:, t[1]], p[:, t[2]])
    lengths = np.concatenate([np.linalg.norm(a - b, axis=0), np.linalg.norm(b - c, axis=0), np.linalg.norm(c - a, axis=0)])
    return float(np.max(lengths))

@dataclass
class FEMSolution:
    cfg: SlabConfig
    order: int
    mesh: Any
    basis: Basis
    u: np.ndarray
    A: csr_matrix
    b: np.ndarray

def solve_slab(cfg: SlabConfig, h_target: float, order: int=1) -> FEMSolution:
    """Solve the Phase-1 TE slab problem using P1 or P2 triangles."""
    if order == 1:
        elem = ElementTriP1()
    elif order == 2:
        elem = ElementTriP2()
    else:
        raise ValueError('order must be 1 or 2')
    mesh = make_aligned_mesh(cfg, h_target=h_target)
    basis = Basis(mesh, elem, intorder=max(4, 2 * order + 2))
    K, M = assemble_te_volume(basis, cfg)
    B, b = assemble_robin_and_source(basis, cfg)
    A = (K - cfg.k0 ** 2 * M + B).tocsr()
    u = spsolve(A.tocsc(), b)
    return FEMSolution(cfg=cfg, order=order, mesh=mesh, basis=basis, u=u, A=A, b=b)

@dataclass(frozen=True)
class BlochReduction:
    """Projection data for left/right Bloch-periodic DOF tying."""
    P: csr_matrix
    left_dofs: np.ndarray
    right_dofs: np.ndarray
    full_to_reduced: np.ndarray
    phase: complex

def build_bloch_projection(basis: Basis, period: float, kx: float, atol: float=1e-10) -> BlochReduction:
    """Build P enforcing

        u(x + period, y)
            = exp(i kx period) u(x, y).

    The full solution is reconstructed using

        u_full = P @ u_reduced.
    """
    left = np.asarray(basis.get_dofs('left').flatten(), dtype=int)
    right = np.asarray(basis.get_dofs('right').flatten(), dtype=int)
    if left.size != right.size:
        raise ValueError(f'Left/right boundary DOF counts do not match: {left.size} != {right.size}')
    y_left = basis.doflocs[1, left]
    y_right = basis.doflocs[1, right]
    left_order = np.argsort(y_left)
    right_order = np.argsort(y_right)
    left = left[left_order]
    right = right[right_order]
    y_left = basis.doflocs[1, left]
    y_right = basis.doflocs[1, right]
    max_pair_error = float(np.max(np.abs(y_left - y_right)))
    if max_pair_error > atol:
        raise ValueError(f'Periodic boundaries are not conforming. Maximum paired y mismatch = {max_pair_error:.3e}')
    phase = np.exp(1j * kx * period)
    n_full = basis.N
    is_right = np.zeros(n_full, dtype=bool)
    is_right[right] = True
    masters = np.flatnonzero(~is_right)
    n_reduced = masters.size
    master_to_reduced = np.full(n_full, -1, dtype=int)
    master_to_reduced[masters] = np.arange(n_reduced, dtype=int)
    full_to_reduced = master_to_reduced.copy()
    for l_dof, r_dof in zip(left, right):
        reduced_index = master_to_reduced[l_dof]
        if reduced_index < 0:
            raise RuntimeError('Left periodic DOF was unexpectedly removed from the master set.')
        full_to_reduced[r_dof] = reduced_index
    if np.any(full_to_reduced < 0):
        raise RuntimeError('Some full DOFs were not mapped to a reduced Bloch DOF.')
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for full_dof in masters:
        rows.append(int(full_dof))
        cols.append(int(master_to_reduced[full_dof]))
        data.append(1.0 + 0j)
    for l_dof, r_dof in zip(left, right):
        rows.append(int(r_dof))
        cols.append(int(master_to_reduced[l_dof]))
        data.append(phase)
    P = coo_matrix((np.asarray(data, dtype=np.complex128), (np.asarray(rows), np.asarray(cols))), shape=(n_full, n_reduced)).tocsr()
    return BlochReduction(P=P, left_dofs=left, right_dofs=right, full_to_reduced=full_to_reduced, phase=phase)

def bloch_mismatch(u: np.ndarray, reduction: BlochReduction) -> float:
    """Relative mismatch in u_R = phase * u_L."""
    u_left = u[reduction.left_dofs]
    u_right = u[reduction.right_dofs]
    error = np.max(np.abs(u_right - reduction.phase * u_left))
    scale = max(float(np.max(np.abs(u))), 1e-30)
    return float(error / scale)

@dataclass(frozen=True)
class BottomPMLConfig(SlabConfig):
    """Phase-2 configuration with a bottom y-directed PML.

    The Phase-1 physical region remains unchanged:

        0 <= y <= total_height

    and the PML is appended below it:

        total_height < y <= computational_height
    """
    pml_bottom: float = 0.6
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

def _segment_nodes_bottom_pml(a: float, b: float, h_target: float) -> np.ndarray:
    """Generate aligned 1D mesh coordinates."""
    n = max(1, int(np.ceil((b - a) / h_target)))
    return np.linspace(a, b, n + 1)

def make_bottom_pml_mesh(cfg: BottomPMLConfig, h_target: float) -> MeshTri:
    """Create a triangular mesh aligned to slab and PML interfaces."""
    nx = max(2, int(np.ceil(cfg.width / h_target)))
    x = np.linspace(0.0, cfg.width, nx + 1)
    yt = _segment_nodes_bottom_pml(0.0, cfg.slab_y0, h_target)
    ys = _segment_nodes_bottom_pml(cfg.slab_y0, cfg.slab_y1, h_target)[1:]
    yb = _segment_nodes_bottom_pml(cfg.slab_y1, cfg.pml_y0, h_target)[1:]
    yp = _segment_nodes_bottom_pml(cfg.pml_y0, cfg.computational_height, h_target)[1:]
    y = np.concatenate([yt, ys, yb, yp])
    mesh = MeshTri.init_tensor(x, y)
    tol = 100.0 * np.finfo(float).eps * max(cfg.computational_height, cfg.width, 1.0)
    mesh = mesh.with_boundaries({'top': lambda X: np.isclose(X[1], 0.0, atol=tol), 'bottom': lambda X: np.isclose(X[1], cfg.computational_height, atol=tol), 'left': lambda X: np.isclose(X[0], 0.0, atol=tol), 'right': lambda X: np.isclose(X[0], cfg.width, atol=tol)})
    mesh = mesh.with_subdomains({'top_air': lambda X: X[1] < cfg.slab_y0 - tol, 'slab': lambda X: (X[1] > cfg.slab_y0 - tol) & (X[1] < cfg.slab_y1 + tol), 'bottom_air': lambda X: (X[1] > cfg.slab_y1 + tol) & (X[1] < cfg.pml_y0 - tol), 'bottom_pml': lambda X: X[1] > cfg.pml_y0 - tol})
    return mesh

def assemble_physical_volume(basis: Basis, cfg: BottomPMLConfig) -> tuple[csr_matrix, csr_matrix]:
    """Assemble K and M only over the non-PML physical region."""
    mesh = basis.mesh
    K = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    M = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    for tag, n in (('top_air', cfg.n_inc), ('slab', cfg.n_slab), ('bottom_air', cfg.n_out)):
        subbasis = basis.with_elements(mesh.subdomains[tag])
        K = K + asm(stiffness, subbasis)
        M = M + n ** 2 * asm(scalar_mass, subbasis)
    return (K.tocsr(), M.tocsr())

def assemble_bottom_pml(basis: Basis, cfg: BottomPMLConfig) -> csr_matrix:
    """Assemble the complex-coordinate-stretched PML operator.

    y-directed coordinate stretch:

        s_y = 1 + i sigma(y)

    gives the transformed TE Helmholtz weak form:

        integral [
            s_y u_x v_x
            + (1/s_y) u_y v_y
            - k0^2 n^2 s_y u v
        ] dOmega
    """
    mesh = basis.mesh
    pml_basis = basis.with_elements(mesh.subdomains['bottom_pml'])

    @BilinearForm(dtype=np.complex128)
    def pml_form(u, v, w):
        y = w.x[1]
        xi = (y - cfg.pml_y0) / cfg.pml_bottom
        xi = np.clip(xi, 0.0, 1.0)
        sigma = cfg.pml_sigma_max * xi ** cfg.pml_order
        sy = 1.0 + 1j * sigma
        gu = grad(u)
        gv = grad(v)
        gradient_term = sy * gu[0] * gv[0] + 1.0 / sy * gu[1] * gv[1]
        mass_term = cfg.k0 ** 2 * cfg.n_out ** 2 * sy * u * v
        return gradient_term - mass_term
    return asm(pml_form, pml_basis).tocsr()

def assemble_top_source(basis: Basis, cfg: BottomPMLConfig) -> tuple[csr_matrix, np.ndarray]:
    """Keep the validated Phase-1 Robin injection at the top only."""
    mesh = basis.mesh
    elem = basis.elem
    top = FacetBasis(mesh, elem, facets=mesh.boundaries['top'])
    k_top = cfg.k0 * cfg.n_inc
    B_top = -1j * k_top * asm(boundary_mass, top)

    @LinearForm(dtype=np.complex128)
    def incident(v, w):
        u_inc = np.exp(1j * k_top * w.x[1])
        return -2j * k_top * u_inc * v
    b = asm(incident, top)
    return (B_top.tocsr(), np.asarray(b, dtype=np.complex128))

@dataclass
class BlochSolveResult:
    """Full FEM solution plus Bloch reduction information."""
    solution: FEMSolution
    reduction: BlochReduction
    reduced_dofs: int
    free_reduced_dofs: int

def solve_slab_pml_bloch(cfg: BottomPMLConfig, h_target: float, order: int=2, kx: float=0.0) -> BlochSolveResult:
    """Solve slab + bottom PML + lateral Bloch constraint.

    For this validation step the optical source is still
    normal incidence, so we deliberately require kx = 0.

    Non-zero kx will be enabled in the next step when the
    incident field is upgraded consistently for oblique
    incidence.
    """
    if abs(kx) > 1e-14:
        raise ValueError('Non-zero kx is intentionally disabled in this validation step. We must upgrade the incident source to oblique incidence first.')
    if order == 1:
        elem = ElementTriP1()
    elif order == 2:
        elem = ElementTriP2()
    else:
        raise ValueError('order must be 1 or 2')
    mesh = make_bottom_pml_mesh(cfg, h_target=h_target)
    basis = Basis(mesh, elem, intorder=max(4, 2 * order + 2))
    K, M = assemble_physical_volume(basis, cfg)
    A_physical = K - cfg.k0 ** 2 * M
    A_pml = assemble_bottom_pml(basis, cfg)
    B_top, b = assemble_top_source(basis, cfg)
    A_full = (A_physical + A_pml + B_top).tocsr()
    reduction = build_bloch_projection(basis=basis, period=cfg.width, kx=kx)
    P = reduction.P
    PH = P.conjugate().transpose()
    A_red = (PH @ A_full @ P).tocsr()
    b_red = np.asarray(PH @ b, dtype=np.complex128).reshape(-1)
    bottom_full = np.asarray(basis.get_dofs('bottom').flatten(), dtype=int)
    bottom_red = np.unique(reduction.full_to_reduced[bottom_full])
    all_red = np.arange(A_red.shape[0], dtype=int)
    free_red = np.setdiff1d(all_red, bottom_red)
    z = np.zeros(A_red.shape[0], dtype=np.complex128)
    A_free = A_red[free_red][:, free_red]
    b_free = b_red[free_red]
    z[free_red] = spsolve(A_free.tocsc(), b_free)
    u_full = np.asarray(P @ z, dtype=np.complex128).reshape(-1)
    solution = FEMSolution(cfg=cfg, order=order, mesh=mesh, basis=basis, u=u_full, A=A_full, b=b)
    return BlochSolveResult(solution=solution, reduction=reduction, reduced_dofs=A_red.shape[0], free_reduced_dofs=free_red.size)

def sample_centerline(sol: FEMSolution, y: ArrayLike) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.full_like(y, 0.5 * sol.cfg.width)
    pts = np.vstack([x, y])
    return np.asarray(sol.basis.interpolator(sol.u)(pts), dtype=complex)

def relative_l2_profile_error(sol: FEMSolution, npts: int=4001) -> float:
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
    yb = np.linspace(cfg.slab_y1 + 0.18 * cfg.air_bottom, cfg.total_height - 0.12 * cfg.air_bottom, 80)
    ut = sample_centerline(sol, yt)
    ub = sample_centerline(sol, yb)
    kt = cfg.k0 * cfg.n_inc
    kb = cfg.k0 * cfg.n_out
    Xt = np.column_stack([np.exp(1j * kt * yt), np.exp(-1j * kt * yt)])
    Xb = np.column_stack([np.exp(1j * kb * yb), np.exp(-1j * kb * yb)])
    a_inc, a_refl = np.linalg.lstsq(Xt, ut, rcond=None)[0]
    a_trans, a_in_bottom = np.linalg.lstsq(Xb, ub, rcond=None)[0]
    R = float(abs(a_refl / a_inc) ** 2)
    T = float(np.real(cfg.n_out) / np.real(cfg.n_inc) * abs(a_trans / a_inc) ** 2)
    return {'a_inc': a_inc, 'a_refl': a_refl, 'a_trans': a_trans, 'a_in_bottom': a_in_bottom, 'R': R, 'T': T, 'R_plus_T': R + T, 'bottom_incoming_ratio': float(abs(a_in_bottom / a_inc))}

@dataclass(frozen=True)
class FullPMLConfig(SlabConfig):
    """Slab configuration with PMLs above and below the physical domain."""
    pml_top: float = 0.6
    pml_bottom: float = 0.6
    pml_order: int = 3
    pml_sigma_max: float = 6.0

    @property
    def computational_ymin(self) -> float:
        return -self.pml_top

    @property
    def computational_ymax(self) -> float:
        return self.total_height + self.pml_bottom

def _segment_nodes_full_pml(a: float, b: float, h_target: float) -> np.ndarray:
    n = max(1, int(np.ceil((b - a) / h_target)))
    return np.linspace(a, b, n + 1)

def make_full_pml_mesh(cfg: FullPMLConfig, h_target: float) -> MeshTri:
    """Create an interface-aligned mesh with top and bottom PMLs."""
    nx = max(2, int(np.ceil(cfg.width / h_target)))
    x = np.linspace(0.0, cfg.width, nx + 1)
    yp_top = _segment_nodes_full_pml(cfg.computational_ymin, 0.0, h_target)
    yt = _segment_nodes_full_pml(0.0, cfg.slab_y0, h_target)[1:]
    ys = _segment_nodes_full_pml(cfg.slab_y0, cfg.slab_y1, h_target)[1:]
    yb = _segment_nodes_full_pml(cfg.slab_y1, cfg.total_height, h_target)[1:]
    yp_bottom = _segment_nodes_full_pml(cfg.total_height, cfg.computational_ymax, h_target)[1:]
    y = np.concatenate([yp_top, yt, ys, yb, yp_bottom])
    mesh = MeshTri.init_tensor(x, y)
    tol = 100.0 * np.finfo(float).eps * max(abs(cfg.computational_ymin), abs(cfg.computational_ymax), cfg.width, 1.0)
    mesh = mesh.with_boundaries({'top': lambda X: np.isclose(X[1], cfg.computational_ymin, atol=tol), 'bottom': lambda X: np.isclose(X[1], cfg.computational_ymax, atol=tol), 'left': lambda X: np.isclose(X[0], 0.0, atol=tol), 'right': lambda X: np.isclose(X[0], cfg.width, atol=tol)})
    mesh = mesh.with_subdomains({'top_pml': lambda X: X[1] < -tol, 'top_air': lambda X: (X[1] > -tol) & (X[1] < cfg.slab_y0 - tol), 'slab': lambda X: (X[1] > cfg.slab_y0 - tol) & (X[1] < cfg.slab_y1 + tol), 'bottom_air': lambda X: (X[1] > cfg.slab_y1 + tol) & (X[1] < cfg.total_height + tol), 'bottom_pml': lambda X: X[1] > cfg.total_height + tol})
    return mesh

def outgoing_ky(k0: float, n: complex, kx: complex) -> complex:
    """Return the downward/outgoing y-wavenumber.

    Convention:

        exp(-i omega t)

    Downward propagation:

        exp(+i ky y)

    For passive/evanescent waves we choose Im(ky) >= 0
    so the field decays as y increases.
    """
    ky = complex(np.sqrt((k0 * n) ** 2 - kx ** 2 + 0j))
    if ky.imag < 0.0:
        ky = -ky
    elif abs(ky.imag) < 1e-14 and ky.real < 0.0:
        ky = -ky
    return ky

def incident_wavevector(cfg: SlabConfig, angle_deg: float) -> tuple[complex, complex]:
    """Return kx and ky in the incident medium."""
    theta = np.deg2rad(angle_deg)
    k_inc = cfg.k0 * cfg.n_inc
    kx = k_inc * np.sin(theta)
    ky = outgoing_ky(cfg.k0, cfg.n_inc, kx)
    return (complex(kx), complex(ky))

def slab_rt_te_oblique(cfg: SlabConfig, angle_deg: float) -> tuple[complex, complex, float, float]:
    """Exact TE thin-film R/T at oblique incidence."""
    kx, ky0 = incident_wavevector(cfg, angle_deg)
    ky1 = outgoing_ky(cfg.k0, cfg.n_slab, kx)
    ky2 = outgoing_ky(cfg.k0, cfg.n_out, kx)
    y0 = ky0 / cfg.k0
    y1 = ky1 / cfg.k0
    y2 = ky2 / cfg.k0
    r01 = (y0 - y1) / (y0 + y1)
    r12 = (y1 - y2) / (y1 + y2)
    t01 = 2.0 * y0 / (y0 + y1)
    t12 = 2.0 * y1 / (y1 + y2)
    delta = ky1 * cfg.slab_thickness
    denominator = 1.0 + r01 * r12 * np.exp(2j * delta)
    r = (r01 + r12 * np.exp(2j * delta)) / denominator
    t = t01 * t12 * np.exp(1j * delta) / denominator
    R = float(abs(r) ** 2)
    T = float(np.real(y2) / np.real(y0) * abs(t) ** 2)
    return (r, t, R, T)

def assemble_full_physical_volume(basis: Basis, cfg: FullPMLConfig) -> tuple[csr_matrix, csr_matrix]:
    """Ordinary TE Helmholtz operator in the non-PML physical region."""
    mesh = basis.mesh
    K = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    M = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    for tag, n in (('top_air', cfg.n_inc), ('slab', cfg.n_slab), ('bottom_air', cfg.n_out)):
        subbasis = basis.with_elements(mesh.subdomains[tag])
        K = K + asm(stiffness, subbasis)
        M = M + n ** 2 * asm(scalar_mass, subbasis)
    return (K.tocsr(), M.tocsr())

def _assemble_one_y_pml(basis: Basis, cfg: FullPMLConfig, tag: str, interface_y: float, thickness: float, n_medium: complex, top: bool) -> csr_matrix:
    """Assemble one y-directed PML."""
    mesh = basis.mesh
    pml_basis = basis.with_elements(mesh.subdomains[tag])

    @BilinearForm(dtype=np.complex128)
    def pml_form(u, v, w):
        y = w.x[1]
        if top:
            xi = (interface_y - y) / thickness
        else:
            xi = (y - interface_y) / thickness
        xi = np.clip(xi, 0.0, 1.0)
        sigma = cfg.pml_sigma_max * xi ** cfg.pml_order
        sy = 1.0 + 1j * sigma
        gu = grad(u)
        gv = grad(v)
        gradient_term = sy * gu[0] * gv[0] + 1.0 / sy * gu[1] * gv[1]
        mass_term = cfg.k0 ** 2 * n_medium ** 2 * sy * u * v
        return gradient_term - mass_term
    return asm(pml_form, pml_basis).tocsr()

def assemble_full_pml_operator(basis: Basis, cfg: FullPMLConfig) -> csr_matrix:
    """Assemble both top and bottom PML operators."""
    top_pml = _assemble_one_y_pml(basis=basis, cfg=cfg, tag='top_pml', interface_y=0.0, thickness=cfg.pml_top, n_medium=cfg.n_inc, top=True)
    bottom_pml = _assemble_one_y_pml(basis=basis, cfg=cfg, tag='bottom_pml', interface_y=cfg.total_height, thickness=cfg.pml_bottom, n_medium=cfg.n_out, top=False)
    return (top_pml + bottom_pml).tocsr()

def assemble_scattered_field_source(basis: Basis, cfg: FullPMLConfig, angle_deg: float) -> tuple[np.ndarray, complex, complex]:
    """Volume source for the TE scattered-field formulation.

    Assumes the background above and below the slab is the
    same homogeneous medium.
    """
    if not np.isclose(cfg.n_inc, cfg.n_out):
        raise ValueError('This scattered-field source currently requires n_inc == n_out.')
    kx, ky = incident_wavevector(cfg, angle_deg)
    slab_basis = basis.with_elements(basis.mesh.subdomains['slab'])
    eps_background = cfg.n_inc ** 2
    eps_slab = cfg.n_slab ** 2
    contrast = eps_slab - eps_background

    @LinearForm(dtype=np.complex128)
    def source(v, w):
        x = w.x[0]
        y = w.x[1]
        u_inc = np.exp(1j * (kx * x + ky * y))
        return cfg.k0 ** 2 * contrast * u_inc * v
    b = asm(source, slab_basis)
    return (np.asarray(b, dtype=np.complex128), kx, ky)

def polynomial_sigma(distance: np.ndarray, thickness: float, sigma_max: float, order: int) -> np.ndarray:
    """Polynomial PML attenuation profile.

    sigma = sigma_max * (distance / thickness)**order
    """
    if thickness <= 0.0:
        return np.zeros_like(distance, dtype=float)
    xi = np.clip(distance / thickness, 0.0, 1.0)
    return sigma_max * xi ** order

def stretch_1d(coordinate: np.ndarray, physical_min: float, physical_max: float, pml_low: float, pml_high: float, sigma_max: float, order: int) -> np.ndarray:
    """Return complex coordinate stretch s = 1 + i sigma.

    The physical region is

        physical_min <= coordinate <= physical_max

    with optional PMLs on either side.
    """
    sigma = np.zeros_like(coordinate, dtype=float)
    if pml_low > 0.0:
        mask = coordinate < physical_min
        distance = physical_min - coordinate[mask]
        sigma[mask] = polynomial_sigma(distance, pml_low, sigma_max, order)
    if pml_high > 0.0:
        mask = coordinate > physical_max
        distance = coordinate[mask] - physical_max
        sigma[mask] = polynomial_sigma(distance, pml_high, sigma_max, order)
    return 1.0 + 1j * sigma

def assemble_te_generic_pml(basis: Basis, cfg: FullPMLConfig) -> csr_matrix:
    """Assemble the complete TE Helmholtz + PML operator.

    For complex coordinate stretches

        sx = d(x_tilde)/dx
        sy = d(y_tilde)/dy

    the transformed scalar TE weak form is

        integral [
            (sy/sx) ux vx
            + (sx/sy) uy vy
            - k0^2 eps_r sx sy u v
        ] dOmega.

    Currently:

        sx = 1

    because the lateral boundaries are Bloch-periodic.

    sy contains both the top and bottom PMLs.
    """

    @BilinearForm(dtype=np.complex128)
    def helmholtz_pml(u, v, w):
        x = w.x[0]
        y = w.x[1]
        sx = np.ones_like(x, dtype=np.complex128)
        sy = stretch_1d(coordinate=y, physical_min=0.0, physical_max=cfg.total_height, pml_low=cfg.pml_top, pml_high=cfg.pml_bottom, sigma_max=cfg.pml_sigma_max, order=cfg.pml_order)
        eps_r = np.full(y.shape, cfg.n_inc ** 2, dtype=np.complex128)
        slab_mask = (y >= cfg.slab_y0) & (y <= cfg.slab_y1)
        bottom_mask = y > cfg.slab_y1
        eps_r[slab_mask] = cfg.n_slab ** 2
        eps_r[bottom_mask] = cfg.n_out ** 2
        gu = grad(u)
        gv = grad(v)
        x_term = sy / sx * gu[0] * gv[0]
        y_term = sx / sy * gu[1] * gv[1]
        mass_term = cfg.k0 ** 2 * eps_r * sx * sy * u * v
        return x_term + y_term - mass_term
    A = asm(helmholtz_pml, basis)
    return A.tocsr()

@dataclass
class FullPMLSolveResult:
    """Scattered-field solution using PMLs above and below."""
    scattered: FEMSolution
    reduction: BlochReduction
    kx: complex
    ky_inc: complex
    angle_deg: float
    reduced_dofs: int
    free_reduced_dofs: int

def solve_full_pml_te(cfg: FullPMLConfig, angle_deg: float, h_target: float, order: int=2) -> FullPMLSolveResult:
    """Solve TE scattering with top/bottom PMLs."""
    if order == 1:
        elem = ElementTriP1()
    elif order == 2:
        elem = ElementTriP2()
    else:
        raise ValueError('order must be 1 or 2')
    mesh = make_full_pml_mesh(cfg, h_target=h_target)
    basis = Basis(mesh, elem, intorder=max(4, 2 * order + 2))
    K, M = assemble_full_physical_volume(basis, cfg)
    A_physical = K - cfg.k0 ** 2 * M
    A_pml = assemble_full_pml_operator(basis, cfg)
    b, kx, ky_inc = assemble_scattered_field_source(basis, cfg, angle_deg)
    A_full = (A_physical + A_pml).tocsr()
    reduction = build_bloch_projection(basis=basis, period=cfg.width, kx=float(np.real(kx)))
    P = reduction.P
    PH = P.conjugate().transpose()
    A_red = (PH @ A_full @ P).tocsr()
    b_red = np.asarray(PH @ b, dtype=np.complex128).reshape(-1)
    top_full = np.asarray(basis.get_dofs('top').flatten(), dtype=int)
    bottom_full = np.asarray(basis.get_dofs('bottom').flatten(), dtype=int)
    outer_full = np.unique(np.concatenate([top_full, bottom_full]))
    outer_red = np.unique(reduction.full_to_reduced[outer_full])
    all_red = np.arange(A_red.shape[0], dtype=int)
    free_red = np.setdiff1d(all_red, outer_red)
    z = np.zeros(A_red.shape[0], dtype=np.complex128)
    z[free_red] = spsolve(A_red[free_red][:, free_red].tocsc(), b_red[free_red])
    u_scattered = np.asarray(P @ z, dtype=np.complex128).reshape(-1)
    scattered = FEMSolution(cfg=cfg, order=order, mesh=mesh, basis=basis, u=u_scattered, A=A_full, b=b)
    return FullPMLSolveResult(scattered=scattered, reduction=reduction, kx=kx, ky_inc=ky_inc, angle_deg=angle_deg, reduced_dofs=A_red.shape[0], free_reduced_dofs=free_red.size)

@dataclass
class GenericPMLSolveResult:
    scattered: FEMSolution
    reduction: BlochReduction
    kx: complex
    ky_inc: complex
    angle_deg: float
    reduced_dofs: int
    free_reduced_dofs: int

def solve_generic_pml_te(cfg: FullPMLConfig, angle_deg: float, h_target: float, order: int=2) -> GenericPMLSolveResult:
    """TE scattered-field solve with generic coordinate stretching."""
    if order == 1:
        elem = ElementTriP1()
    elif order == 2:
        elem = ElementTriP2()
    else:
        raise ValueError('order must be 1 or 2')
    mesh = make_full_pml_mesh(cfg, h_target=h_target)
    basis = Basis(mesh, elem, intorder=max(4, 2 * order + 2))
    A_full = assemble_te_generic_pml(basis, cfg)
    b, kx, ky_inc = assemble_scattered_field_source(basis, cfg, angle_deg)
    reduction = build_bloch_projection(basis=basis, period=cfg.width, kx=float(np.real(kx)))
    P = reduction.P
    PH = P.conjugate().transpose()
    A_red = (PH @ A_full @ P).tocsr()
    b_red = np.asarray(PH @ b, dtype=np.complex128).reshape(-1)
    top_full = np.asarray(basis.get_dofs('top').flatten(), dtype=int)
    bottom_full = np.asarray(basis.get_dofs('bottom').flatten(), dtype=int)
    outer_full = np.unique(np.concatenate([top_full, bottom_full]))
    outer_red = np.unique(reduction.full_to_reduced[outer_full])
    all_red = np.arange(A_red.shape[0], dtype=int)
    free_red = np.setdiff1d(all_red, outer_red)
    z = np.zeros(A_red.shape[0], dtype=np.complex128)
    z[free_red] = spsolve(A_red[free_red][:, free_red].tocsc(), b_red[free_red])
    u_scattered = np.asarray(P @ z, dtype=np.complex128).reshape(-1)
    scattered = FEMSolution(cfg=cfg, order=order, mesh=mesh, basis=basis, u=u_scattered, A=A_full, b=b)
    return GenericPMLSolveResult(scattered=scattered, reduction=reduction, kx=kx, ky_inc=ky_inc, angle_deg=angle_deg, reduced_dofs=A_red.shape[0], free_reduced_dofs=free_red.size)

def sample_scattered_centerline(result: FullPMLSolveResult, y: Sequence[float] | np.ndarray) -> np.ndarray:
    """Sample the scattered FEM field at x = width / 2."""
    sol = result.scattered
    cfg = sol.cfg
    y = np.asarray(y, dtype=float)
    x = np.full_like(y, 0.5 * cfg.width)
    pts = np.vstack([x, y])
    return np.asarray(sol.basis.interpolator(sol.u)(pts), dtype=np.complex128)

def sample_total_centerline(result: FullPMLSolveResult, y: Sequence[float] | np.ndarray) -> np.ndarray:
    """Total field in the physical region.

    total = incident + scattered
    """
    sol = result.scattered
    cfg = sol.cfg
    y = np.asarray(y, dtype=float)
    if np.any(y < 0.0) or np.any(y > cfg.total_height):
        raise ValueError('Total-field sampling is defined here only inside the physical region.')
    x = np.full_like(y, 0.5 * cfg.width)
    u_scattered = sample_scattered_centerline(result, y)
    u_incident = np.exp(1j * (result.kx * x + result.ky_inc * y))
    return u_incident + u_scattered

def fit_full_pml_rt(result: FullPMLSolveResult) -> dict[str, complex | float]:
    """Extract TE R/T from the total physical field."""
    sol = result.scattered
    cfg = sol.cfg
    yt = np.linspace(0.12 * cfg.air_top, 0.82 * cfg.air_top, 100)
    yb = np.linspace(cfg.slab_y1 + 0.18 * cfg.air_bottom, cfg.total_height - 0.12 * cfg.air_bottom, 100)
    ut = sample_total_centerline(result, yt)
    ub = sample_total_centerline(result, yb)
    ky_top = outgoing_ky(cfg.k0, cfg.n_inc, result.kx)
    ky_bottom = outgoing_ky(cfg.k0, cfg.n_out, result.kx)
    Xt = np.column_stack([np.exp(1j * ky_top * yt), np.exp(-1j * ky_top * yt)])
    Xb = np.column_stack([np.exp(1j * ky_bottom * yb), np.exp(-1j * ky_bottom * yb)])
    a_inc, a_refl = np.linalg.lstsq(Xt, ut, rcond=None)[0]
    a_trans, a_in_bottom = np.linalg.lstsq(Xb, ub, rcond=None)[0]
    R = float(abs(a_refl / a_inc) ** 2)
    T = float(np.real(ky_bottom) / np.real(ky_top) * abs(a_trans / a_inc) ** 2)
    return {'a_inc': a_inc, 'a_refl': a_refl, 'a_trans': a_trans, 'a_in_bottom': a_in_bottom, 'R': R, 'T': T, 'R_plus_T': R + T, 'bottom_incoming_ratio': float(abs(a_in_bottom / a_inc))}

@dataclass(frozen=True)
class Objective3Config(FullPMLConfig):
    """Fixed geometry for Objective 3.

    Geometry:
        period = width = 1 um
        slab thickness = 1 um
        circular air hole diameter = 0.5 um
    """
    hole_diameter: float = 0.5

    @property
    def hole_radius(self) -> float:
        return 0.5 * self.hole_diameter

    @property
    def hole_center_x(self) -> float:
        return 0.5 * self.width

    @property
    def hole_center_y(self) -> float:
        return 0.5 * (self.slab_y0 + self.slab_y1)

@dataclass
class Objective3SolveResult:
    scattered: FEMSolution
    reduction: BlochReduction
    reduced_dofs: int
    free_reduced_dofs: int

def build_bodyfitted_gmsh_mesh(filename: str | Path, cfg: Objective3Config, h_bulk: float=0.018, h_hole: float=0.008, refine_distance: float=0.2, force: bool=False) -> Path:
    """Generate one fixed, body-fitted Gmsh mesh.

    The circular interface is an actual geometric boundary.
    The left/right boundary meshes are made periodic copies.

    IMPORTANT:
        This mesh is intentionally independent of wavelength.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.exists() and (not force):
        return filename
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError('The body-fitted Objective 3 mesh requires gmsh. Install with: python -m pip install gmsh meshio') from exc
    gmsh.initialize()
    try:
        gmsh.model.add('objective3_bodyfitted')
        geo = gmsh.model.geo
        x0 = 0.0
        x1 = cfg.width
        levels = [-cfg.pml_top, 0.0, cfg.slab_y0, cfg.slab_y1, cfg.total_height, cfg.total_height + cfg.pml_bottom]
        p_left = []
        p_right = []
        for y in levels:
            p_left.append(geo.addPoint(x0, y, 0.0, h_bulk))
            p_right.append(geo.addPoint(x1, y, 0.0, h_bulk))
        horizontal = []
        for i in range(len(levels)):
            horizontal.append(geo.addLine(p_left[i], p_right[i]))
        left_vertical = []
        right_vertical = []
        for i in range(len(levels) - 1):
            left_vertical.append(geo.addLine(p_left[i], p_left[i + 1]))
            right_vertical.append(geo.addLine(p_right[i], p_right[i + 1]))
        xc = cfg.hole_center_x
        yc = cfg.hole_center_y
        r = cfg.hole_radius
        pc = geo.addPoint(xc, yc, 0.0, h_hole)
        pr = geo.addPoint(xc + r, yc, 0.0, h_hole)
        pt = geo.addPoint(xc, yc + r, 0.0, h_hole)
        pl = geo.addPoint(xc - r, yc, 0.0, h_hole)
        pb = geo.addPoint(xc, yc - r, 0.0, h_hole)
        circle_arcs = [geo.addCircleArc(pr, pc, pt), geo.addCircleArc(pt, pc, pl), geo.addCircleArc(pl, pc, pb), geo.addCircleArc(pb, pc, pr)]
        circle_loop = geo.addCurveLoop(circle_arcs)
        surfaces: dict[str, int] = {}

        def outer_loop(i: int) -> int:
            return geo.addCurveLoop([horizontal[i], right_vertical[i], -horizontal[i + 1], -left_vertical[i]])
        surfaces['top_pml'] = geo.addPlaneSurface([outer_loop(0)])
        surfaces['top_air'] = geo.addPlaneSurface([outer_loop(1)])
        slab_outer = outer_loop(2)
        surfaces['slab'] = geo.addPlaneSurface([slab_outer, circle_loop])
        surfaces['hole'] = geo.addPlaneSurface([circle_loop])
        surfaces['bottom_air'] = geo.addPlaneSurface([outer_loop(3)])
        surfaces['bottom_pml'] = geo.addPlaneSurface([outer_loop(4)])
        geo.synchronize()
        for name, tag in surfaces.items():
            ptag = gmsh.model.addPhysicalGroup(2, [tag])
            gmsh.model.setPhysicalName(2, ptag, name)
        for name, curves in (('top', [horizontal[0]]), ('bottom', [horizontal[-1]]), ('left', left_vertical), ('right', right_vertical), ('hole_boundary', circle_arcs)):
            ptag = gmsh.model.addPhysicalGroup(1, curves)
            gmsh.model.setPhysicalName(1, ptag, name)
        affine = [1.0, 0.0, 0.0, cfg.width, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        gmsh.model.mesh.setPeriodic(1, right_vertical, left_vertical, affine)
        distance = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', circle_arcs)
        gmsh.model.mesh.field.setNumber(distance, 'Sampling', 200)
        threshold = gmsh.model.mesh.field.add('Threshold')
        gmsh.model.mesh.field.setNumber(threshold, 'InField', distance)
        gmsh.model.mesh.field.setNumber(threshold, 'SizeMin', h_hole)
        gmsh.model.mesh.field.setNumber(threshold, 'SizeMax', h_bulk)
        gmsh.model.mesh.field.setNumber(threshold, 'DistMin', 0.0)
        gmsh.model.mesh.field.setNumber(threshold, 'DistMax', refine_distance)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber('Mesh.MeshSizeMin', h_hole)
        gmsh.option.setNumber('Mesh.MeshSizeMax', h_bulk)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.Smoothing', 5)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber('Mesh.Binary', 0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(filename))
    finally:
        gmsh.finalize()
    return filename

def load_bodyfitted_mesh(filename: str | Path, cfg: Objective3Config) -> MeshTri:
    """Load the fixed Gmsh mesh and attach robust tags."""
    mesh = MeshTri.load(str(filename))
    scale = max(cfg.width, cfg.total_height + cfg.pml_top + cfg.pml_bottom, 1.0)
    tol = 1e-08 * scale
    mesh = mesh.with_boundaries({'top': lambda X: np.isclose(X[1], -cfg.pml_top, atol=tol), 'bottom': lambda X: np.isclose(X[1], cfg.total_height + cfg.pml_bottom, atol=tol), 'left': lambda X: np.isclose(X[0], 0.0, atol=tol), 'right': lambda X: np.isclose(X[0], cfg.width, atol=tol)})

    def in_hole(X):
        r2 = (X[0] - cfg.hole_center_x) ** 2 + (X[1] - cfg.hole_center_y) ** 2
        return (X[1] > cfg.slab_y0 + tol) & (X[1] < cfg.slab_y1 - tol) & (r2 < cfg.hole_radius ** 2)

    def in_slab(X):
        r2 = (X[0] - cfg.hole_center_x) ** 2 + (X[1] - cfg.hole_center_y) ** 2
        return (X[1] > cfg.slab_y0 + tol) & (X[1] < cfg.slab_y1 - tol) & (r2 > cfg.hole_radius ** 2)
    mesh = mesh.with_subdomains({'top_pml': lambda X: X[1] < -tol, 'top_air': lambda X: (X[1] > -tol) & (X[1] < cfg.slab_y0 - tol), 'slab': in_slab, 'hole': in_hole, 'bottom_air': lambda X: (X[1] > cfg.slab_y1 + tol) & (X[1] < cfg.total_height - tol), 'bottom_pml': lambda X: X[1] > cfg.total_height + tol})
    return mesh

def _assemble_operator(basis: Basis, cfg: Objective3Config) -> csr_matrix:
    """TE Helmholtz operator on the body-fitted geometry."""
    A = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    region_index = {'top_pml': cfg.n_inc, 'top_air': cfg.n_inc, 'slab': cfg.n_slab, 'hole': cfg.n_inc, 'bottom_air': cfg.n_out, 'bottom_pml': cfg.n_out}
    for region, n in region_index.items():
        subbasis = basis.with_elements(basis.mesh.subdomains[region])
        eps_r = n ** 2

        @BilinearForm(dtype=np.complex128)
        def form(u, v, w):
            y = w.x[1]
            sy = stretch_1d(coordinate=y, physical_min=0.0, physical_max=cfg.total_height, pml_low=cfg.pml_top, pml_high=cfg.pml_bottom, sigma_max=cfg.pml_sigma_max, order=cfg.pml_order)
            gu = grad(u)
            gv = grad(v)
            return sy * gu[0] * gv[0] + 1.0 / sy * gu[1] * gv[1] - cfg.k0 ** 2 * eps_r * sy * u * v
        A = A + asm(form, subbasis)
    return A.tocsr()

def _assemble_source(basis: Basis, cfg: Objective3Config) -> np.ndarray:
    """Scattered-field source.

    Only the dielectric material is different from the
    air background.  The circular hole is background air.
    """
    slab_basis = basis.with_elements(basis.mesh.subdomains['slab'])
    contrast = cfg.n_slab ** 2 - cfg.n_inc ** 2

    @LinearForm(dtype=np.complex128)
    def source(v, w):
        y = w.x[1]
        u_inc = np.exp(1j * cfg.k0 * cfg.n_inc * y)
        return cfg.k0 ** 2 * contrast * u_inc * v
    return np.asarray(asm(source, slab_basis), dtype=np.complex128)

def solve_objective3_bodyfitted(cfg: Objective3Config, mesh_file: str | Path) -> Objective3SolveResult:
    """Solve Objective 3 on the fixed body-fitted mesh."""
    mesh = load_bodyfitted_mesh(mesh_file, cfg)
    basis = Basis(mesh, ElementTriP2(), intorder=8)
    A_full = _assemble_operator(basis, cfg)
    b = _assemble_source(basis, cfg)
    reduction = build_bloch_projection(basis=basis, period=cfg.width, kx=0.0)
    P = reduction.P
    PH = P.conjugate().transpose()
    A_red = (PH @ A_full @ P).tocsr()
    b_red = np.asarray(PH @ b, dtype=np.complex128).reshape(-1)
    top_full = np.asarray(basis.get_dofs('top').flatten(), dtype=int)
    bottom_full = np.asarray(basis.get_dofs('bottom').flatten(), dtype=int)
    outer_full = np.unique(np.concatenate([top_full, bottom_full]))
    outer_red = np.unique(reduction.full_to_reduced[outer_full])
    all_red = np.arange(A_red.shape[0], dtype=int)
    free_red = np.setdiff1d(all_red, outer_red)
    z = np.zeros(A_red.shape[0], dtype=np.complex128)
    z[free_red] = spsolve(A_red[free_red][:, free_red].tocsc(), b_red[free_red])
    u_scattered = np.asarray(P @ z, dtype=np.complex128).reshape(-1)
    solution = FEMSolution(cfg=cfg, order=2, mesh=mesh, basis=basis, u=u_scattered, A=A_full, b=b)
    return Objective3SolveResult(scattered=solution, reduction=reduction, reduced_dofs=A_red.shape[0], free_reduced_dofs=free_red.size)

def sample_horizontal_line_objective3(sol: FEMSolution, y: float, width: float, npoints: int=512, chunk: int=32) -> tuple[np.ndarray, np.ndarray]:
    """Memory-safe midpoint sampling over one period."""
    dx = width / npoints
    x = (np.arange(npoints, dtype=float) + 0.5) * dx
    values = np.empty(npoints, dtype=np.complex128)
    interpolator = sol.basis.interpolator(sol.u)
    for start in range(0, npoints, chunk):
        stop = min(start + chunk, npoints)
        xx = x[start:stop]
        yy = np.full_like(xx, y)
        values[start:stop] = np.asarray(interpolator(np.vstack([xx, yy])), dtype=np.complex128)
    return (x, values)

def diffraction_orders_bodyfitted(result: Objective3SolveResult, cfg: Objective3Config, npoints: int=512, cutoff_rel_tol: float=1e-09) -> dict:
    """Rayleigh/Fourier diffraction-order decomposition.

    Normal incidence, air above and below.
    """
    sol = result.scattered
    if not np.isclose(cfg.n_inc, cfg.n_out):
        raise ValueError('Current Objective 3 extractor assumes n_inc == n_out.')
    period = cfg.width
    k_air = float(np.real(cfg.k0 * cfg.n_inc))
    ky_inc = k_air
    y_top = 0.5 * cfg.air_top
    y_bottom = cfg.slab_y1 + 0.5 * cfg.air_bottom
    x, us_top = sample_horizontal_line_objective3(sol, y_top, period, npoints=npoints)
    _, us_bottom = sample_horizontal_line_objective3(sol, y_bottom, period, npoints=npoints)
    u_inc_bottom = np.exp(1j * cfg.k0 * cfg.n_inc * y_bottom)
    u_total_bottom = us_bottom + u_inc_bottom
    m_prop_est = int(np.floor(k_air * period / (2.0 * np.pi)))
    m_values = np.arange(-m_prop_est - 3, m_prop_est + 4, dtype=int)
    rows = []
    cutoff_orders = []
    scale = k_air ** 2
    for m in m_values:
        kx_m = 2.0 * np.pi * m / period
        ky2 = k_air ** 2 - kx_m ** 2
        near_cutoff = abs(ky2) <= cutoff_rel_tol * scale
        propagating = ky2 > cutoff_rel_tol * scale
        if near_cutoff:
            cutoff_orders.append(int(m))
        if propagating:
            ky_m = float(np.sqrt(ky2))
        else:
            ky_m = complex(np.sqrt(ky2 + 0j))
        phase = np.exp(-1j * kx_m * x)
        r_m = np.mean(us_top * phase)
        t_m = np.mean(u_total_bottom * phase)
        if propagating:
            factor = ky_m / ky_inc
            R_m = float(factor * abs(r_m) ** 2)
            T_m = float(factor * abs(t_m) ** 2)
        else:
            R_m = 0.0
            T_m = 0.0
        rows.append({'m': int(m), 'kx': kx_m, 'ky': ky_m, 'propagating': bool(propagating), 'near_cutoff': bool(near_cutoff), 'r': r_m, 't': t_m, 'R': R_m, 'T': T_m})
    R_total = float(sum((row['R'] for row in rows)))
    T_total = float(sum((row['T'] for row in rows)))
    lookup = {row['m']: row for row in rows}
    symmetry_abs = []
    for m in range(1, max((abs(int(x)) for x in m_values)) + 1):
        if m in lookup and -m in lookup and lookup[m]['propagating'] and lookup[-m]['propagating']:
            symmetry_abs.append(abs(lookup[m]['R'] - lookup[-m]['R']))
            symmetry_abs.append(abs(lookup[m]['T'] - lookup[-m]['T']))
    symmetry_max = max(symmetry_abs) if symmetry_abs else 0.0
    return {'orders': rows, 'R': R_total, 'T': T_total, 'R_plus_T': R_total + T_total, 'energy_error': abs(R_total + T_total - 1.0), 'symmetry_abs_max': float(symmetry_max), 'cutoff_orders': cutoff_orders, 'y_top': y_top, 'y_bottom': y_bottom}

@dataclass(frozen=True)
class DisorderedConfig:
    wavelength: float
    width: float = 7.0
    xmin: float = -3.5
    xmax: float = 3.5
    scatter_ymin: float = -3.5
    scatter_ymax: float = 3.5
    physical_ymin: float = -4.0
    physical_ymax: float = 4.0
    pml_low: float = 1.0
    pml_high: float = 1.0
    pml_order: int = 3
    pml_sigma_max: float = 8.0
    n_background: complex = 1.0 + 0j
    n_disk: complex = 3.0 + 0j

    @property
    def k0(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @property
    def ymin(self) -> float:
        return self.physical_ymin - self.pml_low

    @property
    def ymax(self) -> float:
        return self.physical_ymax + self.pml_high

    @property
    def top_monitor_y(self) -> float:
        return 0.5 * (self.scatter_ymax + self.physical_ymax)

    @property
    def bottom_monitor_y(self) -> float:
        return 0.5 * (self.scatter_ymin + self.physical_ymin)

@dataclass
class SolveTimings:
    mesh_and_basis: float
    periodic_setup: float
    matrix_assembly: float
    source_assembly: float
    linear_solve: float
    reconstruction: float
    total: float

@dataclass
class DisorderedSolveResult:
    cfg: DisorderedConfig
    mesh: MeshTri
    basis: Basis
    u_scattered: np.ndarray
    full_dofs: int
    reduced_dofs: int
    free_reduced_dofs: int
    timings: SolveTimings

def _require_named_groups(mesh: MeshTri) -> None:
    required_subdomains = {'bottom_pml', 'bottom_air', 'scatter_background', 'top_air', 'top_pml', 'disks'}
    required_boundaries = {'left', 'right', 'top', 'bottom'}
    actual_subdomains = set(mesh.subdomains.keys() if mesh.subdomains is not None else [])
    actual_boundaries = set(mesh.boundaries.keys() if mesh.boundaries is not None else [])
    missing_subdomains = required_subdomains - actual_subdomains
    missing_boundaries = required_boundaries - actual_boundaries
    if missing_subdomains or missing_boundaries:
        raise RuntimeError(f'Mesh physical groups were not loaded as expected.\nAvailable subdomains: {sorted(actual_subdomains)}\nAvailable boundaries: {sorted(actual_boundaries)}\nMissing subdomains: {sorted(missing_subdomains)}\nMissing boundaries: {sorted(missing_boundaries)}')

def load_disordered_mesh(mesh_file: str | Path) -> MeshTri:
    mesh = MeshTri.load(str(mesh_file))
    _require_named_groups(mesh)
    return mesh

def build_normal_incidence_periodic_projection(basis: Basis, *, atol: float=2e-07):
    """
    Identify right-boundary P2 DOFs with the matching left-boundary DOFs.

    Normal incidence => Bloch phase = 1.

    This performs periodicity only.  No mirror/symmetry reduction is used.
    """
    left = np.asarray(basis.get_dofs('left').flatten(), dtype=int)
    right = np.asarray(basis.get_dofs('right').flatten(), dtype=int)
    if left.size != right.size:
        raise RuntimeError(f'Periodic DOF count mismatch: left={left.size}, right={right.size}')
    doflocs = basis.doflocs
    left_order = np.argsort(doflocs[1, left], kind='stable')
    right_order = np.argsort(doflocs[1, right], kind='stable')
    left = left[left_order]
    right = right[right_order]
    y_left = doflocs[1, left]
    y_right = doflocs[1, right]
    max_y_mismatch = float(np.max(np.abs(y_left - y_right)))
    if max_y_mismatch > atol:
        raise RuntimeError(f'Periodic left/right P2 DOFs do not match in y. max mismatch = {max_y_mismatch:.3e} um')
    N = basis.N
    is_slave = np.zeros(N, dtype=bool)
    is_slave[right] = True
    masters_and_interior = np.flatnonzero(~is_slave)
    full_to_reduced = np.full(N, -1, dtype=int)
    full_to_reduced[masters_and_interior] = np.arange(masters_and_interior.size, dtype=int)
    full_to_reduced[right] = full_to_reduced[left]
    if np.any(full_to_reduced < 0):
        raise RuntimeError('Periodic projection construction left unmapped DOFs.')
    rows = np.arange(N, dtype=int)
    cols = full_to_reduced
    data = np.ones(N, dtype=np.complex128)
    P = coo_matrix((data, (rows, cols)), shape=(N, masters_and_interior.size)).tocsr()
    return (P, full_to_reduced, left, right, max_y_mismatch)

def _stretch_y(y, cfg: DisorderedConfig):
    return stretch_1d(coordinate=y, physical_min=cfg.physical_ymin, physical_max=cfg.physical_ymax, pml_low=cfg.pml_low, pml_high=cfg.pml_high, sigma_max=cfg.pml_sigma_max, order=cfg.pml_order)

def assemble_static_matrices(basis: Basis, cfg: DisorderedConfig):
    """
    TE/Ez operator:

        A(k0) = K - k0^2 M

    with y-directed PML stretch sy.
    """

    @BilinearForm(dtype=np.complex128)
    def stiffness(u, v, w):
        sy = _stretch_y(w.x[1], cfg)
        gu = grad(u)
        gv = grad(v)
        return sy * gu[0] * gv[0] + 1.0 / sy * gu[1] * gv[1]
    K = asm(stiffness, basis).tocsr()
    M = csr_matrix((basis.N, basis.N), dtype=np.complex128)
    region_index = {'bottom_pml': cfg.n_background, 'bottom_air': cfg.n_background, 'scatter_background': cfg.n_background, 'top_air': cfg.n_background, 'top_pml': cfg.n_background, 'disks': cfg.n_disk}
    for region, n in region_index.items():
        subbasis = basis.with_elements(basis.mesh.subdomains[region])
        eps_r = n ** 2

        @BilinearForm(dtype=np.complex128)
        def mass(u, v, w):
            sy = _stretch_y(w.x[1], cfg)
            return eps_r * sy * u * v
        M = M + asm(mass, subbasis)
    return (K.tocsr(), M.tocsr())

def assemble_scattered_source(basis: Basis, cfg: DisorderedConfig) -> np.ndarray:
    """
    Scattered-field source for a downward plane wave.

    Time convention: exp(-i omega t)
    Downward (-y) incident field: exp(-i k y)

    Only the n=3 disks differ from the n=1 background.
    """
    disk_basis = basis.with_elements(basis.mesh.subdomains['disks'])
    contrast = cfg.n_disk ** 2 - cfg.n_background ** 2

    @LinearForm(dtype=np.complex128)
    def source(v, w):
        y = w.x[1]
        u_inc = np.exp(-1j * cfg.k0 * cfg.n_background * y)
        return cfg.k0 ** 2 * contrast * u_inc * v
    return np.asarray(asm(source, disk_basis), dtype=np.complex128)

def solve_disordered(cfg: DisorderedConfig, mesh_file: str | Path, *, intorder: int=8) -> DisorderedSolveResult:
    total_start = perf_counter()
    t0 = perf_counter()
    mesh = load_disordered_mesh(mesh_file)
    basis = Basis(mesh, ElementTriP2(), intorder=intorder)
    t1 = perf_counter()
    P, full_to_reduced, left_dofs, right_dofs, periodic_y_mismatch = build_normal_incidence_periodic_projection(basis)
    PH = P.conjugate().transpose().tocsr()
    top_full = np.asarray(basis.get_dofs('top').flatten(), dtype=int)
    bottom_full = np.asarray(basis.get_dofs('bottom').flatten(), dtype=int)
    outer_full = np.unique(np.concatenate([top_full, bottom_full]))
    outer_red = np.unique(full_to_reduced[outer_full])
    reduced_dofs = P.shape[1]
    all_red = np.arange(reduced_dofs, dtype=int)
    free_red = np.setdiff1d(all_red, outer_red)
    t2 = perf_counter()
    K_full, M_full = assemble_static_matrices(basis, cfg)
    K_red = (PH @ K_full @ P).tocsr()
    M_red = (PH @ M_full @ P).tocsr()
    K_free = K_red[free_red][:, free_red].tocsc()
    M_free = M_red[free_red][:, free_red].tocsc()
    t3 = perf_counter()
    b_full = assemble_scattered_source(basis, cfg)
    b_red = np.asarray(PH @ b_full, dtype=np.complex128).reshape(-1)
    b_free = b_red[free_red]
    t4 = perf_counter()
    A_free = (K_free - cfg.k0 ** 2 * M_free).tocsc()
    z_free = spsolve(A_free, b_free)
    t5 = perf_counter()
    z = np.zeros(reduced_dofs, dtype=np.complex128)
    z[free_red] = z_free
    u_scattered = np.asarray(P @ z, dtype=np.complex128).reshape(-1)
    t6 = perf_counter()
    timings = SolveTimings(mesh_and_basis=t1 - t0, periodic_setup=t2 - t1, matrix_assembly=t3 - t2, source_assembly=t4 - t3, linear_solve=t5 - t4, reconstruction=t6 - t5, total=t6 - total_start)
    timings.periodic_y_mismatch = periodic_y_mismatch
    timings.left_periodic_dofs = int(left_dofs.size)
    timings.right_periodic_dofs = int(right_dofs.size)
    return DisorderedSolveResult(cfg=cfg, mesh=mesh, basis=basis, u_scattered=u_scattered, full_dofs=basis.N, reduced_dofs=reduced_dofs, free_reduced_dofs=free_red.size, timings=timings)

def sample_horizontal_line_disordered(result: DisorderedSolveResult, y: float, *, npoints: int=1024, chunk: int=32):
    """
    Midpoint samples across x = [-3.5, +3.5].
    """
    cfg = result.cfg
    dx = cfg.width / npoints
    x = cfg.xmin + (np.arange(npoints, dtype=float) + 0.5) * dx
    values = np.empty(npoints, dtype=np.complex128)
    interpolator = result.basis.interpolator(result.u_scattered)
    for start in range(0, npoints, chunk):
        stop = min(start + chunk, npoints)
        xx = x[start:stop]
        yy = np.full_like(xx, y)
        values[start:stop] = np.asarray(interpolator(np.vstack([xx, yy])), dtype=np.complex128)
    return (x, values)

def diffraction_orders(result: DisorderedSolveResult, *, npoints: int=1024, cutoff_rel_tol: float=1e-09) -> dict:
    cfg = result.cfg
    if not np.isclose(cfg.n_background.imag, 0.0):
        raise ValueError('Current R/T extractor assumes a lossless background.')
    k_air = float(np.real(cfg.k0 * cfg.n_background))
    ky_inc = k_air
    x, us_top = sample_horizontal_line_disordered(result, cfg.top_monitor_y, npoints=npoints)
    _, us_bottom = sample_horizontal_line_disordered(result, cfg.bottom_monitor_y, npoints=npoints)
    u_inc_bottom = np.exp(-1j * cfg.k0 * cfg.n_background * cfg.bottom_monitor_y)
    u_total_bottom = us_bottom + u_inc_bottom
    m_prop_est = int(np.floor(k_air * cfg.width / (2.0 * np.pi)))
    m_values = np.arange(-m_prop_est - 3, m_prop_est + 4, dtype=int)
    scale = k_air ** 2
    rows = []
    cutoff_orders = []
    for m in m_values:
        kx_m = 2.0 * np.pi * m / cfg.width
        ky2 = k_air ** 2 - kx_m ** 2
        near_cutoff = abs(ky2) <= cutoff_rel_tol * scale
        propagating = ky2 > cutoff_rel_tol * scale
        if near_cutoff:
            cutoff_orders.append(int(m))
        if propagating:
            ky_m = float(np.sqrt(ky2))
        else:
            ky_m = complex(np.sqrt(ky2 + 0j))
        phase = np.exp(-1j * kx_m * x)
        r_m = np.mean(us_top * phase)
        t_m = np.mean(u_total_bottom * phase)
        if propagating:
            factor = ky_m / ky_inc
            R_m = float(factor * abs(r_m) ** 2)
            T_m = float(factor * abs(t_m) ** 2)
        else:
            R_m = 0.0
            T_m = 0.0
        rows.append({'m': int(m), 'kx': kx_m, 'ky': ky_m, 'propagating': bool(propagating), 'near_cutoff': bool(near_cutoff), 'r': r_m, 't': t_m, 'R': R_m, 'T': T_m})
    R_total = float(sum((row['R'] for row in rows)))
    T_total = float(sum((row['T'] for row in rows)))
    return {'orders': rows, 'R': R_total, 'T': T_total, 'R_plus_T': R_total + T_total, 'energy_error': abs(R_total + T_total - 1.0), 'cutoff_orders': cutoff_orders, 'top_monitor_y': cfg.top_monitor_y, 'bottom_monitor_y': cfg.bottom_monitor_y}

@dataclass
class CachedDisorderedTimings:
    source_assembly: float
    matrix_build: float
    linear_solve: float
    reconstruction: float
    total: float

class CachedDisorderedSolver:
    """
    Cached Ez/TE solver for the fixed 196-disk disordered geometry.

    Cached once per process:
      - mesh
      - P2 basis
      - periodic x projection
      - top/bottom constrained DOFs
      - static K and M matrices
      - reduced/free K and M blocks

    Repeated per wavelength:
      - scattered-field source
      - A = K - k0^2 M
      - sparse solve
      - field reconstruction
    """

    def __init__(self, cfg_template: DisorderedConfig, mesh_file: str | Path, *, intorder: int=8):
        self.cfg_template = cfg_template
        self.mesh_file = Path(mesh_file)
        self.intorder = int(intorder)
        setup_start = perf_counter()
        self.mesh = load_disordered_mesh(self.mesh_file)
        self.basis = Basis(self.mesh, ElementTriP2(), intorder=self.intorder)
        self.P, self.full_to_reduced, self.left_dofs, self.right_dofs, self.periodic_y_mismatch = build_normal_incidence_periodic_projection(self.basis)
        self.P = self.P.tocsr()
        self.PH = self.P.conjugate().transpose().tocsr()
        top_full = np.asarray(self.basis.get_dofs('top').flatten(), dtype=int)
        bottom_full = np.asarray(self.basis.get_dofs('bottom').flatten(), dtype=int)
        outer_full = np.unique(np.concatenate([top_full, bottom_full]))
        self.outer_red = np.unique(self.full_to_reduced[outer_full])
        self.reduced_dofs = self.P.shape[1]
        all_red = np.arange(self.reduced_dofs, dtype=int)
        self.free_red = np.setdiff1d(all_red, self.outer_red)
        self.free_reduced_dofs = self.free_red.size
        matrix_start = perf_counter()
        self.K_full, self.M_full = assemble_static_matrices(self.basis, cfg_template)
        self.K_red = (self.PH @ self.K_full @ self.P).tocsr()
        self.M_red = (self.PH @ self.M_full @ self.P).tocsr()
        self.K_free = self.K_red[self.free_red][:, self.free_red].tocsc()
        self.M_free = self.M_red[self.free_red][:, self.free_red].tocsc()
        self.static_matrix_setup_time = perf_counter() - matrix_start
        self.setup_time = perf_counter() - setup_start
        self.last_timings = None

    def _check_cfg(self, cfg: DisorderedConfig) -> None:
        fixed_attrs = ['width', 'xmin', 'xmax', 'scatter_ymin', 'scatter_ymax', 'physical_ymin', 'physical_ymax', 'pml_low', 'pml_high', 'pml_sigma_max', 'n_background', 'n_disk']
        for name in fixed_attrs:
            a = complex(getattr(self.cfg_template, name))
            b = complex(getattr(cfg, name))
            if not np.isclose(a, b, rtol=0.0, atol=1e-14):
                raise ValueError(f'Cached solver cannot change {name}: cached={a}, requested={b}.')
        if int(self.cfg_template.pml_order) != int(cfg.pml_order):
            raise ValueError('Cached solver cannot change pml_order.')

    def solve(self, cfg: DisorderedConfig) -> DisorderedSolveResult:
        self._check_cfg(cfg)
        total_start = perf_counter()
        t0 = perf_counter()
        b_full = assemble_scattered_source(self.basis, cfg)
        b_red = np.asarray(self.PH @ b_full, dtype=np.complex128).reshape(-1)
        b_free = b_red[self.free_red]
        t1 = perf_counter()
        A_free = (self.K_free - cfg.k0 ** 2 * self.M_free).tocsc()
        t2 = perf_counter()
        z_free = spsolve(A_free, b_free)
        t3 = perf_counter()
        z = np.zeros(self.reduced_dofs, dtype=np.complex128)
        z[self.free_red] = z_free
        u_scattered = np.asarray(self.P @ z, dtype=np.complex128).reshape(-1)
        t4 = perf_counter()
        self.last_timings = CachedDisorderedTimings(source_assembly=t1 - t0, matrix_build=t2 - t1, linear_solve=t3 - t2, reconstruction=t4 - t3, total=t4 - total_start)
        timings = SolveTimings(mesh_and_basis=0.0, periodic_setup=0.0, matrix_assembly=0.0, source_assembly=t1 - t0, linear_solve=t3 - t2, reconstruction=t4 - t3, total=t4 - total_start)
        timings.periodic_y_mismatch = self.periodic_y_mismatch
        timings.left_periodic_dofs = int(self.left_dofs.size)
        timings.right_periodic_dofs = int(self.right_dofs.size)
        return DisorderedSolveResult(cfg=cfg, mesh=self.mesh, basis=self.basis, u_scattered=u_scattered, full_dofs=self.basis.N, reduced_dofs=self.reduced_dofs, free_reduced_dofs=self.free_reduced_dofs, timings=timings)
