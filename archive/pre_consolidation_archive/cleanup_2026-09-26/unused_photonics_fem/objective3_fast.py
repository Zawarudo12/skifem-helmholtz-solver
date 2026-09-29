from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from skfem import Basis, BilinearForm, ElementTriP2, LinearForm, asm
from skfem.helpers import grad

from .bloch import build_bloch_projection
from .objective3_bodyfitted import (
    Objective3Config,
    Objective3SolveResult,
    load_bodyfitted_mesh,
)
from .pml_generic import stretch_1d
from .solver import FEMSolution


@dataclass
class FastSolveTimings:
    source_assembly: float
    matrix_build: float
    linear_solve: float
    reconstruction: float
    total: float


class Objective3FastSolver:
    """
    Cached solver for the validated Objective 3 body-fitted TE problem.

    What is cached once
    -------------------
    - Mesh loading / geometric tags
    - P2 Basis
    - Bloch projection
    - Top/bottom constrained-DOF mapping
    - Wavelength-independent PML stiffness matrix K
    - Wavelength-independent material mass matrix M
    - Reduced free-DOF matrices

    For every wavelength we only do
    --------------------------------
    - Assemble the wavelength-dependent scattered-field source
    - Form A(k0) = K - k0^2 M
    - Solve the reduced sparse complex linear system
    - Reconstruct the full scattered field

    This keeps the same physics as objective3_bodyfitted.py.
    """

    def __init__(
        self,
        cfg_template: Objective3Config,
        mesh_file: str | Path,
        *,
        intorder: int = 8,
    ):
        self.cfg_template = cfg_template
        self.mesh_file = Path(mesh_file)
        self.intorder = int(intorder)

        setup_start = perf_counter()

        # ------------------------------------------------------------
        # 1. Load mesh ONCE.
        # ------------------------------------------------------------
        self.mesh = load_bodyfitted_mesh(
            self.mesh_file,
            cfg_template,
        )

        # ------------------------------------------------------------
        # 2. Build P2 FEM basis ONCE.
        # ------------------------------------------------------------
        self.basis = Basis(
            self.mesh,
            ElementTriP2(),
            intorder=self.intorder,
        )

        # Slab basis is also fixed.
        self.slab_basis = self.basis.with_elements(
            self.mesh.subdomains["slab"]
        )

        # ------------------------------------------------------------
        # 3. Build normal-incidence Bloch map ONCE.
        # ------------------------------------------------------------
        self.reduction = build_bloch_projection(
            basis=self.basis,
            period=cfg_template.width,
            kx=0.0,
        )

        self.P = self.reduction.P.tocsr()

        self.PH = (
            self.P
            .conjugate()
            .transpose()
            .tocsr()
        )

        # ------------------------------------------------------------
        # 4. Determine fixed free reduced DOFs ONCE.
        # ------------------------------------------------------------
        top_full = np.asarray(
            self.basis
            .get_dofs("top")
            .flatten(),
            dtype=int,
        )

        bottom_full = np.asarray(
            self.basis
            .get_dofs("bottom")
            .flatten(),
            dtype=int,
        )

        outer_full = np.unique(
            np.concatenate(
                [
                    top_full,
                    bottom_full,
                ]
            )
        )

        self.outer_red = np.unique(
            self.reduction.full_to_reduced[
                outer_full
            ]
        )

        self.reduced_dofs = self.P.shape[1]

        all_red = np.arange(
            self.reduced_dofs,
            dtype=int,
        )

        self.free_red = np.setdiff1d(
            all_red,
            self.outer_red,
        )

        self.free_reduced_dofs = self.free_red.size

        # ------------------------------------------------------------
        # 5. Preassemble exact static operator pieces.
        #
        # Original operator:
        #
        # A = K - k0^2 M
        #
        # PML stretch sy is independent of wavelength in the current
        # implementation, therefore K and M are both static.
        # ------------------------------------------------------------
        matrix_start = perf_counter()

        self.K_full = self._assemble_stiffness_matrix(
            cfg_template
        )

        self.M_full = self._assemble_mass_matrix(
            cfg_template
        )

        # Reduce both pieces only once.
        self.K_red = (
            self.PH
            @ self.K_full
            @ self.P
        ).tocsr()

        self.M_red = (
            self.PH
            @ self.M_full
            @ self.P
        ).tocsr()

        # Only the unconstrained block is ever solved.
        self.K_free = self.K_red[
            self.free_red
        ][:, self.free_red].tocsc()

        self.M_free = self.M_red[
            self.free_red
        ][:, self.free_red].tocsc()

        self.static_matrix_setup_time = (
            perf_counter()
            - matrix_start
        )

        self.setup_time = (
            perf_counter()
            - setup_start
        )

        self.last_timings: FastSolveTimings | None = None

    # -----------------------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------------------

    def _check_cfg(
        self,
        cfg: Objective3Config,
    ) -> None:
        """
        The cache is valid only while geometry, materials and PML
        settings remain unchanged.  Wavelength/k0 may change.
        """

        fixed_float_attrs = [
            "width",
            "total_height",
            "slab_y0",
            "slab_y1",
            "pml_top",
            "pml_bottom",
            "pml_sigma_max",
            "n_inc",
            "n_slab",
            "n_out",
            "hole_diameter",
        ]

        for name in fixed_float_attrs:
            # Some optical/PML parameters may legitimately be complex
            # (for example a complex refractive index).  Compare scalars
            # without forcing them through float().
            a = complex(
                getattr(
                    self.cfg_template,
                    name,
                )
            )

            b = complex(
                getattr(
                    cfg,
                    name,
                )
            )

            if not np.isclose(
                a,
                b,
                rtol=0.0,
                atol=1e-14,
            ):
                raise ValueError(
                    f"Cached Objective3FastSolver cannot change "
                    f"{name}: cached={a}, requested={b}. "
                    f"Create a new solver cache for changed geometry, "
                    f"materials or PML settings."
                )

        if int(self.cfg_template.pml_order) != int(cfg.pml_order):
            raise ValueError(
                "Cached Objective3FastSolver cannot change pml_order. "
                "Create a new solver cache."
            )

    # -----------------------------------------------------------------
    # PML helper
    # -----------------------------------------------------------------

    @staticmethod
    def _sy(
        y,
        cfg: Objective3Config,
    ):
        return stretch_1d(
            coordinate=y,
            physical_min=0.0,
            physical_max=cfg.total_height,
            pml_low=cfg.pml_top,
            pml_high=cfg.pml_bottom,
            sigma_max=cfg.pml_sigma_max,
            order=cfg.pml_order,
        )

    # -----------------------------------------------------------------
    # Static matrix assembly
    # -----------------------------------------------------------------

    def _assemble_stiffness_matrix(
        self,
        cfg: Objective3Config,
    ) -> csr_matrix:
        """
        Assemble wavelength-independent PML stiffness:

            K = int [
                  sy * ux * vx
                + (1/sy) * uy * vy
            ] dA
        """

        @BilinearForm(
            dtype=np.complex128
        )
        def stiffness(
            u,
            v,
            w,
        ):
            sy = self._sy(
                w.x[1],
                cfg,
            )

            gu = grad(u)
            gv = grad(v)

            return (
                sy
                * gu[0]
                * gv[0]

                +

                (1.0 / sy)
                * gu[1]
                * gv[1]
            )

        return asm(
            stiffness,
            self.basis,
        ).tocsr()

    def _assemble_mass_matrix(
        self,
        cfg: Objective3Config,
    ) -> csr_matrix:
        """
        Assemble wavelength-independent material/PML mass:

            M = int eps_r * sy * u * v dA

        so that

            A(k0) = K - k0^2 M
        """

        M = csr_matrix(
            (
                self.basis.N,
                self.basis.N,
            ),
            dtype=np.complex128,
        )

        region_index = {
            "top_pml": cfg.n_inc,
            "top_air": cfg.n_inc,
            "slab": cfg.n_slab,
            "hole": cfg.n_inc,
            "bottom_air": cfg.n_out,
            "bottom_pml": cfg.n_out,
        }

        for region, n in region_index.items():
            subbasis = self.basis.with_elements(
                self.mesh.subdomains[
                    region
                ]
            )

            eps_r = n ** 2

            @BilinearForm(
                dtype=np.complex128
            )
            def mass(
                u,
                v,
                w,
            ):
                sy = self._sy(
                    w.x[1],
                    cfg,
                )

                return (
                    eps_r
                    * sy
                    * u
                    * v
                )

            M = (
                M
                + asm(
                    mass,
                    subbasis,
                )
            )

        return M.tocsr()

    # -----------------------------------------------------------------
    # Wavelength-dependent source
    # -----------------------------------------------------------------

    def _assemble_source(
        self,
        cfg: Objective3Config,
    ) -> np.ndarray:
        """
        Same scattered-field source as the reference solver.

        Only the dielectric slab contributes because the air hole is
        the same material as the incident background.
        """

        contrast = (
            cfg.n_slab ** 2
            - cfg.n_inc ** 2
        )

        @LinearForm(
            dtype=np.complex128
        )
        def source(
            v,
            w,
        ):
            y = w.x[1]

            u_inc = np.exp(
                1j
                * cfg.k0
                * cfg.n_inc
                * y
            )

            return (
                cfg.k0 ** 2
                * contrast
                * u_inc
                * v
            )

        return np.asarray(
            asm(
                source,
                self.slab_basis,
            ),
            dtype=np.complex128,
        )

    # -----------------------------------------------------------------
    # Solve one wavelength
    # -----------------------------------------------------------------

    def solve(
        self,
        cfg: Objective3Config,
    ) -> Objective3SolveResult:
        self._check_cfg(
            cfg
        )

        total_start = perf_counter()

        # -------------------------------------------------------------
        # Source
        # -------------------------------------------------------------
        t0 = perf_counter()

        b_full = self._assemble_source(
            cfg
        )

        b_red = np.asarray(
            self.PH @ b_full,
            dtype=np.complex128,
        ).reshape(-1)

        b_free = b_red[
            self.free_red
        ]

        t1 = perf_counter()

        # -------------------------------------------------------------
        # A(k0) = K - k0^2 M
        # Sparse linear combination only; no FEM reassembly.
        # -------------------------------------------------------------
        A_free = (
            self.K_free
            - cfg.k0 ** 2
            * self.M_free
        ).tocsc()

        t2 = perf_counter()

        # -------------------------------------------------------------
        # Sparse complex solve.
        # This is still expected to be the main bottleneck.
        # -------------------------------------------------------------
        z_free = spsolve(
            A_free,
            b_free,
        )

        t3 = perf_counter()

        # -------------------------------------------------------------
        # Reconstruct reduced and then full scattered field.
        # -------------------------------------------------------------
        z = np.zeros(
            self.reduced_dofs,
            dtype=np.complex128,
        )

        z[
            self.free_red
        ] = z_free

        u_scattered = np.asarray(
            self.P @ z,
            dtype=np.complex128,
        ).reshape(-1)

        # Preserve the same FEMSolution interface as the trusted solver.
        A_full = (
            self.K_full
            - cfg.k0 ** 2
            * self.M_full
        ).tocsr()

        solution = FEMSolution(
            cfg=cfg,
            order=2,
            mesh=self.mesh,
            basis=self.basis,
            u=u_scattered,
            A=A_full,
            b=b_full,
        )

        t4 = perf_counter()

        self.last_timings = FastSolveTimings(
            source_assembly=t1 - t0,
            matrix_build=t2 - t1,
            linear_solve=t3 - t2,
            reconstruction=t4 - t3,
            total=t4 - total_start,
        )

        return Objective3SolveResult(
            scattered=solution,
            reduction=self.reduction,
            reduced_dofs=self.reduced_dofs,
            free_reduced_dofs=self.free_reduced_dofs,
        )

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    def timing_report(
        self,
    ) -> str:
        if self.last_timings is None:
            return (
                "No wavelength has been solved yet."
            )

        t = self.last_timings

        return (
            f"source assembly : {t.source_assembly:.6f} s\n"
            f"matrix build    : {t.matrix_build:.6f} s\n"
            f"linear solve    : {t.linear_solve:.6f} s\n"
            f"reconstruction  : {t.reconstruction:.6f} s\n"
            f"total solve     : {t.total:.6f} s"
        )
