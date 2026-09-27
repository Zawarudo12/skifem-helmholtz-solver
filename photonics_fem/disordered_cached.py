from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from scipy.sparse.linalg import spsolve
from skfem import Basis, ElementTriP2

from .disordered_fast import (
    DisorderedConfig,
    DisorderedSolveResult,
    SolveTimings,
    assemble_scattered_source,
    assemble_static_matrices,
    build_normal_incidence_periodic_projection,
    load_disordered_mesh,
)


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

    def __init__(
        self,
        cfg_template: DisorderedConfig,
        mesh_file: str | Path,
        *,
        intorder: int = 8,
    ):
        self.cfg_template = cfg_template
        self.mesh_file = Path(mesh_file)
        self.intorder = int(intorder)

        setup_start = perf_counter()

        self.mesh = load_disordered_mesh(
            self.mesh_file
        )

        self.basis = Basis(
            self.mesh,
            ElementTriP2(),
            intorder=self.intorder,
        )

        (
            self.P,
            self.full_to_reduced,
            self.left_dofs,
            self.right_dofs,
            self.periodic_y_mismatch,
        ) = build_normal_incidence_periodic_projection(
            self.basis
        )

        self.P = self.P.tocsr()

        self.PH = (
            self.P
            .conjugate()
            .transpose()
            .tocsr()
        )

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
            self.full_to_reduced[
                outer_full
            ]
        )

        self.reduced_dofs = (
            self.P.shape[1]
        )

        all_red = np.arange(
            self.reduced_dofs,
            dtype=int,
        )

        self.free_red = np.setdiff1d(
            all_red,
            self.outer_red,
        )

        self.free_reduced_dofs = (
            self.free_red.size
        )

        matrix_start = perf_counter()

        (
            self.K_full,
            self.M_full,
        ) = assemble_static_matrices(
            self.basis,
            cfg_template,
        )

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

        self.last_timings = None

    def _check_cfg(
        self,
        cfg: DisorderedConfig,
    ) -> None:

        fixed_attrs = [
            "width",
            "xmin",
            "xmax",
            "scatter_ymin",
            "scatter_ymax",
            "physical_ymin",
            "physical_ymax",
            "pml_low",
            "pml_high",
            "pml_sigma_max",
            "n_background",
            "n_disk",
        ]

        for name in fixed_attrs:

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
                    f"Cached solver cannot change {name}: "
                    f"cached={a}, requested={b}."
                )

        if (
            int(
                self.cfg_template.pml_order
            )
            != int(
                cfg.pml_order
            )
        ):
            raise ValueError(
                "Cached solver cannot change pml_order."
            )

    def solve(
        self,
        cfg: DisorderedConfig,
    ) -> DisorderedSolveResult:

        self._check_cfg(
            cfg
        )

        total_start = perf_counter()

        t0 = perf_counter()

        b_full = assemble_scattered_source(
            self.basis,
            cfg,
        )

        b_red = np.asarray(
            self.PH @ b_full,
            dtype=np.complex128,
        ).reshape(-1)

        b_free = b_red[
            self.free_red
        ]

        t1 = perf_counter()

        A_free = (
            self.K_free
            - cfg.k0 ** 2
            * self.M_free
        ).tocsc()

        t2 = perf_counter()

        z_free = spsolve(
            A_free,
            b_free,
        )

        t3 = perf_counter()

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

        t4 = perf_counter()

        self.last_timings = (
            CachedDisorderedTimings(
                source_assembly=t1 - t0,
                matrix_build=t2 - t1,
                linear_solve=t3 - t2,
                reconstruction=t4 - t3,
                total=t4 - total_start,
            )
        )

        # Preserve the existing result interface.
        timings = SolveTimings(
            mesh_and_basis=0.0,
            periodic_setup=0.0,
            matrix_assembly=0.0,
            source_assembly=t1 - t0,
            linear_solve=t3 - t2,
            reconstruction=t4 - t3,
            total=t4 - total_start,
        )

        timings.periodic_y_mismatch = (
            self.periodic_y_mismatch
        )

        timings.left_periodic_dofs = int(
            self.left_dofs.size
        )

        timings.right_periodic_dofs = int(
            self.right_dofs.size
        )

        return DisorderedSolveResult(
            cfg=cfg,
            mesh=self.mesh,
            basis=self.basis,
            u_scattered=u_scattered,
            full_dofs=self.basis.N,
            reduced_dofs=self.reduced_dofs,
            free_reduced_dofs=self.free_reduced_dofs,
            timings=timings,
        )
