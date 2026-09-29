from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scikits.umfpack

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders


MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_buffer1_pml2.msh"
)

# Representative points: difficult short-wave region, stop band,
# transition, and long-wave region.
WAVELENGTHS = np.array([
    1.064516129032258,
    1.170212765957447,
    1.500000000000000,
    1.755319148936170,
    2.062500000000000,
    2.323943661971831,
])


def make_cfg(wavelength: float) -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=float(wavelength),
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,
        physical_ymin=-4.5,
        physical_ymax=+4.5,
        pml_low=2.0,
        pml_high=2.0,
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def main():
    print("=" * 110)
    print("UMFPACK SPEED BENCHMARK — IMPROVED DISORDERED-MEDIA MESH")
    print("=" * 110)
    print("scikit-umfpack import: OK")
    print(f"Mesh: {MESH_FILE}")
    print()

    if not MESH_FILE.exists():
        raise FileNotFoundError(MESH_FILE)

    t_setup0 = perf_counter()

    solver = CachedDisorderedSolver(
        make_cfg(1.5),
        MESH_FILE,
        intorder=8,
    )

    setup_time = perf_counter() - t_setup0

    print(f"Full P2 DOFs      : {solver.basis.N}")
    print(f"Reduced DOFs      : {solver.reduced_dofs}")
    print(f"Free reduced DOFs : {solver.free_reduced_dofs}")
    print(f"Cache setup       : {setup_time:.3f} s")
    print()

    print(
        f"{'lambda':>10} "
        f"{'R':>10} "
        f"{'T':>10} "
        f"{'source':>9} "
        f"{'matrix':>9} "
        f"{'linear':>9} "
        f"{'recon':>9} "
        f"{'RT-post':>9} "
        f"{'total':>9}"
    )
    print("-" * 110)

    totals = []

    wall0 = perf_counter()

    for wl in WAVELENGTHS:
        cfg = make_cfg(float(wl))

        p0 = perf_counter()
        result = solver.solve(cfg)
        p1 = perf_counter()

        diff = diffraction_orders(
            result,
            npoints=1024,
        )

        p2 = perf_counter()

        tm = solver.last_timings
        post = p2 - p1
        point_total = p2 - p0

        totals.append([
            tm.source_assembly,
            tm.matrix_build,
            tm.linear_solve,
            tm.reconstruction,
            post,
            point_total,
        ])

        print(
            f"{wl:10.6f} "
            f"{diff['R']:10.6f} "
            f"{diff['T']:10.6f} "
            f"{tm.source_assembly:9.3f} "
            f"{tm.matrix_build:9.3f} "
            f"{tm.linear_solve:9.3f} "
            f"{tm.reconstruction:9.3f} "
            f"{post:9.3f} "
            f"{point_total:9.3f}"
        )

    wall = perf_counter() - wall0
    totals = np.asarray(totals)

    labels = [
        "Source assembly",
        "Matrix build",
        "Linear solve",
        "Reconstruction",
        "R/T post-process",
        "Point total",
    ]

    print()
    print("=" * 110)
    print("AVERAGE PER WAVELENGTH")
    print("=" * 110)

    for i, label in enumerate(labels):
        print(
            f"{label:<22}: "
            f"{np.mean(totals[:, i]):9.3f} s"
        )

    print()
    linear_fraction = (
        np.sum(totals[:, 2])
        / np.sum(totals[:, 5])
        * 100.0
    )

    print(f"Linear solve share     : {linear_fraction:.1f}%")
    print(f"6-point measured wall  : {wall:.3f} s")

    projected_97 = (
        np.mean(totals[:, 5]) * 97.0
    )

    print(
        f"Simple 1-worker 97-pt projection: "
        f"{projected_97:.1f} s"
    )

    print()
    print(
        "NOTE: this projection excludes multiprocessing effects and "
        "is only for identifying the bottleneck."
    )


if __name__ == "__main__":
    main()
