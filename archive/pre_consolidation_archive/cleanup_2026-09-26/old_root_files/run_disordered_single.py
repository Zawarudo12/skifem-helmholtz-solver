from __future__ import annotations

# Keep BLAS/OpenMP from oversubscribing the CPU during this first test.
import os

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
)

from pathlib import Path
from time import perf_counter

import numpy as np

from photonics_fem.disordered_fast import (
    DisorderedConfig,
    diffraction_orders,
    solve_disordered,
)


MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_pilot.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

WAVELENGTH = 1.600


def load_fdtd_reference():

    if not FDTD_FILE.exists():
        return None

    try:

        raw = np.loadtxt(
            FDTD_FILE,
            delimiter=",",
        )

    except ValueError:

        raw = np.loadtxt(
            FDTD_FILE
        )

    if raw.ndim == 1:

        raw = raw.reshape(
            1,
            -1,
        )

    if raw.shape[1] < 3:

        raise ValueError(
            "FDTD file must have at least three columns: wavelength, R, T."
        )

    data = raw[
        :,
        :3,
    ]

    order = np.argsort(
        data[:, 0]
    )

    return data[
        order
    ]


def main() -> None:

    if not MESH_FILE.exists():

        raise FileNotFoundError(
            f"Could not find mesh: {MESH_FILE}"
        )

    cfg = DisorderedConfig(
        wavelength=WAVELENGTH,

        width=7.0,
        xmin=-3.5,
        xmax=3.5,

        scatter_ymin=-3.5,
        scatter_ymax=3.5,

        physical_ymin=-4.0,
        physical_ymax=4.0,

        pml_low=1.0,
        pml_high=1.0,

        pml_order=3,
        pml_sigma_max=8.0,

        # Background was not explicitly specified in the message.
        # Air is the current pilot assumption.
        n_background=1.0 + 0.0j,

        n_disk=3.0 + 0.0j,
    )

    print()
    print("=" * 92)
    print("DISORDERED MEDIA — SINGLE-WAVELENGTH PILOT")
    print("=" * 92)

    print(
        f"Wavelength        = "
        f"{cfg.wavelength:.4f} um"
    )

    print(
        f"Polarization      = "
        "Ez / TE"
    )

    print(
        f"Cell width        = "
        f"{cfg.width:.3f} um"
    )

    print(
        f"Disk index        = "
        f"{cfg.n_disk.real:.3f}"
    )

    print(
        f"Background index  = "
        f"{cfg.n_background.real:.3f} "
        "(pilot assumption)"
    )

    print(
        "x boundary        = periodic, normal-incidence Bloch phase = 1"
    )

    print(
        "y boundary        = 0.5 um air buffer + 1.0 um PML each side"
    )

    print(
        f"Mesh              = "
        f"{MESH_FILE}"
    )

    print("=" * 92)
    print()

    total_start = perf_counter()

    result = solve_disordered(
        cfg,
        MESH_FILE,
        intorder=8,
    )

    diff = diffraction_orders(
        result,
        npoints=1024,
    )

    total_time = (
        perf_counter()
        - total_start
    )

    print()
    print("=" * 92)
    print("FEM SYSTEM")
    print("=" * 92)

    print(
        f"Full P2 DOFs          = "
        f"{result.full_dofs}"
    )

    print(
        f"Periodic reduced DOFs = "
        f"{result.reduced_dofs}"
    )

    print(
        f"Free reduced DOFs     = "
        f"{result.free_reduced_dofs}"
    )

    print(
        f"Periodic edge DOFs    = "
        f"{result.timings.left_periodic_dofs} left / "
        f"{result.timings.right_periodic_dofs} right"
    )

    print(
        f"Max periodic y error  = "
        f"{result.timings.periodic_y_mismatch:.3e} um"
    )

    print()
    print("=" * 92)
    print("R / T")
    print("=" * 92)

    print(
        f"R                     = "
        f"{diff['R']:.10f}"
    )

    print(
        f"T                     = "
        f"{diff['T']:.10f}"
    )

    print(
        f"R + T                 = "
        f"{diff['R_plus_T']:.10f}"
    )

    print(
        f"|R + T - 1|           = "
        f"{diff['energy_error']:.6e}"
    )

    print(
        f"Top monitor y         = "
        f"{diff['top_monitor_y']:.3f} um"
    )

    print(
        f"Bottom monitor y      = "
        f"{diff['bottom_monitor_y']:.3f} um"
    )

    print(
        f"Cutoff orders         = "
        f"{diff['cutoff_orders']}"
    )

    propagating = [
        row
        for row in diff[
            "orders"
        ]
        if row[
            "propagating"
        ]
    ]

    print()
    print(
        "Propagating diffraction orders:"
    )

    print(
        f"{'m':>4} "
        f"{'R_m':>14} "
        f"{'T_m':>14}"
    )

    for row in propagating:

        print(
            f"{row['m']:4d} "
            f"{row['R']:14.8e} "
            f"{row['T']:14.8e}"
        )

    reference = load_fdtd_reference()

    if reference is not None:

        R_fdtd = float(
            np.interp(
                cfg.wavelength,
                reference[:, 0],
                reference[:, 1],
            )
        )

        T_fdtd = float(
            np.interp(
                cfg.wavelength,
                reference[:, 0],
                reference[:, 2],
            )
        )

        print()
        print("=" * 92)
        print("FDTD REFERENCE AT SAME WAVELENGTH")
        print("=" * 92)

        print(
            f"FDTD R (interpolated) = "
            f"{R_fdtd:.10f}"
        )

        print(
            f"FDTD T (interpolated) = "
            f"{T_fdtd:.10f}"
        )

        print(
            f"|dR| FEM-FDTD         = "
            f"{abs(diff['R'] - R_fdtd):.6e}"
        )

        print(
            f"|dT| FEM-FDTD         = "
            f"{abs(diff['T'] - T_fdtd):.6e}"
        )

    t = result.timings

    print()
    print("=" * 92)
    print("TIMING")
    print("=" * 92)

    print(
        f"Mesh + P2 basis       = "
        f"{t.mesh_and_basis:.3f} s"
    )

    print(
        f"Periodic setup        = "
        f"{t.periodic_setup:.3f} s"
    )

    print(
        f"K/M assembly          = "
        f"{t.matrix_assembly:.3f} s"
    )

    print(
        f"Source assembly       = "
        f"{t.source_assembly:.3f} s"
    )

    print(
        f"Sparse linear solve   = "
        f"{t.linear_solve:.3f} s"
    )

    print(
        f"Reconstruction        = "
        f"{t.reconstruction:.3f} s"
    )

    print(
        f"Core solve total      = "
        f"{t.total:.3f} s"
    )

    print(
        f"With R/T postprocess  = "
        f"{total_time:.3f} s"
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
