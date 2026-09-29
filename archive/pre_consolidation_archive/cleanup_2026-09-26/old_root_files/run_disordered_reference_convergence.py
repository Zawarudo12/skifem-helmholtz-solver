from __future__ import annotations

import gc
import os
from pathlib import Path
from time import perf_counter

# Keep sparse/BLAS helpers from oversubscribing RAM/CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders


REFERENCE_MESH = Path(
    "meshes_disordered/disordered_196disks_reference.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

OUT = Path(
    "results_disordered_reference_convergence"
)
OUT.mkdir(exist_ok=True)

# Only the two mesh-sensitive, NON-cutoff wavelengths.
WAVELENGTHS = np.array(
    [
        1.0645161290322580,
        2.0625000000000000,
    ],
    dtype=float,
)

# Results already obtained on the material-aware mesh.
# Keeping them here avoids re-solving ~590k DOFs just to compute the delta.
MATERIAL_FINE = {
    1.0645161290322580: {
        "R": 0.8324896805951029,
        "T": 0.16740861322443162,
        "energy": 1.0170618046556612e-4,
    },
    2.0625000000000000: {
        "R": 0.7404141132038413,
        "T": 0.2591297843762537,
        "energy": 4.5610241990501343e-4,
    },
}


def make_cfg(wavelength: float) -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=float(wavelength),
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
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def load_fdtd() -> np.ndarray:
    try:
        data = np.loadtxt(FDTD_FILE, delimiter=",")
    except ValueError:
        data = np.loadtxt(FDTD_FILE)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data[:, :3]


def nearest_fdtd(data: np.ndarray, wavelength: float):
    idx = int(np.argmin(np.abs(data[:, 0] - wavelength)))
    return (
        float(data[idx, 0]),
        float(data[idx, 1]),
        float(data[idx, 2]),
    )


def main() -> None:
    for path in (REFERENCE_MESH, FDTD_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find: {path}"
            )

    fdtd = load_fdtd()

    print()
    print("=" * 118)
    print("FINAL REFERENCE-MESH CONVERGENCE CHECK")
    print("=" * 118)
    print(f"Mesh: {REFERENCE_MESH}")

    setup_start = perf_counter()

    solver = CachedDisorderedSolver(
        make_cfg(1.5),
        REFERENCE_MESH,
        intorder=8,
    )

    setup_time = perf_counter() - setup_start

    print()
    print(f"Full P2 DOFs          = {solver.basis.N}")
    print(f"Reduced DOFs          = {solver.reduced_dofs}")
    print(f"Free reduced DOFs     = {solver.free_reduced_dofs}")
    print(f"Cache setup           = {setup_time:.3f} s")

    print()
    print(
        f"{'lambda':>10} "
        f"{'R_mat':>11} "
        f"{'R_ref':>11} "
        f"{'|dR|':>11} "
        f"{'T_mat':>11} "
        f"{'T_ref':>11} "
        f"{'|dT|':>11} "
        f"{'E_ref':>10} "
        f"{'R_FDTD':>11} "
        f"{'|Rref-FDTD|':>13} "
        f"{'solve[s]':>10}"
    )

    rows = []

    for wavelength in WAVELENGTHS:
        wl = float(wavelength)
        previous = MATERIAL_FINE[wl]

        cfg = make_cfg(wl)

        t0 = perf_counter()
        result = solver.solve(cfg)
        diff = diffraction_orders(result, npoints=1024)
        elapsed = perf_counter() - t0

        fdtd_wl, R_fdtd, T_fdtd = nearest_fdtd(
            fdtd,
            wl,
        )

        R_ref = float(diff["R"])
        T_ref = float(diff["T"])
        energy_ref = float(diff["energy_error"])

        dR = abs(R_ref - previous["R"])
        dT = abs(T_ref - previous["T"])
        dR_fdtd = abs(R_ref - R_fdtd)
        dT_fdtd = abs(T_ref - T_fdtd)

        print(
            f"{wl:10.6f} "
            f"{previous['R']:11.7f} "
            f"{R_ref:11.7f} "
            f"{dR:11.3e} "
            f"{previous['T']:11.7f} "
            f"{T_ref:11.7f} "
            f"{dT:11.3e} "
            f"{energy_ref:10.3e} "
            f"{R_fdtd:11.7f} "
            f"{dR_fdtd:13.3e} "
            f"{elapsed:10.3f}"
        )

        rows.append(
            [
                wl,
                previous["R"],
                previous["T"],
                previous["energy"],
                R_ref,
                T_ref,
                energy_ref,
                dR,
                dT,
                fdtd_wl,
                R_fdtd,
                T_fdtd,
                dR_fdtd,
                dT_fdtd,
                elapsed,
            ]
        )

    rows = np.asarray(rows, dtype=float)

    out_file = OUT / "materialfine_vs_reference.csv"

    np.savetxt(
        out_file,
        rows,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_materialfine,T_materialfine,energy_materialfine,"
            "R_reference,T_reference,energy_reference,"
            "abs_dR_mesh,abs_dT_mesh,"
            "fdtd_wavelength_um,R_fdtd,T_fdtd,"
            "abs_dR_reference_fdtd,abs_dT_reference_fdtd,"
            "reference_solve_time_s"
        ),
        comments="",
    )

    print()
    print("=" * 118)
    print("HOW TO READ THIS")
    print("=" * 118)
    print(
        "Compare the new |dR| with the previous fine -> material-fine changes:"
    )
    print(
        "  1.064516 um : previous |dR| = 7.853e-02"
    )
    print(
        "  2.062500 um : previous |dR| = 4.993e-02"
    )
    print(
        "If the new |dR| values are much smaller again, practical mesh convergence is being reached."
    )
    print(
        "Do not use this expensive reference mesh for the 97-point production sweep."
    )
    print()
    print(f"Saved: {out_file}")

    del solver
    gc.collect()


if __name__ == "__main__":
    main()
