from __future__ import annotations

import gc
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders


PILOT_MESH = Path(
    "meshes_disordered/disordered_196disks_pilot.msh"
)

FINE_MESH = Path(
    "meshes_disordered/disordered_196disks_fine.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

OUT = Path(
    "results_disordered_convergence"
)
OUT.mkdir(exist_ok=True)

# Exact wavelengths already present in the supplied FDTD dataset.
# They deliberately include clean points, large FEM/FDTD mismatches,
# and wavelengths close to Rayleigh cutoffs.
WAVELENGTHS = np.array(
    [
        1.0645161290322580,   # large mismatch, good energy on pilot
        1.1702127659574468,   # close to 7/6 = 1.16667 um cutoff
        1.5000000000000000,   # clean stop-band control
        1.7553191489361701,   # close to 7/4 = 1.75 um cutoff
        2.0625000000000000,   # large mismatch, good-ish energy
        2.3239436619718310,   # close to 7/3 = 2.33333 um cutoff
    ],
    dtype=float,
)


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
    return float(data[idx, 1]), float(data[idx, 2]), float(data[idx, 0])


def cutoff_distance(wavelength: float):
    candidates = []
    for m in range(1, 10):
        lam_c = 7.0 / m
        if 1.0 <= lam_c <= 2.4:
            candidates.append((abs(wavelength - lam_c), m, lam_c))
    return min(candidates)


def solve_mesh(mesh_file: Path, label: str):
    print()
    print("=" * 108)
    print(f"{label.upper()} MESH")
    print("=" * 108)
    print(f"Mesh: {mesh_file}")

    setup_start = perf_counter()
    solver = CachedDisorderedSolver(
        make_cfg(1.5),
        mesh_file,
        intorder=8,
    )
    setup_time = perf_counter() - setup_start

    print(f"Full P2 DOFs          = {solver.basis.N}")
    print(f"Reduced DOFs          = {solver.reduced_dofs}")
    print(f"Free reduced DOFs     = {solver.free_reduced_dofs}")
    print(f"Cache setup           = {setup_time:.3f} s")
    print()

    rows = []

    print(
        f"{'lambda':>10} "
        f"{'R':>12} "
        f"{'T':>12} "
        f"{'R+T':>12} "
        f"{'energy':>11} "
        f"{'solve[s]':>10}"
    )

    for wl in WAVELENGTHS:
        cfg = make_cfg(float(wl))

        t0 = perf_counter()
        result = solver.solve(cfg)
        diff = diffraction_orders(result, npoints=1024)
        elapsed = perf_counter() - t0

        rows.append(
            {
                "wavelength": float(wl),
                "R": float(diff["R"]),
                "T": float(diff["T"]),
                "R_plus_T": float(diff["R_plus_T"]),
                "energy": float(diff["energy_error"]),
                "solve_time": float(elapsed),
            }
        )

        print(
            f"{wl:10.6f} "
            f"{diff['R']:12.8f} "
            f"{diff['T']:12.8f} "
            f"{diff['R_plus_T']:12.8f} "
            f"{diff['energy_error']:11.3e} "
            f"{elapsed:10.3f}"
        )

    # Explicitly drop the large sparse matrices before loading the next mesh.
    del solver
    gc.collect()

    return rows


def main() -> None:
    for path in (PILOT_MESH, FINE_MESH, FDTD_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Could not find: {path}")

    fdtd = load_fdtd()

    pilot = solve_mesh(PILOT_MESH, "pilot")
    fine = solve_mesh(FINE_MESH, "fine")

    print()
    print("=" * 132)
    print("PILOT vs FINE CONVERGENCE")
    print("=" * 132)
    print(
        f"{'lambda':>10} "
        f"{'near cutoff':>12} "
        f"{'R_pilot':>11} "
        f"{'R_fine':>11} "
        f"{'|dR|':>11} "
        f"{'T_pilot':>11} "
        f"{'T_fine':>11} "
        f"{'|dT|':>11} "
        f"{'E_fine':>10} "
        f"{'R_FDTD':>11}"
    )

    output = []

    for p, f in zip(pilot, fine):
        wl = p["wavelength"]
        dist, m, lam_c = cutoff_distance(wl)
        near_cutoff = dist < 0.015
        R_fdtd, T_fdtd, wl_fdtd = nearest_fdtd(fdtd, wl)

        dR = abs(p["R"] - f["R"])
        dT = abs(p["T"] - f["T"])

        print(
            f"{wl:10.6f} "
            f"{('m='+str(m)) if near_cutoff else '-':>12} "
            f"{p['R']:11.7f} "
            f"{f['R']:11.7f} "
            f"{dR:11.3e} "
            f"{p['T']:11.7f} "
            f"{f['T']:11.7f} "
            f"{dT:11.3e} "
            f"{f['energy']:10.3e} "
            f"{R_fdtd:11.7f}"
        )

        output.append(
            [
                wl,
                float(near_cutoff),
                float(m if near_cutoff else 0),
                lam_c,
                p["R"],
                p["T"],
                p["energy"],
                f["R"],
                f["T"],
                f["energy"],
                dR,
                dT,
                wl_fdtd,
                R_fdtd,
                T_fdtd,
                abs(f["R"] - R_fdtd),
                abs(f["T"] - T_fdtd),
            ]
        )

    out_file = OUT / "pilot_vs_fine_checkpoints.csv"

    np.savetxt(
        out_file,
        np.asarray(output, dtype=float),
        delimiter=",",
        header=(
            "wavelength_um,near_cutoff,cutoff_order,cutoff_wavelength_um,"
            "R_pilot,T_pilot,energy_pilot,"
            "R_fine,T_fine,energy_fine,abs_dR_mesh,abs_dT_mesh,"
            "fdtd_wavelength_um,R_fdtd,T_fdtd,abs_dR_fine_fdtd,abs_dT_fine_fdtd"
        ),
        comments="",
    )

    noncutoff_dR = []
    noncutoff_dT = []

    for row in output:
        if row[1] < 0.5:
            noncutoff_dR.append(row[10])
            noncutoff_dT.append(row[11])

    print()
    print("=" * 132)
    print("SUMMARY")
    print("=" * 132)

    if noncutoff_dR:
        print(
            f"Non-cutoff max |R_pilot-R_fine| = "
            f"{max(noncutoff_dR):.6e}"
        )
        print(
            f"Non-cutoff max |T_pilot-T_fine| = "
            f"{max(noncutoff_dT):.6e}"
        )

    print()
    print("Interpretation:")
    print("  - Tiny pilot/fine differences away from cutoffs => pilot mesh is already converged there.")
    print("  - Large pilot/fine differences away from cutoffs => mesh needs refinement before judging FEM vs FDTD.")
    print("  - Large energy error only near marked cutoffs => likely grazing-order/PML issue, not ordinary mesh error.")
    print()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
