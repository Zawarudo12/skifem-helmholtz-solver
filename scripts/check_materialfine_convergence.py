from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter

import numpy as np

from solver.photonics_fem_solver import CachedDisorderedSolver
from solver.photonics_fem_solver import DisorderedConfig, diffraction_orders

COARSE_MESH = Path(
    "meshes_disordered/disordered_196disks_buffer2_pml3.msh"
)

REFINED_MESH = Path(
    "meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh"
)

FDTD_FILE = Path(
    "comparison_data/Disordered/dataset_disordered_disks_fdtd.csv"
)

COMSOL_FILE = Path(
    "comparison_data/Disordered/Disorderd1.csv"
)

WAVELENGTHS = np.array(
    [
        1.064516129032258,
        1.170212765957447,
        1.500000000000000,
        1.755319148936170,
        2.062500000000000,
        2.323943661971831,
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
        physical_ymin=-5.5,
        physical_ymax=+5.5,
        pml_low=3.0,
        pml_high=3.0,
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )

def load_fdtd():
    try:
        a = np.loadtxt(FDTD_FILE, delimiter=",")
    except ValueError:
        a = np.loadtxt(FDTD_FILE)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, :3]

def load_comsol():
    a = np.genfromtxt(
        COMSOL_FILE,
        delimiter=",",
        comments="%",
        dtype=float,
    )
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a = a[np.all(np.isfinite(a[:, :3]), axis=1)]
    order = np.argsort(a[:, 0])
    return a[order]

def nearest(data, wavelength):
    i = int(np.argmin(np.abs(data[:, 0] - wavelength)))
    return data[i]

def solve_mesh(mesh_file: Path, label: str):
    print()
    print("=" * 126)
    print(label)
    print("=" * 126)

    t0 = perf_counter()

    solver = CachedDisorderedSolver(
        make_cfg(1.5),
        mesh_file,
        intorder=8,
    )

    setup = perf_counter() - t0

    print(f"Mesh                 : {mesh_file}")
    print(f"Full P2 DOFs         : {solver.basis.N}")
    print(f"Free reduced DOFs    : {solver.free_reduced_dofs}")
    print(f"Cache setup          : {setup:.3f} s")
    print()

    rows = []

    for wl in WAVELENGTHS:
        p0 = perf_counter()

        result = solver.solve(
            make_cfg(float(wl))
        )

        diff = diffraction_orders(
            result,
            npoints=1024,
        )

        elapsed = perf_counter() - p0

        rows.append(
            (
                float(wl),
                float(diff["R"]),
                float(diff["T"]),
                float(diff["energy_error"]),
                float(elapsed),
            )
        )

        print(
            f"lambda={wl:.6f}  "
            f"R={diff['R']:.9f}  "
            f"T={diff['T']:.9f}  "
            f"E={diff['energy_error']:.3e}  "
            f"time={elapsed:.2f}s"
        )

    return np.array(rows, dtype=float)

def main():
    for p in (
        COARSE_MESH,
        REFINED_MESH,
        FDTD_FILE,
        COMSOL_FILE,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    fdtd = load_fdtd()
    comsol = load_comsol()

    coarse = solve_mesh(
        COARSE_MESH,
        "COARSE 2um AIR / 3um PML",
    )

    refined = solve_mesh(
        REFINED_MESH,
        "MATERIAL-AWARE REFINED 2um AIR / 3um PML",
    )

    print()
    print("=" * 150)
    print("CONVERGENCE TOWARD COMSOL / FDTD")
    print("=" * 150)

    print(
        f"{'lambda':>9} "
        f"{'R coarse':>10} "
        f"{'R refined':>10} "
        f"{'R COMSOL':>10} "
        f"{'R FDTD':>10} "
        f"{'|dR| coarse-C':>15} "
        f"{'|dR| fine-C':>13} "
        f"{'T coarse':>10} "
        f"{'T refined':>10} "
        f"{'T COMSOL':>10}"
    )

    print("-" * 150)

    for i, wl in enumerate(WAVELENGTHS):
        c = nearest(comsol, wl)
        f = nearest(fdtd, wl)

        Rc = float(c[1])
        Tc = float(c[2])

        print(
            f"{wl:9.6f} "
            f"{coarse[i,1]:10.6f} "
            f"{refined[i,1]:10.6f} "
            f"{Rc:10.6f} "
            f"{float(f[1]):10.6f} "
            f"{abs(coarse[i,1]-Rc):15.6f} "
            f"{abs(refined[i,1]-Rc):13.6f} "
            f"{coarse[i,2]:10.6f} "
            f"{refined[i,2]:10.6f} "
            f"{Tc:10.6f}"
        )

    mae_coarse_R = np.mean(
        np.abs(
            coarse[:,1]
            - np.array(
                [nearest(comsol, w)[1] for w in WAVELENGTHS]
            )
        )
    )

    mae_refined_R = np.mean(
        np.abs(
            refined[:,1]
            - np.array(
                [nearest(comsol, w)[1] for w in WAVELENGTHS]
            )
        )
    )

    mae_coarse_T = np.mean(
        np.abs(
            coarse[:,2]
            - np.array(
                [nearest(comsol, w)[2] for w in WAVELENGTHS]
            )
        )
    )

    mae_refined_T = np.mean(
        np.abs(
            refined[:,2]
            - np.array(
                [nearest(comsol, w)[2] for w in WAVELENGTHS]
            )
        )
    )

    print()
    print(f"Selected-point R MAE vs COMSOL: coarse={mae_coarse_R:.6e}, refined={mae_refined_R:.6e}")
    print(f"Selected-point T MAE vs COMSOL: coarse={mae_coarse_T:.6e}, refined={mae_refined_T:.6e}")

    if mae_refined_R < mae_coarse_R:
        print("R RESULT: refinement moved the selected wavelengths toward COMSOL.")
    else:
        print("R RESULT: refinement did NOT reduce selected-point MAE; investigate formulation/geometry next.")

    if mae_refined_T < mae_coarse_T:
        print("T RESULT: refinement moved the selected wavelengths toward COMSOL.")
    else:
        print("T RESULT: refinement did NOT reduce selected-point MAE; investigate formulation/geometry next.")

if __name__ == "__main__":
    main()




