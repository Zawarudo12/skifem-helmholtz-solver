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


FINE_MESH = Path(
    "meshes_disordered/disordered_196disks_fine.msh"
)

MATERIAL_MESH = Path(
    "meshes_disordered/disordered_196disks_materialfine.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

OUT = Path(
    "results_disordered_material_convergence"
)
OUT.mkdir(exist_ok=True)

# Three intentionally chosen points:
#   1.064516 : strong pre-band resonance / previous mesh sensitivity
#   1.500000 : stop-band control point
#   2.062500 : post-band resonance / previous mesh sensitivity
WAVELENGTHS = np.array(
    [
        1.0645161290322580,
        1.5000000000000000,
        2.0625000000000000,
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
    return (
        float(data[idx, 0]),
        float(data[idx, 1]),
        float(data[idx, 2]),
    )


def solve_mesh(mesh_file: Path, label: str):
    print()
    print("=" * 104)
    print(f"{label.upper()}")
    print("=" * 104)
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

    print(
        f"{'lambda':>10} "
        f"{'R':>12} "
        f"{'T':>12} "
        f"{'R+T':>12} "
        f"{'energy':>11} "
        f"{'solve[s]':>10}"
    )

    rows = []

    for wavelength in WAVELENGTHS:
        cfg = make_cfg(float(wavelength))

        t0 = perf_counter()
        result = solver.solve(cfg)
        diff = diffraction_orders(result, npoints=1024)
        elapsed = perf_counter() - t0

        row = {
            "wavelength": float(wavelength),
            "R": float(diff["R"]),
            "T": float(diff["T"]),
            "R_plus_T": float(diff["R_plus_T"]),
            "energy": float(diff["energy_error"]),
            "solve_time": float(elapsed),
        }
        rows.append(row)

        print(
            f"{wavelength:10.6f} "
            f"{row['R']:12.8f} "
            f"{row['T']:12.8f} "
            f"{row['R_plus_T']:12.8f} "
            f"{row['energy']:11.3e} "
            f"{row['solve_time']:10.3f}"
        )

    del solver
    gc.collect()

    return rows


def main() -> None:
    for path in (
        FINE_MESH,
        MATERIAL_MESH,
        FDTD_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find: {path}"
            )

    fdtd = load_fdtd()

    fine = solve_mesh(
        FINE_MESH,
        "PREVIOUS FINE MESH",
    )

    material = solve_mesh(
        MATERIAL_MESH,
        "MATERIAL-AWARE FINE MESH",
    )

    print()
    print("=" * 146)
    print("FINE -> MATERIAL-AWARE CONVERGENCE")
    print("=" * 146)

    print(
        f"{'lambda':>10} "
        f"{'R_fine':>11} "
        f"{'R_mat':>11} "
        f"{'|dR|':>11} "
        f"{'T_fine':>11} "
        f"{'T_mat':>11} "
        f"{'|dT|':>11} "
        f"{'E_mat':>10} "
        f"{'R_FDTD':>11} "
        f"{'|Rmat-FDTD|':>13} "
        f"{'T_FDTD':>11}"
    )

    output = []

    for f, m in zip(fine, material):
        wl = f["wavelength"]
        fdtd_wl, R_fdtd, T_fdtd = nearest_fdtd(
            fdtd,
            wl,
        )

        dR = abs(f["R"] - m["R"])
        dT = abs(f["T"] - m["T"])
        dR_fdtd = abs(m["R"] - R_fdtd)
        dT_fdtd = abs(m["T"] - T_fdtd)

        print(
            f"{wl:10.6f} "
            f"{f['R']:11.7f} "
            f"{m['R']:11.7f} "
            f"{dR:11.3e} "
            f"{f['T']:11.7f} "
            f"{m['T']:11.7f} "
            f"{dT:11.3e} "
            f"{m['energy']:10.3e} "
            f"{R_fdtd:11.7f} "
            f"{dR_fdtd:13.3e} "
            f"{T_fdtd:11.7f}"
        )

        output.append(
            [
                wl,
                f["R"],
                f["T"],
                f["energy"],
                m["R"],
                m["T"],
                m["energy"],
                dR,
                dT,
                fdtd_wl,
                R_fdtd,
                T_fdtd,
                dR_fdtd,
                dT_fdtd,
                f["solve_time"],
                m["solve_time"],
            ]
        )

    output = np.asarray(
        output,
        dtype=float,
    )

    out_file = (
        OUT
        / "fine_vs_materialfine_checkpoints.csv"
    )

    np.savetxt(
        out_file,
        output,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fine,T_fine,energy_fine,"
            "R_materialfine,T_materialfine,energy_materialfine,"
            "abs_dR_mesh,abs_dT_mesh,"
            "fdtd_wavelength_um,R_fdtd,T_fdtd,"
            "abs_dR_materialfine_fdtd,abs_dT_materialfine_fdtd,"
            "fine_solve_time_s,materialfine_solve_time_s"
        ),
        comments="",
    )

    print()
    print("=" * 146)
    print("INTERPRETATION GUIDE")
    print("=" * 146)
    print(
        "1.064516 / 2.062500: if |dR| is now much smaller than the previous pilot->fine change, "
        "the solution is approaching mesh convergence."
    )
    print(
        "1.500000: should remain essentially R=1, T=0; this is the stop-band control."
    )
    print(
        "If material-aware results move closer to FDTD again, spatial discretization was a major source of the old mismatch."
    )
    print(
        "If they stabilize away from FDTD, investigate the exact FDTD geometry/source/mesh/settings instead of endlessly refining FEM."
    )
    print()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
