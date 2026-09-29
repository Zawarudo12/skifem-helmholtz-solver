from __future__ import annotations

# Set before NumPy/SciPy imports.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders


MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_buffer1_pml2.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

# Previous baseline sweep, if it exists.
OLD_SWEEP_FILE = Path(
    "results_disordered_sweep/disordered_full_2workers.csv"
)

OUT = Path(
    "results_disordered_improved_boundary"
)
OUT.mkdir(exist_ok=True)

NPOINTS_RT = 1024

_WORKER_SOLVER = None


def make_cfg(wavelength: float) -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=float(wavelength),
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,

        # NEW open-boundary setup:
        # 1.0 um ordinary air on each side of the 7x7 disk region.
        physical_ymin=-4.5,
        physical_ymax=+4.5,

        # 2.0 um PML above and below.
        pml_low=2.0,
        pml_high=2.0,

        pml_order=3,
        pml_sigma_max=8.0,

        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def load_fdtd_reference():
    if not FDTD_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {FDTD_FILE}"
        )

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
        raw = raw.reshape(1, -1)

    if raw.shape[1] < 3:
        raise ValueError(
            "FDTD file must contain wavelength_um, R, T."
        )

    data = raw[:, :3]

    mask = (
        (data[:, 0] >= 1.0 - 1e-12)
        &
        (data[:, 0] <= 2.4 + 1e-12)
    )

    data = data[mask]
    return data[np.argsort(data[:, 0])]


def worker_init(mesh_file: str):
    global _WORKER_SOLVER

    _WORKER_SOLVER = CachedDisorderedSolver(
        make_cfg(1.5),
        Path(mesh_file),
        intorder=8,
    )


def worker_solve(wavelength: float):
    global _WORKER_SOLVER

    if _WORKER_SOLVER is None:
        raise RuntimeError(
            "Worker cache not initialized."
        )

    cfg = make_cfg(wavelength)

    t0 = perf_counter()

    result = _WORKER_SOLVER.solve(
        cfg
    )

    t1 = perf_counter()

    diff = diffraction_orders(
        result,
        npoints=NPOINTS_RT,
    )

    t2 = perf_counter()

    return {
        "wavelength": float(wavelength),
        "R": float(diff["R"]),
        "T": float(diff["T"]),
        "R_plus_T": float(diff["R_plus_T"]),
        "energy_error": float(diff["energy_error"]),
        "cutoff_flag": 1.0 if diff["cutoff_orders"] else 0.0,
        "solve_time": float(t1 - t0),
        "post_time": float(t2 - t1),
        "point_time": float(t2 - t0),
    }


def load_old_sweep():
    if not OLD_SWEEP_FILE.exists():
        return None

    try:
        data = np.genfromtxt(
            OLD_SWEEP_FILE,
            delimiter=",",
            names=True,
        )
    except Exception:
        return None

    if data.size == 0:
        return None

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"Missing improved mesh: {MESH_FILE}\n"
            "Run build_disordered_mesh_buffer1_pml2.py first."
        )

    fdtd = load_fdtd_reference()
    wavelengths = fdtd[:, 0].copy()

    print()
    print("=" * 104)
    print("DISORDERED MEDIA — IMPROVED BOUNDARY FULL SPECTRUM")
    print("=" * 104)
    print(f"Mesh              = {MESH_FILE}")
    print("Air buffer        = 1.0 um each side")
    print("PML thickness     = 2.0 um each side")
    print("Polarization      = Ez / TE")
    print("x boundary        = periodic, Bloch phase = 1")
    print("Mirror symmetry   = none")
    print("Diffraction R/T   = all propagating orders, dynamic")
    print(f"FDTD points       = {len(wavelengths)} exact wavelengths")
    print(f"Workers           = {args.workers}")
    print("=" * 104)
    print()

    wall_start = perf_counter()

    rows = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(MESH_FILE),),
    ) as executor:

        futures = {
            executor.submit(
                worker_solve,
                float(wl),
            ): float(wl)
            for wl in wavelengths
        }

        for i, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            row = future.result()
            rows.append(row)

            print(
                f"[{i:03d}/{len(wavelengths):03d}] "
                f"lambda={row['wavelength']:.6f} um  "
                f"R={row['R']:.6f}  "
                f"T={row['T']:.6e}  "
                f"R+T={row['R_plus_T']:.6f}  "
                f"E={row['energy_error']:.3e}  "
                f"time={row['point_time']:.2f}s"
            )

    wall_time = perf_counter() - wall_start

    rows.sort(
        key=lambda r: r["wavelength"]
    )

    w = np.array(
        [r["wavelength"] for r in rows]
    )

    R = np.array(
        [r["R"] for r in rows]
    )

    T = np.array(
        [r["T"] for r in rows]
    )

    RT = np.array(
        [r["R_plus_T"] for r in rows]
    )

    E = np.array(
        [r["energy_error"] for r in rows]
    )

    cutoff = np.array(
        [r["cutoff_flag"] for r in rows]
    )

    point_time = np.array(
        [r["point_time"] for r in rows]
    )

    R_fdtd = np.interp(
        w,
        fdtd[:, 0],
        fdtd[:, 1],
    )

    T_fdtd = np.interp(
        w,
        fdtd[:, 0],
        fdtd[:, 2],
    )

    dR = np.abs(R - R_fdtd)
    dT = np.abs(T - T_fdtd)

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------
    data = np.column_stack(
        [
            w,
            R,
            T,
            RT,
            E,
            cutoff,
            R_fdtd,
            T_fdtd,
            dR,
            dT,
            point_time,
        ]
    )

    csv_path = (
        OUT
        / "disordered_improved_boundary_full.csv"
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fem,T_fem,R_plus_T,"
            "energy_error,cutoff_flag,"
            "R_fdtd,T_fdtd,"
            "abs_dR,abs_dT,"
            "point_time_s"
        ),
        comments="",
    )

    # --------------------------------------------------------
    # Main final comparison plot.
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10.5, 6.3)
    )

    ax.axvspan(
        1.4,
        1.8,
        alpha=0.10,
        label="Target stop band 1.4-1.8 um",
    )

    ax.plot(
        w,
        R,
        linewidth=2.2,
        label="FEM R — improved boundary",
    )

    ax.plot(
        w,
        T,
        linewidth=2.2,
        label="FEM T — improved boundary",
    )

    ax.plot(
        w,
        R_fdtd,
        "--",
        linewidth=1.7,
        label="FDTD R",
    )

    ax.plot(
        w,
        T_fdtd,
        "--",
        linewidth=1.7,
        label="FDTD T",
    )

    ax.set_xlim(
        1.0,
        2.4,
    )

    ax.set_ylim(
        -0.04,
        1.16,
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Power fraction"
    )

    ax.set_title(
        "Disordered disks: improved FEM boundary treatment vs FDTD"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    final_plot = (
        OUT
        / "01_improved_FEM_vs_FDTD_RT.png"
    )

    fig.savefig(
        final_plot,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Old vs new FEM comparison, if old CSV is available.
    # --------------------------------------------------------
    old = load_old_sweep()

    oldnew_plot = None

    if old is not None:
        try:
            old_w = np.asarray(
                old["wavelength_um"],
                dtype=float,
            )
            old_R = np.asarray(
                old["R_fem"],
                dtype=float,
            )
            old_T = np.asarray(
                old["T_fem"],
                dtype=float,
            )

            order = np.argsort(old_w)
            old_w = old_w[order]
            old_R = old_R[order]
            old_T = old_T[order]

            fig, ax = plt.subplots(
                figsize=(10.5, 6.3)
            )

            ax.axvspan(
                1.4,
                1.8,
                alpha=0.10,
                label="Target stop band",
            )

            ax.plot(
                old_w,
                old_R,
                ":",
                linewidth=1.4,
                label="Old FEM R — air0.5/PML1.0",
            )

            ax.plot(
                w,
                R,
                linewidth=2.2,
                label="New FEM R — air1.0/PML2.0",
            )

            ax.plot(
                w,
                R_fdtd,
                "--",
                linewidth=1.7,
                label="FDTD R",
            )

            ax.set_xlim(1.0, 2.4)
            ax.set_ylim(-0.04, 1.16)
            ax.set_xlabel("Wavelength [um]")
            ax.set_ylabel("Reflectance")
            ax.set_title(
                "Reflectance: old vs improved boundary treatment"
            )
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()

            oldnew_plot = (
                OUT
                / "02_old_vs_new_vs_FDTD_reflectance.png"
            )

            fig.savefig(
                oldnew_plot,
                dpi=250,
                bbox_inches="tight",
            )

            plt.close(fig)

            fig, ax = plt.subplots(
                figsize=(10.5, 6.3)
            )

            ax.axvspan(
                1.4,
                1.8,
                alpha=0.10,
                label="Target stop band",
            )

            ax.plot(
                old_w,
                old_T,
                ":",
                linewidth=1.4,
                label="Old FEM T — air0.5/PML1.0",
            )

            ax.plot(
                w,
                T,
                linewidth=2.2,
                label="New FEM T — air1.0/PML2.0",
            )

            ax.plot(
                w,
                T_fdtd,
                "--",
                linewidth=1.7,
                label="FDTD T",
            )

            ax.set_xlim(1.0, 2.4)
            ax.set_ylim(-0.04, 1.05)
            ax.set_xlabel("Wavelength [um]")
            ax.set_ylabel("Transmittance")
            ax.set_title(
                "Transmittance: old vs improved boundary treatment"
            )
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()

            trans_plot = (
                OUT
                / "03_old_vs_new_vs_FDTD_transmittance.png"
            )

            fig.savefig(
                trans_plot,
                dpi=250,
                bbox_inches="tight",
            )

            plt.close(fig)

        except Exception as exc:
            print(
                f"Old sweep overlay skipped: {exc}"
            )

    # --------------------------------------------------------
    # Energy error plot with theoretical Rayleigh cutoffs.
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10.5, 5.8)
    )

    ax.semilogy(
        w,
        E,
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        label="Improved-boundary FEM",
    )

    for m in range(3, 8):
        cutoff_wl = 7.0 / m

        if 1.0 <= cutoff_wl <= 2.4:
            ax.axvline(
                cutoff_wl,
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
            )

            ax.text(
                cutoff_wl,
                max(np.min(E) * 1.5, 1e-7),
                f"m={m}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
            )

    ax.set_xlim(
        1.0,
        2.4,
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        r"Energy error $|R+T-1|$"
    )

    ax.set_title(
        "Energy conservation with 1.0 um air buffer + 2.0 um PML"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    energy_plot = (
        OUT
        / "04_energy_error_with_Rayleigh_cutoffs.png"
    )

    fig.savefig(
        energy_plot,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Console metrics.
    # --------------------------------------------------------
    stop = (
        (w >= 1.4)
        &
        (w <= 1.8)
    )

    print()
    print("=" * 104)
    print("FINAL SUMMARY")
    print("=" * 104)

    print(
        f"Wall clock                = "
        f"{wall_time:.3f} s "
        f"({wall_time / 60.0:.2f} min)"
    )

    print(
        f"R MAE vs FDTD             = "
        f"{np.mean(dR):.6e}"
    )

    print(
        f"T MAE vs FDTD             = "
        f"{np.mean(dT):.6e}"
    )

    print(
        f"Max |R+T-1|               = "
        f"{np.max(E):.6e}"
    )

    print(
        f"Mean |R+T-1|              = "
        f"{np.mean(E):.6e}"
    )

    if np.any(stop):
        print()
        print("Target stop band 1.4-1.8 um:")
        print(
            f"  mean FEM R              = "
            f"{np.mean(R[stop]):.9f}"
        )
        print(
            f"  mean FEM T              = "
            f"{np.mean(T[stop]):.9e}"
        )
        print(
            f"  mean FDTD R             = "
            f"{np.mean(R_fdtd[stop]):.9f}"
        )
        print(
            f"  mean FDTD T             = "
            f"{np.mean(T_fdtd[stop]):.9e}"
        )
        print(
            f"  max FEM T in band       = "
            f"{np.max(T[stop]):.9e}"
        )

    print()
    print(f"CSV                       = {csv_path}")
    print(f"Main comparison plot      = {final_plot}")

    if oldnew_plot is not None:
        print(f"Old/new reflectance plot  = {oldnew_plot}")
        print(
            f"Old/new transmittance plot= "
            f"{OUT / '03_old_vs_new_vs_FDTD_transmittance.png'}"
        )

    print(f"Energy-error plot         = {energy_plot}")


if __name__ == "__main__":
    main()
