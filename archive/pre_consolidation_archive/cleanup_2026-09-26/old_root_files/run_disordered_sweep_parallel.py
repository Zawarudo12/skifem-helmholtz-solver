from __future__ import annotations

# IMPORTANT: set these before numpy/scipy imports.
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

from photonics_fem.disordered_cached import (
    CachedDisorderedSolver,
)

from photonics_fem.disordered_fast import (
    DisorderedConfig,
    diffraction_orders,
)


MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_pilot.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

OUT = Path(
    "results_disordered_sweep"
)

OUT.mkdir(
    exist_ok=True
)

NPOINTS_RT = 1024

_WORKER_SOLVER = None


def make_cfg(
    wavelength: float,
) -> DisorderedConfig:

    return DisorderedConfig(
        wavelength=float(
            wavelength
        ),

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

        raw = raw.reshape(
            1,
            -1,
        )

    if (
        raw.ndim != 2
        or raw.shape[1] < 3
    ):

        raise ValueError(
            "FDTD file must contain "
            "wavelength_um, R, T."
        )

    data = raw[
        :,
        :3,
    ]

    mask = (
        (data[:, 0] >= 1.0 - 1e-12)
        &
        (data[:, 0] <= 2.4 + 1e-12)
    )

    data = data[
        mask
    ]

    order = np.argsort(
        data[:, 0]
    )

    return data[
        order
    ]


def worker_init(
    mesh_file: str,
) -> None:

    global _WORKER_SOLVER

    cfg_template = make_cfg(
        1.6
    )

    _WORKER_SOLVER = (
        CachedDisorderedSolver(
            cfg_template,
            Path(
                mesh_file
            ),
            intorder=8,
        )
    )


def worker_solve(
    wavelength: float,
) -> dict:

    global _WORKER_SOLVER

    if _WORKER_SOLVER is None:

        raise RuntimeError(
            "Worker cache was not initialized."
        )

    cfg = make_cfg(
        wavelength
    )

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

    timing = (
        _WORKER_SOLVER.last_timings
    )

    return {
        "wavelength": float(
            wavelength
        ),
        "R": float(
            diff["R"]
        ),
        "T": float(
            diff["T"]
        ),
        "R_plus_T": float(
            diff["R_plus_T"]
        ),
        "energy_error": float(
            diff["energy_error"]
        ),
        "cutoff_flag": (
            1.0
            if diff[
                "cutoff_orders"
            ]
            else 0.0
        ),
        "solve_time": float(
            t1 - t0
        ),
        "post_time": float(
            t2 - t1
        ),
        "point_time": float(
            t2 - t0
        ),
        "source_time": float(
            timing.source_assembly
        ),
        "matrix_time": float(
            timing.matrix_build
        ),
        "linear_time": float(
            timing.linear_solve
        ),
        "reconstruction_time": float(
            timing.reconstruction
        ),
    }


def error_metrics(
    a: np.ndarray,
    b: np.ndarray,
):

    d = (
        a
        - b
    )

    ad = np.abs(
        d
    )

    idx = int(
        np.argmax(
            ad
        )
    )

    return {
        "mae": float(
            np.mean(
                ad
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    d ** 2
                )
            )
        ),
        "max": float(
            ad[
                idx
            ]
        ),
        "idx": idx,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of wavelength worker processes. "
            "Start with 2 for this ~200k-DOF problem."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Solve only 9 representative FDTD wavelengths "
            "before attempting the full 97-point sweep."
        ),
    )

    args = parser.parse_args()

    if not MESH_FILE.exists():

        raise FileNotFoundError(
            f"Could not find mesh: {MESH_FILE}"
        )

    fdtd = load_fdtd_reference()

    wavelengths = (
        fdtd[:, 0].copy()
    )

    if args.quick:

        indices = np.unique(
            np.round(
                np.linspace(
                    0,
                    len(wavelengths) - 1,
                    9,
                )
            ).astype(int)
        )

        wavelengths = wavelengths[
            indices
        ]

    print()
    print("=" * 100)
    print("DISORDERED MEDIA — CACHED PARALLEL FEM SWEEP")
    print("=" * 100)

    print(
        f"Workers             = "
        f"{args.workers}"
    )

    print(
        f"Mode                = "
        f"{'QUICK' if args.quick else 'FULL'}"
    )

    print(
        f"Wavelength points   = "
        f"{len(wavelengths)}"
    )

    print(
        f"Range               = "
        f"{wavelengths.min():.6f} "
        f"to "
        f"{wavelengths.max():.6f} um"
    )

    print(
        "Grid                = exact supplied FDTD wavelengths"
    )

    print(
        f"Mesh                = "
        f"{MESH_FILE}"
    )

    print(
        "Physics             = Ez/TE, n_disk=3, n_background=1"
    )

    print(
        "Boundaries          = periodic x, PML y, no mirror symmetry"
    )

    print("=" * 100)
    print()

    wall_start = perf_counter()

    results = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(
            str(
                MESH_FILE
            ),
        ),
    ) as executor:

        future_map = {
            executor.submit(
                worker_solve,
                float(
                    wavelength
                ),
            ): float(
                wavelength
            )

            for wavelength
            in wavelengths
        }

        completed = 0

        for future in as_completed(
            future_map
        ):

            row = future.result()

            results.append(
                row
            )

            completed += 1

            print(
                f"[{completed:03d}/{len(wavelengths):03d}] "
                f"lambda={row['wavelength']:.6f} um  "
                f"R={row['R']:.6f}  "
                f"T={row['T']:.3e}  "
                f"R+T={row['R_plus_T']:.6f}  "
                f"solve={row['solve_time']:.2f}s"
            )

    wall_time = (
        perf_counter()
        - wall_start
    )

    results.sort(
        key=lambda row: row[
            "wavelength"
        ]
    )

    w = np.asarray(
        [
            row["wavelength"]
            for row in results
        ],
        dtype=float,
    )

    R = np.asarray(
        [
            row["R"]
            for row in results
        ],
        dtype=float,
    )

    T = np.asarray(
        [
            row["T"]
            for row in results
        ],
        dtype=float,
    )

    RplusT = np.asarray(
        [
            row["R_plus_T"]
            for row in results
        ],
        dtype=float,
    )

    energy = np.asarray(
        [
            row["energy_error"]
            for row in results
        ],
        dtype=float,
    )

    cutoff = np.asarray(
        [
            row["cutoff_flag"]
            for row in results
        ],
        dtype=float,
    )

    solve_time = np.asarray(
        [
            row["solve_time"]
            for row in results
        ],
        dtype=float,
    )

    post_time = np.asarray(
        [
            row["post_time"]
            for row in results
        ],
        dtype=float,
    )

    linear_time = np.asarray(
        [
            row["linear_time"]
            for row in results
        ],
        dtype=float,
    )

    # Reference at the exact solved wavelengths.
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

    r_metrics = error_metrics(
        R,
        R_fdtd,
    )

    t_metrics = error_metrics(
        T,
        T_fdtd,
    )

    stopband = (
        (w >= 1.4)
        &
        (w <= 1.8)
    )

    print()
    print("=" * 100)
    print("SPEED")
    print("=" * 100)

    print(
        f"Wall-clock total       = "
        f"{wall_time:.3f} s "
        f"({wall_time / 60.0:.3f} min)"
    )

    print(
        f"Mean FEM solve         = "
        f"{np.mean(solve_time):.3f} s"
    )

    print(
        f"Mean sparse solve      = "
        f"{np.mean(linear_time):.3f} s"
    )

    print(
        f"Mean R/T postprocess   = "
        f"{np.mean(post_time):.3f} s"
    )

    print()
    print("=" * 100)
    print("NUMERICAL CHECK")
    print("=" * 100)

    print(
        f"Max |R+T-1|            = "
        f"{np.max(energy):.6e}"
    )

    print(
        f"Mean |R+T-1|           = "
        f"{np.mean(energy):.6e}"
    )

    print(
        f"Cutoff points          = "
        f"{int(np.sum(cutoff > 0.5))}"
    )

    print()
    print("=" * 100)
    print("FEM vs FDTD")
    print("=" * 100)

    print(
        f"R MAE                  = "
        f"{r_metrics['mae']:.6e}"
    )

    print(
        f"R RMSE                 = "
        f"{r_metrics['rmse']:.6e}"
    )

    print(
        f"R MAX                  = "
        f"{r_metrics['max']:.6e} "
        f"@ {w[r_metrics['idx']]:.6f} um"
    )

    print(
        f"T MAE                  = "
        f"{t_metrics['mae']:.6e}"
    )

    print(
        f"T RMSE                 = "
        f"{t_metrics['rmse']:.6e}"
    )

    print(
        f"T MAX                  = "
        f"{t_metrics['max']:.6e} "
        f"@ {w[t_metrics['idx']]:.6f} um"
    )

    if np.any(
        stopband
    ):

        print()
        print(
            "Stop-band region 1.4-1.8 um:"
        )

        print(
            f"  FEM mean R           = "
            f"{np.mean(R[stopband]):.8f}"
        )

        print(
            f"  FEM mean T           = "
            f"{np.mean(T[stopband]):.3e}"
        )

        print(
            f"  FDTD mean R          = "
            f"{np.mean(R_fdtd[stopband]):.8f}"
        )

        print(
            f"  FDTD mean T          = "
            f"{np.mean(T_fdtd[stopband]):.3e}"
        )

    data = np.column_stack(
        [
            w,
            R,
            T,
            RplusT,
            energy,
            cutoff,
            R_fdtd,
            T_fdtd,
            np.abs(
                R
                - R_fdtd
            ),
            np.abs(
                T
                - T_fdtd
            ),
            solve_time,
            linear_time,
            post_time,
        ]
    )

    mode_name = (
        "quick"
        if args.quick
        else "full"
    )

    csv_path = (
        OUT
        / (
            f"disordered_{mode_name}_"
            f"{args.workers}workers.csv"
        )
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
            "solve_time_s,"
            "linear_solve_time_s,"
            "post_time_s"
        ),
        comments="",
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        w,
        R,
        linewidth=2.0,
        label="FEM R",
    )

    plt.plot(
        w,
        R_fdtd,
        "--",
        linewidth=1.8,
        label="FDTD R",
    )

    plt.plot(
        w,
        T,
        linewidth=2.0,
        label="FEM T",
    )

    plt.plot(
        w,
        T_fdtd,
        "--",
        linewidth=1.8,
        label="FDTD T",
    )

    plt.xlabel(
        "Wavelength [um]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Disordered disks: FEM vs FDTD"
    )

    plt.xlim(
        1.0,
        2.4,
    )

    plt.ylim(
        -0.03,
        1.05,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        OUT
        / (
            f"disordered_{mode_name}_"
            f"{args.workers}workers_RT.png"
        )
    )

    plt.savefig(
        plot_path,
        dpi=220,
    )

    plt.close()

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"CSV  = {csv_path}"
    )

    print(
        f"Plot = {plot_path}"
    )


if __name__ == "__main__":
    main()
