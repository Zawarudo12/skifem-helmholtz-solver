from __future__ import annotations

# IMPORTANT:
# Keep numerical libraries single-threaded inside each worker process.
# We parallelize over wavelengths instead.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    diffraction_orders_bodyfitted,
)

from photonics_fem.objective3_fast import (
    Objective3FastSolver,
)


# ============================================================
# SETTINGS
# ============================================================

WORKERS = 4

MESH = Path(
    "meshes_speedtest/mesh_28_14.msh"
)

COMSOL_FILE = Path(
    "Comsol1.csv"
)

FDTD_FILE = Path(
    "dataset_hollow_core_fdtd(1).csv"
)

# Optional: trusted 18/8 scikit-fem spectrum from the old run.
REFERENCE_FEM_FILE = Path(
    "results_objective3_redo/objective3_bodyfitted_total_RT.csv"
)

OUT = Path(
    "results_fast_threeway"
)

OUT.mkdir(
    exist_ok=True
)

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    101,
)

# Previous measured workflow timings.
OLD_FULL_WORKFLOW_SECONDS = 1512.8988
UNCACHED_28_14_SECONDS = 213.0
CACHED_SERIAL_SECONDS = 147.640


# ============================================================
# WORKER CACHE
# ============================================================

_WORKER_SOLVER = None


def _worker_init(
    mesh_file: str,
) -> None:

    global _WORKER_SOLVER

    cfg_template = make_cfg(
        0.600
    )

    _WORKER_SOLVER = Objective3FastSolver(
        cfg_template,
        Path(mesh_file),
    )


def _worker_solve(
    wavelength: float,
) -> dict:

    global _WORKER_SOLVER

    if _WORKER_SOLVER is None:
        raise RuntimeError(
            "Worker solver cache was not initialized."
        )

    cfg = make_cfg(
        float(wavelength)
    )

    t0 = perf_counter()

    result = _WORKER_SOLVER.solve(
        cfg
    )

    t1 = perf_counter()

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    t2 = perf_counter()

    timings = _WORKER_SOLVER.last_timings

    propagating_orders = tuple(
        int(row["m"])
        for row in diff["orders"]
        if row["propagating"]
    )

    cutoff_orders = tuple(
        int(m)
        for m in diff["cutoff_orders"]
    )

    return {
        "wavelength": float(wavelength),
        "R": float(diff["R"]),
        "T": float(diff["T"]),
        "R_plus_T": float(diff["R_plus_T"]),
        "energy_error": float(diff["energy_error"]),
        "symmetry_error": float(diff["symmetry_abs_max"]),
        "cutoff_flag": 1.0 if cutoff_orders else 0.0,
        "propagating_orders": propagating_orders,
        "cutoff_orders": cutoff_orders,
        "fem_time": float(t1 - t0),
        "post_time": float(t2 - t1),
        "total_point_time": float(t2 - t0),
        "source_time": float(timings.source_assembly),
        "matrix_build_time": float(timings.matrix_build),
        "linear_solve_time": float(timings.linear_solve),
        "reconstruction_time": float(timings.reconstruction),
    }


# ============================================================
# REFERENCE LOADERS
# Same format as the original comparison script.
# ============================================================

def load_fdtd_reference():

    if not FDTD_FILE.exists():

        print()
        print(
            f"FDTD file not found: {FDTD_FILE}"
        )

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

    if (
        raw.ndim != 2
        or raw.shape[1] < 3
    ):
        raise ValueError(
            "FDTD file must contain at least "
            "three columns: wavelength_um, R, T."
        )

    fdtd = raw[:, :3]

    order = np.argsort(
        fdtd[:, 0]
    )

    return fdtd[
        order
    ]


def load_comsol_reference():

    if not COMSOL_FILE.exists():

        print()
        print(
            f"COMSOL file not found: {COMSOL_FILE}"
        )

        return None

    # Original COMSOL export:
    # column 0 = wavelength [um]
    # column 2 = TOTAL reflectance
    # column 4 = TOTAL transmittance
    # lines beginning with % are metadata/header rows.
    raw = np.loadtxt(
        COMSOL_FILE,
        delimiter=",",
        comments="%",
    )

    if raw.ndim == 1:
        raw = raw.reshape(
            1,
            -1,
        )

    if (
        raw.ndim != 2
        or raw.shape[1] < 5
    ):
        raise ValueError(
            "COMSOL CSV must contain wavelength "
            "plus total R/T columns."
        )

    comsol = np.column_stack(
        [
            raw[:, 0],
            raw[:, 2],
            raw[:, 4],
        ]
    )

    order = np.argsort(
        comsol[:, 0]
    )

    return comsol[
        order
    ]


def load_reference_fem():

    if not REFERENCE_FEM_FILE.exists():
        return None

    raw = np.loadtxt(
        REFERENCE_FEM_FILE,
        delimiter=",",
        skiprows=1,
    )

    if raw.ndim == 1:
        raw = raw.reshape(
            1,
            -1,
        )

    if raw.shape[1] < 3:
        return None

    return raw[:, :3]


# ============================================================
# METRICS
# ============================================================

def error_metrics(
    a: np.ndarray,
    b: np.ndarray,
):

    error = (
        a
        - b
    )

    abs_error = np.abs(
        error
    )

    max_index = int(
        np.argmax(
            abs_error
        )
    )

    return {
        "mae": float(
            np.mean(
                abs_error
            )
        ),

        "rmse": float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),

        "max": float(
            abs_error[
                max_index
            ]
        ),

        "max_index": max_index,
    }


def print_pair_metrics(
    title: str,
    wavelength_um: np.ndarray,
    R_a: np.ndarray,
    T_a: np.ndarray,
    R_b: np.ndarray,
    T_b: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:

    if mask is None:
        mask = np.ones(
            wavelength_um.shape,
            dtype=bool,
        )

    w = wavelength_um[
        mask
    ]

    r = error_metrics(
        R_a[mask],
        R_b[mask],
    )

    t = error_metrics(
        T_a[mask],
        T_b[mask],
    )

    print()
    print(title)

    print(
        f"  R: MAE={r['mae']:.6e}  "
        f"RMSE={r['rmse']:.6e}  "
        f"MAX={r['max']:.6e} "
        f"@ {w[r['max_index']] * 1000.0:.1f} nm"
    )

    print(
        f"  T: MAE={t['mae']:.6e}  "
        f"RMSE={t['rmse']:.6e}  "
        f"MAX={t['max']:.6e} "
        f"@ {w[t['max_index']] * 1000.0:.1f} nm"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 100)
    print("OBJECTIVE 3 — FAST SCIKIT-FEM vs COMSOL vs FDTD")
    print("=" * 100)

    print(
        f"Workers       = {WORKERS}"
    )

    print(
        f"Mesh          = {MESH}"
    )

    print(
        f"Wavelengths   = {len(WAVELENGTHS)} "
        "(300-800 nm, 5 nm step)"
    )

    print("=" * 100)
    print()

    if not MESH.exists():
        raise FileNotFoundError(
            f"Fast mesh not found: {MESH}"
        )

    # --------------------------------------------------------
    # Run NEW scikit-fem sweep.
    # --------------------------------------------------------

    wall_start = perf_counter()

    results = []

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=_worker_init,
        initargs=(str(MESH),),
    ) as executor:

        future_map = {
            executor.submit(
                _worker_solve,
                float(wavelength),
            ): float(wavelength)

            for wavelength in WAVELENGTHS
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
                f"[{completed:03d}/101] "
                f"{row['wavelength'] * 1000.0:6.1f} nm  "
                f"R={row['R']:.6f}  "
                f"T={row['T']:.6f}  "
                f"solve={row['fem_time']:.3f}s"
            )

    wall_time = (
        perf_counter()
        - wall_start
    )

    results.sort(
        key=lambda x: x["wavelength"]
    )

    fem_w = np.asarray(
        [
            row["wavelength"]
            for row in results
        ],
        dtype=float,
    )

    fem_R = np.asarray(
        [
            row["R"]
            for row in results
        ],
        dtype=float,
    )

    fem_T = np.asarray(
        [
            row["T"]
            for row in results
        ],
        dtype=float,
    )

    fem_RplusT = np.asarray(
        [
            row["R_plus_T"]
            for row in results
        ],
        dtype=float,
    )

    energy_error = np.asarray(
        [
            row["energy_error"]
            for row in results
        ],
        dtype=float,
    )

    symmetry_error = np.asarray(
        [
            row["symmetry_error"]
            for row in results
        ],
        dtype=float,
    )

    cutoff_flag = np.asarray(
        [
            row["cutoff_flag"]
            for row in results
        ],
        dtype=float,
    )

    point_time = np.asarray(
        [
            row["total_point_time"]
            for row in results
        ],
        dtype=float,
    )

    linear_time = np.asarray(
        [
            row["linear_solve_time"]
            for row in results
        ],
        dtype=float,
    )

    noncut = (
        cutoff_flag
        < 0.5
    )

    # --------------------------------------------------------
    # Load COMSOL + FDTD.
    # --------------------------------------------------------

    comsol = load_comsol_reference()
    fdtd = load_fdtd_reference()
    reference_fem = load_reference_fem()

    comsol_R = None
    comsol_T = None
    fdtd_R = None
    fdtd_T = None

    if comsol is not None:

        comsol_R = np.interp(
            fem_w,
            comsol[:, 0],
            comsol[:, 1],
        )

        comsol_T = np.interp(
            fem_w,
            comsol[:, 0],
            comsol[:, 2],
        )

    if fdtd is not None:

        fdtd_R = np.interp(
            fem_w,
            fdtd[:, 0],
            fdtd[:, 1],
        )

        fdtd_T = np.interp(
            fem_w,
            fdtd[:, 0],
            fdtd[:, 2],
        )

    # --------------------------------------------------------
    # SPEED REPORT
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("SPEED")
    print("=" * 100)

    print(
        f"NEW 4-worker wall time     = "
        f"{wall_time:.3f} s"
    )

    print(
        f"NEW 4-worker wall time     = "
        f"{wall_time / 60.0:.3f} min"
    )

    print(
        f"Mean point work time       = "
        f"{np.mean(point_time):.4f} s"
    )

    print(
        f"Mean linear-solve time     = "
        f"{np.mean(linear_time):.4f} s"
    )

    print()
    print(
        f"vs old full workflow       = "
        f"{OLD_FULL_WORKFLOW_SECONDS / wall_time:.2f}x "
        "less wall-clock time"
    )

    print(
        f"vs uncached 28/14 sweep    = "
        f"{UNCACHED_28_14_SECONDS / wall_time:.2f}x"
    )

    print(
        f"vs cached serial sweep     = "
        f"{CACHED_SERIAL_SECONDS / wall_time:.2f}x"
    )

    # --------------------------------------------------------
    # INTERNAL NUMERICAL DIAGNOSTICS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("SCIKIT-FEM INTERNAL NUMERICAL CHECKS")
    print("=" * 100)

    print(
        f"Max |R+T-1|, all points   = "
        f"{np.max(energy_error):.6e}"
    )

    print(
        f"Max |R+T-1|, non-cutoff   = "
        f"{np.max(energy_error[noncut]):.6e}"
    )

    print(
        f"Mean |R+T-1|, non-cutoff  = "
        f"{np.mean(energy_error[noncut]):.6e}"
    )

    print(
        f"Max symmetry error         = "
        f"{np.max(symmetry_error):.6e}"
    )

    cutoff_w = (
        fem_w[
            ~noncut
        ]
        * 1000.0
    )

    if cutoff_w.size:
        print(
            "Exact cutoff wavelength(s)  = "
            + ", ".join(
                f"{x:.1f} nm"
                for x in cutoff_w
            )
        )

    # --------------------------------------------------------
    # THREE-SOLVER ACCURACY
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("THREE-SOLVER ACCURACY")
    print("=" * 100)

    if comsol_R is not None:

        print_pair_metrics(
            "scikit-fem vs COMSOL — all 101 points",
            fem_w,
            fem_R,
            fem_T,
            comsol_R,
            comsol_T,
        )

        print_pair_metrics(
            "scikit-fem vs COMSOL — excluding exact Rayleigh cutoff",
            fem_w,
            fem_R,
            fem_T,
            comsol_R,
            comsol_T,
            mask=noncut,
        )

    if fdtd_R is not None:

        print_pair_metrics(
            "scikit-fem vs FDTD — all 101 points",
            fem_w,
            fem_R,
            fem_T,
            fdtd_R,
            fdtd_T,
        )

        print_pair_metrics(
            "scikit-fem vs FDTD — excluding exact Rayleigh cutoff",
            fem_w,
            fem_R,
            fem_T,
            fdtd_R,
            fdtd_T,
            mask=noncut,
        )

    if (
        comsol_R is not None
        and fdtd_R is not None
    ):

        print_pair_metrics(
            "COMSOL vs FDTD — all 101 points",
            fem_w,
            comsol_R,
            comsol_T,
            fdtd_R,
            fdtd_T,
        )

    # --------------------------------------------------------
    # FAST MESH vs TRUSTED 18/8 SCIKIT-FEM
    # --------------------------------------------------------

    if reference_fem is not None:

        ref_R = np.interp(
            fem_w,
            reference_fem[:, 0],
            reference_fem[:, 1],
        )

        ref_T = np.interp(
            fem_w,
            reference_fem[:, 0],
            reference_fem[:, 2],
        )

        print()
        print("=" * 100)
        print("FAST 28/14 MESH vs TRUSTED 18/8 SCIKIT-FEM")
        print("=" * 100)

        print_pair_metrics(
            "28/14 vs 18/8",
            fem_w,
            fem_R,
            fem_T,
            ref_R,
            ref_T,
        )

    # --------------------------------------------------------
    # SAVE COMBINED CSV
    # --------------------------------------------------------

    columns = [
        fem_w,
        fem_R,
        fem_T,
        fem_RplusT,
        energy_error,
        symmetry_error,
        cutoff_flag,
        point_time,
        linear_time,
    ]

    headers = [
        "wavelength_um",
        "R_skfem_fast",
        "T_skfem_fast",
        "R_plus_T_skfem",
        "energy_error",
        "symmetry_error",
        "cutoff_flag",
        "point_work_time_s",
        "linear_solve_time_s",
    ]

    if comsol_R is not None:

        columns.extend(
            [
                comsol_R,
                comsol_T,
                np.abs(
                    fem_R
                    - comsol_R
                ),
                np.abs(
                    fem_T
                    - comsol_T
                ),
            ]
        )

        headers.extend(
            [
                "R_comsol",
                "T_comsol",
                "abs_dR_skfem_comsol",
                "abs_dT_skfem_comsol",
            ]
        )

    if fdtd_R is not None:

        columns.extend(
            [
                fdtd_R,
                fdtd_T,
                np.abs(
                    fem_R
                    - fdtd_R
                ),
                np.abs(
                    fem_T
                    - fdtd_T
                ),
            ]
        )

        headers.extend(
            [
                "R_fdtd",
                "T_fdtd",
                "abs_dR_skfem_fdtd",
                "abs_dT_skfem_fdtd",
            ]
        )

    combined = np.column_stack(
        columns
    )

    np.savetxt(
        OUT
        / "fast_skfem_comsol_fdtd.csv",
        combined,
        delimiter=",",
        header=",".join(
            headers
        ),
        comments="",
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    wavelength_nm = (
        fem_w
        * 1000.0
    )

    # Reflectance
    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        wavelength_nm,
        fem_R,
        linewidth=2.0,
        label="scikit-fem fast R",
    )

    if comsol_R is not None:
        plt.plot(
            wavelength_nm,
            comsol_R,
            "--",
            linewidth=2.0,
            label="COMSOL R",
        )

    if fdtd_R is not None:
        plt.plot(
            wavelength_nm,
            fdtd_R,
            "-.",
            linewidth=2.0,
            label="FDTD R",
        )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(
        "Objective 3: fast scikit-fem vs COMSOL vs FDTD"
    )

    plt.xlim(
        300,
        800,
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_R_fast_skfem_comsol_fdtd.png",
        dpi=200,
    )

    plt.close()

    # Transmittance
    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        wavelength_nm,
        fem_T,
        linewidth=2.0,
        label="scikit-fem fast T",
    )

    if comsol_T is not None:
        plt.plot(
            wavelength_nm,
            comsol_T,
            "--",
            linewidth=2.0,
            label="COMSOL T",
        )

    if fdtd_T is not None:
        plt.plot(
            wavelength_nm,
            fdtd_T,
            "-.",
            linewidth=2.0,
            label="FDTD T",
        )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Transmittance"
    )

    plt.title(
        "Objective 3: fast scikit-fem vs COMSOL vs FDTD"
    )

    plt.xlim(
        300,
        800,
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "02_T_fast_skfem_comsol_fdtd.png",
        dpi=200,
    )

    plt.close()

    # Error vs references
    plt.figure(
        figsize=(10, 6)
    )

    if comsol_R is not None:
        plt.semilogy(
            wavelength_nm,
            np.abs(
                fem_R
                - comsol_R
            ),
            label="|R FEM-COMSOL|",
        )

        plt.semilogy(
            wavelength_nm,
            np.abs(
                fem_T
                - comsol_T
            ),
            label="|T FEM-COMSOL|",
        )

    if fdtd_R is not None:
        plt.semilogy(
            wavelength_nm,
            np.abs(
                fem_R
                - fdtd_R
            ),
            "--",
            label="|R FEM-FDTD|",
        )

        plt.semilogy(
            wavelength_nm,
            np.abs(
                fem_T
                - fdtd_T
            ),
            "--",
            label="|T FEM-FDTD|",
        )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Absolute difference"
    )

    plt.title(
        "Fast scikit-fem error vs external solvers"
    )

    plt.xlim(
        300,
        800,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "03_absolute_error_vs_comsol_fdtd.png",
        dpi=200,
    )

    plt.close()

    # Timing
    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        wavelength_nm,
        point_time,
        label="point work time",
    )

    plt.plot(
        wavelength_nm,
        linear_time,
        label="linear solve time",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Time [s]"
    )

    plt.title(
        "Fast scikit-fem per-wavelength timing"
    )

    plt.xlim(
        300,
        800,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "04_fast_solver_timing.png",
        dpi=200,
    )

    plt.close()

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"Results folder: {OUT}"
    )

    print(
        "CSV: "
        f"{OUT / 'fast_skfem_comsol_fdtd.csv'}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
