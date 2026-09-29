from __future__ import annotations

# Keep each worker to one BLAS/OpenMP thread.
# This avoids CPU oversubscription when several Python processes solve at once.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    diffraction_orders_bodyfitted,
)

from photonics_fem.objective3_fast import (
    Objective3FastSolver,
)


# ---------------------------------------------------------------------
# Worker-global cached solver.
#
# On Windows, every worker is a separate process.  Each process builds
# ONE cache and then reuses it for every wavelength assigned to it.
# ---------------------------------------------------------------------

_WORKER_SOLVER = None


def _worker_init(mesh_file: str) -> None:
    global _WORKER_SOLVER

    cfg_template = make_cfg(0.600)

    _WORKER_SOLVER = Objective3FastSolver(
        cfg_template,
        Path(mesh_file),
    )


def _worker_solve(wavelength: float) -> dict:
    global _WORKER_SOLVER

    if _WORKER_SOLVER is None:
        raise RuntimeError(
            "Worker cache was not initialized."
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

    return {
        "wavelength": float(wavelength),

        "R": float(diff["R"]),
        "T": float(diff["T"]),
        "R_plus_T": float(diff["R_plus_T"]),
        "energy_error": float(diff["energy_error"]),
        "symmetry_error": float(diff["symmetry_abs_max"]),

        "fem_time": float(t1 - t0),
        "post_time": float(t2 - t1),
        "total_point_time": float(t2 - t0),

        "source_time": float(timings.source_assembly),
        "matrix_build_time": float(timings.matrix_build),
        "linear_solve_time": float(timings.linear_solve),
        "reconstruction_time": float(timings.reconstruction),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Parallel cached Objective 3 wavelength sweep."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of worker processes. "
            "Start with 2 on Windows."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run only 21 wavelengths as a stability/speed test."
        ),
    )

    args = parser.parse_args()

    workers = max(
        1,
        int(args.workers),
    )

    mesh = Path(
        "meshes_speedtest/mesh_28_14.msh"
    )

    if not mesh.exists():
        raise FileNotFoundError(
            f"Mesh not found: {mesh}"
        )

    if args.quick:
        # 300, 325, ..., 800 nm
        wavelengths = np.linspace(
            0.300,
            0.800,
            21,
        )

        mode_name = "QUICK 21-POINT TEST"

    else:
        # 300, 305, ..., 800 nm
        wavelengths = np.linspace(
            0.300,
            0.800,
            101,
        )

        mode_name = "FULL 101-POINT SWEEP"

    out = Path(
        "results_parallel_28_14"
    )

    out.mkdir(
        exist_ok=True
    )

    print()
    print("=" * 96)
    print("PARALLEL CACHED OBJECTIVE 3")
    print("=" * 96)
    print(
        f"Mode       : {mode_name}"
    )
    print(
        f"Workers    : {workers}"
    )
    print(
        f"Mesh       : {mesh}"
    )
    print(
        f"Wavelengths: {len(wavelengths)}"
    )
    print("=" * 96)
    print()

    wall_start = perf_counter()

    results = []

    # Windows requires this entire executor section to be under
    # if __name__ == "__main__", which main() is.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(mesh),),
    ) as executor:

        future_to_wavelength = {
            executor.submit(
                _worker_solve,
                float(wavelength),
            ): float(wavelength)

            for wavelength in wavelengths
        }

        completed = 0

        for future in as_completed(
            future_to_wavelength
        ):

            wavelength = future_to_wavelength[
                future
            ]

            try:
                row = future.result()

            except Exception as exc:
                print()
                print(
                    f"FAILED at "
                    f"{wavelength * 1000:.1f} nm"
                )
                raise exc

            results.append(
                row
            )

            completed += 1

            print(
                f"[{completed:03d}/{len(wavelengths):03d}] "
                f"{row['wavelength'] * 1000:6.1f} nm  "
                f"R={row['R']:.6f}  "
                f"T={row['T']:.6f}  "
                f"solve={row['fem_time']:.3f}s"
            )

    wall_time = (
        perf_counter()
        - wall_start
    )

    # Restore wavelength order because futures complete out of order.
    results.sort(
        key=lambda x: x["wavelength"]
    )

    data = np.asarray(
        [
            [
                row["wavelength"],
                row["R"],
                row["T"],
                row["R_plus_T"],
                row["energy_error"],
                row["symmetry_error"],
                row["fem_time"],
                row["post_time"],
                row["total_point_time"],
                row["source_time"],
                row["matrix_build_time"],
                row["linear_solve_time"],
                row["reconstruction_time"],
            ]

            for row in results
        ],
        dtype=float,
    )

    if args.quick:
        filename = (
            out
            / f"parallel_quick_{workers}workers.csv"
        )
    else:
        filename = (
            out
            / f"parallel_full_{workers}workers.csv"
        )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R,"
            "T,"
            "R_plus_T,"
            "energy_error,"
            "symmetry_error,"
            "fem_time_s,"
            "postprocess_time_s,"
            "total_point_time_s,"
            "source_time_s,"
            "matrix_build_time_s,"
            "linear_solve_time_s,"
            "reconstruction_time_s"
        ),
        comments="",
    )

    # -----------------------------------------------------------------
    # Compare against the already-computed serial cached sweep if it
    # exists.  This proves that multiprocessing changed only scheduling.
    # -----------------------------------------------------------------

    serial_file = Path(
        "results_cached_28_14/cached_28_14_fullsweep.csv"
    )

    max_dR = None
    max_dT = None

    if serial_file.exists():

        serial = np.loadtxt(
            serial_file,
            delimiter=",",
            skiprows=1,
        )

        serial_w = serial[:, 0]
        serial_R = serial[:, 1]
        serial_T = serial[:, 2]

        reference_R = np.interp(
            data[:, 0],
            serial_w,
            serial_R,
        )

        reference_T = np.interp(
            data[:, 0],
            serial_w,
            serial_T,
        )

        max_dR = float(
            np.max(
                np.abs(
                    data[:, 1]
                    - reference_R
                )
            )
        )

        max_dT = float(
            np.max(
                np.abs(
                    data[:, 2]
                    - reference_T
                )
            )
        )

    # Sum of per-task times estimates how long those same tasks would
    # take if they were executed one after another.
    serial_work = float(
        np.sum(
            data[:, 8]
        )
    )

    effective_speedup = (
        serial_work
        / wall_time
    )

    print()
    print("=" * 96)
    print("PARALLEL SWEEP SUMMARY")
    print("=" * 96)

    print(
        f"Workers                 = "
        f"{workers}"
    )

    print(
        f"Wavelength count        = "
        f"{len(wavelengths)}"
    )

    print()
    print(
        f"Wall-clock runtime      = "
        f"{wall_time:.3f} s"
    )

    print(
        f"Wall-clock runtime      = "
        f"{wall_time / 60.0:.3f} min"
    )

    print()
    print(
        f"Sum of point runtimes   = "
        f"{serial_work:.3f} s"
    )

    print(
        f"Effective parallel gain = "
        f"{effective_speedup:.2f}x"
    )

    print()
    print(
        f"Mean FEM solve          = "
        f"{np.mean(data[:, 6]):.4f} s"
    )

    print(
        f"Mean postprocess        = "
        f"{np.mean(data[:, 7]):.4f} s"
    )

    print(
        f"Mean linear solve       = "
        f"{np.mean(data[:, 11]):.4f} s"
    )

    print()
    print(
        f"Max energy error        = "
        f"{np.max(data[:, 4]):.6e}"
    )

    if (
        max_dR is not None
        and max_dT is not None
    ):
        print()
        print(
            f"Max |dR| vs serial      = "
            f"{max_dR:.6e}"
        )

        print(
            f"Max |dT| vs serial      = "
            f"{max_dT:.6e}"
        )

        if (
            max_dR < 1e-10
            and max_dT < 1e-10
        ):
            print(
                "Numerical check          = PASS"
            )
        else:
            print(
                "Numerical check          = CHECK"
            )

    print()
    print(
        f"Saved                    = "
        f"{filename}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
