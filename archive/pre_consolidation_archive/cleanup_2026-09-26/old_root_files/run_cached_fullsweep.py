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


# ============================================================
# SETTINGS
# ============================================================

MESH = Path(
    "meshes_speedtest/mesh_28_14.msh"
)

OUT = Path(
    "results_cached_28_14"
)

OUT.mkdir(
    exist_ok=True
)

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    101,
)


# ============================================================
# BUILD CACHED FEM PROBLEM ONCE
# ============================================================

print()
print("=" * 90)
print("BUILDING CACHED OBJECTIVE 3 FEM PROBLEM")
print("=" * 90)

cfg_template = make_cfg(
    0.600
)

t0 = perf_counter()

solver = Objective3FastSolver(
    cfg_template,
    MESH,
)

setup_time = (
    perf_counter()
    - t0
)

print()
print(
    f"Cache setup time = "
    f"{setup_time:.3f} s"
)

print(
    f"Full DOFs        = "
    f"{solver.basis.N}"
)

print(
    f"Reduced DOFs     = "
    f"{solver.reduced_dofs}"
)

print(
    f"Free DOFs        = "
    f"{solver.free_reduced_dofs}"
)


# ============================================================
# FULL 101-POINT SWEEP
# ============================================================

print()
print("=" * 90)
print("CACHED 101-WAVELENGTH SWEEP")
print("=" * 90)
print()

rows = []

sweep_start = perf_counter()


for i, wavelength in enumerate(
    WAVELENGTHS,
    start=1,
):

    cfg = make_cfg(
        float(wavelength)
    )

    point_start = perf_counter()

    result = solver.solve(
        cfg
    )

    solve_finished = perf_counter()

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    point_finished = perf_counter()

    fem_time = (
        solve_finished
        - point_start
    )

    post_time = (
        point_finished
        - solve_finished
    )

    total_point_time = (
        point_finished
        - point_start
    )

    timings = solver.last_timings

    rows.append(
        [
            wavelength,

            diff["R"],
            diff["T"],
            diff["R_plus_T"],
            diff["energy_error"],
            diff["symmetry_abs_max"],

            fem_time,
            post_time,
            total_point_time,

            timings.source_assembly,
            timings.matrix_build,
            timings.linear_solve,
            timings.reconstruction,
        ]
    )

    print(
        f"[{i:03d}/101] "
        f"{wavelength * 1000:6.1f} nm  "
        f"R={diff['R']:.6f}  "
        f"T={diff['T']:.6f}  "
        f"solve={fem_time:.3f}s  "
        f"total={total_point_time:.3f}s"
    )


sweep_time = (
    perf_counter()
    - sweep_start
)

total_time = (
    setup_time
    + sweep_time
)


# ============================================================
# SAVE RESULTS
# ============================================================

data = np.asarray(
    rows,
    dtype=float,
)

np.savetxt(
    OUT / "cached_28_14_fullsweep.csv",
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


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("CACHED FULL-SWEEP SUMMARY")
print("=" * 90)

print()
print(
    f"Cache setup time    = "
    f"{setup_time:.3f} s"
)

print(
    f"Sweep runtime       = "
    f"{sweep_time:.3f} s"
)

print(
    f"Total runtime       = "
    f"{total_time:.3f} s"
)

print(
    f"Total runtime       = "
    f"{total_time / 60:.3f} min"
)

print()
print(
    f"Mean FEM solve      = "
    f"{np.mean(data[:, 6]):.4f} s"
)

print(
    f"Mean postprocess    = "
    f"{np.mean(data[:, 7]):.4f} s"
)

print(
    f"Mean point runtime  = "
    f"{np.mean(data[:, 8]):.4f} s"
)

print()
print(
    f"Mean source time    = "
    f"{np.mean(data[:, 9]):.4f} s"
)

print(
    f"Mean matrix build   = "
    f"{np.mean(data[:, 10]):.4f} s"
)

print(
    f"Mean linear solve   = "
    f"{np.mean(data[:, 11]):.4f} s"
)

print(
    f"Mean reconstruction = "
    f"{np.mean(data[:, 12]):.4f} s"
)

print()
print(
    f"Max energy error    = "
    f"{np.max(data[:, 4]):.6e}"
)

print(
    f"Mean energy error   = "
    f"{np.mean(data[:, 4]):.6e}"
)

print()
print(
    f"Saved to: "
    f"{OUT / 'cached_28_14_fullsweep.csv'}"
)

print()