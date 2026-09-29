from pathlib import Path
from time import perf_counter

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)

from photonics_fem.objective3_fast import (
    Objective3FastSolver,
)


MESH = Path(
    "meshes_speedtest/mesh_28_14.msh"
)

TEST_WAVELENGTHS = [
    0.300,
    0.415,
    0.600,
    0.720,
    0.800,
]


print()
print("=" * 88)
print("CACHED OBJECTIVE 3 SOLVER VALIDATION")
print("=" * 88)

# ---------------------------------------------------------------------
# Build the cache once.
# Geometry/material/PML settings come from the normal 600 nm config.
# ---------------------------------------------------------------------

cfg_template = make_cfg(
    0.600
)

print()
print("Building cached FEM problem...")

t0 = perf_counter()

fast_solver = Objective3FastSolver(
    cfg_template,
    MESH,
)

cache_time = (
    perf_counter()
    - t0
)

print(
    f"Cache setup time = "
    f"{cache_time:.6f} s"
)

print(
    f"Static matrix setup inside cache = "
    f"{fast_solver.static_matrix_setup_time:.6f} s"
)

print(
    f"Full DOFs = "
    f"{fast_solver.basis.N}"
)

print(
    f"Reduced DOFs = "
    f"{fast_solver.reduced_dofs}"
)

print(
    f"Free reduced DOFs = "
    f"{fast_solver.free_reduced_dofs}"
)


max_dR = 0.0
max_dT = 0.0

old_total = 0.0
fast_total = 0.0


for wavelength in TEST_WAVELENGTHS:

    cfg = make_cfg(
        wavelength
    )

    print()
    print("-" * 88)
    print(
        f"Wavelength = "
        f"{wavelength * 1000:.1f} nm"
    )
    print("-" * 88)

    # -----------------------------------------------------------------
    # Trusted original solver
    # -----------------------------------------------------------------

    t_old = perf_counter()

    old_result = solve_objective3_bodyfitted(
        cfg,
        MESH,
    )

    old_diff = diffraction_orders_bodyfitted(
        old_result,
        cfg,
        npoints=512,
    )

    old_time = (
        perf_counter()
        - t_old
    )

    old_total += old_time

    # -----------------------------------------------------------------
    # New cached solver
    # -----------------------------------------------------------------

    t_fast = perf_counter()

    fast_result = fast_solver.solve(
        cfg
    )

    fast_diff = diffraction_orders_bodyfitted(
        fast_result,
        cfg,
        npoints=512,
    )

    fast_time = (
        perf_counter()
        - t_fast
    )

    fast_total += fast_time

    # -----------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------

    dR = abs(
        fast_diff["R"]
        - old_diff["R"]
    )

    dT = abs(
        fast_diff["T"]
        - old_diff["T"]
    )

    max_dR = max(
        max_dR,
        dR,
    )

    max_dT = max(
        max_dT,
        dT,
    )

    print(
        f"ORIGINAL  "
        f"R={old_diff['R']:.10f}  "
        f"T={old_diff['T']:.10f}  "
        f"time={old_time:.4f}s"
    )

    print(
        f"CACHED    "
        f"R={fast_diff['R']:.10f}  "
        f"T={fast_diff['T']:.10f}  "
        f"time={fast_time:.4f}s"
    )

    print(
        f"|dR| = {dR:.3e}"
    )

    print(
        f"|dT| = {dT:.3e}"
    )

    print()
    print(
        fast_solver.timing_report()
    )


print()
print("=" * 88)
print("SUMMARY")
print("=" * 88)

print(
    f"Max |dR| = "
    f"{max_dR:.6e}"
)

print(
    f"Max |dT| = "
    f"{max_dT:.6e}"
)

print()

print(
    f"Original total for test wavelengths = "
    f"{old_total:.3f} s"
)

print(
    f"Cached total for test wavelengths   = "
    f"{fast_total:.3f} s"
)

print(
    f"Cached speedup after setup           = "
    f"{old_total / fast_total:.2f}x"
)

print()

if (
    max_dR < 1e-10
    and max_dT < 1e-10
):
    print(
        "RESULT: PASS - cached solver is numerically equivalent "
        "to the trusted solver at the tested wavelengths."
    )
else:
    print(
        "RESULT: CHECK - differences are larger than 1e-10. "
        "Do not use the cached solver for production yet."
    )
