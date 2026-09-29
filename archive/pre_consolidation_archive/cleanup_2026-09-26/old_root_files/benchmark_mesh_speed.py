from pathlib import Path
from time import perf_counter

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    build_bodyfitted_gmsh_mesh,
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)


cfg = make_cfg(0.600)

REFERENCE_R = 0.1842511785
REFERENCE_T = 0.8157459920

tests = [
    ("18_8",  0.018,  0.008),
    ("22_10", 0.022, 0.010),
    ("25_12", 0.025, 0.012),
    ("30_15", 0.030, 0.015),
]

mesh_dir = Path("meshes_speedtest")
mesh_dir.mkdir(exist_ok=True)

print()
print("=" * 100)
print("MESH SPEED / ACCURACY BENCHMARK @ 600 nm")
print("=" * 100)

for name, h_bulk, h_hole in tests:

    mesh_file = mesh_dir / f"mesh_{name}.msh"

    print()
    print("-" * 100)
    print(
        f"Mesh {name}: "
        f"h_bulk={h_bulk*1000:.1f} nm, "
        f"h_hole={h_hole*1000:.1f} nm"
    )

    # Build mesh only if it doesn't already exist
    build_bodyfitted_gmsh_mesh(
        mesh_file,
        cfg,
        h_bulk=h_bulk,
        h_hole=h_hole,
        refine_distance=0.20,
        force=False,
    )

    t0 = perf_counter()

    result = solve_objective3_bodyfitted(
        cfg,
        mesh_file,
    )

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    runtime = perf_counter() - t0

    R = diff["R"]
    T = diff["T"]

    dR = abs(R - REFERENCE_R)
    dT = abs(T - REFERENCE_T)

    print(f"DOFs       = {result.scattered.basis.N}")
    print(f"Runtime    = {runtime:.3f} s")
    print(f"R          = {R:.10f}")
    print(f"T          = {T:.10f}")
    print(f"R + T      = {diff['R_plus_T']:.10f}")
    print(f"|dR|       = {dR:.6e}")
    print(f"|dT|       = {dT:.6e}")