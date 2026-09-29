from pathlib import Path
from time import perf_counter

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)


MESH = Path(
    "meshes_speedtest/mesh_28_14.msh"
)

WAVELENGTH = 0.600


print()
print("=" * 70)
print("28/14 nm FAST-MESH SINGLE-WAVELENGTH BENCHMARK")
print("=" * 70)

cfg = make_cfg(
    WAVELENGTH
)

t0 = perf_counter()

result = solve_objective3_bodyfitted(
    cfg,
    MESH,
)

t1 = perf_counter()

diff = diffraction_orders_bodyfitted(
    result,
    cfg,
    npoints=512,
)

t2 = perf_counter()


solve_time = t1 - t0
post_time = t2 - t1
total_time = t2 - t0


print()
print(f"Wavelength      = {WAVELENGTH * 1000:.1f} nm")
print(f"DOFs            = {result.scattered.basis.N}")

print()
print(f"FEM solve time  = {solve_time:.6f} s")
print(f"R/T postprocess = {post_time:.6f} s")
print(f"Total runtime   = {total_time:.6f} s")

print()
print(f"R               = {diff['R']:.10f}")
print(f"T               = {diff['T']:.10f}")
print(f"R + T           = {diff['R_plus_T']:.10f}")

print()
print("=" * 70)