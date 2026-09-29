from time import perf_counter

from run_objective3_compare_all import (
    BASE_MESH,
    build_meshes,
    solve_one,
)

print("Loading/building mesh...")
t0 = perf_counter()

build_meshes()

t1 = perf_counter()

print(f"Mesh preparation: {t1 - t0:.3f} s")

print()
print("Solving 600 nm...")

t2 = perf_counter()

cfg, result, diff = solve_one(
    0.600,
    BASE_MESH,
)

t3 = perf_counter()

print()
print("=" * 60)
print("SINGLE-WAVELENGTH BENCHMARK")
print("=" * 60)
print(f"Runtime = {t3 - t2:.3f} s")
print(f"R       = {diff['R']:.10f}")
print(f"T       = {diff['T']:.10f}")
print(f"R + T   = {diff['R_plus_T']:.10f}")
print(f"DOFs    = {result.scattered.basis.N}")