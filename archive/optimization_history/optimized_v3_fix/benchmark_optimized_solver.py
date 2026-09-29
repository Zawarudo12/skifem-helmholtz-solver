from pathlib import Path
from time import perf_counter

import numpy as np

from photonics_fem_solver import CachedDisorderedSolver, DisorderedConfig, diffraction_orders

MESH = Path("meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh")
WAVELENGTHS = [1.064516129032258, 1.170212765957447, 1.5, 1.755319148936170, 2.323943661971831]

def cfg(w):
    return DisorderedConfig(
        wavelength=float(w), width=7.0, xmin=-3.5, xmax=3.5,
        scatter_ymin=-3.5, scatter_ymax=3.5,
        physical_ymin=-5.5, physical_ymax=5.5,
        pml_low=3.0, pml_high=3.0, pml_order=3, pml_sigma_max=8.0,
        n_background=1.0 + 0j, n_disk=3.0 + 0j,
    )

def main():
    t0 = perf_counter()
    solver = CachedDisorderedSolver(cfg(1.5), MESH, intorder=8)
    print(f"setup: {perf_counter()-t0:.3f} s")
    print(f"backend: {solver.linear_backend}")
    print(f"symbolic reuse: {solver.symbolic_reuse_enabled}")
    print(f"K nnz: {solver.K_nnz:,}; M nnz: {solver.M_nnz:,}; union nnz: {solver.union_nnz}")
    print(f"original patterns match: {solver.original_patterns_match}")
    if solver.backend_failure:
        print(f"UMFPACK initialization failure: {solver.backend_failure}")
    if not solver.symbolic_reuse_enabled:
        raise SystemExit("UMFPACK reuse is NOT active. Stopping before expensive solves.")
    print(f"DOFs: full={solver.basis.N:,} free={solver.free_reduced_dofs:,}")
    print()
    totals=[]
    for w in WAVELENGTHS:
        p0=perf_counter()
        result=solver.solve(cfg(w))
        p1=perf_counter()
        d=diffraction_orders(result, npoints=1024)
        p2=perf_counter()
        tm=solver.last_timings
        totals.append(p2-p0)
        print(f"lambda={w:.6f} R={d['R']:.9f} T={d['T']:.9f} source={tm.source_assembly:.3f}s matrix={tm.matrix_build:.3f}s numeric+solve={tm.linear_solve:.3f}s recon={tm.reconstruction:.3f}s post={p2-p1:.3f}s total={p2-p0:.3f}s")
    print()
    print(f"mean point total: {np.mean(totals):.3f} s")
    print(f"5-point solve wall: {np.sum(totals):.3f} s")

if __name__ == "__main__":
    main()
