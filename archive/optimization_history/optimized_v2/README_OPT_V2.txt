Optimization V2

- Same consolidated physics solver.
- CachedDisorderedSolver now explicitly uses scikit-umfpack low-level API when available.
- UMFPACK symbolic analysis is performed once per cached solver/process and reused for each wavelength.
- K and M CSC arrays are reused in-place when their sparsity patterns match, avoiding per-wavelength sparse matrix allocation.
- Falls back to scipy.spsolve automatically if scikit-umfpack is unavailable or K/M patterns differ.

Run validate_singlefile_solver.py first. Then run benchmark_optimized_solver.py.
