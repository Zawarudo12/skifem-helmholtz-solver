V3: Align K and M on the union of their CSC sparsity patterns.
This allows UMFPACK symbolic reuse even if K and M have different patterns.
A symbolic initialization failure is printed explicitly; the benchmark exits
rather than silently benchmarking the fallback.

From project root, after backing up V2:
  Copy-Item ".\optimized_v3\photonics_fem_solver.py" ".\photonics_fem_solver.py" -Force
  Copy-Item ".\optimized_v3\benchmark_optimized_solver.py" ".\benchmark_optimized_solver.py" -Force
  python validate_singlefile_solver.py
  python benchmark_optimized_solver.py

Do not remove the original photonics_fem folder yet.  The real FEM benchmark
must be run in the local femwell environment (not available in this sandbox).
