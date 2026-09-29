V4: UMFPACK AUTO + BEST ordering with symbolic reuse.

Changes from V3:
- Keeps the exact K - k0^2 M pencil and shared CSC sparsity pattern.
- Requests UMFPACK STRATEGY_AUTO + ORDERING_BEST before the one-time symbolic analysis.
- Reuses that symbolic analysis for every wavelength.
- If BEST tuning is unavailable/fails, falls back to the normal UMFPACK symbolic path; SciPy remains the final safety fallback.
- Reports requested/used ordering and symbolic time in the benchmark.

Validate first:
  python validate_singlefile_solver.py

Then benchmark:
  python benchmark_optimized_solver.py
