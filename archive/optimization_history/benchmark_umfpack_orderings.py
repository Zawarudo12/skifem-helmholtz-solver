from pathlib import Path
from time import perf_counter
import numpy as np
import scikits.umfpack as umf

from photonics_fem_solver import CachedDisorderedSolver, DisorderedConfig, assemble_scattered_source

MESH = Path('meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh')
LAM = 1.5

cfg0 = DisorderedConfig(
    wavelength=1.5,
    width=7.0,
    xmin=-3.5,
    xmax=3.5,
    scatter_ymin=-3.5,
    scatter_ymax=3.5,
    physical_ymin=-5.5,
    physical_ymax=5.5,
    pml_low=3.0,
    pml_high=3.0,
    pml_sigma_max=8.0,
    pml_order=3,
    n_background=1.0,
    n_disk=3.0,
)

print('Building cached FEM system once...')
t0 = perf_counter()
solver = CachedDisorderedSolver(cfg0, MESH, intorder=8)
print(f'setup: {perf_counter()-t0:.3f} s')
print(f'DOFs: {solver.free_reduced_dofs:,}; union nnz: {solver._A_work.nnz:,}')

cfg = DisorderedConfig(**{**cfg0.__dict__, 'wavelength': LAM})
b_full = assemble_scattered_source(solver.basis, cfg)
b_red = np.asarray(solver.PH @ b_full, dtype=np.complex128).reshape(-1)
b = b_red[solver.free_red]
A = solver._A_work.copy()
A.data[:] = solver._K_aligned - cfg.k0**2 * solver._M_aligned

strategy_i = getattr(umf, 'UMFPACK_STRATEGY', 5)
ordering_i = getattr(umf, 'UMFPACK_ORDERING', 10)
strategy_used_i = getattr(umf, 'UMFPACK_STRATEGY_USED', 18)
ordering_used_i = getattr(umf, 'UMFPACK_ORDERING_USED', 19)
lnz_i = getattr(umf, 'UMFPACK_LNZ', 43)
unz_i = getattr(umf, 'UMFPACK_UNZ', 44)

strategies = [
    ('AUTO', getattr(umf, 'UMFPACK_STRATEGY_AUTO', 0)),
    ('UNSYMMETRIC', getattr(umf, 'UMFPACK_STRATEGY_UNSYMMETRIC', 1)),
    ('SYMMETRIC', getattr(umf, 'UMFPACK_STRATEGY_SYMMETRIC', 3)),
]
orderings = [
    ('AMD', getattr(umf, 'UMFPACK_ORDERING_AMD', 1)),
    ('METIS', getattr(umf, 'UMFPACK_ORDERING_METIS', 3)),
    ('BEST', getattr(umf, 'UMFPACK_ORDERING_BEST', 4)),
]

reference = None
rows = []
for sname, sval in strategies:
    for oname, oval in orderings:
        ctx = umf.UmfpackContext('zi')
        ctx.control[strategy_i] = sval
        ctx.control[ordering_i] = oval
        try:
            ts = perf_counter(); ctx.symbolic(A); symbolic = perf_counter()-ts
            tn = perf_counter(); ctx.numeric(A); numeric = perf_counter()-tn
            tv = perf_counter(); x = ctx.solve(umf.UMFPACK_A, A, b, autoTranspose=False); solve = perf_counter()-tv
            if reference is None:
                reference = np.asarray(x).copy()
            delta = float(np.max(np.abs(np.asarray(x)-reference)))
            used_s = int(round(ctx.info[strategy_used_i])) if strategy_used_i < len(ctx.info) else -1
            used_o = int(round(ctx.info[ordering_used_i])) if ordering_used_i < len(ctx.info) else -1
            lnz = int(round(ctx.info[lnz_i])) if lnz_i < len(ctx.info) else -1
            unz = int(round(ctx.info[unz_i])) if unz_i < len(ctx.info) else -1
            total_repeat = numeric + solve
            rows.append((total_repeat, sname, oname, symbolic, numeric, solve, delta, used_s, used_o, lnz+unz))
            print(f'{sname:11s} {oname:5s} symbolic={symbolic:7.3f}s numeric={numeric:7.3f}s solve={solve:6.3f}s repeat={total_repeat:7.3f}s dfield={delta:.3e} used=({used_s},{used_o}) LU_nnz={lnz+unz:,}')
        except Exception as exc:
            print(f'{sname:11s} {oname:5s} FAILED: {type(exc).__name__}: {exc}')

if rows:
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    print('\nBest repeated-wavelength candidate:')
    print(f'  strategy={best[1]} ordering={best[2]} numeric+solve={best[0]:.3f}s symbolic(one-time)={best[3]:.3f}s dfield={best[6]:.3e}')
    print('\nNOTE: dfield is compared with the first successful UMFPACK configuration. Small floating-point differences are normal; the winning configuration still needs the full old-vs-new validator and R/T check before adoption.')
