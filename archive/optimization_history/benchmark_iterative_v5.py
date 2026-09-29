from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import photonics_fem_solver as pf

MESH = Path("meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh")
WAVELENGTHS = [
    1.447368421052632,
    1.460176991150442,
    1.473214285714286,
    1.486486486486486,
    1.500000000000000,
    1.513761467889908,
    1.527777777777778,
    1.542056074766355,
    1.556603773584906,
]
REF_WAVELENGTH = 1.5
RTOL = 1e-10
RESTART = 30
MAXITER = 20


def cfg(w):
    return pf.DisorderedConfig(
        wavelength=float(w), width=7.0, xmin=-3.5, xmax=3.5,
        scatter_ymin=-3.5, scatter_ymax=3.5,
        physical_ymin=-5.5, physical_ymax=5.5,
        pml_low=3.0, pml_high=3.0, pml_order=3, pml_sigma_max=8.0,
        n_background=1.0 + 0j, n_disk=3.0 + 0j,
    )


def make_A(solver, c):
    A = solver._A_work.copy()
    A.data[:] = solver._K_aligned - c.k0**2 * solver._M_aligned
    return A


def make_rhs(solver, c):
    b_full = pf.assemble_scattered_source(solver.basis, c)
    b_red = np.asarray(solver.PH @ b_full, dtype=np.complex128).reshape(-1)
    return b_red[solver.free_red]


def reconstruct(solver, z_free):
    z = np.zeros(solver.reduced_dofs, dtype=np.complex128)
    z[solver.free_red] = z_free
    return np.asarray(solver.P @ z, dtype=np.complex128).reshape(-1)


def make_result(solver, c, u):
    return pf.DisorderedSolveResult(
        cfg=c, mesh=solver.mesh, basis=solver.basis,
        u_scattered=u, full_dofs=solver.basis.N,
        reduced_dofs=solver.reduced_dofs,
        free_reduced_dofs=solver.free_reduced_dofs,
        timings=pf.SolveTimings(0, 0, 0, 0, 0, 0, 0),
    )


def main():
    if pf._umfpack is None:
        raise SystemExit("scikits.umfpack is unavailable; V5 benchmark requires it.")

    print("V5 EXPERIMENT — GMRES + reusable exact-reference LU preconditioner")
    print("V4 remains unchanged; this script only benchmarks an alternative solve path.\n")

    t0 = perf_counter()
    solver = pf.CachedDisorderedSolver(cfg(REF_WAVELENGTH), MESH, intorder=8)
    print(f"FEM setup: {perf_counter()-t0:.3f} s")
    print(f"V4 backend: {solver.linear_backend}")
    print(f"free DOFs: {solver.free_reduced_dofs:,}; union nnz: {solver.union_nnz:,}\n")
    if not solver.symbolic_reuse_enabled:
        raise SystemExit(f"V4 UMFPACK backend not active: {solver.backend_failure}")

    # Build a separate fixed LU at lambda=1.5.  GMRES applies this inverse as M.
    cref = cfg(REF_WAVELENGTH)
    Aref = make_A(solver, cref)
    family = "zi" if Aref.indices.dtype == np.int32 else "zl"
    ctx = pf._umfpack.UmfpackContext(family)
    strategy_i = getattr(pf._umfpack, "UMFPACK_STRATEGY", 5)
    ordering_i = getattr(pf._umfpack, "UMFPACK_ORDERING", 10)
    ctx.control[strategy_i] = getattr(pf._umfpack, "UMFPACK_STRATEGY_AUTO", 0)
    ctx.control[ordering_i] = getattr(pf._umfpack, "UMFPACK_ORDERING_BEST", 4)

    p0 = perf_counter()
    ctx.symbolic(Aref)
    p1 = perf_counter()
    ctx.numeric(Aref)
    p2 = perf_counter()
    print(f"preconditioner symbolic: {p1-p0:.3f} s")
    print(f"preconditioner numeric:  {p2-p1:.3f} s")
    print(f"preconditioner build:    {p2-p0:.3f} s (one-time for this block)\n")

    def psolve(x):
        return np.asarray(ctx.solve(
            pf._umfpack.UMFPACK_A, Aref,
            np.asarray(x, dtype=np.complex128),
            autoTranspose=False,
        ), dtype=np.complex128)

    M = LinearOperator(Aref.shape, matvec=psolve, dtype=np.complex128)

    x0 = None
    direct_times = []
    iter_times = []
    iterations = []
    max_field_delta = 0.0
    max_rt_delta = 0.0
    all_pass = True

    print("lambda      direct    GMRES   iters   relres       max|du|       |dR|+|dT|   status")
    print("-" * 94)

    for w in WAVELENGTHS:
        c = cfg(w)

        d0 = perf_counter()
        direct = solver.solve(c)
        dd = pf.diffraction_orders(direct, npoints=1024)
        d1 = perf_counter()
        direct_time = d1 - d0

        A = make_A(solver, c)
        b = make_rhs(solver, c)
        niter = [0]
        def cb(_):
            niter[0] += 1

        g0 = perf_counter()
        x, info = gmres(
            A, b, x0=x0, M=M,
            restart=RESTART, maxiter=MAXITER,
            rtol=RTOL, atol=0.0,
            callback=cb, callback_type="pr_norm",
        )
        g1 = perf_counter()
        iter_time = g1 - g0

        relres = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
        u = reconstruct(solver, x)
        du = float(np.max(np.abs(u - direct.u_scattered)))
        itres = make_result(solver, c, u)
        di = pf.diffraction_orders(itres, npoints=1024)
        drt = abs(di["R"] - dd["R"]) + abs(di["T"] - dd["T"])

        ok = (info == 0 and relres <= 5e-10 and du <= 1e-8 and drt <= 1e-9)
        all_pass &= ok
        max_field_delta = max(max_field_delta, du)
        max_rt_delta = max(max_rt_delta, drt)
        direct_times.append(direct_time)
        iter_times.append(iter_time)
        iterations.append(niter[0])
        x0 = x.copy() if info == 0 else None

        status = "PASS" if ok else f"FAIL(info={info})"
        print(f"{w:8.6f}  {direct_time:7.3f}s {iter_time:7.3f}s  {niter[0]:5d}  {relres:9.2e}  {du:11.3e}  {drt:11.3e}  {status}")

    direct_sum = float(np.sum(direct_times))
    iter_sum = float(np.sum(iter_times))
    precond_build = p2 - p0
    iter_with_build = iter_sum + precond_build

    print("\nSUMMARY")
    print("-" * 94)
    print(f"V4 direct solve+R/T total:             {direct_sum:.3f} s")
    print(f"V5 GMRES solve-only total:             {iter_sum:.3f} s")
    print(f"V5 GMRES + one preconditioner build:   {iter_with_build:.3f} s")
    print(f"mean V4 point:                         {np.mean(direct_times):.3f} s")
    print(f"mean V5 GMRES point:                   {np.mean(iter_times):.3f} s")
    print(f"mean GMRES iterations:                 {np.mean(iterations):.2f}")
    print(f"max field difference:                  {max_field_delta:.3e}")
    print(f"max |dR|+|dT|:                         {max_rt_delta:.3e}")
    if iter_sum > 0:
        print(f"steady solve-path speedup (excl build): {direct_sum/iter_sum:.2f}x")
    if iter_with_build > 0:
        print(f"block speedup (incl build):             {direct_sum/iter_with_build:.2f}x")

    print()
    if all_pass and iter_with_build < direct_sum:
        print("RESULT: PROMISING — V5 is faster for this block and passes accuracy checks.")
        print("Next test should use several blocks across the full spectrum before adoption.")
    elif all_pass:
        print("RESULT: ACCURATE BUT NOT FASTER — keep V4 as production backend.")
    else:
        print("RESULT: REJECT V5 — convergence/accuracy criterion failed; keep V4.")


if __name__ == "__main__":
    main()
