from pathlib import Path
import numpy as np

import photonics_fem_solver as new
from photonics_fem.pml import BottomPMLConfig as OldBottomPMLConfig
from photonics_fem.solver_bloch import solve_slab_pml_bloch as old_solve_slab_pml_bloch
from photonics_fem.full_pml import FullPMLConfig as OldFullPMLConfig
from photonics_fem.solver_generic_pml import solve_generic_pml_te as old_solve_generic_pml_te
from photonics_fem.objective3_bodyfitted import Objective3Config as OldObjective3Config
from photonics_fem.objective3_bodyfitted import solve_objective3_bodyfitted as old_solve_objective3_bodyfitted
from photonics_fem.objective3_bodyfitted import diffraction_orders_bodyfitted as old_diffraction_orders_bodyfitted
from photonics_fem.disordered_fast import DisorderedConfig as OldDisorderedConfig
from photonics_fem.disordered_fast import diffraction_orders as old_diffraction_orders
from photonics_fem.disordered_cached import CachedDisorderedSolver as OldCachedDisorderedSolver

ATOL = 1e-10


def check(name, a, b, atol=ATOL):
    a = np.asarray(a)
    b = np.asarray(b)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = bool(err <= atol)
    print(f"{name:<38} max |delta| = {err:.3e}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"{name} failed: {err} > {atol}")


def objective1():
    kw = dict(
        wavelength=0.633,
        n_inc=1.0 + 0.0j,
        n_slab=1.5 + 0.0j,
        n_out=1.5 + 0.0j,
        air_top=0.50,
        slab_thickness=0.50,
        air_bottom=0.50,
        width=1.0,
        pml_bottom=0.80,
        pml_order=3,
        pml_sigma_max=6.0,
    )
    old_cfg = OldBottomPMLConfig(**kw)
    new_cfg = new.BottomPMLConfig(**kw)
    h = old_cfg.lambda_min_material / 12
    old_result = old_solve_slab_pml_bloch(old_cfg, h_target=h, order=2, kx=0.0)
    new_result = new.solve_slab_pml_bloch(new_cfg, h_target=h, order=2, kx=0.0)
    check("Objective 1 field", old_result.solution.u, new_result.solution.u)


def objective2():
    kw = dict(
        wavelength=0.633,
        n_inc=1.0 + 0.0j,
        n_slab=1.5 + 0.0j,
        n_out=1.0 + 0.0j,
        slab_thickness=1.0,
        air_top=0.50,
        air_bottom=0.50,
        width=1.0,
        pml_top=0.80,
        pml_bottom=0.80,
        pml_order=3,
        pml_sigma_max=6.0,
    )
    old_cfg = OldFullPMLConfig(**kw)
    new_cfg = new.FullPMLConfig(**kw)
    h = old_cfg.lambda_min_material / 12
    old_result = old_solve_generic_pml_te(old_cfg, angle_deg=0.0, h_target=h, order=2)
    new_result = new.solve_generic_pml_te(new_cfg, angle_deg=0.0, h_target=h, order=2)
    check("Objective 2 scattered field", old_result.scattered.u, new_result.scattered.u)


def objective3():
    mesh = Path("meshes/objective3_bodyfitted_fixed.msh")
    if not mesh.exists():
        raise FileNotFoundError(mesh)
    kw = dict(
        wavelength=0.600,
        n_inc=1.0 + 0.0j,
        n_slab=1.5 + 0.0j,
        n_out=1.0 + 0.0j,
        slab_thickness=1.0,
        air_top=0.50,
        air_bottom=0.50,
        width=1.0,
        hole_diameter=0.50,
        pml_top=1.00,
        pml_bottom=1.00,
        pml_order=3,
        pml_sigma_max=8.0,
    )
    old_cfg = OldObjective3Config(**kw)
    new_cfg = new.Objective3Config(**kw)
    old_result = old_solve_objective3_bodyfitted(old_cfg, mesh)
    new_result = new.solve_objective3_bodyfitted(new_cfg, mesh)
    check("Objective 3 scattered field", old_result.scattered.u, new_result.scattered.u)
    old_rt = old_diffraction_orders_bodyfitted(old_result, old_cfg, npoints=512)
    new_rt = new.diffraction_orders_bodyfitted(new_result, new_cfg, npoints=512)
    check("Objective 3 R,T", [old_rt["R"], old_rt["T"]], [new_rt["R"], new_rt["T"]])


def disordered():
    mesh = Path("meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh")
    if not mesh.exists():
        raise FileNotFoundError(mesh)
    kw = dict(
        wavelength=1.500,
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,
        physical_ymin=-5.5,
        physical_ymax=5.5,
        pml_low=3.0,
        pml_high=3.0,
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )
    old_cfg = OldDisorderedConfig(**kw)
    new_cfg = new.DisorderedConfig(**kw)
    old_solver = OldCachedDisorderedSolver(old_cfg, mesh)
    new_solver = new.CachedDisorderedSolver(new_cfg, mesh)
    old_result = old_solver.solve(old_cfg)
    new_result = new_solver.solve(new_cfg)
    check("Disordered scattered field", old_result.u_scattered, new_result.u_scattered)
    old_rt = old_diffraction_orders(old_result, npoints=1024)
    new_rt = new.diffraction_orders(new_result, npoints=1024)
    check("Disordered R,T", [old_rt["R"], old_rt["T"]], [new_rt["R"], new_rt["T"]])


if __name__ == "__main__":
    print("Single-file solver equivalence validation")
    print("Tolerance:", ATOL)
    objective1()
    objective2()
    objective3()
    disordered()
    print("ALL CHECKS PASSED")
