from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from photonics_fem.analytics import fabry_perot_resonances, slab_field, slab_rt
from photonics_fem.config import SlabConfig
from photonics_fem.mesh import max_edge_length
from photonics_fem.postprocess import fit_plane_waves, relative_l2_profile_error, sample_centerline
from photonics_fem.solver import solve_slab

OUT = Path(__file__).resolve().parent / "results_scikit_fem"
OUT.mkdir(exist_ok=True)


def one_solution_plot(cfg: SlabConfig, order: int = 2, ppw: int = 18) -> None:
    h = cfg.lambda_min_material / ppw
    sol = solve_slab(cfg, h_target=h, order=order)
    y = np.linspace(0.0, cfg.total_height, 2000)
    uh = sample_centerline(sol, y)
    ue = slab_field(y, cfg)
    rt = fit_plane_waves(sol)
    _, _, Ra, Ta = slab_rt(cfg)
    err = relative_l2_profile_error(sol)

    print(f"single solve P{order}: dofs={sol.basis.N}, hmax={max_edge_length(sol.mesh):.5g} um")
    print(f"relative L2 profile error = {err:.6e}")
    print(f"FEM R={rt['R']:.8f}, T={rt['T']:.8f}, R+T={rt['R_plus_T']:.10f}")
    print(f"ANA R={Ra:.8f}, T={Ta:.8f}, R+T={Ra+Ta:.10f}")
    print(f"bottom incoming amplitude ratio={rt['bottom_incoming_ratio']:.3e}")

    plt.figure(figsize=(7.2, 4.5))
    plt.plot(y, np.real(ue), label="analytic Re(Ez)")
    plt.plot(y, np.real(uh), "--", label=f"FEM P{order} Re(Ez)")
    plt.plot(y, np.abs(ue), label="analytic |Ez|", alpha=0.8)
    plt.plot(y, np.abs(uh), "--", label=f"FEM P{order} |Ez|", alpha=0.8)
    plt.axvspan(cfg.slab_y0, cfg.slab_y1, alpha=0.12, label="dielectric slab")
    plt.xlabel("depth y [um]")
    plt.ylabel("field amplitude")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "field_profile.png", dpi=180)
    plt.close()


def convergence(cfg: SlabConfig) -> None:
    ppws = [6, 8, 12, 16, 24, 32]
    rows = []
    plt.figure(figsize=(6.4, 4.6))
    for order in (1, 2):
        hs, errs = [], []
        for ppw in ppws:
            h = cfg.lambda_min_material / ppw
            sol = solve_slab(cfg, h_target=h, order=order)
            hmax = max_edge_length(sol.mesh)
            err = relative_l2_profile_error(sol)
            rt = fit_plane_waves(sol)
            hs.append(hmax)
            errs.append(err)
            rows.append([order, ppw, sol.basis.N, hmax, err,
                         rt["R"], rt["T"], rt["R_plus_T"]])
        slope = np.polyfit(np.log(hs[-4:]), np.log(errs[-4:]), 1)[0]
        print(f"P{order} fitted L2 slope (last 4 meshes): {slope:.3f}")
        plt.loglog(hs, errs, "o-", label=f"P{order}, slope~{slope:.2f}")

    with (OUT / "error_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "ppw_in_slab", "dofs", "hmax_um", "rel_L2", "R", "T", "R_plus_T"])
        w.writerows(rows)

    plt.gca().invert_xaxis()
    plt.xlabel("max triangle edge h [um]")
    plt.ylabel("relative L2 field-profile error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "convergence.png", dpi=180)
    plt.close()


def wavelength_sweep(cfg: SlabConfig) -> None:
    wavelengths = np.linspace(0.45, 0.95, 81)
    Rf, Tf, Ra, Ta = [], [], [], []
    # Fixed resolution based on the shortest wavelength in the sweep.
    h = min(wavelengths) / abs(cfg.n_slab) / 18.0
    for lam in wavelengths:
        c = cfg.with_wavelength(float(lam))
        sol = solve_slab(c, h_target=h, order=2)
        rt = fit_plane_waves(sol)
        _, _, r_a, t_a = slab_rt(c)
        Rf.append(rt["R"]); Tf.append(rt["T"])
        Ra.append(r_a); Ta.append(t_a)

    res = fabry_perot_resonances(cfg, np.arange(1, 8))
    res = res[(res >= wavelengths.min()) & (res <= wavelengths.max())]
    print("Fabry-Perot resonances in sweep [um]:", np.round(res, 6))

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(wavelengths, Ra, label="analytic R")
    plt.plot(wavelengths, Ta, label="analytic T")
    plt.plot(wavelengths, Rf, "--", label="FEM R")
    plt.plot(wavelengths, Tf, "--", label="FEM T")
    for rr in res:
        plt.axvline(rr, linewidth=0.8, alpha=0.35)
    plt.xlabel("vacuum wavelength [um]")
    plt.ylabel("power fraction")
    plt.ylim(-0.03, 1.05)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "rt_sweep.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    cfg = SlabConfig()
    one_solution_plot(cfg)
    convergence(cfg)
    wavelength_sweep(cfg)
    print(f"Wrote results to {OUT}")
