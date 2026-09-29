from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from photonics_fem.full_pml import FullPMLConfig
from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
    sample_total_centerline,
)
from photonics_fem.solver_generic_pml import solve_generic_pml_te


OUT = Path("results_top_incidence_visual")
OUT.mkdir(exist_ok=True)

ANGLE_DEG = 0.0
WAVELENGTH = 0.633


def main() -> None:
    cfg = FullPMLConfig(
        wavelength=WAVELENGTH,

        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=0.50,

        air_top=0.50,
        air_bottom=0.50,

        width=0.50,

        pml_top=0.60,
        pml_bottom=0.60,

        pml_order=3,
        pml_sigma_max=6.0,
    )

    h_target = cfg.lambda_min_material / 18

    print(
        f"Solving top-incidence visual case: "
        f"lambda={cfg.wavelength:.3f} um, "
        f"angle={ANGLE_DEG:.1f} deg"
    )

    result = solve_generic_pml_te(
        cfg,
        angle_deg=ANGLE_DEG,
        h_target=h_target,
        order=2,
    )

    sol = result.scattered

    fem = fit_full_pml_rt(result)

    print()
    print("R/T:")
    print(f"R = {fem['R']:.12f}")
    print(f"T = {fem['T']:.12f}")
    print(f"R + T = {fem['R_plus_T']:.12f}")
    print(f"bottom incoming ratio = {fem['bottom_incoming_ratio']:.6e}")

    # ========================================================
    # Refine/interpolate FEM solution for memory-safe plotting
    # ========================================================

    rmesh, u_scattered_refined = sol.basis.refinterp(
        sol.u,
        nrefs=2,
    )

    x_ref = rmesh.p[0]
    y_ref = rmesh.p[1]
    triangles = rmesh.t.T

    tri_all = mtri.Triangulation(
        x_ref,
        y_ref,
        triangles,
    )

    # Incident field added analytically
    u_incident_refined = np.exp(
        1j * (
            result.kx * x_ref
            + result.ky_inc * y_ref
        )
    )

    u_total_refined = (
        u_scattered_refined
        + u_incident_refined
    )

    # Mask triangles outside physical region for total-field plots
    triangle_y = (
        y_ref[rmesh.t[0]]
        + y_ref[rmesh.t[1]]
        + y_ref[rmesh.t[2]]
    ) / 3.0

    outside_physical = (
        (triangle_y < 0.0)
        | (triangle_y > cfg.total_height)
    )

    tri_physical = mtri.Triangulation(
        x_ref,
        y_ref,
        triangles,
    )

    tri_physical.set_mask(outside_physical)

    # ========================================================
    # 1) 2D total field: Re(E)
    # ========================================================

    plt.figure(figsize=(7.5, 6.2))

    plt.tripcolor(
        tri_physical,
        np.real(u_total_refined),
        shading="gouraud",
    )

    plt.colorbar(label="Re(E_total)")

    plt.axhline(cfg.slab_y0, linestyle="--", linewidth=1.0)
    plt.axhline(cfg.slab_y1, linestyle="--", linewidth=1.0)

    plt.text(
        0.02, 0.20,
        "Top air",
    )
    plt.text(
        0.02, 0.75,
        "Dielectric slab",
    )
    plt.text(
        0.02, 1.25,
        "Bottom air",
    )

    plt.annotate(
        "incident wave from top",
        xy=(0.25, 0.05),
        xytext=(0.25, 0.20),
        ha="center",
        arrowprops=dict(arrowstyle="->"),
    )

    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.title(
        "2D total field (real part)\n"
        "Light incident from top"
    )

    plt.ylim(cfg.total_height, 0.0)

    plt.tight_layout()
    plt.savefig(OUT / "01_total_field_real.png", dpi=180)
    plt.close()

    # ========================================================
    # 2) 2D total field: magnitude
    # ========================================================

    plt.figure(figsize=(7.5, 6.2))

    plt.tripcolor(
        tri_physical,
        np.abs(u_total_refined),
        shading="gouraud",
    )

    plt.colorbar(label="|E_total|")

    plt.axhline(cfg.slab_y0, linestyle="--", linewidth=1.0)
    plt.axhline(cfg.slab_y1, linestyle="--", linewidth=1.0)

    plt.text(
        0.02, 0.20,
        "Top air",
    )
    plt.text(
        0.02, 0.75,
        "Dielectric slab",
    )
    plt.text(
        0.02, 1.25,
        "Bottom air",
    )

    plt.annotate(
        "incident wave from top",
        xy=(0.25, 0.05),
        xytext=(0.25, 0.20),
        ha="center",
        arrowprops=dict(arrowstyle="->"),
    )

    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.title(
        "2D total field magnitude\n"
        "Light incident from top"
    )

    plt.ylim(cfg.total_height, 0.0)

    plt.tight_layout()
    plt.savefig(OUT / "02_total_field_magnitude.png", dpi=180)
    plt.close()

    # ========================================================
    # 3) Centerline total field magnitude
    # ========================================================

    y_phys = np.linspace(
        0.0,
        cfg.total_height,
        2500,
    )

    u_total_center = sample_total_centerline(
        result,
        y_phys,
    )

    plt.figure(figsize=(8.0, 5.0))

    plt.plot(
        y_phys,
        np.abs(u_total_center),
        label="|E_total|",
    )

    plt.axvspan(
        cfg.slab_y0,
        cfg.slab_y1,
        alpha=0.15,
        label="dielectric slab",
    )

    plt.xlabel("y [um]")
    plt.ylabel("|E_total|")
    plt.title(
        "Centerline total field magnitude\n"
        "Light incident from top"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "03_centerline_total_magnitude.png", dpi=180)
    plt.close()

    # ========================================================
    # 4) Centerline real field
    # ========================================================

    plt.figure(figsize=(8.0, 5.0))

    plt.plot(
        y_phys,
        np.real(u_total_center),
        label="Re(E_total)",
    )

    plt.axvspan(
        cfg.slab_y0,
        cfg.slab_y1,
        alpha=0.15,
        label="dielectric slab",
    )

    plt.xlabel("y [um]")
    plt.ylabel("Re(E_total)")
    plt.title(
        "Centerline total field (real part)\n"
        "Light incident from top"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "04_centerline_total_real.png", dpi=180)
    plt.close()

    # ========================================================
    # 5) Scattered field including PMLs
    # ========================================================

    plt.figure(figsize=(7.5, 7.0))

    plt.tripcolor(
        tri_all,
        np.log10(np.abs(u_scattered_refined) + 1e-12),
        shading="gouraud",
    )

    plt.colorbar(label="log10 |E_scattered|")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axhline(cfg.slab_y0, linestyle=":", linewidth=1.0)
    plt.axhline(cfg.slab_y1, linestyle=":", linewidth=1.0)
    plt.axhline(cfg.total_height, linestyle="--", linewidth=1.0)

    plt.text(0.02, -0.25, "Top PML")
    plt.text(0.02, 0.20, "Top air")
    plt.text(0.02, 0.75, "Slab")
    plt.text(0.02, 1.25, "Bottom air")
    plt.text(0.02, 1.85, "Bottom PML")

    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.title("2D scattered field including PMLs")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT / "05_scattered_field_with_pmls.png", dpi=180)
    plt.close()

    # ========================================================
    # 6) Scattered-field centerline through full domain
    # ========================================================

    y_full = np.linspace(
        cfg.computational_ymin,
        cfg.computational_ymax,
        2500,
    )

    u_s_center = sample_scattered_centerline(
        result,
        y_full,
    )

    plt.figure(figsize=(8.0, 5.0))

    plt.plot(
        y_full,
        np.abs(u_s_center),
        label="|E_scattered|",
    )

    plt.axvspan(
        cfg.computational_ymin,
        0.0,
        alpha=0.10,
        label="top PML",
    )

    plt.axvspan(
        cfg.slab_y0,
        cfg.slab_y1,
        alpha=0.15,
        label="slab",
    )

    plt.axvspan(
        cfg.total_height,
        cfg.computational_ymax,
        alpha=0.10,
        label="bottom PML",
    )

    plt.xlabel("y [um]")
    plt.ylabel("|E_scattered|")
    plt.title("Scattered-field magnitude through full domain")

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "06_scattered_centerline_full_domain.png", dpi=180)
    plt.close()

    print()
    print("Plots written to:")
    for path in sorted(OUT.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()