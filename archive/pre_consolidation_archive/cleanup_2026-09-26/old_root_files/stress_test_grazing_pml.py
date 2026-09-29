from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from photonics_fem.full_pml import FullPMLConfig

from photonics_fem.oblique import (
    slab_rt_te_oblique,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
    sample_total_centerline,
)

from photonics_fem.solver_generic_pml import (
    solve_generic_pml_te,
)


OUT = Path("results_grazing_stress_test")
OUT.mkdir(exist_ok=True)


ANGLE_DEG = 70.0
WAVELENGTH = 0.633


def make_cfg(
    pml_thickness: float,
    sigma_max: float,
) -> FullPMLConfig:

    return FullPMLConfig(
        wavelength=WAVELENGTH,

        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=0.50,

        air_top=0.50,
        air_bottom=0.50,

        width=0.50,

        pml_top=pml_thickness,
        pml_bottom=pml_thickness,

        pml_order=3,
        pml_sigma_max=sigma_max,
    )


def relative_l2(
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> float:

    num = np.trapezoid(
        np.abs(a - b) ** 2,
        x,
    )

    den = np.trapezoid(
        np.abs(b) ** 2,
        x,
    )

    return float(
        np.sqrt(
            num / den
        )
    )


def solve_case(
    cfg: FullPMLConfig,
    name: str,
):

    h_target = (
        cfg.lambda_min_material
        / 18
    )

    print()
    print(
        f"Solving {name}: "
        f"L={cfg.pml_top:.2f}, "
        f"sigma={cfg.pml_sigma_max:.1f}, "
        f"angle={ANGLE_DEG:.1f} deg"
    )

    result = solve_generic_pml_te(
        cfg,
        angle_deg=ANGLE_DEG,
        h_target=h_target,
        order=2,
    )

    rt = fit_full_pml_rt(
        result
    )

    return result, rt


def make_total_field_plot(
    result,
    filename: str,
    title: str,
) -> None:

    sol = result.scattered
    cfg = sol.cfg

    rmesh, u_s = sol.basis.refinterp(
        sol.u,
        nrefs=2,
    )

    x = rmesh.p[0]
    y = rmesh.p[1]

    u_inc = np.exp(
        1j
        * (
            result.kx * x
            + result.ky_inc * y
        )
    )

    u_total = (
        u_inc
        + u_s
    )

    tri_y = (
        y[rmesh.t[0]]
        + y[rmesh.t[1]]
        + y[rmesh.t[2]]
    ) / 3.0

    mask = (
        (tri_y < 0.0)
        |
        (
            tri_y
            > cfg.total_height
        )
    )

    tri = mtri.Triangulation(
        x,
        y,
        rmesh.t.T,
    )

    tri.set_mask(
        mask
    )

    plt.figure(
        figsize=(7.5, 6.0)
    )

    plt.tripcolor(
        tri,
        np.real(u_total),
        shading="gouraud",
    )

    plt.colorbar(
        label="Re(E_total)"
    )

    plt.axhline(
        cfg.slab_y0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.axhline(
        cfg.slab_y1,
        linestyle="--",
        linewidth=1.0,
    )

    # Correct downward arrow.
    plt.annotate(
        "incident light",
        xy=(
            0.25,
            0.28,
        ),
        xytext=(
            0.25,
            0.06,
        ),
        ha="center",
        arrowprops=dict(
            arrowstyle="->"
        ),
    )

    plt.text(
        0.02,
        0.20,
        "Top air",
    )

    plt.text(
        0.02,
        0.75,
        "Slab",
    )

    plt.text(
        0.02,
        1.25,
        "Bottom air",
    )

    plt.xlabel(
        "x [um]"
    )

    plt.ylabel(
        "y [um]"
    )

    plt.title(
        title
    )

    plt.ylim(
        cfg.total_height,
        0.0,
    )

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180,
    )

    plt.close()


def make_scattered_pml_plot(
    result,
    filename: str,
    title: str,
) -> None:

    sol = result.scattered
    cfg = sol.cfg

    rmesh, u_s = sol.basis.refinterp(
        sol.u,
        nrefs=2,
    )

    x = rmesh.p[0]
    y = rmesh.p[1]

    tri = mtri.Triangulation(
        x,
        y,
        rmesh.t.T,
    )

    plt.figure(
        figsize=(7.5, 7.0)
    )

    plt.tripcolor(
        tri,
        np.log10(
            np.abs(u_s)
            + 1e-12
        ),
        shading="gouraud",
        vmin=-4.0,
        vmax=0.5,
    )

    plt.colorbar(
        label="log10 |E_scattered|"
    )

    # PML entrances
    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.axhline(
        cfg.total_height,
        linestyle="--",
        linewidth=1.0,
    )

    # Slab
    plt.axhline(
        cfg.slab_y0,
        linestyle=":",
        linewidth=1.0,
    )

    plt.axhline(
        cfg.slab_y1,
        linestyle=":",
        linewidth=1.0,
    )

    plt.xlabel(
        "x [um]"
    )

    plt.ylabel(
        "y [um]"
    )

    plt.title(
        title
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180,
    )

    plt.close()


def main() -> None:

    # ========================================================
    # CONFIGURATIONS
    # ========================================================

    cfg_normal = make_cfg(
        pml_thickness=0.60,
        sigma_max=6.0,
    )

    cfg_strong = make_cfg(
        pml_thickness=1.00,
        sigma_max=8.0,
    )

    # ========================================================
    # ANALYTIC ANSWER
    # ========================================================

    _, _, R_exact, T_exact = (
        slab_rt_te_oblique(
            cfg_normal,
            ANGLE_DEG,
        )
    )

    print("=" * 76)
    print("70 DEGREE GRAZING-INCIDENCE PML STRESS TEST")
    print("=" * 76)

    print()
    print("Analytic TE thin-film answer:")

    print(
        f"R exact = "
        f"{R_exact:.12f}"
    )

    print(
        f"T exact = "
        f"{T_exact:.12f}"
    )

    # ========================================================
    # SOLVES
    # ========================================================

    normal, rt_normal = solve_case(
        cfg_normal,
        "NORMAL PML",
    )

    strong, rt_strong = solve_case(
        cfg_strong,
        "STRONG PML",
    )

    # ========================================================
    # NUMERICAL COMPARISON
    # ========================================================

    print()
    print("=" * 76)
    print("R/T COMPARISON")
    print("=" * 76)

    print()

    print("NORMAL PML")

    print(
        f"R       = "
        f"{rt_normal['R']:.12f}"
    )

    print(
        f"T       = "
        f"{rt_normal['T']:.12f}"
    )

    print(
        f"R + T   = "
        f"{rt_normal['R_plus_T']:.12f}"
    )

    print(
        f"|dR|    = "
        f"{abs(rt_normal['R'] - R_exact):.6e}"
    )

    print(
        f"|dT|    = "
        f"{abs(rt_normal['T'] - T_exact):.6e}"
    )

    print()

    print("STRONG PML")

    print(
        f"R       = "
        f"{rt_strong['R']:.12f}"
    )

    print(
        f"T       = "
        f"{rt_strong['T']:.12f}"
    )

    print(
        f"R + T   = "
        f"{rt_strong['R_plus_T']:.12f}"
    )

    print(
        f"|dR|    = "
        f"{abs(rt_strong['R'] - R_exact):.6e}"
    )

    print(
        f"|dT|    = "
        f"{abs(rt_strong['T'] - T_exact):.6e}"
    )

    print()

    print("PML-TO-PML DIFFERENCE")

    print(
        f"|R normal - R strong| = "
        f"{abs(rt_normal['R'] - rt_strong['R']):.6e}"
    )

    print(
        f"|T normal - T strong| = "
        f"{abs(rt_normal['T'] - rt_strong['T']):.6e}"
    )

    # ========================================================
    # PHYSICAL FIELD COMPARISON
    # ========================================================

    y_phys = np.linspace(
        0.0,
        cfg_normal.total_height,
        3000,
    )

    u_normal = (
        sample_total_centerline(
            normal,
            y_phys,
        )
    )

    u_strong = (
        sample_total_centerline(
            strong,
            y_phys,
        )
    )

    field_error = relative_l2(
        u_normal,
        u_strong,
        y_phys,
    )

    print()

    print(
        f"Physical-region centerline L2 difference = "
        f"{field_error:.6e}"
    )

    # ========================================================
    # PLOT 1:
    # PHYSICAL FIELD COMPARISON
    # ========================================================

    plt.figure(
        figsize=(8.5, 5.0)
    )

    plt.plot(
        y_phys,
        np.real(u_normal),
        label="normal PML",
    )

    plt.plot(
        y_phys,
        np.real(u_strong),
        "--",
        label="strong PML",
    )

    plt.axvspan(
        cfg_normal.slab_y0,
        cfg_normal.slab_y1,
        alpha=0.15,
        label="slab",
    )

    plt.xlabel(
        "y [um]"
    )

    plt.ylabel(
        "Re(E_total)"
    )

    plt.title(
        "70 degree incidence:\n"
        "physical field should not depend on PML choice"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_physical_field_pml_comparison.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # PLOT 2 + 3:
    # REAL TOTAL FIELD
    # ========================================================

    make_total_field_plot(
        normal,
        "02_total_field_normal_pml.png",
        "70 degree incidence - normal PML",
    )

    make_total_field_plot(
        strong,
        "03_total_field_strong_pml.png",
        "70 degree incidence - strong PML",
    )

    # ========================================================
    # PLOT 4 + 5:
    # FULL PML SCATTERED FIELD
    # ========================================================

    make_scattered_pml_plot(
        normal,
        "04_scattered_normal_pml.png",
        "70 degree incidence - normal PML",
    )

    make_scattered_pml_plot(
        strong,
        "05_scattered_strong_pml.png",
        "70 degree incidence - strong PML",
    )

    # ========================================================
    # PLOT 6:
    # HOW MUCH FIELD REMAINS THROUGH THE PMLS
    # ========================================================

    # Normal PML
    d_normal = np.linspace(
        0.0,
        cfg_normal.pml_bottom,
        600,
    )

    y_bottom_normal = (
        cfg_normal.total_height
        + d_normal
    )

    us_normal = (
        sample_scattered_centerline(
            normal,
            y_bottom_normal,
        )
    )

    norm_normal = (
        np.abs(us_normal)
        / max(
            float(
                np.abs(us_normal[0])
            ),
            1e-30,
        )
    )

    # Strong PML
    d_strong = np.linspace(
        0.0,
        cfg_strong.pml_bottom,
        600,
    )

    y_bottom_strong = (
        cfg_strong.total_height
        + d_strong
    )

    us_strong = (
        sample_scattered_centerline(
            strong,
            y_bottom_strong,
        )
    )

    norm_strong = (
        np.abs(us_strong)
        / max(
            float(
                np.abs(us_strong[0])
            ),
            1e-30,
        )
    )

    plt.figure(
        figsize=(8.0, 5.0)
    )

    plt.semilogy(
        d_normal,
        norm_normal,
        label="normal PML",
    )

    plt.semilogy(
        d_strong,
        norm_strong,
        label="strong PML",
    )

    plt.xlabel(
        "distance into bottom PML [um]"
    )

    plt.ylabel(
        "relative |E_scattered|"
    )

    plt.title(
        "70 degree incidence:\n"
        "PML absorption stress test"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "06_pml_strength_comparison.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 76)

    passed = (
        abs(
            rt_strong["R"]
            - R_exact
        ) < 5e-4
        and
        abs(
            rt_strong["T"]
            - T_exact
        ) < 5e-4
        and
        abs(
            rt_strong["R_plus_T"]
            - 1.0
        ) < 5e-4
        and
        field_error < 5e-3
    )

    print(
        "STRESS TEST:",
        "PASS"
        if passed
        else "CHECK RESULTS",
    )

    print("=" * 76)

    print()
    print("Plots written to:")

    for path in sorted(
        OUT.glob("*.png")
    ):
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()