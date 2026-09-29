from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from photonics_fem.full_pml import (
    FullPMLConfig,
)

from photonics_fem.oblique import (
    outgoing_ky,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
    sample_total_centerline,
)

from photonics_fem.solver_generic_pml import (
    solve_generic_pml_te,
)


# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUT = Path(
    "results_visual_diagnostics"
)

OUT.mkdir(
    exist_ok=True
)


ANGLE_DEG = 30.0
WAVELENGTH = 0.633


# ============================================================
# PML THEORY
# ============================================================

def predicted_decay_curve(
    ky: complex,
    thickness: float,
    sigma_max: float,
    order: int,
    distance: np.ndarray,
) -> np.ndarray:
    """Predict plane-wave amplitude decay inside a polynomial PML.

    PML profile:

        sigma(d)
            = sigma_max * (d / L)^order

    where

        d = distance from PML entrance
        L = PML thickness

    For the e^(-i omega t) convention and

        s = 1 + i sigma

    the amplitude decays as

        exp(
            -Re(ky)
            * integral sigma(d) dd
        ).
    """

    xi = np.clip(
        distance / thickness,
        0.0,
        1.0,
    )

    integrated_sigma = (
        sigma_max
        * thickness
        / (order + 1)
        * xi ** (order + 1)
    )

    return np.exp(
        -abs(np.real(ky))
        * integrated_sigma
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Physical + PML configuration
    # --------------------------------------------------------

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

    # About 18 elements per shortest wavelength.
    h_target = (
        cfg.lambda_min_material
        / 18
    )

    print(
        f"Solving diagnostics case: "
        f"lambda={cfg.wavelength:.3f} um, "
        f"angle={ANGLE_DEG:.1f} deg"
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    result = solve_generic_pml_te(
        cfg,
        angle_deg=ANGLE_DEG,
        h_target=h_target,
        order=2,
    )

    sol = result.scattered

    # --------------------------------------------------------
    # R/T extraction
    # --------------------------------------------------------

    fem = fit_full_pml_rt(
        result
    )

    print()

    print("R/T:")

    print(
        f"R = "
        f"{fem['R']:.12f}"
    )

    print(
        f"T = "
        f"{fem['T']:.12f}"
    )

    print(
        f"R + T = "
        f"{fem['R_plus_T']:.12f}"
    )

    print(
        f"bottom incoming ratio = "
        f"{fem['bottom_incoming_ratio']:.6e}"
    )

    # --------------------------------------------------------
    # Direct Bloch mismatch from actual paired FEM DOFs
    # --------------------------------------------------------

    left_dofs = (
        result.reduction.left_dofs
    )

    right_dofs = (
        result.reduction.right_dofs
    )

    phase = (
        result.reduction.phase
    )

    u_left = (
        sol.u[left_dofs]
    )

    u_right = (
        sol.u[right_dofs]
    )

    bloch_error = (
        u_right
        - phase * u_left
    )

    bloch_scale = max(
        float(
            np.max(
                np.abs(sol.u)
            )
        ),
        1e-30,
    )

    bloch_mismatch = float(
        np.max(
            np.abs(bloch_error)
        )
        / bloch_scale
    )

    print(
        f"Bloch mismatch = "
        f"{bloch_mismatch:.6e}"
    )

    # ========================================================
    # 1. TOTAL FIELD MAGNITUDE ALONG CENTERLINE
    # ========================================================

    y_phys = np.linspace(
        0.0,
        cfg.total_height,
        2500,
    )

    u_total = (
        sample_total_centerline(
            result,
            y_phys,
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        y_phys,
        np.abs(u_total),
        label="|E_total|",
    )

    plt.axvspan(
        cfg.slab_y0,
        cfg.slab_y1,
        alpha=0.15,
        label="dielectric slab",
    )

    plt.xlabel(
        "y [um]"
    )

    plt.ylabel(
        "|E_total|"
    )

    plt.title(
        f"Centerline total field\n"
        f"{ANGLE_DEG:.1f} deg, "
        f"lambda={cfg.wavelength:.3f} um"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_centerline_total_magnitude.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 2. PLANE-WAVE FIT IN TOP AIR
    # ========================================================

    ky_top = outgoing_ky(
        cfg.k0,
        cfg.n_inc,
        result.kx,
    )

    ky_bottom = outgoing_ky(
        cfg.k0,
        cfg.n_out,
        result.kx,
    )

    yt = np.linspace(
        0.12 * cfg.air_top,
        0.82 * cfg.air_top,
        400,
    )

    ut = sample_total_centerline(
        result,
        yt,
    )

    top_fit = (
        fem["a_inc"]
        * np.exp(
            1j
            * ky_top
            * yt
        )
        +
        fem["a_refl"]
        * np.exp(
            -1j
            * ky_top
            * yt
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        yt,
        np.real(ut),
        label="FEM Re(E)",
    )

    plt.plot(
        yt,
        np.real(top_fit),
        "--",
        label="incident + reflected fit",
    )

    plt.xlabel(
        "y [um]"
    )

    plt.ylabel(
        "Re(E)"
    )

    plt.title(
        "Top-air plane-wave fit"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "02_top_air_fit_check.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 3. PLANE-WAVE FIT IN BOTTOM AIR
    # ========================================================

    yb = np.linspace(
        cfg.slab_y1
        + 0.18 * cfg.air_bottom,

        cfg.total_height
        - 0.12 * cfg.air_bottom,

        400,
    )

    ub = sample_total_centerline(
        result,
        yb,
    )

    bottom_fit = (
        fem["a_trans"]
        * np.exp(
            1j
            * ky_bottom
            * yb
        )
        +
        fem["a_in_bottom"]
        * np.exp(
            -1j
            * ky_bottom
            * yb
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        yb,
        np.real(ub),
        label="FEM Re(E)",
    )

    plt.plot(
        yb,
        np.real(bottom_fit),
        "--",
        label="transmitted-wave fit",
    )

    plt.xlabel(
        "y [um]"
    )

    plt.ylabel(
        "Re(E)"
    )

    plt.title(
        "Bottom-air plane-wave fit"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "03_bottom_air_fit_check.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 4. PML DECAY: FEM VS THEORY
    # ========================================================

    # --------------------------------------------------------
    # Top PML
    #
    # d = 0 at physical/PML interface
    # y = -d
    # --------------------------------------------------------

    dist_top = np.linspace(
        0.0,
        0.90 * cfg.pml_top,
        400,
    )

    y_top = (
        -dist_top
    )

    u_top = (
        sample_scattered_centerline(
            result,
            y_top,
        )
    )

    top_relative = (
        np.abs(u_top)
        / max(
            float(
                np.abs(u_top[0])
            ),
            1e-30,
        )
    )

    top_theory = (
        predicted_decay_curve(
            ky=ky_top,

            thickness=cfg.pml_top,

            sigma_max=cfg.pml_sigma_max,

            order=cfg.pml_order,

            distance=dist_top,
        )
    )

    # --------------------------------------------------------
    # Bottom PML
    #
    # d = 0 at physical/PML interface
    # y = total_height + d
    # --------------------------------------------------------

    dist_bottom = np.linspace(
        0.0,
        0.90 * cfg.pml_bottom,
        400,
    )

    y_bottom = (
        cfg.total_height
        + dist_bottom
    )

    u_bottom = (
        sample_scattered_centerline(
            result,
            y_bottom,
        )
    )

    bottom_relative = (
        np.abs(u_bottom)
        / max(
            float(
                np.abs(u_bottom[0])
            ),
            1e-30,
        )
    )

    bottom_theory = (
        predicted_decay_curve(
            ky=ky_bottom,

            thickness=cfg.pml_bottom,

            sigma_max=cfg.pml_sigma_max,

            order=cfg.pml_order,

            distance=dist_bottom,
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogy(
        dist_top,
        top_relative,
        label="top PML FEM",
    )

    plt.semilogy(
        dist_top,
        top_theory,
        "--",
        label="top PML theory",
    )

    plt.semilogy(
        dist_bottom,
        bottom_relative,
        label="bottom PML FEM",
    )

    plt.semilogy(
        dist_bottom,
        bottom_theory,
        "--",
        label="bottom PML theory",
    )

    plt.xlabel(
        "distance into PML [um]"
    )

    plt.ylabel(
        "relative |E_scattered|"
    )

    plt.title(
        "PML attenuation: FEM vs theory"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "04_pml_decay_check.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 5. MEMORY-SAFE 2D SCATTERED-FIELD MAP
    # ========================================================

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT probe a huge arbitrary Cartesian grid here.
    #
    # refinterp() refines the existing FEM representation
    # directly and avoids the massive element-search array
    # that previously attempted to allocate ~24.6 GB.
    # --------------------------------------------------------

    rmesh, u_scattered_refined = (
        sol.basis.refinterp(
            sol.u,
            nrefs=2,
        )
    )

    x_ref = (
        rmesh.p[0]
    )

    y_ref = (
        rmesh.p[1]
    )

    triangles = (
        rmesh.t.T
    )

    tri = mtri.Triangulation(
        x_ref,
        y_ref,
        triangles,
    )

    plt.figure(
        figsize=(7.5, 7.0)
    )

    plt.tripcolor(
        tri,
        np.log10(
            np.abs(
                u_scattered_refined
            )
            + 1e-12
        ),
        shading="gouraud",
    )

    plt.colorbar(
        label="log10 |E_scattered|"
    )

    # Physical/PML interfaces
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

    # Slab interfaces
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
        "2D scattered field including PMLs"
    )

    # Show propagation downward on screen.
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "05_scattered_field_map.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 6. 2D TOTAL FIELD IN PHYSICAL REGION
    # ========================================================

    # The analytic incident field is added only for
    # visualisation inside the physical region.

    u_incident_refined = np.exp(
        1j
        * (
            result.kx
            * x_ref
            +
            result.ky_inc
            * y_ref
        )
    )

    u_total_refined = (
        u_scattered_refined
        + u_incident_refined
    )

    # --------------------------------------------------------
    # Mask all triangles outside:
    #
    # 0 <= y <= total_height
    # --------------------------------------------------------

    triangle_y = (
        y_ref[
            rmesh.t[0]
        ]
        +
        y_ref[
            rmesh.t[1]
        ]
        +
        y_ref[
            rmesh.t[2]
        ]
    ) / 3.0

    outside_physical = (
        (triangle_y < 0.0)
        |
        (
            triangle_y
            > cfg.total_height
        )
    )

    tri_physical = (
        mtri.Triangulation(
            x_ref,
            y_ref,
            triangles,
        )
    )

    tri_physical.set_mask(
        outside_physical
    )

    plt.figure(
        figsize=(7.5, 6.0)
    )

    plt.tripcolor(
        tri_physical,
        np.real(
            u_total_refined
        ),
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

    plt.xlabel(
        "x [um]"
    )

    plt.ylabel(
        "y [um]"
    )

    plt.title(
        f"2D total field in physical region\n"
        f"{ANGLE_DEG:.1f} deg incidence"
    )

    plt.ylim(
        cfg.total_height,
        0.0,
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "06_total_field_map_physical.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # 7. BLOCH BOUNDARY VISUAL CHECK
    # ========================================================

    # Use the actual paired FEM boundary DOFs directly.
    # This is both cheaper and more exact than probing
    # arbitrary points along x = 0 and x = width.

    y_left = (
        sol.basis.doflocs[
            1,
            left_dofs,
        ]
    )

    bloch_abs_error = (
        np.abs(
            bloch_error
        )
    )

    sort_index = (
        np.argsort(
            y_left
        )
    )

    y_left = (
        y_left[
            sort_index
        ]
    )

    bloch_abs_error = (
        bloch_abs_error[
            sort_index
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.semilogy(
        y_left,
        bloch_abs_error
        + 1e-30,
    )

    plt.xlabel(
        "y [um]"
    )

    plt.ylabel(
        "|u_right - phase * u_left|"
    )

    plt.title(
        "Bloch boundary mismatch"
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "07_bloch_mismatch_profile.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()

    print(
        "Visual diagnostics complete."
    )

    print()

    print(
        "Measured top PML amplitude at 90% depth:"
    )

    print(
        f"    FEM    = "
        f"{top_relative[-1]:.6e}"
    )

    print(
        f"    theory = "
        f"{top_theory[-1]:.6e}"
    )

    print()

    print(
        "Measured bottom PML amplitude at 90% depth:"
    )

    print(
        f"    FEM    = "
        f"{bottom_relative[-1]:.6e}"
    )

    print(
        f"    theory = "
        f"{bottom_theory[-1]:.6e}"
    )

    print()

    print(
        "Maximum absolute Bloch boundary error:"
    )

    print(
        f"    {np.max(bloch_abs_error):.6e}"
    )

    print()

    print(
        "Plots written to:"
    )

    for path in sorted(
        OUT.glob("*.png")
    ):
        print(
            f"    {path}"
        )


if __name__ == "__main__":
    main()