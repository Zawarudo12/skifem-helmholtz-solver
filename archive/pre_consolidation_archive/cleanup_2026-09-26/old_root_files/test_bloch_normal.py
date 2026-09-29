import numpy as np

from photonics_fem.analytics import slab_rt

from photonics_fem.bloch import (
    bloch_mismatch,
)

from photonics_fem.pml import (
    BottomPMLConfig,
)

from photonics_fem.postprocess import (
    fit_plane_waves,
    sample_centerline,
)

from photonics_fem.solver_bloch import (
    solve_slab_pml_bloch,
)

from photonics_fem.solver_pml import (
    solve_slab_bottom_pml,
)


def relative_difference(
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
) -> float:
    """Relative L2 difference between two sampled fields."""

    numerator = np.trapezoid(
        np.abs(a - b) ** 2,
        y,
    )

    denominator = np.trapezoid(
        np.abs(b) ** 2,
        y,
    )

    return float(
        np.sqrt(
            numerator / denominator
        )
    )


def main() -> None:

    cfg = BottomPMLConfig(
        wavelength=0.633,

        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=0.50,
        air_top=0.50,
        air_bottom=0.50,
        width=0.50,

        pml_bottom=0.60,
        pml_order=3,
        pml_sigma_max=6.0,
    )

    ppw = 18

    h_target = (
        cfg.lambda_min_material
        / ppw
    )

    # --------------------------------------------------
    # Reference:
    #
    # Validated PML solver with natural lateral BCs.
    # --------------------------------------------------

    print(
        "Solving reference PML model..."
    )

    reference = solve_slab_bottom_pml(
        cfg,
        h_target=h_target,
        order=2,
    )

    # --------------------------------------------------
    # New:
    #
    # Identical model but left/right DOFs are periodic.
    #
    # Normal incidence:
    #
    # kx = 0
    #
    # phase = exp(i kx Lambda) = 1
    # --------------------------------------------------

    print(
        "Solving Bloch-periodic PML model..."
    )

    result = solve_slab_pml_bloch(
        cfg,
        h_target=h_target,
        order=2,
        kx=0.0,
    )

    bloch = result.solution

    # --------------------------------------------------
    # R/T extraction
    # --------------------------------------------------

    ref_rt = fit_plane_waves(
        reference
    )

    bloch_rt = fit_plane_waves(
        bloch
    )

    _, _, R_exact, T_exact = slab_rt(
        cfg
    )

    # --------------------------------------------------
    # Compare physical-region centerline fields.
    # --------------------------------------------------

    y = np.linspace(
        0.0,
        cfg.total_height,
        2501,
    )

    u_ref = sample_centerline(
        reference,
        y,
    )

    u_bloch = sample_centerline(
        bloch,
        y,
    )

    field_difference = relative_difference(
        u_bloch,
        u_ref,
        y,
    )

    # --------------------------------------------------
    # Direct Bloch-periodicity check.
    # --------------------------------------------------

    mismatch = bloch_mismatch(
        bloch.u,
        result.reduction,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 68)
    print("NORMAL-INCIDENCE BLOCH VALIDATION")
    print("=" * 68)

    print()
    print("SYSTEM SIZE")

    print(
        f"Full FEM DOFs          = "
        f"{bloch.basis.N}"
    )

    print(
        f"Bloch reduced DOFs     = "
        f"{result.reduced_dofs}"
    )

    print(
        f"Free reduced DOFs      = "
        f"{result.free_reduced_dofs}"
    )

    print()
    print("BLOCH CONSTRAINT")

    print(
        f"phase                  = "
        f"{result.reduction.phase}"
    )

    print(
        f"left/right DOF pairs   = "
        f"{len(result.reduction.left_dofs)}"
    )

    print(
        f"relative mismatch      = "
        f"{mismatch:.6e}"
    )

    print()
    print("ANALYTIC")

    print(
        f"R                      = "
        f"{R_exact:.12f}"
    )

    print(
        f"T                      = "
        f"{T_exact:.12f}"
    )

    print()
    print("REFERENCE PML")

    print(
        f"R                      = "
        f"{ref_rt['R']:.12f}"
    )

    print(
        f"T                      = "
        f"{ref_rt['T']:.12f}"
    )

    print()
    print("BLOCH + PML")

    print(
        f"R                      = "
        f"{bloch_rt['R']:.12f}"
    )

    print(
        f"T                      = "
        f"{bloch_rt['T']:.12f}"
    )

    print(
        f"R + T                  = "
        f"{bloch_rt['R_plus_T']:.12f}"
    )

    print(
        f"bottom incoming        = "
        f"{bloch_rt['bottom_incoming_ratio']:.6e}"
    )

    print()
    print("DIFFERENCES")

    print(
        f"|R_bloch - R_ref|      = "
        f"{abs(bloch_rt['R'] - ref_rt['R']):.6e}"
    )

    print(
        f"|T_bloch - T_ref|      = "
        f"{abs(bloch_rt['T'] - ref_rt['T']):.6e}"
    )

    print(
        f"field relative L2 diff = "
        f"{field_difference:.6e}"
    )

    # --------------------------------------------------
    # Pass/fail
    # --------------------------------------------------

    passed = (
        mismatch < 1e-12
        and abs(
            bloch_rt["R"]
            - ref_rt["R"]
        ) < 1e-6
        and abs(
            bloch_rt["T"]
            - ref_rt["T"]
        ) < 1e-6
        and field_difference < 1e-6
        and abs(
            bloch_rt["R_plus_T"] - 1.0
        ) < 5e-4
        and abs(
            bloch_rt["R"] - R_exact
        ) < 5e-4
    )

    print()

    print(
        "RESULT:",
        "PASS"
        if passed
        else "FAIL",
    )

if __name__ == "__main__":
    main()