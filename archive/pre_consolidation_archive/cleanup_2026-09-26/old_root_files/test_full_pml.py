import numpy as np

from photonics_fem.bloch import (
    bloch_mismatch,
)

from photonics_fem.full_pml import (
    FullPMLConfig,
)

from photonics_fem.oblique import (
    slab_rt_te_oblique,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
)

from photonics_fem.solver_full_pml import (
    solve_full_pml_te,
)


def run_case(
    angle_deg: float,
) -> bool:

    cfg = FullPMLConfig(
        wavelength=0.633,

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

    h_target = (
        cfg.lambda_min_material
        / 18
    )

    print()
    print(
        f"Solving full-PML TE case at "
        f"{angle_deg:.1f} degrees..."
    )

    result = solve_full_pml_te(
        cfg,
        angle_deg=angle_deg,
        h_target=h_target,
        order=2,
    )

    fem = fit_full_pml_rt(
        result
    )

    _, _, R_exact, T_exact = (
        slab_rt_te_oblique(
            cfg,
            angle_deg,
        )
    )

    mismatch = bloch_mismatch(
        result.scattered.u,
        result.reduction,
    )

    # --------------------------------------------------
    # PML decay check:
    #
    # We measure the SCATTERED field because the
    # analytically known incident field is not part
    # of the FEM PML field.
    # --------------------------------------------------

    y_top = np.linspace(
        -0.90 * cfg.pml_top,
        0.0,
        400,
    )

    us_top = (
        sample_scattered_centerline(
            result,
            y_top,
        )
    )

    # PML entrance is last point (y = 0).
    top_entrance = abs(
        us_top[-1]
    )

    top_deep = abs(
        us_top[0]
    )

    top_decay = (
        top_deep
        / max(
            top_entrance,
            1e-30,
        )
    )

    y_bottom = np.linspace(
        cfg.total_height,
        cfg.total_height
        + 0.90 * cfg.pml_bottom,
        400,
    )

    us_bottom = (
        sample_scattered_centerline(
            result,
            y_bottom,
        )
    )

    bottom_entrance = abs(
        us_bottom[0]
    )

    bottom_deep = abs(
        us_bottom[-1]
    )

    bottom_decay = (
        bottom_deep
        / max(
            bottom_entrance,
            1e-30,
        )
    )

    R_error = abs(
        fem["R"]
        - R_exact
    )

    T_error = abs(
        fem["T"]
        - T_exact
    )

    print()
    print("=" * 70)
    print(
        f"FULL PML VALIDATION: "
        f"{angle_deg:.1f} DEG"
    )
    print("=" * 70)

    print()
    print("SYSTEM")

    print(
        f"full FEM DOFs          = "
        f"{result.scattered.basis.N}"
    )

    print(
        f"reduced DOFs           = "
        f"{result.reduced_dofs}"
    )

    print()
    print("BLOCH")

    print(
        f"phase                  = "
        f"{result.reduction.phase}"
    )

    print(
        f"mismatch               = "
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
    print("FULL PML FEM")

    print(
        f"R                      = "
        f"{fem['R']:.12f}"
    )

    print(
        f"T                      = "
        f"{fem['T']:.12f}"
    )

    print(
        f"R + T                  = "
        f"{fem['R_plus_T']:.12f}"
    )

    print()
    print("ERROR")

    print(
        f"|dR|                   = "
        f"{R_error:.6e}"
    )

    print(
        f"|dT|                   = "
        f"{T_error:.6e}"
    )

    print()
    print("SCATTERED-FIELD PML DECAY")

    print(
        f"top 90%/entrance       = "
        f"{top_decay:.6e}"
    )

    print(
        f"bottom 90%/entrance    = "
        f"{bottom_decay:.6e}"
    )

    passed = (
        mismatch < 1e-12
        and R_error < 5e-4
        and T_error < 5e-4
        and abs(
            fem["R_plus_T"]
            - 1.0
        ) < 5e-4
        and top_decay < 1e-2
        and bottom_decay < 1e-2
    )

    print()

    print(
        "RESULT:",
        "PASS"
        if passed
        else "FAIL",
    )

    return passed


def main() -> None:

    normal_pass = run_case(
        0.0
    )

    oblique_pass = run_case(
        30.0
    )

    print()
    print("=" * 70)

    print(
        "OVERALL:",
        "ALL FULL-PML TESTS PASS"
        if (
            normal_pass
            and oblique_pass
        )
        else "SOME FULL-PML TESTS FAILED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()