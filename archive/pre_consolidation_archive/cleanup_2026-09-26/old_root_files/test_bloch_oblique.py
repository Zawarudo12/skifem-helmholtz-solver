import numpy as np

from photonics_fem.bloch import (
    bloch_mismatch,
)

from photonics_fem.oblique import (
    slab_rt_te_oblique,
)

from photonics_fem.pml import (
    BottomPMLConfig,
)

from photonics_fem.postprocess_oblique import (
    fit_oblique_plane_waves,
)

from photonics_fem.solver_oblique import (
    solve_oblique_te,
)


def main() -> None:

    angle_deg = 30.0

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

    h_target = (
        cfg.lambda_min_material
        / 18
    )

    print(
        f"Solving TE incidence at "
        f"{angle_deg:.1f} degrees..."
    )

    result = solve_oblique_te(
        cfg,
        angle_deg=angle_deg,
        h_target=h_target,
        order=2,
    )

    sol = result.solution

    fem = fit_oblique_plane_waves(
        sol,
        result.kx,
    )

    r_exact, t_exact, R_exact, T_exact = (
        slab_rt_te_oblique(
            cfg,
            angle_deg,
        )
    )

    mismatch = bloch_mismatch(
        sol.u,
        result.reduction,
    )

    expected_phase = np.exp(
        1j
        * result.kx
        * cfg.width
    )

    print()
    print("=" * 68)
    print("OBLIQUE TE + BLOCH + PML VALIDATION")
    print("=" * 68)

    print()
    print("INCIDENCE")

    print(
        f"angle                  = "
        f"{angle_deg:.3f} deg"
    )

    print(
        f"kx                     = "
        f"{result.kx}"
    )

    print(
        f"ky incident            = "
        f"{result.ky_inc}"
    )

    print()
    print("BLOCH")

    print(
        f"expected phase         = "
        f"{expected_phase}"
    )

    print(
        f"implemented phase      = "
        f"{result.reduction.phase}"
    )

    print(
        f"relative mismatch      = "
        f"{mismatch:.6e}"
    )

    print()
    print("SYSTEM")

    print(
        f"full DOFs              = "
        f"{sol.basis.N}"
    )

    print(
        f"reduced DOFs           = "
        f"{result.reduced_dofs}"
    )

    print()
    print("ANALYTIC TE")

    print(
        f"R                      = "
        f"{R_exact:.12f}"
    )

    print(
        f"T                      = "
        f"{T_exact:.12f}"
    )

    print(
        f"R + T                  = "
        f"{R_exact + T_exact:.12f}"
    )

    print()
    print("FEM")

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

    print(
        f"bottom incoming        = "
        f"{fem['bottom_incoming_ratio']:.6e}"
    )

    print()
    print("ERROR")

    R_error = abs(
        fem["R"]
        - R_exact
    )

    T_error = abs(
        fem["T"]
        - T_exact
    )

    print(
        f"|R_FEM - R_exact|      = "
        f"{R_error:.6e}"
    )

    print(
        f"|T_FEM - T_exact|      = "
        f"{T_error:.6e}"
    )

    passed = (
        mismatch < 1e-12
        and R_error < 5e-4
        and T_error < 5e-4
        and abs(
            fem["R_plus_T"]
            - 1.0
        ) < 5e-4
        and fem[
            "bottom_incoming_ratio"
        ] < 1e-3
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