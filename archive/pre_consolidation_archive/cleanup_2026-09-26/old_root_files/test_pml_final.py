import numpy as np

from photonics_fem.bloch import bloch_mismatch

from photonics_fem.full_pml import FullPMLConfig

from photonics_fem.oblique import (
    outgoing_ky,
    slab_rt_te_oblique,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
)

from photonics_fem.solver_generic_pml import (
    solve_generic_pml_te,
)


def predicted_pml_decay(
    cfg: FullPMLConfig,
    angle_deg: float,
    n_medium: complex,
    thickness: float,
) -> float:
    """Expected plane-wave amplitude ratio at 90% PML depth."""

    theta = np.deg2rad(angle_deg)

    kx = (
        cfg.k0
        * cfg.n_inc
        * np.sin(theta)
    )

    ky = outgoing_ky(
        cfg.k0,
        n_medium,
        kx,
    )

    fraction = 0.90

    integrated_sigma = (
        cfg.pml_sigma_max
        * thickness
        / (cfg.pml_order + 1)
        * fraction ** (cfg.pml_order + 1)
    )

    return float(
        np.exp(
            -abs(np.real(ky))
            * integrated_sigma
        )
    )


def pml_decay_ratios(
    result,
) -> tuple[float, float]:
    """Measure scattered-field decay 90% into both PMLs."""

    cfg = result.scattered.cfg

    # --------------------------------------------------
    # Top PML
    # --------------------------------------------------

    y_top = np.linspace(
        -0.90 * cfg.pml_top,
        0.0,
        400,
    )

    u_top = sample_scattered_centerline(
        result,
        y_top,
    )

    top_deep = abs(
        u_top[0]
    )

    top_entrance = abs(
        u_top[-1]
    )

    top_ratio = (
        top_deep
        / max(
            top_entrance,
            1e-30,
        )
    )

    # --------------------------------------------------
    # Bottom PML
    # --------------------------------------------------

    y_bottom = np.linspace(
        cfg.total_height,
        cfg.total_height
        + 0.90 * cfg.pml_bottom,
        400,
    )

    u_bottom = sample_scattered_centerline(
        result,
        y_bottom,
    )

    bottom_entrance = abs(
        u_bottom[0]
    )

    bottom_deep = abs(
        u_bottom[-1]
    )

    bottom_ratio = (
        bottom_deep
        / max(
            bottom_entrance,
            1e-30,
        )
    )

    return (
        float(top_ratio),
        float(bottom_ratio),
    )


def relative_decay_error(
    measured: float,
    expected: float,
) -> float:
    """Relative error in measured PML attenuation."""

    return float(
        abs(
            measured - expected
        )
        / max(
            abs(expected),
            1e-30,
        )
    )


def solve_case(
    wavelength: float,
    angle_deg: float,
    pml_thickness: float = 0.60,
    sigma_max: float = 6.0,
    pml_order: int = 3,
) -> dict[str, float]:

    cfg = FullPMLConfig(
        wavelength=wavelength,

        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=0.50,

        air_top=0.50,
        air_bottom=0.50,

        width=0.50,

        pml_top=pml_thickness,
        pml_bottom=pml_thickness,

        pml_order=pml_order,
        pml_sigma_max=sigma_max,
    )

    # --------------------------------------------------
    # Keep the FEM resolution relative to the shortest
    # physical wavelength.
    # --------------------------------------------------

    h_target = (
        cfg.lambda_min_material
        / 18
    )

    result = solve_generic_pml_te(
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

    top_decay, bottom_decay = (
        pml_decay_ratios(
            result
        )
    )

    expected_top_decay = (
        predicted_pml_decay(
            cfg,
            angle_deg,
            cfg.n_inc,
            cfg.pml_top,
        )
    )

    expected_bottom_decay = (
        predicted_pml_decay(
            cfg,
            angle_deg,
            cfg.n_out,
            cfg.pml_bottom,
        )
    )

    top_decay_error = (
        relative_decay_error(
            top_decay,
            expected_top_decay,
        )
    )

    bottom_decay_error = (
        relative_decay_error(
            bottom_decay,
            expected_bottom_decay,
        )
    )

    return {
        "R": fem["R"],
        "T": fem["T"],

        "R_exact": R_exact,
        "T_exact": T_exact,

        "R_error": abs(
            fem["R"]
            - R_exact
        ),

        "T_error": abs(
            fem["T"]
            - T_exact
        ),

        "energy_error": abs(
            fem["R_plus_T"]
            - 1.0
        ),

        "bloch_mismatch": mismatch,

        "top_decay": top_decay,
        "bottom_decay": bottom_decay,

        "expected_top_decay": expected_top_decay,
        "expected_bottom_decay": expected_bottom_decay,

        "top_decay_error": top_decay_error,
        "bottom_decay_error": bottom_decay_error,

        "dofs": result.scattered.basis.N,
    }


def optical_regression() -> bool:
    """Validate wavelength and angle dependence."""

    cases = [
        (0.500, 0.0),
        (0.500, 30.0),

        (0.633, 0.0),
        (0.633, 30.0),
        (0.633, 50.0),

        # Fabry-Perot resonance
        (0.850, 0.0),
    ]

    print()
    print("=" * 146)
    print("FINAL PML OPTICAL REGRESSION")
    print("=" * 146)

    print(
        f"{'lambda':>7} "
        f"{'angle':>7} "
        f"{'R FEM':>12} "
        f"{'R exact':>12} "
        f"{'|dR|':>10} "
        f"{'|dT|':>10} "
        f"{'energy':>10} "
        f"{'top FEM':>10} "
        f"{'top theory':>11} "
        f"{'bot FEM':>10} "
        f"{'bot theory':>11}"
    )

    all_pass = True

    for wavelength, angle in cases:

        result = solve_case(
            wavelength=wavelength,
            angle_deg=angle,
        )

        # --------------------------------------------------
        # Here we no longer require an arbitrary fixed
        # attenuation such as < 0.02.
        #
        # Instead, we check whether the measured PML decay
        # agrees with the coordinate-stretch prediction.
        # --------------------------------------------------

        passed = (
            result["R_error"] < 5e-4
            and result["T_error"] < 5e-4
            and result["energy_error"] < 5e-4
            and result["bloch_mismatch"] < 1e-12
            and result["top_decay_error"] < 0.10
            and result["bottom_decay_error"] < 0.10
        )

        all_pass &= passed

        print(
            f"{wavelength:7.3f} "
            f"{angle:7.1f} "
            f"{result['R']:12.9f} "
            f"{result['R_exact']:12.9f} "
            f"{result['R_error']:10.3e} "
            f"{result['T_error']:10.3e} "
            f"{result['energy_error']:10.3e} "
            f"{result['top_decay']:10.3e} "
            f"{result['expected_top_decay']:11.3e} "
            f"{result['bottom_decay']:10.3e} "
            f"{result['expected_bottom_decay']:11.3e}"
        )

    print()

    print(
        "OPTICAL REGRESSION:",
        "PASS"
        if all_pass
        else "FAIL",
    )

    return all_pass


def pml_parameter_regression() -> bool:
    """Check stability against arbitrary PML parameters."""

    settings = [
        # thickness, sigma_max, order
        (0.40, 6.0, 3),
        (0.60, 4.0, 3),
        (0.60, 6.0, 3),
        (0.80, 6.0, 3),
    ]

    wavelength = 0.633
    angle = 30.0

    print()
    print("=" * 144)
    print("FINAL PML PARAMETER REGRESSION")
    print("=" * 144)

    print(
        f"{'L':>5} "
        f"{'sigma':>7} "
        f"{'m':>3} "
        f"{'R':>13} "
        f"{'T':>13} "
        f"{'|dR|':>10} "
        f"{'|dT|':>10} "
        f"{'top FEM':>10} "
        f"{'top theory':>11} "
        f"{'bot FEM':>10} "
        f"{'bot theory':>11}"
    )

    all_pass = True
    results = []

    for thickness, sigma, order in settings:

        result = solve_case(
            wavelength=wavelength,
            angle_deg=angle,

            pml_thickness=thickness,
            sigma_max=sigma,
            pml_order=order,
        )

        results.append(
            result
        )

        # Weak PML configurations are allowed to attenuate
        # less strongly. What matters is:
        #
        # 1. their decay follows the theoretical stretch;
        # 2. R/T remains acceptably accurate;
        # 3. the strong configurations converge to a plateau.

        passed = (
            result["R_error"] < 5e-4
            and result["T_error"] < 5e-4
            and result["energy_error"] < 5e-4
            and result["bloch_mismatch"] < 1e-12
            and result["top_decay_error"] < 0.10
            and result["bottom_decay_error"] < 0.10
        )

        all_pass &= passed

        print(
            f"{thickness:5.2f} "
            f"{sigma:7.2f} "
            f"{order:3d} "
            f"{result['R']:13.10f} "
            f"{result['T']:13.10f} "
            f"{result['R_error']:10.3e} "
            f"{result['T_error']:10.3e} "
            f"{result['top_decay']:10.3e} "
            f"{result['expected_top_decay']:11.3e} "
            f"{result['bottom_decay']:10.3e} "
            f"{result['expected_bottom_decay']:11.3e}"
        )

    # --------------------------------------------------
    # Strong-PML convergence plateau.
    #
    # Compare:
    #
    # L = 0.60, sigma = 6
    #
    # against
    #
    # L = 0.80, sigma = 6
    # --------------------------------------------------

    strong_a = results[2]
    strong_b = results[3]

    plateau_dR = abs(
        strong_a["R"]
        - strong_b["R"]
    )

    plateau_dT = abs(
        strong_a["T"]
        - strong_b["T"]
    )

    print()
    print("CONVERGENCE PLATEAU")

    print(
        f"|R(L=0.6) - R(L=0.8)| = "
        f"{plateau_dR:.6e}"
    )

    print(
        f"|T(L=0.6) - T(L=0.8)| = "
        f"{plateau_dT:.6e}"
    )

    plateau_pass = (
        plateau_dR < 1e-5
        and plateau_dT < 1e-5
    )

    all_pass &= plateau_pass

    print()

    print(
        "PML PARAMETER REGRESSION:",
        "PASS"
        if all_pass
        else "FAIL",
    )

    return all_pass


def main() -> None:

    optical_pass = (
        optical_regression()
    )

    parameter_pass = (
        pml_parameter_regression()
    )

    print()
    print("=" * 146)

    if (
        optical_pass
        and parameter_pass
    ):
        print(
            "FINAL RESULT: FULL PML IMPLEMENTATION PASS"
        )

    else:
        print(
            "FINAL RESULT: PML VALIDATION FAILED"
        )

    print("=" * 146)


if __name__ == "__main__":
    main()