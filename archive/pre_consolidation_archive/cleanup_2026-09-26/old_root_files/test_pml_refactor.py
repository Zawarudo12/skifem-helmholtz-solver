import numpy as np

from photonics_fem.full_pml import (
    FullPMLConfig,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
    sample_scattered_centerline,
)

from photonics_fem.solver_full_pml import (
    solve_full_pml_te,
)

from photonics_fem.solver_generic_pml import (
    solve_generic_pml_te,
)


def relative_l2(
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
) -> float:

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
        f"Testing generic PML refactor at "
        f"{angle_deg:.1f} deg..."
    )

    old = solve_full_pml_te(
        cfg,
        angle_deg=angle_deg,
        h_target=h_target,
        order=2,
    )

    new = solve_generic_pml_te(
        cfg,
        angle_deg=angle_deg,
        h_target=h_target,
        order=2,
    )

    # The postprocessor only requires the result object
    # to contain scattered, kx and ky_inc, so the new
    # result is compatible.

    old_rt = fit_full_pml_rt(
        old
    )

    new_rt = fit_full_pml_rt(
        new
    )

    y = np.linspace(
        cfg.computational_ymin,
        cfg.computational_ymax,
        3501,
    )

    old_field = sample_scattered_centerline(
        old,
        y,
    )

    new_field = sample_scattered_centerline(
        new,
        y,
    )

    field_error = relative_l2(
        new_field,
        old_field,
        y,
    )

    dR = abs(
        new_rt["R"]
        - old_rt["R"]
    )

    dT = abs(
        new_rt["T"]
        - old_rt["T"]
    )

    print()
    print("=" * 70)

    print(
        f"PML REFACTOR VALIDATION: "
        f"{angle_deg:.1f} DEG"
    )

    print("=" * 70)

    print()
    print("OLD FULL PML")

    print(
        f"R = {old_rt['R']:.12f}"
    )

    print(
        f"T = {old_rt['T']:.12f}"
    )

    print()
    print("NEW GENERIC PML")

    print(
        f"R = {new_rt['R']:.12f}"
    )

    print(
        f"T = {new_rt['T']:.12f}"
    )

    print()
    print("DIFFERENCES")

    print(
        f"|dR|                   = "
        f"{dR:.6e}"
    )

    print(
        f"|dT|                   = "
        f"{dT:.6e}"
    )

    print(
        f"field relative L2      = "
        f"{field_error:.6e}"
    )

    passed = (
        dR < 1e-8
        and dT < 1e-8
        and field_error < 1e-8
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

    normal = run_case(
        0.0
    )

    oblique = run_case(
        30.0
    )

    print()
    print("=" * 70)

    print(
        "OVERALL:",
        "GENERIC PML REFACTOR PASS"
        if (
            normal
            and oblique
        )
        else "GENERIC PML REFACTOR FAIL",
    )

    print("=" * 70)


if __name__ == "__main__":
    main()