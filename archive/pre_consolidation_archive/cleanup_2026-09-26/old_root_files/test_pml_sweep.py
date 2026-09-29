import numpy as np

from photonics_fem.analytics import slab_rt
from photonics_fem.pml import BottomPMLConfig
from photonics_fem.postprocess import fit_plane_waves
from photonics_fem.solver_pml import solve_slab_bottom_pml


def run_case(
    thickness: float,
    sigma_max: float,
    order: int = 3,
) -> dict[str, float]:

    cfg = BottomPMLConfig(
        wavelength=0.633,
        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=0.50,
        air_top=0.50,
        air_bottom=0.50,
        width=0.50,

        pml_bottom=thickness,
        pml_order=order,
        pml_sigma_max=sigma_max,
    )

    ppw = 18

    h_target = (
        cfg.lambda_min_material
        / ppw
    )

    sol = solve_slab_bottom_pml(
        cfg,
        h_target=h_target,
        order=2,
    )

    rt = fit_plane_waves(sol)

    _, _, R_exact, T_exact = slab_rt(cfg)

    return {
        "R": rt["R"],
        "T": rt["T"],
        "R_error": abs(rt["R"] - R_exact),
        "T_error": abs(rt["T"] - T_exact),
        "incoming": rt["bottom_incoming_ratio"],
        "dofs": sol.basis.N,
    }


def main() -> None:

    thicknesses = [
        0.20,
        0.30,
        0.40,
        0.60,
        0.80,
    ]

    sigma_values = [
        1.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]

    print()
    print("=" * 94)
    print("PML THICKNESS SWEEP")
    print("=" * 94)

    print(
        f"{'L':>6} "
        f"{'R':>14} "
        f"{'T':>14} "
        f"{'|dR|':>12} "
        f"{'|dT|':>12} "
        f"{'bottom-in':>12}"
    )

    for L in thicknesses:

        result = run_case(
            thickness=L,
            sigma_max=6.0,
            order=3,
        )

        print(
            f"{L:6.2f} "
            f"{result['R']:14.10f} "
            f"{result['T']:14.10f} "
            f"{result['R_error']:12.3e} "
            f"{result['T_error']:12.3e} "
            f"{result['incoming']:12.3e}"
        )

    print()
    print("=" * 94)
    print("PML SIGMA_MAX SWEEP")
    print("=" * 94)

    print(
        f"{'sigma':>6} "
        f"{'R':>14} "
        f"{'T':>14} "
        f"{'|dR|':>12} "
        f"{'|dT|':>12} "
        f"{'bottom-in':>12}"
    )

    for sigma in sigma_values:

        result = run_case(
            thickness=0.60,
            sigma_max=sigma,
            order=3,
        )

        print(
            f"{sigma:6.1f} "
            f"{result['R']:14.10f} "
            f"{result['T']:14.10f} "
            f"{result['R_error']:12.3e} "
            f"{result['T_error']:12.3e} "
            f"{result['incoming']:12.3e}"
        )


if __name__ == "__main__":
    main()