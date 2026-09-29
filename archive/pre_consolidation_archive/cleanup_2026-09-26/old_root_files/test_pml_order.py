from photonics_fem.analytics import slab_rt
from photonics_fem.pml import BottomPMLConfig
from photonics_fem.postprocess import fit_plane_waves
from photonics_fem.solver_pml import solve_slab_bottom_pml


def main() -> None:

    L = 0.60

    # Match the integrated PML strength of:
    #
    # m = 3
    # sigma_max = 6
    #
    # integral sigma dy = sigma_max * L / (m + 1)
    #                   = 0.9

    target_integral = 0.90

    orders = [1, 2, 3, 4, 5]

    print()
    print("=" * 90)
    print("PML POLYNOMIAL-ORDER SWEEP")
    print("=" * 90)

    print(
        f"{'m':>4} "
        f"{'sigma_max':>10} "
        f"{'R':>14} "
        f"{'T':>14} "
        f"{'|dR|':>12} "
        f"{'bottom-in':>12}"
    )

    for m in orders:

        sigma_max = (
            target_integral
            * (m + 1)
            / L
        )

        cfg = BottomPMLConfig(
            wavelength=0.633,
            n_inc=1.0 + 0.0j,
            n_slab=1.7 + 0.0j,
            n_out=1.0 + 0.0j,

            slab_thickness=0.50,
            air_top=0.50,
            air_bottom=0.50,
            width=0.50,

            pml_bottom=L,
            pml_order=m,
            pml_sigma_max=sigma_max,
        )

        h_target = (
            cfg.lambda_min_material
            / 18
        )

        sol = solve_slab_bottom_pml(
            cfg,
            h_target=h_target,
            order=2,
        )

        rt = fit_plane_waves(sol)

        _, _, R_exact, _ = slab_rt(cfg)

        print(
            f"{m:4d} "
            f"{sigma_max:10.3f} "
            f"{rt['R']:14.10f} "
            f"{rt['T']:14.10f} "
            f"{abs(rt['R'] - R_exact):12.3e} "
            f"{rt['bottom_incoming_ratio']:12.3e}"
        )


if __name__ == "__main__":
    main()