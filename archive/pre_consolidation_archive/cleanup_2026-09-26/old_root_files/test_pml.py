from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.analytics import (
    slab_rt,
)

from photonics_fem.postprocess import (
    fit_plane_waves,
    relative_l2_profile_error,
    sample_centerline,
)

from photonics_fem.solver import (
    solve_slab,
)

from photonics_fem.solver_pml import (
    solve_slab_bottom_pml,
)

from photonics_fem.pml import (
    BottomPMLConfig,
)


OUT = Path("results_pml")
OUT.mkdir(exist_ok=True)


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

    print(
        "Solving old Robin model..."
    )

    robin = solve_slab(
        cfg,
        h_target=h_target,
        order=2,
    )

    print(
        "Solving new PML model..."
    )

    pml = solve_slab_bottom_pml(
        cfg,
        h_target=h_target,
        order=2,
    )

    robin_rt = fit_plane_waves(
        robin
    )

    pml_rt = fit_plane_waves(
        pml
    )

    _, _, R_exact, T_exact = slab_rt(
        cfg
    )

    pml_l2 = relative_l2_profile_error(
        pml
    )

    print()
    print("=" * 62)
    print("BOTTOM PML VALIDATION")
    print("=" * 62)

    print()
    print("ANALYTIC")
    print(
        f"R = {R_exact:.12f}"
    )
    print(
        f"T = {T_exact:.12f}"
    )

    print()
    print("OLD ROBIN")
    print(
        f"R = {robin_rt['R']:.12f}"
    )
    print(
        f"T = {robin_rt['T']:.12f}"
    )
    print(
        f"R+T = "
        f"{robin_rt['R_plus_T']:.12f}"
    )

    print()
    print("NEW PML")
    print(
        f"DOFs = {pml.basis.N}"
    )
    print(
        f"R = {pml_rt['R']:.12f}"
    )
    print(
        f"T = {pml_rt['T']:.12f}"
    )
    print(
        f"R+T = "
        f"{pml_rt['R_plus_T']:.12f}"
    )

    print(
        f"bottom incoming ratio = "
        f"{pml_rt['bottom_incoming_ratio']:.6e}"
    )

    print(
        f"relative physical-region L2 error = "
        f"{pml_l2:.6e}"
    )

    print()
    print("PML VS ANALYTIC")

    R_error = abs(
        pml_rt["R"]
        - R_exact
    )

    T_error = abs(
        pml_rt["T"]
        - T_exact
    )

    print(
        f"|R - R_exact| = "
        f"{R_error:.6e}"
    )

    print(
        f"|T - T_exact| = "
        f"{T_error:.6e}"
    )

    # --------------------------------------------------
    # Examine decay inside the PML.
    # --------------------------------------------------

    y_pml = np.linspace(
        cfg.pml_y0,
        cfg.computational_height,
        500,
    )

    u_pml = sample_centerline(
        pml,
        y_pml,
    )

    entrance_amp = abs(
        u_pml[0]
    )

    i90 = int(
        0.90
        * (len(y_pml) - 1)
    )

    amp_90 = abs(
        u_pml[i90]
    )

    relative_amp_90 = (
        amp_90
        / entrance_amp
    )

    print()
    print("PML DECAY")

    print(
        f"|E| at PML entrance = "
        f"{entrance_amp:.6e}"
    )

    print(
        f"|E| at 90% depth    = "
        f"{amp_90:.6e}"
    )

    print(
        f"relative amplitude   = "
        f"{relative_amp_90:.6e}"
    )

    # --------------------------------------------------
    # Plot physical field + PML decay.
    # --------------------------------------------------

    y_all = np.linspace(
        0.0,
        cfg.computational_height,
        2500,
    )

    u_all = sample_centerline(
        pml,
        y_all,
    )

    plt.figure(
        figsize=(7.5, 4.7)
    )

    plt.plot(
        y_all,
        np.abs(u_all),
        label="PML FEM |Ez|",
    )

    plt.axvspan(
        cfg.slab_y0,
        cfg.slab_y1,
        alpha=0.12,
        label="dielectric slab",
    )

    plt.axvspan(
        cfg.pml_y0,
        cfg.computational_height,
        alpha=0.18,
        label="PML",
    )

    plt.axvline(
        cfg.pml_y0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "depth y [um]"
    )

    plt.ylabel(
        "|Ez|"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "pml_decay.png",
        dpi=180,
    )

    plt.close()

    # --------------------------------------------------
    # Initial pass/fail thresholds.
    #
    # These are deliberately not ultra-tight yet.
    # First we want a functioning, stable PML.
    # Then we will optimize it.
    # --------------------------------------------------

    passed = (
        R_error < 5e-4
        and T_error < 5e-4
        and abs(
            pml_rt["R_plus_T"] - 1.0
        ) < 5e-4
        and pml_rt[
            "bottom_incoming_ratio"
        ] < 1e-3
        and pml_l2 < 1e-3
        and relative_amp_90 < 1e-2
    )

    print()
    print(
        "RESULT:",
        "PASS"
        if passed
        else "FAIL",
    )

    print()
    print(
        "Plot written to:",
        OUT / "pml_decay.png",
    )


if __name__ == "__main__":
    main()