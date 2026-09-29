from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.pml import (
    BottomPMLConfig,
)

from photonics_fem.solver_bloch import (
    solve_slab_pml_bloch,
)

from photonics_fem.postprocess import (
    sample_centerline,
)

from photonics_fem.full_pml import (
    FullPMLConfig,
)

from photonics_fem.solver_generic_pml import (
    solve_generic_pml_te,
)

from photonics_fem.postprocess_full_pml import (
    fit_full_pml_rt,
)

from photonics_fem.oblique import (
    slab_rt_te_oblique,
)

OUT = Path("results_objectives_1_2")
OUT.mkdir(exist_ok=True)

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    26,
)

PPW = 12

PERIOD = 1.0

def fresnel_single_interface(
    n1: complex,
    n2: complex,
) -> tuple[float, float]:

    r = (
        n1 - n2
    ) / (
        n1 + n2
    )

    t = (
        2.0 * n1
    ) / (
        n1 + n2
    )

    R = float(
        abs(r) ** 2
    )

    T = float(
        np.real(n2)
        / np.real(n1)
        * abs(t) ** 2
    )

    return R, T

def extract_interface_rt(
    sol,
    cfg,
) -> dict[str, float]:

    yt = np.linspace(
        0.08 * cfg.air_top,
        0.82 * cfg.air_top,
        100,
    )

    ut = sample_centerline(
        sol,
        yt,
    )

    yb = np.linspace(
        cfg.slab_y1
        + 0.15 * cfg.air_bottom,

        cfg.total_height
        - 0.10 * cfg.air_bottom,

        100,
    )

    ub = sample_centerline(
        sol,
        yb,
    )

    k_top = (
        cfg.k0
        * cfg.n_inc
    )

    k_bottom = (
        cfg.k0
        * cfg.n_out
    )

    Xt = np.column_stack(
        [
            np.exp(
                1j
                * k_top
                * yt
            ),

            np.exp(
                -1j
                * k_top
                * yt
            ),
        ]
    )

    a_inc, a_refl = np.linalg.lstsq(
        Xt,
        ut,
        rcond=None,
    )[0]

    Xb = np.column_stack(
        [
            np.exp(
                1j
                * k_bottom
                * yb
            ),

            np.exp(
                -1j
                * k_bottom
                * yb
            ),
        ]
    )

    a_trans, a_in_bottom = np.linalg.lstsq(
        Xb,
        ub,
        rcond=None,
    )[0]

    R = float(
        abs(
            a_refl
            / a_inc
        ) ** 2
    )

    T = float(
        np.real(k_bottom)
        / np.real(k_top)
        * abs(
            a_trans
            / a_inc
        ) ** 2
    )

    return {
        "R": R,
        "T": T,
        "energy": R + T,

        "bottom_incoming": float(
            abs(
                a_in_bottom
                / a_inc
            )
        ),
    }

def run_objective_1():

    print()
    print("=" * 76)
    print("OBJECTIVE 1")
    print("AIR -> SEMI-INFINITE n = 1.5")
    print("=" * 76)

    R_exact, T_exact = (
        fresnel_single_interface(
            1.0,
            1.5,
        )
    )

    print()

    print(
        f"Analytic Fresnel: "
        f"R={R_exact:.8f}, "
        f"T={T_exact:.8f}"
    )

    rows = []

    for i, wavelength in enumerate(
        WAVELENGTHS,
        start=1,
    ):

        cfg = BottomPMLConfig(
            wavelength=wavelength,

            n_inc=1.0 + 0.0j,

            n_slab=1.5 + 0.0j,

            n_out=1.5 + 0.0j,

            air_top=0.50,

            slab_thickness=0.50,

            air_bottom=0.50,

            width=PERIOD,

            pml_bottom=0.80,

            pml_order=3,

            pml_sigma_max=6.0,
        )

        h_target = (
            cfg.lambda_min_material
            / PPW
        )

        result = solve_slab_pml_bloch(
            cfg,
            h_target=h_target,
            order=2,
            kx=0.0,
        )

        rt = extract_interface_rt(
            result.solution,
            cfg,
        )

        dR = abs(
            rt["R"]
            - R_exact
        )

        dT = abs(
            rt["T"]
            - T_exact
        )

        rows.append(
            [
                wavelength,
                rt["R"],
                rt["T"],
                R_exact,
                T_exact,
                dR,
                dT,
                abs(
                    rt["energy"]
                    - 1.0
                ),
                rt["bottom_incoming"],
            ]
        )

        print(
            f"[{i:02d}/{len(WAVELENGTHS)}] "
            f"lambda={wavelength * 1000:6.1f} nm  "
            f"R={rt['R']:.6f}  "
            f"T={rt['T']:.6f}  "
            f"|dR|={dR:.2e}"
        )

    data = np.asarray(
        rows,
        dtype=float,
    )

    np.savetxt(
        OUT / "objective1_interface.csv",
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fem,"
            "T_fem,"
            "R_exact,"
            "T_exact,"
            "R_error,"
            "T_error,"
            "energy_error,"
            "bottom_incoming"
        ),
        comments="",
    )

    wavelength_nm = (
        data[:, 0]
        * 1000.0
    )

    plt.figure(
        figsize=(8.5, 5.5)
    )

    plt.plot(
        wavelength_nm,
        data[:, 1],
        "o-",
        label="FEM R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 2],
        "o-",
        label="FEM T",
    )

    plt.plot(
        wavelength_nm,
        data[:, 3],
        "--",
        label="Fresnel R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 4],
        "--",
        label="Fresnel T",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Objective 1: air -> n=1.5 interface"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "objective1_RT.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8.5, 5.0)
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 5],
        "o-",
        label="|R FEM - R exact|",
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 6],
        "o-",
        label="|T FEM - T exact|",
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 7],
        "o-",
        label="|R + T - 1|",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Absolute error"
    )

    plt.title(
        "Objective 1 numerical error"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "objective1_error.png",
        dpi=180,
    )

    plt.close()

    print()

    print(
        "Objective 1 maximum |dR| = "
        f"{np.max(data[:, 5]):.6e}"
    )

    print(
        "Objective 1 maximum |dT| = "
        f"{np.max(data[:, 6]):.6e}"
    )

    return data

def run_objective_2():

    print()
    print("=" * 76)
    print("OBJECTIVE 2")
    print("1 um THICK n = 1.5 SLAB IN AIR")
    print("=" * 76)

    rows = []

    for i, wavelength in enumerate(
        WAVELENGTHS,
        start=1,
    ):

        cfg = FullPMLConfig(
            wavelength=wavelength,

            n_inc=1.0 + 0.0j,

            n_slab=1.5 + 0.0j,

            n_out=1.0 + 0.0j,

            slab_thickness=1.0,

            air_top=0.50,

            air_bottom=0.50,

            width=PERIOD,

            pml_top=0.80,

            pml_bottom=0.80,

            pml_order=3,

            pml_sigma_max=6.0,
        )

        h_target = (
            cfg.lambda_min_material
            / PPW
        )

        result = solve_generic_pml_te(
            cfg,
            angle_deg=0.0,
            h_target=h_target,
            order=2,
        )

        rt = fit_full_pml_rt(
            result
        )

        _, _, R_exact, T_exact = (
            slab_rt_te_oblique(
                cfg,
                0.0,
            )
        )

        dR = abs(
            rt["R"]
            - R_exact
        )

        dT = abs(
            rt["T"]
            - T_exact
        )

        rows.append(
            [
                wavelength,
                rt["R"],
                rt["T"],
                R_exact,
                T_exact,
                dR,
                dT,
                abs(
                    rt["R_plus_T"]
                    - 1.0
                ),
            ]
        )

        print(
            f"[{i:02d}/{len(WAVELENGTHS)}] "
            f"lambda={wavelength * 1000:6.1f} nm  "
            f"R={rt['R']:.6f}  "
            f"T={rt['T']:.6f}  "
            f"|dR|={dR:.2e}"
        )

    data = np.asarray(
        rows,
        dtype=float,
    )

    np.savetxt(
        OUT / "objective2_slab.csv",
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fem,"
            "T_fem,"
            "R_exact,"
            "T_exact,"
            "R_error,"
            "T_error,"
            "energy_error"
        ),
        comments="",
    )

    wavelength_nm = (
        data[:, 0]
        * 1000.0
    )

    plt.figure(
        figsize=(9.0, 5.5)
    )

    plt.plot(
        wavelength_nm,
        data[:, 1],
        "o",
        label="FEM R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 2],
        "o",
        label="FEM T",
    )

    plt.plot(
        wavelength_nm,
        data[:, 3],
        "-",
        label="analytic R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 4],
        "-",
        label="analytic T",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Objective 2: "
        "1 um n=1.5 dielectric slab"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "objective2_RT.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(8.5, 5.0)
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 5],
        "o-",
        label="|R FEM - R exact|",
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 6],
        "o-",
        label="|T FEM - T exact|",
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 7],
        "o-",
        label="|R + T - 1|",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Absolute error"
    )

    plt.title(
        "Objective 2 numerical error"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "objective2_error.png",
        dpi=180,
    )

    plt.close()

    print()

    print(
        "Objective 2 maximum |dR| = "
        f"{np.max(data[:, 5]):.6e}"
    )

    print(
        "Objective 2 maximum |dT| = "
        f"{np.max(data[:, 6]):.6e}"
    )

    return data

def main() -> None:

    print()
    print("=" * 76)
    print("ORIGINAL PROJECT OBJECTIVES: 300-800 nm")
    print("=" * 76)

    run_objective_1()

    run_objective_2()

    print()
    print("=" * 76)

    print(
        "OBJECTIVES 1 AND 2 COMPLETE."
    )

    print(
        "Next: Objective 3 periodic circular hole "
        "+ diffraction-order R/T."
    )

    print("=" * 76)

    print()

    print(
        f"Results written to: {OUT}"
    )

if __name__ == "__main__":
    main()
