from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.objective3_bodyfitted import (
    Objective3Config,
    diffraction_orders_bodyfitted,
    solve_objective3_bodyfitted,
)


# ============================================================
# PATHS
# ============================================================

OUT = Path(
    "results_objective3_neff"
)

OUT.mkdir(
    exist_ok=True
)

MESH_FILE = Path(
    "meshes/objective3_bodyfitted_fixed.msh"
)

FDTD_FILE = Path(
    "dataset_hollow_core_fdtd(1).csv"
)


# ============================================================
# COARSE EFFECTIVE-INDEX TEST
#
# Keep EVERYTHING fixed except n_slab.
#
# This is intentionally coarse first.  If one region looks
# promising, refine around that index afterward.
# ============================================================

N_VALUES = np.arange(
    1.10,
    2.51,
    0.20,
)


# Diagnostic wavelengths only.
#
# Deliberately exclude 0.500 um because that is the exact
# m=+-2 Rayleigh cutoff in the present 1-um-period 2D model.
#
# 0.720 um is kept because the supplied FDTD reference has
# its striking high-reflection feature in this region.
CHECK_WAVELENGTHS = np.array(
    [
        0.340,
        0.400,
        0.440,
        0.540,
        0.600,
        0.680,
        0.720,
        0.760,
        0.800,
    ],
    dtype=float,
)


def make_cfg(
    wavelength: float,
    n_eff: float,
) -> Objective3Config:

    return Objective3Config(
        wavelength=wavelength,

        n_inc=1.0 + 0.0j,
        n_slab=complex(n_eff),
        n_out=1.0 + 0.0j,

        slab_thickness=1.0,

        air_top=0.50,
        air_bottom=0.50,

        width=1.0,

        hole_diameter=0.50,

        pml_top=1.00,
        pml_bottom=1.00,

        pml_order=3,
        pml_sigma_max=8.0,
    )


def load_fdtd() -> np.ndarray:

    # Try whitespace first because that is what your supplied
    # FDTD reference used in the previous scripts.
    try:
        raw = np.loadtxt(
            FDTD_FILE
        )
    except ValueError:
        raw = np.loadtxt(
            FDTD_FILE,
            delimiter=",",
        )

    raw = np.asarray(
        raw,
        dtype=float,
    )

    if (
        raw.ndim != 2
        or raw.shape[1] < 3
    ):
        raise ValueError(
            "FDTD file must contain at least three columns: "
            "wavelength_um, R, T"
        )

    raw = raw[
        :,
        :3
    ]

    order = np.argsort(
        raw[:, 0]
    )

    return raw[
        order
    ]


def main():

    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"Missing fixed body-fitted mesh:\n{MESH_FILE}\n\n"
            "Run run_objective3_redo.py first."
        )

    if not FDTD_FILE.exists():
        raise FileNotFoundError(
            f"Missing FDTD reference:\n{FDTD_FILE}"
        )

    fdtd = load_fdtd()

    fdtd_R_check = np.interp(
        CHECK_WAVELENGTHS,
        fdtd[:, 0],
        fdtd[:, 1],
    )

    fdtd_T_check = np.interp(
        CHECK_WAVELENGTHS,
        fdtd[:, 0],
        fdtd[:, 2],
    )

    print()
    print("=" * 104)
    print("OBJECTIVE 3 EFFECTIVE-INDEX SWEEP")
    print("FIXED 2D TE GEOMETRY -- ONLY n_eff CHANGES")
    print("=" * 104)

    print()
    print(
        "n_eff candidates:",
        np.array2string(
            N_VALUES,
            precision=2,
        )
    )

    print(
        "check wavelengths [um]:",
        np.array2string(
            CHECK_WAVELENGTHS,
            precision=3,
        )
    )

    summary_rows = []
    detailed_rows = []

    best = None

    for ni, n_eff in enumerate(
        N_VALUES,
        start=1,
    ):

        fem_R = []
        fem_T = []
        energy_errors = []
        symmetry_errors = []

        print()
        print("-" * 104)

        print(
            f"[n {ni:02d}/{len(N_VALUES)}] "
            f"n_eff = {n_eff:.3f}"
        )

        for wi, wavelength in enumerate(
            CHECK_WAVELENGTHS,
            start=1,
        ):

            cfg = make_cfg(
                float(wavelength),
                float(n_eff),
            )

            result = solve_objective3_bodyfitted(
                cfg,
                MESH_FILE,
            )

            diff = diffraction_orders_bodyfitted(
                result,
                cfg,
                npoints=512,
            )

            fem_R.append(
                diff["R"]
            )

            fem_T.append(
                diff["T"]
            )

            energy_errors.append(
                diff["energy_error"]
            )

            symmetry_errors.append(
                diff["symmetry_abs_max"]
            )

            detailed_rows.append(
                [
                    n_eff,
                    wavelength,
                    diff["R"],
                    fdtd_R_check[
                        wi - 1
                    ],
                    diff["T"],
                    fdtd_T_check[
                        wi - 1
                    ],
                    diff["energy_error"],
                    diff["symmetry_abs_max"],
                ]
            )

            print(
                f"  [{wi:02d}/{len(CHECK_WAVELENGTHS)}] "
                f"lambda={wavelength*1000:6.1f} nm  "
                f"R={diff['R']:.5f}  "
                f"T={diff['T']:.5f}  "
                f"FDTD R={fdtd_R_check[wi-1]:.5f}  "
                f"FDTD T={fdtd_T_check[wi-1]:.5f}"
            )

        fem_R = np.asarray(
            fem_R,
            dtype=float,
        )

        fem_T = np.asarray(
            fem_T,
            dtype=float,
        )

        dR = np.abs(
            fem_R
            - fdtd_R_check
        )

        dT = np.abs(
            fem_T
            - fdtd_T_check
        )

        mae_R = float(
            np.mean(
                dR
            )
        )

        mae_T = float(
            np.mean(
                dT
            )
        )

        combined_mae = float(
            0.5
            * (
                mae_R
                + mae_T
            )
        )

        rmse_combined = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            (
                                fem_R
                                - fdtd_R_check
                            ) ** 2,
                            (
                                fem_T
                                - fdtd_T_check
                            ) ** 2,
                        ]
                    )
                )
            )
        )

        max_energy = float(
            np.max(
                energy_errors
            )
        )

        max_sym = float(
            np.max(
                symmetry_errors
            )
        )

        summary_rows.append(
            [
                n_eff,
                mae_R,
                mae_T,
                combined_mae,
                rmse_combined,
                max_energy,
                max_sym,
            ]
        )

        print()
        print(
            f"  MAE R        = {mae_R:.6f}"
        )

        print(
            f"  MAE T        = {mae_T:.6f}"
        )

        print(
            f"  combined MAE = {combined_mae:.6f}"
        )

        print(
            f"  combined RMSE= {rmse_combined:.6f}"
        )

        print(
            f"  max energy err = {max_energy:.3e}"
        )

        print(
            f"  max symmetry   = {max_sym:.3e}"
        )

        if (
            best is None
            or combined_mae
            < best["combined_mae"]
        ):

            best = {
                "n_eff": float(
                    n_eff
                ),
                "mae_R": mae_R,
                "mae_T": mae_T,
                "combined_mae": combined_mae,
                "rmse_combined": rmse_combined,
                "fem_R": fem_R.copy(),
                "fem_T": fem_T.copy(),
            }

    summary = np.asarray(
        summary_rows,
        dtype=float,
    )

    details = np.asarray(
        detailed_rows,
        dtype=float,
    )

    np.savetxt(
        OUT
        / "neff_summary.csv",
        summary,
        delimiter=",",
        header=(
            "n_eff,"
            "mae_R,"
            "mae_T,"
            "combined_mae,"
            "combined_rmse,"
            "max_energy_error,"
            "max_symmetry_error"
        ),
        comments="",
    )

    np.savetxt(
        OUT
        / "neff_detailed.csv",
        details,
        delimiter=",",
        header=(
            "n_eff,"
            "wavelength_um,"
            "R_fem,"
            "R_fdtd,"
            "T_fem,"
            "T_fdtd,"
            "energy_error,"
            "symmetry_error"
        ),
        comments="",
    )

    # ========================================================
    # SCORE VS EFFECTIVE INDEX
    # ========================================================

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        summary[:, 0],
        summary[:, 1],
        "o-",
        label="MAE R",
    )

    plt.plot(
        summary[:, 0],
        summary[:, 2],
        "o-",
        label="MAE T",
    )

    plt.plot(
        summary[:, 0],
        summary[:, 3],
        "o-",
        label="Combined MAE",
    )

    plt.axvline(
        1.5,
        linestyle="--",
        linewidth=1,
        label="Original n=1.5",
    )

    plt.xlabel(
        "Effective slab index n_eff"
    )

    plt.ylabel(
        "Mean absolute error"
    )

    plt.title(
        "Can an effective 2D refractive index reproduce FDTD?"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_neff_score.png",
        dpi=200,
    )

    plt.close()

    # ========================================================
    # BEST CHECKPOINT SPECTRUM
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        fdtd[:, 0] * 1000.0,
        fdtd[:, 1],
        "-",
        label="FDTD R",
    )

    plt.plot(
        fdtd[:, 0] * 1000.0,
        fdtd[:, 2],
        "-",
        label="FDTD T",
    )

    plt.plot(
        CHECK_WAVELENGTHS * 1000.0,
        best["fem_R"],
        "o--",
        label=(
            f"2D TE R, n_eff="
            f"{best['n_eff']:.2f}"
        ),
    )

    plt.plot(
        CHECK_WAVELENGTHS * 1000.0,
        best["fem_T"],
        "o--",
        label=(
            f"2D TE T, n_eff="
            f"{best['n_eff']:.2f}"
        ),
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Best coarse effective-index surrogate"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "02_best_neff_checkpoints_vs_fdtd.png",
        dpi=200,
    )

    plt.close()

    # ========================================================
    # ERROR BY WAVELENGTH FOR BEST CANDIDATE
    # ========================================================

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        CHECK_WAVELENGTHS * 1000.0,
        np.abs(
            best["fem_R"]
            - fdtd_R_check
        ),
        "o-",
        label="|dR|",
    )

    plt.plot(
        CHECK_WAVELENGTHS * 1000.0,
        np.abs(
            best["fem_T"]
            - fdtd_T_check
        ),
        "o-",
        label="|dT|",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Absolute error"
    )

    plt.title(
        f"Best n_eff={best['n_eff']:.2f}: "
        "where the 2D surrogate still fails"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "03_best_neff_error_by_wavelength.png",
        dpi=200,
    )

    plt.close()

    print()
    print("=" * 104)
    print("COARSE EFFECTIVE-INDEX RESULT")
    print("=" * 104)

    print(
        f"BEST n_eff       = "
        f"{best['n_eff']:.3f}"
    )

    print(
        f"BEST MAE R       = "
        f"{best['mae_R']:.6f}"
    )

    print(
        f"BEST MAE T       = "
        f"{best['mae_T']:.6f}"
    )

    print(
        f"BEST combined MAE= "
        f"{best['combined_mae']:.6f}"
    )

    print(
        f"BEST combined RMSE= "
        f"{best['rmse_combined']:.6f}"
    )

    print()
    print(
        "Interpretation guide:"
    )

    if best["combined_mae"] < 0.05:

        print(
            "VERY STRONG surrogate: a simple effective-index "
            "2D model reproduces the FDTD surprisingly well."
        )

    elif best["combined_mae"] < 0.10:

        print(
            "PROMISING surrogate: refine n_eff around this value."
        )

    elif best["combined_mae"] < 0.16:

        print(
            "SOME improvement, but index alone is not enough. "
            "Next test n_eff + hole diameter."
        )

    else:

        print(
            "WEAK surrogate: changing only refractive index does "
            "not explain the FDTD spectrum."
        )

        print(
            "If the ~720 nm FDTD resonance is still absent, "
            "that strongly argues against a simple 2D "
            "effective-index explanation."
        )

    print("=" * 104)

    print()
    print(
        f"Results written to: {OUT}"
    )


if __name__ == "__main__":
    main()
