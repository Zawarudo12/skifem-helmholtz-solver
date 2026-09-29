from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.objective3_bodyfitted import (
    Objective3Config,
    build_bodyfitted_gmsh_mesh,
    diffraction_orders_bodyfitted,
    solve_objective3_bodyfitted,
)


# ============================================================
# PATHS
# ============================================================

OUT = Path(
    "results_objective3_period"
)

OUT.mkdir(
    exist_ok=True
)

MESH_DIR = Path(
    "meshes/objective3_period_sweep"
)

MESH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FDTD_FILE = Path(
    "dataset_hollow_core_fdtd(1).csv"
)


# ============================================================
# PERIOD-ONLY SURROGATE TEST
#
# Everything stays fixed except the x-period.
#
# The special value 1/sqrt(2) um is included because a
# square-lattice 3D (1,1) reciprocal vector has magnitude
# sqrt(2) times the fundamental reciprocal vector.
# ============================================================

PERIODS = np.array(
    [
        0.650,
        0.680,
        0.700,
        1.0 / np.sqrt(2.0),
        0.720,
        0.740,
        0.760,
        0.800,
        1.000,  # original 2D model baseline
    ],
    dtype=float,
)


# Dense enough around the suspicious FDTD feature near 720 nm,
# plus several broader checkpoints.
CHECK_WAVELENGTHS = np.array(
    [
        0.340,
        0.400,
        0.440,
        0.540,
        0.600,
        0.660,
        0.680,
        0.700,
        0.720,
        0.740,
        0.760,
        0.780,
        0.800,
    ],
    dtype=float,
)


FULL_WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    26,
)


# Exclude points extremely close to a Rayleigh cutoff from the
# numerical fit score.  We still calculate, save and plot them.
#
# For a 1D periodic structure at normal incidence:
#
#     lambda_cutoff(m) = period / |m|
#
CUTOFF_EXCLUSION_UM = 0.004


# Fixed geometry/material parameters.
N_SLAB = 1.5
HOLE_DIAMETER = 0.50
SLAB_THICKNESS = 1.0


# Fixed mesh resolution for all candidate periods.
H_BULK = 0.018
H_HOLE = 0.008


def make_cfg(
    wavelength: float,
    period: float,
) -> Objective3Config:

    return Objective3Config(
        wavelength=wavelength,

        n_inc=1.0 + 0.0j,
        n_slab=N_SLAB + 0.0j,
        n_out=1.0 + 0.0j,

        slab_thickness=SLAB_THICKNESS,

        air_top=0.50,
        air_bottom=0.50,

        width=period,

        hole_diameter=HOLE_DIAMETER,

        pml_top=1.00,
        pml_bottom=1.00,

        pml_order=3,
        pml_sigma_max=8.0,
    )


def mesh_path(
    period: float,
) -> Path:

    nm = int(
        round(
            period * 1000.0
        )
    )

    return (
        MESH_DIR
        / f"objective3_period_{nm:04d}nm.msh"
    )


def ensure_mesh(
    period: float,
) -> Path:

    path = mesh_path(
        period
    )

    cfg = make_cfg(
        0.600,
        period,
    )

    build_bodyfitted_gmsh_mesh(
        path,
        cfg,

        h_bulk=H_BULK,
        h_hole=H_HOLE,

        refine_distance=0.20,

        force=False,
    )

    return path


def load_fdtd() -> np.ndarray:

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

    raw = raw[:, :3]

    order = np.argsort(
        raw[:, 0]
    )

    return raw[order]


def near_rayleigh_cutoff(
    wavelength: float,
    period: float,
) -> bool:
    """True when lambda is too close to any 1D Rayleigh cutoff."""

    # Enough orders to cover the 300-800 nm range for these
    # candidate periods.
    for m in range(
        1,
        6,
    ):

        cutoff = (
            period
            / m
        )

        if (
            abs(
                wavelength
                - cutoff
            )
            <= CUTOFF_EXCLUSION_UM
        ):
            return True

    return False


def solve_point(
    wavelength: float,
    period: float,
    mesh_file: Path,
):

    cfg = make_cfg(
        wavelength,
        period,
    )

    result = solve_objective3_bodyfitted(
        cfg,
        mesh_file,
    )

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    return diff


def coarse_period_scan(
    fdtd: np.ndarray,
):

    fdtd_R = np.interp(
        CHECK_WAVELENGTHS,
        fdtd[:, 0],
        fdtd[:, 1],
    )

    fdtd_T = np.interp(
        CHECK_WAVELENGTHS,
        fdtd[:, 0],
        fdtd[:, 2],
    )

    summary_rows = []
    detail_rows = []

    best = None

    print()
    print("=" * 110)
    print("OBJECTIVE 3 EFFECTIVE-PERIOD SWEEP")
    print("2D TE BODY-FITTED MODEL -- ONLY PERIOD CHANGES")
    print("=" * 110)

    print()
    print(
        "candidate periods [um]:",
        np.array2string(
            PERIODS,
            precision=6,
        )
    )

    print(
        "special Lambda/sqrt(2) = "
        f"{1.0 / np.sqrt(2.0):.9f} um"
    )

    for pi, period in enumerate(
        PERIODS,
        start=1,
    ):

        print()
        print("-" * 110)

        print(
            f"[period {pi:02d}/{len(PERIODS)}] "
            f"Lambda_eff = {period:.9f} um"
        )

        mesh_file = ensure_mesh(
            float(period)
        )

        fem_R = []
        fem_T = []
        score_mask = []
        energy_errors = []
        symmetry_errors = []

        for wi, wavelength in enumerate(
            CHECK_WAVELENGTHS,
            start=1,
        ):

            diff = solve_point(
                float(wavelength),
                float(period),
                mesh_file,
            )

            cutoff_flag = near_rayleigh_cutoff(
                float(wavelength),
                float(period),
            )

            fem_R.append(
                diff["R"]
            )

            fem_T.append(
                diff["T"]
            )

            score_mask.append(
                not cutoff_flag
            )

            energy_errors.append(
                diff["energy_error"]
            )

            symmetry_errors.append(
                diff["symmetry_abs_max"]
            )

            detail_rows.append(
                [
                    period,
                    wavelength,
                    diff["R"],
                    fdtd_R[wi - 1],
                    diff["T"],
                    fdtd_T[wi - 1],
                    diff["energy_error"],
                    diff["symmetry_abs_max"],
                    1.0 if cutoff_flag else 0.0,
                ]
            )

            flag = (
                "CUT"
                if cutoff_flag
                else "   "
            )

            print(
                f"  [{wi:02d}/{len(CHECK_WAVELENGTHS)}] "
                f"{flag}  "
                f"lambda={wavelength*1000:6.1f} nm  "
                f"R={diff['R']:.5f}  "
                f"T={diff['T']:.5f}  "
                f"FDTD R={fdtd_R[wi-1]:.5f}  "
                f"FDTD T={fdtd_T[wi-1]:.5f}"
            )

        fem_R = np.asarray(
            fem_R,
            dtype=float,
        )

        fem_T = np.asarray(
            fem_T,
            dtype=float,
        )

        score_mask = np.asarray(
            score_mask,
            dtype=bool,
        )

        if not np.any(
            score_mask
        ):
            raise RuntimeError(
                "All diagnostic wavelengths were excluded "
                "by cutoff filtering."
            )

        dR = np.abs(
            fem_R[score_mask]
            - fdtd_R[score_mask]
        )

        dT = np.abs(
            fem_T[score_mask]
            - fdtd_T[score_mask]
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

        combined_rmse = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            (
                                fem_R[score_mask]
                                - fdtd_R[score_mask]
                            ) ** 2,
                            (
                                fem_T[score_mask]
                                - fdtd_T[score_mask]
                            ) ** 2,
                        ]
                    )
                )
            )
        )

        # How well does this candidate specifically capture
        # the suspicious 720-nm FDTD reflectance?
        idx720 = int(
            np.argmin(
                np.abs(
                    CHECK_WAVELENGTHS
                    - 0.720
                )
            )
        )

        R720 = float(
            fem_R[idx720]
        )

        R720_fdtd = float(
            fdtd_R[idx720]
        )

        R720_error = abs(
            R720
            - R720_fdtd
        )

        summary_rows.append(
            [
                period,
                mae_R,
                mae_T,
                combined_mae,
                combined_rmse,
                R720,
                R720_fdtd,
                R720_error,
                float(
                    np.max(
                        energy_errors
                    )
                ),
                float(
                    np.max(
                        symmetry_errors
                    )
                ),
                int(
                    np.sum(
                        score_mask
                    )
                ),
            ]
        )

        print()
        print(
            f"  scored points = "
            f"{np.sum(score_mask)}/"
            f"{len(score_mask)}"
        )

        print(
            f"  MAE R         = "
            f"{mae_R:.6f}"
        )

        print(
            f"  MAE T         = "
            f"{mae_T:.6f}"
        )

        print(
            f"  combined MAE  = "
            f"{combined_mae:.6f}"
        )

        print(
            f"  R(720 nm)     = "
            f"{R720:.6f} "
            f"(FDTD {R720_fdtd:.6f})"
        )

        if (
            best is None
            or combined_mae
            < best["combined_mae"]
        ):
            best = {
                "period": float(
                    period
                ),
                "combined_mae": combined_mae,
                "mae_R": mae_R,
                "mae_T": mae_T,
                "combined_rmse": combined_rmse,
                "fem_R": fem_R.copy(),
                "fem_T": fem_T.copy(),
                "score_mask": score_mask.copy(),
                "R720": R720,
                "R720_error": R720_error,
            }

    summary = np.asarray(
        summary_rows,
        dtype=float,
    )

    details = np.asarray(
        detail_rows,
        dtype=float,
    )

    np.savetxt(
        OUT
        / "period_summary.csv",
        summary,
        delimiter=",",
        header=(
            "period_um,"
            "mae_R,"
            "mae_T,"
            "combined_mae,"
            "combined_rmse,"
            "R_720nm,"
            "R_fdtd_720nm,"
            "R_720_error,"
            "max_energy_error,"
            "max_symmetry_error,"
            "num_scored_points"
        ),
        comments="",
    )

    np.savetxt(
        OUT
        / "period_detailed.csv",
        details,
        delimiter=",",
        header=(
            "period_um,"
            "wavelength_um,"
            "R_fem,"
            "R_fdtd,"
            "T_fem,"
            "T_fdtd,"
            "energy_error,"
            "symmetry_error,"
            "cutoff_excluded"
        ),
        comments="",
    )

    # --------------------------------------------------------
    # Score versus period.
    # --------------------------------------------------------

    plt.figure(
        figsize=(9.5, 5.8)
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
        1.0 / np.sqrt(2.0),
        linestyle="--",
        linewidth=1.2,
        label=r"$1/\sqrt{2}$ um",
    )

    plt.axvline(
        1.0,
        linestyle=":",
        linewidth=1.2,
        label="Original 1.0 um",
    )

    plt.xlabel(
        "Effective period [um]"
    )

    plt.ylabel(
        "Error versus supplied FDTD"
    )

    plt.title(
        "Effective-period surrogate score"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_period_score.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # 720-nm reflectance versus period.
    # --------------------------------------------------------

    plt.figure(
        figsize=(9.5, 5.8)
    )

    plt.plot(
        summary[:, 0],
        summary[:, 5],
        "o-",
        label="2D TE R @ 720 nm",
    )

    plt.axhline(
        summary[0, 6],
        linestyle="--",
        linewidth=1.2,
        label="FDTD R @ 720 nm",
    )

    plt.axvline(
        1.0 / np.sqrt(2.0),
        linestyle=":",
        linewidth=1.2,
        label=r"$1/\sqrt{2}$ um",
    )

    plt.xlabel(
        "Effective period [um]"
    )

    plt.ylabel(
        "Reflectance at 720 nm"
    )

    plt.title(
        "Does the projected 3D diagonal cutoff create "
        "the FDTD-like 720-nm reflection?"
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT
        / "02_R720_vs_period.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Best coarse candidate at diagnostic points.
    # --------------------------------------------------------

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
            "Best 2D R, "
            f"Lambda={best['period']:.4f} um"
        ),
    )

    plt.plot(
        CHECK_WAVELENGTHS * 1000.0,
        best["fem_T"],
        "o--",
        label=(
            "Best 2D T, "
            f"Lambda={best['period']:.4f} um"
        ),
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Best coarse effective-period surrogate"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT
        / "03_best_period_checkpoints_vs_fdtd.png",
        dpi=200,
    )

    plt.close()

    return best, summary


def full_best_sweep(
    best,
    fdtd: np.ndarray,
):

    period = best["period"]

    mesh_file = ensure_mesh(
        period
    )

    print()
    print("=" * 110)
    print(
        "FULL 300-800 nm SWEEP FOR BEST PERIOD "
        f"{period:.9f} um"
    )
    print("=" * 110)

    rows = []

    for i, wavelength in enumerate(
        FULL_WAVELENGTHS,
        start=1,
    ):

        diff = solve_point(
            float(wavelength),
            period,
            mesh_file,
        )

        cutoff_flag = near_rayleigh_cutoff(
            float(wavelength),
            period,
        )

        rows.append(
            [
                wavelength,
                diff["R"],
                diff["T"],
                diff["R_plus_T"],
                diff["energy_error"],
                diff["symmetry_abs_max"],
                1.0 if cutoff_flag else 0.0,
            ]
        )

        print(
            f"[{i:02d}/{len(FULL_WAVELENGTHS)}] "
            f"lambda={wavelength*1000:6.1f} nm  "
            f"R={diff['R']:.6f}  "
            f"T={diff['T']:.6f}  "
            f"R+T={diff['R_plus_T']:.6f}  "
            f"cutoff={'YES' if cutoff_flag else 'no'}"
        )

    data = np.asarray(
        rows,
        dtype=float,
    )

    fdtd_R = np.interp(
        data[:, 0],
        fdtd[:, 0],
        fdtd[:, 1],
    )

    fdtd_T = np.interp(
        data[:, 0],
        fdtd[:, 0],
        fdtd[:, 2],
    )

    score_mask = (
        data[:, 6]
        < 0.5
    )

    mae_R = float(
        np.mean(
            np.abs(
                data[score_mask, 1]
                - fdtd_R[score_mask]
            )
        )
    )

    mae_T = float(
        np.mean(
            np.abs(
                data[score_mask, 2]
                - fdtd_T[score_mask]
            )
        )
    )

    combined_mae = 0.5 * (
        mae_R
        + mae_T
    )

    output = np.column_stack(
        [
            data,
            fdtd_R,
            fdtd_T,
            data[:, 1] - fdtd_R,
            data[:, 2] - fdtd_T,
        ]
    )

    np.savetxt(
        OUT
        / "best_period_full_sweep.csv",
        output,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fem,"
            "T_fem,"
            "R_plus_T,"
            "energy_error,"
            "symmetry_error,"
            "cutoff_excluded,"
            "R_fdtd,"
            "T_fdtd,"
            "R_difference,"
            "T_difference"
        ),
        comments="",
    )

    plt.figure(
        figsize=(10.5, 6.2)
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
        data[:, 0] * 1000.0,
        data[:, 1],
        "o-",
        label=(
            "2D TE R, "
            f"Lambda={period:.4f} um"
        ),
    )

    plt.plot(
        data[:, 0] * 1000.0,
        data[:, 2],
        "o-",
        label=(
            "2D TE T, "
            f"Lambda={period:.4f} um"
        ),
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Best effective-period 2D surrogate vs FDTD"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT
        / "04_best_period_full_vs_fdtd.png",
        dpi=200,
    )

    plt.close()

    print()
    print(
        f"FULL-SWEEP MAE R = "
        f"{mae_R:.6f}"
    )

    print(
        f"FULL-SWEEP MAE T = "
        f"{mae_T:.6f}"
    )

    print(
        f"FULL-SWEEP combined MAE = "
        f"{combined_mae:.6f}"
    )

    return (
        mae_R,
        mae_T,
        combined_mae,
    )


def main():

    if not FDTD_FILE.exists():
        raise FileNotFoundError(
            f"Missing FDTD reference:\n{FDTD_FILE}"
        )

    fdtd = load_fdtd()

    best, summary = coarse_period_scan(
        fdtd
    )

    full_metrics = full_best_sweep(
        best,
        fdtd,
    )

    original_row = summary[
        np.argmin(
            np.abs(
                summary[:, 0]
                - 1.0
            )
        )
    ]

    projected_row = summary[
        np.argmin(
            np.abs(
                summary[:, 0]
                - 1.0 / np.sqrt(2.0)
            )
        )
    ]

    print()
    print("=" * 110)
    print("EFFECTIVE-PERIOD RESULT")
    print("=" * 110)

    print(
        f"BEST period = "
        f"{best['period']:.9f} um"
    )

    print(
        f"BEST coarse combined MAE = "
        f"{best['combined_mae']:.6f}"
    )

    print(
        f"BEST R(720 nm) = "
        f"{best['R720']:.6f}"
    )

    print()
    print(
        f"Original 1.0 um combined MAE = "
        f"{original_row[3]:.6f}"
    )

    print(
        f"Projected 1/sqrt(2) um combined MAE = "
        f"{projected_row[3]:.6f}"
    )

    print(
        f"Projected 1/sqrt(2) um R(720 nm) = "
        f"{projected_row[5]:.6f}"
    )

    print()
    print(
        f"Best full-sweep combined MAE = "
        f"{full_metrics[2]:.6f}"
    )

    print()

    if (
        best["period"]
        < 0.82
        and best["combined_mae"]
        < 0.75 * original_row[3]
    ):
        print(
            "RESULT: changing the effective period gives a "
            "large improvement over the original 2D model."
        )

        print(
            "This supports the idea that missing transverse "
            "diffraction geometry matters."
        )

    elif (
        projected_row[5]
        > 0.60
        and projected_row[3]
        < original_row[3]
    ):
        print(
            "RESULT: the 1/sqrt(2) projection specifically "
            "creates a strong 720-nm response and improves "
            "the fit."
        )

        print(
            "That is suggestive of a missing diagonal "
            "reciprocal-lattice channel, though it is not "
            "proof of a 3D FDTD model."
        )

    else:
        print(
            "RESULT: changing the period does not reproduce "
            "the FDTD spectrum convincingly."
        )

        print(
            "A simple projected-period 2D surrogate is not "
            "enough; different geometry or genuinely 3D/"
            "vector physics remains plausible."
        )

    print("=" * 110)

    print()
    print(
        f"Results written to: {OUT}"
    )


if __name__ == "__main__":
    main()
