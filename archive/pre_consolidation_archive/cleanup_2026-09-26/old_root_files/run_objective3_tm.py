from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.objective3_bodyfitted import (
    Objective3Config,
    diffraction_orders_bodyfitted,
)
from photonics_fem.objective3_tm import (
    solve_objective3_tm,
)


OUT = Path("results_objective3_tm")
OUT.mkdir(exist_ok=True)

MESH_FILE = Path(
    "meshes/objective3_bodyfitted_fixed.msh"
)

FDTD_FILE = Path(
    "dataset_hollow_core_fdtd(1).csv"
)

TE_FILE = Path(
    "results_objective3_redo/"
    "objective3_bodyfitted_total_RT.csv"
)

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    26,
)


def make_cfg(wavelength: float) -> Objective3Config:
    return Objective3Config(
        wavelength=wavelength,

        n_inc=1.0 + 0.0j,
        n_slab=1.5 + 0.0j,
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


def load_fdtd():
    raw = np.loadtxt(
        FDTD_FILE
    )

    raw = raw[:, :3]

    order = np.argsort(
        raw[:, 0]
    )

    return raw[order]


def main():

    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"Missing fixed body-fitted mesh: {MESH_FILE}\n"
            "Run run_objective3_redo.py once first."
        )

    if not FDTD_FILE.exists():
        raise FileNotFoundError(
            f"Missing FDTD reference: {FDTD_FILE}"
        )

    print()
    print("=" * 94)
    print("OBJECTIVE 3 TM TEST")
    print("SAME BODY-FITTED MESH / SAME GEOMETRY / H_z POLARIZATION")
    print("=" * 94)

    rows = []

    for i, wavelength in enumerate(
        WAVELENGTHS,
        start=1,
    ):

        cfg = make_cfg(
            float(wavelength)
        )

        result = solve_objective3_tm(
            cfg,
            MESH_FILE,
        )

        # Because incidence/output medium are both air, the
        # power factor ky/ky_inc used by the existing extractor
        # is valid for H_z TM as well.
        diff = diffraction_orders_bodyfitted(
            result,
            cfg,
            npoints=512,
        )

        rows.append(
            [
                wavelength,
                diff["R"],
                diff["T"],
                diff["R_plus_T"],
                diff["energy_error"],
                diff["symmetry_abs_max"],
                1.0
                if diff["cutoff_orders"]
                else 0.0,
            ]
        )

        print(
            f"[{i:02d}/{len(WAVELENGTHS)}] "
            f"lambda={wavelength*1000:6.1f} nm  "
            f"R={diff['R']:.6f}  "
            f"T={diff['T']:.6f}  "
            f"R+T={diff['R_plus_T']:.6f}  "
            f"sym={diff['symmetry_abs_max']:.2e}  "
            f"cutoff={diff['cutoff_orders']}"
        )

    tm = np.asarray(
        rows,
        dtype=float,
    )

    np.savetxt(
        OUT / "objective3_tm_total_RT.csv",
        tm,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_total,"
            "T_total,"
            "R_plus_T,"
            "energy_error,"
            "symmetry_abs_max,"
            "cutoff_flag"
        ),
        comments="",
    )

    fdtd = load_fdtd()

    fdtd_R = np.interp(
        tm[:, 0],
        fdtd[:, 0],
        fdtd[:, 1],
    )

    fdtd_T = np.interp(
        tm[:, 0],
        fdtd[:, 0],
        fdtd[:, 2],
    )

    mae_R_tm = float(
        np.mean(
            np.abs(
                tm[:, 1]
                - fdtd_R
            )
        )
    )

    mae_T_tm = float(
        np.mean(
            np.abs(
                tm[:, 2]
                - fdtd_T
            )
        )
    )

    print()
    print("=" * 94)
    print("TM vs FDTD")
    print("=" * 94)

    print(
        f"MAE R = {mae_R_tm:.6e}"
    )

    print(
        f"MAE T = {mae_T_tm:.6e}"
    )

    noncut = (
        tm[:, 6]
        < 0.5
    )

    print(
        "Max non-cutoff |R+T-1| = "
        f"{np.max(tm[noncut, 4]):.6e}"
    )

    print(
        "Max non-cutoff symmetry error = "
        f"{np.max(tm[noncut, 5]):.6e}"
    )

    # --------------------------------------------------------
    # Plot TM against FDTD.
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
        tm[:, 0] * 1000.0,
        tm[:, 1],
        "o-",
        label="TM FEM R",
    )

    plt.plot(
        tm[:, 0] * 1000.0,
        tm[:, 2],
        "o-",
        label="TM FEM T",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Objective 3: TM FEM vs supplied FDTD"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / "tm_vs_fdtd.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # If TE results exist, compare TE/TM/FDTD directly.
    # --------------------------------------------------------

    if TE_FILE.exists():

        te = np.genfromtxt(
            TE_FILE,
            delimiter=",",
            names=True,
        )

        te_w = te[
            "wavelength_um"
        ]

        te_R = te[
            "R_total"
        ]

        te_T = te[
            "T_total"
        ]

        te_R_at_tm = np.interp(
            tm[:, 0],
            te_w,
            te_R,
        )

        te_T_at_tm = np.interp(
            tm[:, 0],
            te_w,
            te_T,
        )

        mae_R_te = float(
            np.mean(
                np.abs(
                    te_R_at_tm
                    - fdtd_R
                )
            )
        )

        mae_T_te = float(
            np.mean(
                np.abs(
                    te_T_at_tm
                    - fdtd_T
                )
            )
        )

        print()
        print("=" * 94)
        print("TE vs TM BLACK-BOX COMPARISON")
        print("=" * 94)

        print(
            f"TE MAE R = {mae_R_te:.6e}"
        )

        print(
            f"TE MAE T = {mae_T_te:.6e}"
        )

        print(
            f"TM MAE R = {mae_R_tm:.6e}"
        )

        print(
            f"TM MAE T = {mae_T_tm:.6e}"
        )

        plt.figure(
            figsize=(11, 6.5)
        )

        plt.plot(
            fdtd[:, 0] * 1000.0,
            fdtd[:, 1],
            "-",
            label="FDTD R",
        )

        plt.plot(
            tm[:, 0] * 1000.0,
            te_R_at_tm,
            "o-",
            label="TE FEM R",
        )

        plt.plot(
            tm[:, 0] * 1000.0,
            tm[:, 1],
            "s-",
            label="TM FEM R",
        )

        plt.xlabel(
            "Wavelength [nm]"
        )

        plt.ylabel(
            "Reflectance"
        )

        plt.title(
            "Objective 3 reflectance: FDTD vs TE vs TM"
        )

        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUT / "fdtd_vs_te_vs_tm_R.png",
            dpi=200,
        )

        plt.close()

    print()
    print("=" * 94)

    if mae_R_tm < 0.5 * 0.2209425:
        print(
            "TM is dramatically closer to the FDTD reflectance."
        )
        print(
            "Polarization mismatch becomes a strong explanation."
        )
    elif mae_R_tm < 0.8 * 0.2209425:
        print(
            "TM is noticeably closer to FDTD, but not decisive."
        )
    else:
        print(
            "TM does NOT substantially fix the FDTD mismatch."
        )
        print(
            "That makes a different geometry/dimensionality "
            "(e.g. 3D) much more plausible."
        )

    print("=" * 94)

    print()
    print(
        f"Results written to: {OUT}"
    )


if __name__ == "__main__":
    main()
