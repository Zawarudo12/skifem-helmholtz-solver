from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)


FAST_MESH = Path(
    "meshes_speedtest/mesh_30_15.msh"
)

REFERENCE_FILE = Path(
    "results_objective3_redo/objective3_bodyfitted_total_RT.csv"
)

OUT = Path(
    "results_fast_mesh"
)

OUT.mkdir(
    exist_ok=True
)

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    101,
)


# ------------------------------------------------------------
# Load the already-computed trusted 18/8 nm reference spectrum
# ------------------------------------------------------------

reference = np.loadtxt(
    REFERENCE_FILE,
    delimiter=",",
    skiprows=1,
)

ref_w = reference[:, 0]
ref_R = reference[:, 1]
ref_T = reference[:, 2]


# ------------------------------------------------------------
# Run the fast 30/15 nm mesh
# ------------------------------------------------------------

rows = []

total_start = perf_counter()

for i, wavelength in enumerate(
    WAVELENGTHS,
    start=1,
):

    cfg = make_cfg(
        float(wavelength)
    )

    t0 = perf_counter()

    result = solve_objective3_bodyfitted(
        cfg,
        FAST_MESH,
    )

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    runtime = perf_counter() - t0

    rows.append(
        [
            wavelength,
            diff["R"],
            diff["T"],
            diff["R_plus_T"],
            runtime,
            result.scattered.basis.N,
        ]
    )

    print(
        f"[{i:03d}/101] "
        f"{wavelength * 1000:6.1f} nm  "
        f"R={diff['R']:.6f}  "
        f"T={diff['T']:.6f}  "
        f"time={runtime:.3f}s"
    )


total_runtime = (
    perf_counter()
    - total_start
)

fast = np.asarray(
    rows,
    dtype=float,
)


# ------------------------------------------------------------
# Compare against trusted reference
# ------------------------------------------------------------

fast_w = fast[:, 0]
fast_R = fast[:, 1]
fast_T = fast[:, 2]

ref_R_interp = np.interp(
    fast_w,
    ref_w,
    ref_R,
)

ref_T_interp = np.interp(
    fast_w,
    ref_w,
    ref_T,
)

dR = (
    fast_R
    - ref_R_interp
)

dT = (
    fast_T
    - ref_T_interp
)

abs_dR = np.abs(
    dR
)

abs_dT = np.abs(
    dT
)


print()
print("=" * 90)
print("FULL-SPECTRUM FAST-MESH VALIDATION")
print("=" * 90)

print(
    f"Total runtime = "
    f"{total_runtime:.3f} s"
)

print(
    f"Max |dR| = "
    f"{np.max(abs_dR):.6e}"
)

print(
    f"Max |dT| = "
    f"{np.max(abs_dT):.6e}"
)

print(
    f"Mean |dR| = "
    f"{np.mean(abs_dR):.6e}"
)

print(
    f"Mean |dT| = "
    f"{np.mean(abs_dT):.6e}"
)

print(
    f"RMS dR = "
    f"{np.sqrt(np.mean(dR**2)):.6e}"
)

print(
    f"RMS dT = "
    f"{np.sqrt(np.mean(dT**2)):.6e}"
)


# ------------------------------------------------------------
# Save comparison CSV
# ------------------------------------------------------------

comparison = np.column_stack(
    [
        fast_w,
        ref_R_interp,
        fast_R,
        abs_dR,
        ref_T_interp,
        fast_T,
        abs_dT,
        fast[:, 3],
        fast[:, 4],
    ]
)

np.savetxt(
    OUT / "fast_mesh_vs_reference.csv",
    comparison,
    delimiter=",",
    header=(
        "wavelength_um,"
        "R_reference,"
        "R_fast,"
        "abs_dR,"
        "T_reference,"
        "T_fast,"
        "abs_dT,"
        "R_plus_T_fast,"
        "runtime_s"
    ),
    comments="",
)


# ------------------------------------------------------------
# Reflectance comparison
# ------------------------------------------------------------

w_nm = (
    fast_w
    * 1000.0
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    w_nm,
    ref_R_interp,
    label="Reference 18/8 nm",
)

plt.plot(
    w_nm,
    fast_R,
    "--",
    label="Fast 30/15 nm",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Reflectance"
)

plt.title(
    "Fast mesh validation: Reflectance"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fast_vs_reference_R.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Transmittance comparison
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    w_nm,
    ref_T_interp,
    label="Reference 18/8 nm",
)

plt.plot(
    w_nm,
    fast_T,
    "--",
    label="Fast 30/15 nm",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Transmittance"
)

plt.title(
    "Fast mesh validation: Transmittance"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fast_vs_reference_T.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# Error plot
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.semilogy(
    w_nm,
    abs_dR,
    label="|dR|",
)

plt.semilogy(
    w_nm,
    abs_dT,
    label="|dT|",
)

plt.axhline(
    1e-3,
    linestyle="--",
    label="1e-3 target",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Absolute difference"
)

plt.title(
    "Fast mesh numerical error"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "fast_mesh_error.png",
    dpi=200,
)

plt.close()


print()
print(
    f"Results saved to: {OUT}"
)