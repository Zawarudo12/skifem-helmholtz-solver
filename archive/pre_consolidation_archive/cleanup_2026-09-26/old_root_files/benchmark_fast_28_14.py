from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    build_bodyfitted_gmsh_mesh,
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)


# ============================================================
# PATHS
# ============================================================

FAST_MESH = Path(
    "meshes_speedtest/mesh_28_14.msh"
)

REFERENCE_FILE = Path(
    "results_objective3_redo/objective3_bodyfitted_total_RT.csv"
)

OUT = Path(
    "results_fast_mesh_28_14"
)

OUT.mkdir(
    exist_ok=True
)


# ============================================================
# SWEEP SETTINGS
# 300 nm -> 800 nm in 5 nm steps
# ============================================================

WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    101,
)


# ============================================================
# BUILD 28/14 nm BODY-FITTED MESH
# ============================================================

print()
print("=" * 90)
print("BUILDING / LOADING 28 nm / 14 nm FAST MESH")
print("=" * 90)

reference_cfg = make_cfg(
    0.600
)

build_bodyfitted_gmsh_mesh(
    FAST_MESH,
    reference_cfg,

    h_bulk=0.028,     # 28 nm
    h_hole=0.014,     # 14 nm

    refine_distance=0.20,

    force=False,
)

print()
print(f"Fast mesh: {FAST_MESH}")


# ============================================================
# LOAD EXISTING TRUSTED 18/8 nm REFERENCE RESULTS
# ============================================================

if not REFERENCE_FILE.exists():
    raise FileNotFoundError(
        f"Reference spectrum not found:\n{REFERENCE_FILE}"
    )

reference = np.loadtxt(
    REFERENCE_FILE,
    delimiter=",",
    skiprows=1,
)

ref_w = reference[:, 0]
ref_R = reference[:, 1]
ref_T = reference[:, 2]


# ============================================================
# RUN FULL 28/14 nm SWEEP
# ============================================================

rows = []

print()
print("=" * 90)
print("FULL 28/14 nm FAST-MESH SWEEP")
print("=" * 90)
print()

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

    runtime = (
        perf_counter()
        - t0
    )

    rows.append(
        [
            wavelength,
            diff["R"],
            diff["T"],
            diff["R_plus_T"],
            diff["energy_error"],
            diff["symmetry_abs_max"],
            runtime,
            result.scattered.basis.N,
        ]
    )

    print(
        f"[{i:03d}/101] "
        f"{wavelength * 1000:6.1f} nm  "
        f"R={diff['R']:.6f}  "
        f"T={diff['T']:.6f}  "
        f"R+T={diff['R_plus_T']:.6f}  "
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


# ============================================================
# FAST RESULTS
# ============================================================

fast_w = fast[:, 0]
fast_R = fast[:, 1]
fast_T = fast[:, 2]


# ============================================================
# INTERPOLATE TRUSTED REFERENCE ONTO SAME GRID
# ============================================================

ref_R_at_fast = np.interp(
    fast_w,
    ref_w,
    ref_R,
)

ref_T_at_fast = np.interp(
    fast_w,
    ref_w,
    ref_T,
)


# ============================================================
# ERROR CALCULATION
# ============================================================

dR = (
    fast_R
    - ref_R_at_fast
)

dT = (
    fast_T
    - ref_T_at_fast
)

abs_dR = np.abs(
    dR
)

abs_dT = np.abs(
    dT
)

max_R_index = int(
    np.argmax(abs_dR)
)

max_T_index = int(
    np.argmax(abs_dT)
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("28/14 nm FULL-SPECTRUM VALIDATION")
print("=" * 90)

print()
print(
    f"Total sweep runtime = "
    f"{total_runtime:.3f} s"
)

print(
    f"Total sweep runtime = "
    f"{total_runtime / 60:.3f} min"
)

print()
print(
    f"DOFs = "
    f"{int(fast[0, 7])}"
)

print()
print(
    f"Max |dR| = "
    f"{np.max(abs_dR):.6e}"
)

print(
    f"Max |dR| wavelength = "
    f"{fast_w[max_R_index] * 1000:.1f} nm"
)

print()
print(
    f"Max |dT| = "
    f"{np.max(abs_dT):.6e}"
)

print(
    f"Max |dT| wavelength = "
    f"{fast_w[max_T_index] * 1000:.1f} nm"
)

print()
print(
    f"Mean |dR| = "
    f"{np.mean(abs_dR):.6e}"
)

print(
    f"Mean |dT| = "
    f"{np.mean(abs_dT):.6e}"
)

print()
print(
    f"RMS dR = "
    f"{np.sqrt(np.mean(dR**2)):.6e}"
)

print(
    f"RMS dT = "
    f"{np.sqrt(np.mean(dT**2)):.6e}"
)


# ============================================================
# PASS / FAIL
# ============================================================

TOLERANCE = 1e-3

passed = (
    np.max(abs_dR) < TOLERANCE
    and
    np.max(abs_dT) < TOLERANCE
)

print()
print("=" * 90)

if passed:
    print(
        "FAST MESH VALIDATION: PASS"
    )
    print(
        "28/14 nm stays below 1e-3 "
        "absolute R/T error over the full spectrum."
    )
else:
    print(
        "FAST MESH VALIDATION: CHECK"
    )
    print(
        "At least one wavelength exceeds "
        "1e-3 absolute R/T error."
    )

print("=" * 90)


# ============================================================
# SAVE FAST RAW RESULTS
# ============================================================

np.savetxt(
    OUT / "fast_28_14_raw.csv",
    fast,
    delimiter=",",
    header=(
        "wavelength_um,"
        "R_fast,"
        "T_fast,"
        "R_plus_T,"
        "energy_error,"
        "symmetry_error,"
        "runtime_s,"
        "dofs"
    ),
    comments="",
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison = np.column_stack(
    [
        fast_w,

        ref_R_at_fast,
        fast_R,
        dR,
        abs_dR,

        ref_T_at_fast,
        fast_T,
        dT,
        abs_dT,

        fast[:, 3],
        fast[:, 6],
    ]
)

np.savetxt(
    OUT / "fast_28_14_vs_reference.csv",
    comparison,
    delimiter=",",
    header=(
        "wavelength_um,"
        "R_reference,"
        "R_fast,"
        "dR,"
        "abs_dR,"
        "T_reference,"
        "T_fast,"
        "dT,"
        "abs_dT,"
        "R_plus_T_fast,"
        "runtime_s"
    ),
    comments="",
)


# ============================================================
# PLOTS
# ============================================================

wavelength_nm = (
    fast_w
    * 1000.0
)


# ------------------------------------------------------------
# REFLECTANCE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    wavelength_nm,
    ref_R_at_fast,
    linewidth=2.0,
    label="Reference mesh 18/8 nm",
)

plt.plot(
    wavelength_nm,
    fast_R,
    "--",
    linewidth=2.0,
    label="Fast mesh 28/14 nm",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Reflectance"
)

plt.title(
    "Reflectance: reference vs 28/14 nm fast mesh"
)

plt.xlim(
    300,
    800,
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
    OUT / "01_R_reference_vs_28_14.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# TRANSMITTANCE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    wavelength_nm,
    ref_T_at_fast,
    linewidth=2.0,
    label="Reference mesh 18/8 nm",
)

plt.plot(
    wavelength_nm,
    fast_T,
    "--",
    linewidth=2.0,
    label="Fast mesh 28/14 nm",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Transmittance"
)

plt.title(
    "Transmittance: reference vs 28/14 nm fast mesh"
)

plt.xlim(
    300,
    800,
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
    OUT / "02_T_reference_vs_28_14.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# ABSOLUTE ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.semilogy(
    wavelength_nm,
    abs_dR,
    linewidth=2.0,
    label="|dR|",
)

plt.semilogy(
    wavelength_nm,
    abs_dT,
    linewidth=2.0,
    label="|dT|",
)

plt.axhline(
    1e-3,
    linestyle="--",
    linewidth=1.5,
    label="1e-3 tolerance",
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "Absolute difference"
)

plt.title(
    "28/14 nm mesh error vs reference"
)

plt.xlim(
    300,
    800,
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "03_absolute_error_28_14.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# R + T ENERGY CHECK
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    wavelength_nm,
    fast[:, 3],
    linewidth=2.0,
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1.5,
)

plt.xlabel(
    "Wavelength [nm]"
)

plt.ylabel(
    "R + T"
)

plt.title(
    "28/14 nm fast mesh: energy conservation"
)

plt.xlim(
    300,
    800,
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT / "04_energy_conservation_28_14.png",
    dpi=200,
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print(
    f"Results saved to: {OUT}"
)

print()