from time import perf_counter
from pathlib import Path
import numpy as np

from run_objective3_compare_all import make_cfg

from photonics_fem.objective3_bodyfitted import (
    solve_objective3_bodyfitted,
    diffraction_orders_bodyfitted,
)

# Trusted reference mesh
REF_MESH = Path("meshes/objective3_bodyfitted_fixed.msh")

# Faster candidate meshes we already generated
MESH_25_12 = Path("meshes_speedtest/mesh_25_12.msh")
MESH_30_15 = Path("meshes_speedtest/mesh_30_15.msh")

# Important checkpoints across the spectrum
WAVELENGTHS = [
    0.300,
    0.340,
    0.400,
    0.480,
    0.520,
    0.600,
    0.680,
    0.720,
    0.800,
]


def solve_point(wavelength, mesh_file):
    cfg = make_cfg(wavelength)

    t0 = perf_counter()

    result = solve_objective3_bodyfitted(
        cfg,
        mesh_file,
    )

    diff = diffraction_orders_bodyfitted(
        result,
        cfg,
        npoints=512,
    )

    runtime = perf_counter() - t0

    return {
        "R": diff["R"],
        "T": diff["T"],
        "RT": diff["R_plus_T"],
        "runtime": runtime,
        "dofs": result.scattered.basis.N,
    }


print()
print("=" * 120)
print("SPECTRAL MESH SPEED / ACCURACY TEST")
print("=" * 120)

rows = []

max_dR_25 = 0.0
max_dT_25 = 0.0

max_dR_30 = 0.0
max_dT_30 = 0.0

total_ref_time = 0.0
total_25_time = 0.0
total_30_time = 0.0


for wavelength in WAVELENGTHS:

    print()
    print("=" * 120)
    print(f"Wavelength = {wavelength * 1000:.1f} nm")
    print("=" * 120)

    ref = solve_point(
        wavelength,
        REF_MESH,
    )

    test25 = solve_point(
        wavelength,
        MESH_25_12,
    )

    test30 = solve_point(
        wavelength,
        MESH_30_15,
    )

    dR25 = abs(test25["R"] - ref["R"])
    dT25 = abs(test25["T"] - ref["T"])

    dR30 = abs(test30["R"] - ref["R"])
    dT30 = abs(test30["T"] - ref["T"])

    max_dR_25 = max(max_dR_25, dR25)
    max_dT_25 = max(max_dT_25, dT25)

    max_dR_30 = max(max_dR_30, dR30)
    max_dT_30 = max(max_dT_30, dT30)

    total_ref_time += ref["runtime"]
    total_25_time += test25["runtime"]
    total_30_time += test30["runtime"]

    print(
        f"REF 18/8   "
        f"R={ref['R']:.8f} "
        f"T={ref['T']:.8f} "
        f"time={ref['runtime']:.3f}s "
        f"DOFs={ref['dofs']}"
    )

    print(
        f"25/12      "
        f"R={test25['R']:.8f} "
        f"T={test25['T']:.8f} "
        f"|dR|={dR25:.3e} "
        f"|dT|={dT25:.3e} "
        f"time={test25['runtime']:.3f}s"
    )

    print(
        f"30/15      "
        f"R={test30['R']:.8f} "
        f"T={test30['T']:.8f} "
        f"|dR|={dR30:.3e} "
        f"|dT|={dT30:.3e} "
        f"time={test30['runtime']:.3f}s"
    )

    rows.append([
        wavelength,

        ref["R"],
        ref["T"],

        test25["R"],
        test25["T"],
        dR25,
        dT25,

        test30["R"],
        test30["T"],
        dR30,
        dT30,

        ref["runtime"],
        test25["runtime"],
        test30["runtime"],
    ])


print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print()
print("25 / 12 nm mesh")
print(f"Max |dR| = {max_dR_25:.6e}")
print(f"Max |dT| = {max_dT_25:.6e}")
print(f"Total runtime = {total_25_time:.3f} s")
print(
    f"Speedup vs reference = "
    f"{total_ref_time / total_25_time:.2f}x"
)

print()
print("30 / 15 nm mesh")
print(f"Max |dR| = {max_dR_30:.6e}")
print(f"Max |dT| = {max_dT_30:.6e}")
print(f"Total runtime = {total_30_time:.3f} s")
print(
    f"Speedup vs reference = "
    f"{total_ref_time / total_30_time:.2f}x"
)

data = np.asarray(rows)

np.savetxt(
    "mesh_spectral_benchmark.csv",
    data,
    delimiter=",",
    header=(
        "wavelength_um,"
        "R_ref,T_ref,"
        "R_25_12,T_25_12,dR_25_12,dT_25_12,"
        "R_30_15,T_30_15,dR_30_15,dT_30_15,"
        "time_ref,time_25_12,time_30_15"
    ),
    comments="",
)

print()
print("Saved: mesh_spectral_benchmark.csv")