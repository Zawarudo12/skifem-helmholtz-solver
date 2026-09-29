from __future__ import annotations

import gc
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders


OUT = Path("results_disordered_boundary_sensitivity")
OUT.mkdir(exist_ok=True)

CASES = [
    {
        "label": "baseline air0.5_pml1.0",
        "mesh": Path("meshes_disordered/disordered_196disks_pilot.msh"),
        "air": 0.50,
        "pml": 1.00,
    },
    {
        "label": "air1.0_pml1.5",
        "mesh": Path("meshes_disordered/disordered_196disks_buffer1_pml1p5.msh"),
        "air": 1.00,
        "pml": 1.50,
    },
    {
        "label": "air1.0_pml2.0",
        "mesh": Path("meshes_disordered/disordered_196disks_buffer1_pml2.msh"),
        "air": 1.00,
        "pml": 2.00,
    },
]

# Sensitive Rayleigh-cutoff neighborhoods plus one clean stop-band control.
WAVELENGTHS = np.array(
    [
        1.0645161290322580,  # clean non-cutoff reference
        1.1702127659574468,  # near m=6 cutoff (7/6)
        1.5000000000000000,  # stop-band control
        1.7553191489361701,  # near m=4 cutoff (7/4)
        2.3239436619718310,  # near m=3 cutoff (7/3)
    ],
    dtype=float,
)


def make_cfg(wavelength: float, air: float, pml: float) -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=float(wavelength),
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,
        physical_ymin=-3.5 - float(air),
        physical_ymax=+3.5 + float(air),
        pml_low=float(pml),
        pml_high=float(pml),
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def nearest_cutoff(wavelength: float):
    candidates = []
    for m in range(1, 8):
        wc = 7.0 / m
        candidates.append((abs(wavelength - wc), m, wc))
    return min(candidates, key=lambda r: r[0])


def solve_case(case):
    mesh = case["mesh"]

    if not mesh.exists():
        raise FileNotFoundError(
            f"Missing mesh {mesh}. Build the two new meshes first."
        )

    template = make_cfg(1.5, case["air"], case["pml"])

    print()
    print("=" * 112)
    print(case["label"].upper())
    print("=" * 112)
    print(f"Mesh        = {mesh}")
    print(f"Air buffer  = {case['air']:.2f} um each side")
    print(f"PML         = {case['pml']:.2f} um each side")

    t0 = perf_counter()

    solver = CachedDisorderedSolver(
        template,
        mesh,
        intorder=8,
    )

    setup = perf_counter() - t0

    print(f"Full P2 DOFs = {solver.basis.N}")
    print(f"Cache setup  = {setup:.3f} s")
    print()
    print(
        f"{'lambda':>10} "
        f"{'near cutoff':>12} "
        f"{'R':>12} "
        f"{'T':>12} "
        f"{'R+T':>12} "
        f"{'|R+T-1|':>12} "
        f"{'solve[s]':>10}"
    )

    rows = []

    for wl in WAVELENGTHS:
        cfg = make_cfg(float(wl), case["air"], case["pml"])

        start = perf_counter()
        result = solver.solve(cfg)
        diff = diffraction_orders(result, npoints=1024)
        elapsed = perf_counter() - start

        _, m, cutoff_wl = nearest_cutoff(float(wl))
        cutoff_text = (
            f"m={m}"
            if abs(float(wl) - cutoff_wl) < 0.02
            else "-"
        )

        row = {
            "label": case["label"],
            "air": case["air"],
            "pml": case["pml"],
            "wavelength": float(wl),
            "R": float(diff["R"]),
            "T": float(diff["T"]),
            "R_plus_T": float(diff["R_plus_T"]),
            "energy": float(diff["energy_error"]),
            "solve_time": float(elapsed),
            "cutoff_order": int(m) if cutoff_text != "-" else 0,
            "cutoff_wavelength": float(cutoff_wl),
        }
        rows.append(row)

        print(
            f"{wl:10.6f} "
            f"{cutoff_text:>12} "
            f"{row['R']:12.8f} "
            f"{row['T']:12.8f} "
            f"{row['R_plus_T']:12.8f} "
            f"{row['energy']:12.3e} "
            f"{elapsed:10.3f}"
        )

    del solver
    gc.collect()

    return rows


def main():
    all_rows = []

    for case in CASES:
        all_rows.extend(solve_case(case))

    print()
    print("=" * 132)
    print("BOUNDARY-SENSITIVITY SUMMARY")
    print("=" * 132)
    print(
        f"{'lambda':>10} "
        f"{'baseline E':>13} "
        f"{'air1/pml1.5 E':>15} "
        f"{'air1/pml2 E':>13} "
        f"{'best E':>12} "
        f"{'R baseline':>12} "
        f"{'R thickest':>12}"
    )

    labels = [case["label"] for case in CASES]

    for wl in WAVELENGTHS:
        subset = [
            row for row in all_rows
            if np.isclose(row["wavelength"], wl)
        ]
        by_label = {row["label"]: row for row in subset}

        base = by_label[labels[0]]
        mid = by_label[labels[1]]
        thick = by_label[labels[2]]

        best = min(
            base["energy"],
            mid["energy"],
            thick["energy"],
        )

        print(
            f"{wl:10.6f} "
            f"{base['energy']:13.3e} "
            f"{mid['energy']:15.3e} "
            f"{thick['energy']:13.3e} "
            f"{best:12.3e} "
            f"{base['R']:12.7f} "
            f"{thick['R']:12.7f}"
        )

    # Save all numerical data.
    csv_rows = []
    for row in all_rows:
        csv_rows.append(
            [
                row["wavelength"],
                row["air"],
                row["pml"],
                row["R"],
                row["T"],
                row["R_plus_T"],
                row["energy"],
                row["solve_time"],
                row["cutoff_order"],
                row["cutoff_wavelength"],
            ]
        )

    np.savetxt(
        OUT / "boundary_sensitivity.csv",
        np.asarray(csv_rows, dtype=float),
        delimiter=",",
        header=(
            "wavelength_um,air_buffer_um,pml_thickness_um,"
            "R,T,R_plus_T,energy_error,solve_time_s,"
            "near_cutoff_order,nearest_cutoff_wavelength_um"
        ),
        comments="",
    )

    # Energy-error plot.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for case in CASES:
        rows = [
            row for row in all_rows
            if row["label"] == case["label"]
        ]
        rows.sort(key=lambda r: r["wavelength"])

        ax.semilogy(
            [r["wavelength"] for r in rows],
            [r["energy"] for r in rows],
            marker="o",
            linewidth=1.8,
            label=case["label"],
        )

    for m in (6, 4, 3):
        wc = 7.0 / m
        ax.axvline(
            wc,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel(r"Energy error $|R+T-1|$")
    ax.set_title("Air-buffer / PML sensitivity near Rayleigh cutoffs")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = OUT / "boundary_sensitivity_energy_error.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print()
    print(f"Saved CSV  : {OUT / 'boundary_sensitivity.csv'}")
    print(f"Saved plot : {plot_path}")


if __name__ == "__main__":
    main()
