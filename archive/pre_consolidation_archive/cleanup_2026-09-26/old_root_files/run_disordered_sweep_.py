from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from photonics_fem.disordered_cached import CachedDisorderedSolver
from photonics_fem.disordered_fast import DisorderedConfig, diffraction_orders

MESH_FILE = Path("meshes_disordered/disordered_196disks_buffer2_pml3.msh")
FDTD_FILE = Path("dataset_disordered_disks_fdtd.csv")
OUT = Path("results_disordered_postdoc_air2_pml3")
OUT.mkdir(exist_ok=True)

AIR_BUFFER = 2.0
PML_THICKNESS = 3.0
ORDER_CAP = 5
NPOINTS_RT = 1024

_WORKER_SOLVER = None


def make_cfg(wavelength: float) -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=float(wavelength),
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,
        physical_ymin=-3.5 - AIR_BUFFER,
        physical_ymax=+3.5 + AIR_BUFFER,
        pml_low=PML_THICKNESS,
        pml_high=PML_THICKNESS,
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def load_fdtd_reference() -> np.ndarray:
    if not FDTD_FILE.exists():
        raise FileNotFoundError(f"Could not find {FDTD_FILE}")
    try:
        raw = np.loadtxt(FDTD_FILE, delimiter=",")
    except ValueError:
        raw = np.loadtxt(FDTD_FILE)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        raise ValueError("FDTD file must contain wavelength_um, R, T.")
    data = raw[:, :3]
    mask = ((data[:, 0] >= 1.0 - 1e-12) & (data[:, 0] <= 2.4 + 1e-12))
    data = data[mask]
    return data[np.argsort(data[:, 0])]


def worker_init(mesh_file: str):
    global _WORKER_SOLVER
    _WORKER_SOLVER = CachedDisorderedSolver(
        make_cfg(1.5),
        Path(mesh_file),
        intorder=8,
    )


def summarize_cap(diff: dict, cap: int):
    selected = [
        row for row in diff["orders"]
        if row["propagating"] and abs(int(row["m"])) <= cap
    ]
    omitted = [
        int(row["m"]) for row in diff["orders"]
        if row["propagating"] and abs(int(row["m"])) > cap
    ]
    propagating = [
        int(row["m"]) for row in diff["orders"] if row["propagating"]
    ]
    R_cap = float(sum(row["R"] for row in selected))
    T_cap = float(sum(row["T"] for row in selected))
    return {
        "R_cap": R_cap,
        "T_cap": T_cap,
        "RT_cap": R_cap + T_cap,
        "energy_cap": abs(R_cap + T_cap - 1.0),
        "omitted": omitted,
        "propagating": propagating,
    }


def worker_solve(wavelength: float):
    global _WORKER_SOLVER
    if _WORKER_SOLVER is None:
        raise RuntimeError("Worker cache not initialized.")

    cfg = make_cfg(wavelength)
    t0 = perf_counter()
    result = _WORKER_SOLVER.solve(cfg)
    t1 = perf_counter()

    # Existing extractor already evaluates all propagating Rayleigh orders.
    diff = diffraction_orders(result, npoints=NPOINTS_RT)
    cap = summarize_cap(diff, ORDER_CAP)
    t2 = perf_counter()

    return {
        "wavelength": float(wavelength),
        "R_all": float(diff["R"]),
        "T_all": float(diff["T"]),
        "RT_all": float(diff["R_plus_T"]),
        "energy_all": float(diff["energy_error"]),
        "R_cap": cap["R_cap"],
        "T_cap": cap["T_cap"],
        "RT_cap": cap["RT_cap"],
        "energy_cap": cap["energy_cap"],
        "n_prop_orders": len(cap["propagating"]),
        "omitted_orders": ",".join(str(m) for m in cap["omitted"]),
        "cutoff_flag": 1.0 if diff["cutoff_orders"] else 0.0,
        "point_time": float(t2 - t0),
        "solve_time": float(t1 - t0),
        "post_time": float(t2 - t1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Solve only representative wavelengths before the full 97-point sweep.",
    )
    args = parser.parse_args()

    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"Missing mesh: {MESH_FILE}\n"
            "Run build_disordered_mesh_buffer2_pml3.py first."
        )

    fdtd = load_fdtd_reference()

    if args.quick:
        requested = np.array([
            1.064516129032258,
            1.170212765957447,
            1.500000000000000,
            1.755319148936170,
            2.323943661971831,
        ])
        indices = [int(np.argmin(np.abs(fdtd[:, 0] - w))) for w in requested]
        wavelengths = fdtd[sorted(set(indices)), 0].copy()
    else:
        wavelengths = fdtd[:, 0].copy()

    print()
    print("=" * 118)
    print("DISORDERED MEDIA — POSTDOC BOUNDARY + DIFFRACTION-ORDER TEST")
    print("=" * 118)
    print(f"Mesh                    = {MESH_FILE}")
    print(f"Ordinary-air buffer     = {AIR_BUFFER:.1f} um each side")
    print(f"PML thickness           = {PML_THICKNESS:.1f} um each side")
    print("Polarization            = Ez / TE")
    print("x boundary              = periodic, Bloch phase = 1")
    print("Mirror symmetry         = none")
    print("Primary R/T             = ALL propagating Rayleigh orders")
    print(f"Secondary diagnostic    = only m=-{ORDER_CAP}...+{ORDER_CAP}")
    print(f"Wavelength points       = {len(wavelengths)}")
    print(f"Workers                 = {args.workers}")
    print("=" * 118)
    print()

    wall_start = perf_counter()
    rows = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(MESH_FILE),),
    ) as executor:
        futures = {
            executor.submit(worker_solve, float(wl)): float(wl)
            for wl in wavelengths
        }
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            omitted = row["omitted_orders"] if row["omitted_orders"] else "-"
            print(
                f"[{i:03d}/{len(wavelengths):03d}] "
                f"lambda={row['wavelength']:.6f} um  "
                f"Rall={row['R_all']:.6f}  Tall={row['T_all']:.6f}  "
                f"R5={row['R_cap']:.6f}  T5={row['T_cap']:.6f}  "
                f"Nprop={row['n_prop_orders']:2d}  omitted={omitted:>6s}  "
                f"time={row['point_time']:.2f}s"
            )

    wall_time = perf_counter() - wall_start
    rows.sort(key=lambda r: r["wavelength"])

    w = np.array([r["wavelength"] for r in rows])
    R_all = np.array([r["R_all"] for r in rows])
    T_all = np.array([r["T_all"] for r in rows])
    RT_all = np.array([r["RT_all"] for r in rows])
    E_all = np.array([r["energy_all"] for r in rows])
    R_cap = np.array([r["R_cap"] for r in rows])
    T_cap = np.array([r["T_cap"] for r in rows])
    RT_cap = np.array([r["RT_cap"] for r in rows])
    E_cap = np.array([r["energy_cap"] for r in rows])
    n_prop = np.array([r["n_prop_orders"] for r in rows], dtype=int)
    cutoff = np.array([r["cutoff_flag"] for r in rows])
    point_time = np.array([r["point_time"] for r in rows])
    omitted_flag = np.array([1 if r["omitted_orders"] else 0 for r in rows], dtype=int)

    R_fdtd = np.interp(w, fdtd[:, 0], fdtd[:, 1])
    T_fdtd = np.interp(w, fdtd[:, 0], fdtd[:, 2])

    data = np.column_stack([
        w,
        R_all, T_all, RT_all, E_all,
        R_cap, T_cap, RT_cap, E_cap,
        n_prop, omitted_flag, cutoff,
        R_fdtd, T_fdtd,
        np.abs(R_all - R_fdtd),
        np.abs(T_all - T_fdtd),
        np.abs(R_all - R_cap),
        np.abs(T_all - T_cap),
        point_time,
    ])

    mode = "quick" if args.quick else "full"
    csv_path = OUT / f"postdoc_air2_pml3_{mode}.csv"
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_all,T_all,RplusT_all,energy_all,"
            "R_pm5,T_pm5,RplusT_pm5,energy_pm5,"
            "n_propagating_orders,pm5_omits_propagating_order,cutoff_flag,"
            "R_fdtd,T_fdtd,abs_dR_all_fdtd,abs_dT_all_fdtd,"
            "abs_R_all_minus_pm5,abs_T_all_minus_pm5,point_time_s"
        ),
        comments="",
    )

    omitted_path = OUT / f"postdoc_air2_pml3_{mode}_omitted_orders.txt"
    with omitted_path.open("w", encoding="utf-8") as f:
        f.write("Wavelength [um] : propagating orders omitted by +/-5\n")
        f.write("=" * 68 + "\n")
        for row in rows:
            omitted = row["omitted_orders"] if row["omitted_orders"] else "none"
            f.write(f"{row['wavelength']:.12f} : {omitted}\n")

    # Reflectance plot.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axvspan(1.4, 1.8, alpha=0.10, label="Target stop band 1.4-1.8 um")
    ax.plot(w, R_all, linewidth=2.1, label="scikit-fem R — all propagating orders")
    ax.plot(w, R_cap, ":", linewidth=1.8, label="scikit-fem R — only m=-5...+5")
    ax.plot(w, R_fdtd, "--", linewidth=1.6, label="FDTD R")
    ax.set_xlim(1.0, 2.4)
    ax.set_ylim(-0.04, 1.16)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Reflectance")
    ax.set_title("Reflectance: all propagating orders vs +/-5")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    r_plot = OUT / f"01_reflectance_{mode}.png"
    fig.savefig(r_plot, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Transmittance plot.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axvspan(1.4, 1.8, alpha=0.10, label="Target stop band 1.4-1.8 um")
    ax.plot(w, T_all, linewidth=2.1, label="scikit-fem T — all propagating orders")
    ax.plot(w, T_cap, ":", linewidth=1.8, label="scikit-fem T — only m=-5...+5")
    ax.plot(w, T_fdtd, "--", linewidth=1.6, label="FDTD T")
    ax.set_xlim(1.0, 2.4)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Transmittance")
    ax.set_title("Transmittance: all propagating orders vs +/-5")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    t_plot = OUT / f"02_transmittance_{mode}.png"
    fig.savefig(t_plot, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # What the finite +/-5 sum loses.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.semilogy(
        w,
        np.maximum(np.abs(R_all - R_cap), 1e-15),
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        label="|R(all) - R(+/-5)|",
    )
    ax.semilogy(
        w,
        np.maximum(np.abs(T_all - T_cap), 1e-15),
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        label="|T(all) - T(+/-5)|",
    )
    ax.axvspan(1.4, 1.8, alpha=0.10, label="Target stop band")
    ax.set_xlim(1.0, 2.4)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Absolute difference")
    ax.set_title("Effect of limiting diffraction sum to +/-5")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    diff_plot = OUT / f"03_pm5_loss_{mode}.png"
    fig.savefig(diff_plot, dpi=250, bbox_inches="tight")
    plt.close(fig)

    stop = ((w >= 1.4 - 1e-12) & (w <= 1.8 + 1e-12))

    print()
    print("=" * 118)
    print("FINAL SUMMARY")
    print("=" * 118)
    print(
        f"Wall clock                    = {wall_time:.3f} s "
        f"({int(wall_time // 60)} min {wall_time % 60:.1f} s)"
    )
    print(f"All-order max |R+T-1|         = {np.max(E_all):.6e}")
    print(f"All-order mean |R+T-1|        = {np.mean(E_all):.6e}")
    print(f"Max |R(all)-R(+/-5)|          = {np.max(np.abs(R_all-R_cap)):.6e}")
    print(f"Max |T(all)-T(+/-5)|          = {np.max(np.abs(T_all-T_cap)):.6e}")
    print(
        f"Wavelengths where +/-5 omits a propagating order = "
        f"{int(np.sum(omitted_flag))}/{len(w)}"
    )

    if np.any(stop):
        print()
        print("Target stop band 1.4-1.8 um:")
        print(f"  mean R all orders           = {np.mean(R_all[stop]):.9f}")
        print(f"  mean T all orders           = {np.mean(T_all[stop]):.9e}")
        print(f"  max |Rall-Rpm5| in band     = {np.max(np.abs(R_all[stop]-R_cap[stop])):.3e}")
        print(f"  max |Tall-Tpm5| in band     = {np.max(np.abs(T_all[stop]-T_cap[stop])):.3e}")

    print()
    print(f"CSV                           = {csv_path}")
    print(f"Omitted-order report          = {omitted_path}")
    print(f"Reflectance plot              = {r_plot}")
    print(f"Transmittance plot            = {t_plot}")
    print(f"+/-5 loss plot                = {diff_plot}")


if __name__ == "__main__":
    main()
