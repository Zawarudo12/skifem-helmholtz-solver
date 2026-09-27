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
from photonics_fem.disordered_fast import (
    DisorderedConfig,
    diffraction_orders,
)


# ============================================================
# FILES
# ============================================================

MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_buffer2_pml3_materialfine.msh"
)

FDTD_FILE = Path(
    "dataset_disordered_disks_fdtd.csv"
)

COMSOL_FILE = Path(
    "Disorderd1.csv"
)

OUT = Path(
    "results_disordered_postdoc_3way_materialfine"
)
OUT.mkdir(exist_ok=True)


# ============================================================
# PHYSICS / NUMERICS
# ============================================================

AIR_BUFFER = 2.0
PML_THICKNESS = 3.0

ORDER_CAP = 5
NPOINTS_RT = 1024

STOP_MIN = 1.4
STOP_MAX = 1.8

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


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

def load_fdtd():
    if not FDTD_FILE.exists():
        raise FileNotFoundError(
            f"Missing FDTD file: {FDTD_FILE}"
        )

    try:
        data = np.loadtxt(
            FDTD_FILE,
            delimiter=",",
        )
    except ValueError:
        data = np.loadtxt(
            FDTD_FILE
        )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            "FDTD file must contain wavelength, R, T."
        )

    data = data[:, :3]

    mask = (
        (data[:, 0] >= 1.0 - 1e-12)
        &
        (data[:, 0] <= 2.4 + 1e-12)
    )

    data = data[mask]

    order = np.argsort(
        data[:, 0]
    )

    return data[order]


def load_comsol():
    if not COMSOL_FILE.exists():
        raise FileNotFoundError(
            f"Missing COMSOL export: {COMSOL_FILE}\n"
            "Put Disorderd1.csv in the project root."
        )

    data = np.genfromtxt(
        COMSOL_FILE,
        delimiter=",",
        comments="%",
        dtype=float,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Remove any malformed / NaN rows.
    good = np.all(
        np.isfinite(data[:, :3]),
        axis=1,
    )
    data = data[good]

    if data.shape[1] < 3:
        raise ValueError(
            "COMSOL file must contain wavelength, R, T."
        )

    w = data[:, 0]
    R = data[:, 1]
    T = data[:, 2]

    if data.shape[1] >= 4:
        RT = data[:, 3]
    else:
        RT = R + T

    order = np.argsort(w)

    return (
        w[order],
        R[order],
        T[order],
        RT[order],
    )


# ============================================================
# WORKERS
# ============================================================

def worker_init(mesh_file: str):
    global _WORKER_SOLVER

    _WORKER_SOLVER = CachedDisorderedSolver(
        make_cfg(1.5),
        Path(mesh_file),
        intorder=8,
    )


def summarize_pm5(diff: dict):
    selected = [
        row
        for row in diff["orders"]
        if row["propagating"]
        and abs(int(row["m"])) <= ORDER_CAP
    ]

    omitted = [
        int(row["m"])
        for row in diff["orders"]
        if row["propagating"]
        and abs(int(row["m"])) > ORDER_CAP
    ]

    R5 = float(
        sum(row["R"] for row in selected)
    )

    T5 = float(
        sum(row["T"] for row in selected)
    )

    return R5, T5, omitted


def worker_solve(wavelength: float):
    global _WORKER_SOLVER

    if _WORKER_SOLVER is None:
        raise RuntimeError(
            "Worker cache not initialized."
        )

    cfg = make_cfg(
        wavelength
    )

    t0 = perf_counter()

    result = _WORKER_SOLVER.solve(
        cfg
    )

    t1 = perf_counter()

    diff = diffraction_orders(
        result,
        npoints=NPOINTS_RT,
    )

    R5, T5, omitted = summarize_pm5(
        diff
    )

    t2 = perf_counter()

    return {
        "wavelength": float(wavelength),

        # MAIN result: physically complete sum over all
        # propagating Rayleigh orders.
        "R_fem": float(diff["R"]),
        "T_fem": float(diff["T"]),
        "RT_fem": float(diff["R_plus_T"]),
        "energy_fem": float(diff["energy_error"]),

        # Secondary +/-5 diagnostic.
        "R_pm5": R5,
        "T_pm5": T5,

        "omitted_pm5": (
            ",".join(
                str(m)
                for m in omitted
            )
        ),

        "cutoff_flag": (
            1.0 if diff["cutoff_orders"] else 0.0
        ),

        "solve_time": float(t1 - t0),
        "post_time": float(t2 - t1),
        "point_time": float(t2 - t0),
    }


# ============================================================
# METRICS
# ============================================================

def metrics(a, b):
    d = np.asarray(a) - np.asarray(b)
    ad = np.abs(d)

    return {
        "mae": float(np.mean(ad)),
        "rmse": float(
            np.sqrt(
                np.mean(d ** 2)
            )
        ),
        "max": float(np.max(ad)),
    }


def print_metric(label, m):
    print(
        f"{label:<30} "
        f"MAE={m['mae']:.6e}  "
        f"RMSE={m['rmse']:.6e}  "
        f"MAX={m['max']:.6e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 5 representative wavelengths.",
    )

    args = parser.parse_args()

    for path in (
        MESH_FILE,
        FDTD_FILE,
        COMSOL_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

    fdtd = load_fdtd()

    (
        w_comsol,
        R_comsol_raw,
        T_comsol_raw,
        RT_comsol_raw,
    ) = load_comsol()

    # Use the exact FDTD wavelength grid as the scikit-fem grid.
    if args.quick:
        requested = np.array(
            [
                1.064516129032258,
                1.170212765957447,
                1.500000000000000,
                1.755319148936170,
                2.323943661971831,
            ],
            dtype=float,
        )

        indices = [
            int(
                np.argmin(
                    np.abs(
                        fdtd[:, 0] - w
                    )
                )
            )
            for w in requested
        ]

        indices = sorted(
            set(indices)
        )

        wavelengths = (
            fdtd[indices, 0]
            .copy()
        )

    else:
        wavelengths = (
            fdtd[:, 0]
            .copy()
        )

    print()
    print("=" * 118)
    print(
        "DISORDERED MEDIA — MATERIAL-FINE scikit-fem vs COMSOL vs FDTD"
    )
    print("=" * 118)
    print(f"Mesh                  = {MESH_FILE}")
    print(f"Air buffer            = {AIR_BUFFER:.1f} um each side")
    print(f"PML thickness         = {PML_THICKNESS:.1f} um each side")
    print("Polarization          = Ez / TE")
    print("x boundary            = periodic, Bloch phase = 1")
    print("scikit-fem R/T        = ALL propagating diffraction orders")
    print("Secondary diagnostic  = m=-5...+5 only")
    print(f"Wavelength points     = {len(wavelengths)}")
    print(f"Workers               = {args.workers}")
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
            executor.submit(
                worker_solve,
                float(wl),
            ): float(wl)
            for wl in wavelengths
        }

        for i, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            row = future.result()
            rows.append(row)

            omitted = (
                row["omitted_pm5"]
                if row["omitted_pm5"]
                else "-"
            )

            print(
                f"[{i:03d}/{len(wavelengths):03d}] "
                f"lambda={row['wavelength']:.6f}  "
                f"R={row['R_fem']:.6f}  "
                f"T={row['T_fem']:.6f}  "
                f"R5={row['R_pm5']:.6f}  "
                f"T5={row['T_pm5']:.6f}  "
                f"omitted={omitted:>6s}  "
                f"time={row['point_time']:.2f}s"
            )

    wall_time = (
        perf_counter()
        - wall_start
    )

    rows.sort(
        key=lambda r: r["wavelength"]
    )

    w = np.array(
        [
            r["wavelength"]
            for r in rows
        ]
    )

    R_fem = np.array(
        [
            r["R_fem"]
            for r in rows
        ]
    )

    T_fem = np.array(
        [
            r["T_fem"]
            for r in rows
        ]
    )

    RT_fem = np.array(
        [
            r["RT_fem"]
            for r in rows
        ]
    )

    E_fem = np.array(
        [
            r["energy_fem"]
            for r in rows
        ]
    )

    R5 = np.array(
        [
            r["R_pm5"]
            for r in rows
        ]
    )

    T5 = np.array(
        [
            r["T_pm5"]
            for r in rows
        ]
    )

    point_time = np.array(
        [
            r["point_time"]
            for r in rows
        ]
    )

    # Interpolate COMSOL and FDTD onto the exact FEM grid.
    R_comsol = np.interp(
        w,
        w_comsol,
        R_comsol_raw,
    )

    T_comsol = np.interp(
        w,
        w_comsol,
        T_comsol_raw,
    )

    RT_comsol = np.interp(
        w,
        w_comsol,
        RT_comsol_raw,
    )

    R_fdtd = np.interp(
        w,
        fdtd[:, 0],
        fdtd[:, 1],
    )

    T_fdtd = np.interp(
        w,
        fdtd[:, 0],
        fdtd[:, 2],
    )

    RT_fdtd = (
        R_fdtd
        + T_fdtd
    )

    # --------------------------------------------------------
    # Save aligned 3-way data.
    # --------------------------------------------------------
    data = np.column_stack(
        [
            w,

            R_fem,
            T_fem,
            RT_fem,
            E_fem,

            R_comsol,
            T_comsol,
            RT_comsol,

            R_fdtd,
            T_fdtd,
            RT_fdtd,

            R5,
            T5,

            np.abs(
                R_fem
                - R_comsol
            ),
            np.abs(
                T_fem
                - T_comsol
            ),

            np.abs(
                R_fem
                - R_fdtd
            ),
            np.abs(
                T_fem
                - T_fdtd
            ),

            np.abs(
                R_comsol
                - R_fdtd
            ),
            np.abs(
                T_comsol
                - T_fdtd
            ),

            np.abs(
                R_fem
                - R5
            ),
            np.abs(
                T_fem
                - T5
            ),

            point_time,
        ]
    )

    mode = (
        "quick"
        if args.quick
        else "full"
    )

    csv_path = (
        OUT
        / f"threeway_{mode}.csv"
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_skfem,T_skfem,RplusT_skfem,energy_skfem,"
            "R_comsol,T_comsol,RplusT_comsol,"
            "R_fdtd,T_fdtd,RplusT_fdtd,"
            "R_pm5,T_pm5,"
            "abs_dR_skfem_comsol,abs_dT_skfem_comsol,"
            "abs_dR_skfem_fdtd,abs_dT_skfem_fdtd,"
            "abs_dR_comsol_fdtd,abs_dT_comsol_fdtd,"
            "abs_dR_all_pm5,abs_dT_all_pm5,"
            "point_time_s"
        ),
        comments="",
    )

    # --------------------------------------------------------
    # REFLECTANCE — MAIN FINAL PLOT
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10.8, 6.4)
    )

    ax.axvspan(
        STOP_MIN,
        STOP_MAX,
        alpha=0.10,
        label="Target stop band 1.4-1.8 um",
    )

    ax.plot(
        w,
        R_comsol,
        linewidth=2.2,
        label="COMSOL R",
    )

    ax.plot(
        w,
        R_fem,
        linewidth=2.1,
        label="scikit-fem R — all propagating orders",
    )

    ax.plot(
        w,
        R_fdtd,
        "--",
        linewidth=1.8,
        label="FDTD R",
    )

    ax.set_xlim(
        1.0,
        2.4,
    )

    ax.set_ylim(
        -0.04,
        1.16,
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Reflectance"
    )

    ax.set_title(
        "Reflectance: COMSOL vs full scikit-fem vs FDTD"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    r_plot = (
        OUT
        / f"01_reflectance_COMSOL_skFEM_FDTD_{mode}.png"
    )

    fig.savefig(
        r_plot,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # TRANSMITTANCE — MAIN FINAL PLOT
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10.8, 6.4)
    )

    ax.axvspan(
        STOP_MIN,
        STOP_MAX,
        alpha=0.10,
        label="Target stop band 1.4-1.8 um",
    )

    ax.plot(
        w,
        T_comsol,
        linewidth=2.2,
        label="COMSOL T",
    )

    ax.plot(
        w,
        T_fem,
        linewidth=2.1,
        label="scikit-fem T — all propagating orders",
    )

    ax.plot(
        w,
        T_fdtd,
        "--",
        linewidth=1.8,
        label="FDTD T",
    )

    ax.set_xlim(
        1.0,
        2.4,
    )

    ax.set_ylim(
        -0.04,
        1.10,
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Transmittance"
    )

    ax.set_title(
        "Transmittance: COMSOL vs full scikit-fem vs FDTD"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    t_plot = (
        OUT
        / f"02_transmittance_COMSOL_skFEM_FDTD_{mode}.png"
    )

    fig.savefig(
        t_plot,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # +/-5 DIAGNOSTIC ONLY
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10.8, 5.8)
    )

    ax.semilogy(
        w,
        np.maximum(
            np.abs(
                R_fem - R5
            ),
            1e-15,
        ),
        linewidth=1.8,
        label="|R(all) - R(+/-5)|",
    )

    ax.semilogy(
        w,
        np.maximum(
            np.abs(
                T_fem - T5
            ),
            1e-15,
        ),
        linewidth=1.8,
        label="|T(all) - T(+/-5)|",
    )

    ax.axvspan(
        STOP_MIN,
        STOP_MAX,
        alpha=0.10,
        label="Target stop band",
    )

    ax.set_xlim(
        1.0,
        2.4,
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Absolute difference"
    )

    ax.set_title(
        "Diagnostic only: full diffraction sum vs +/-5"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    pm5_plot = (
        OUT
        / f"03_pm5_diagnostic_{mode}.png"
    )

    fig.savefig(
        pm5_plot,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m_sc_R = metrics(
        R_fem,
        R_comsol,
    )
    m_sc_T = metrics(
        T_fem,
        T_comsol,
    )

    m_sf_R = metrics(
        R_fem,
        R_fdtd,
    )
    m_sf_T = metrics(
        T_fem,
        T_fdtd,
    )

    m_cf_R = metrics(
        R_comsol,
        R_fdtd,
    )
    m_cf_T = metrics(
        T_comsol,
        T_fdtd,
    )

    stop = (
        (w >= STOP_MIN - 1e-12)
        &
        (w <= STOP_MAX + 1e-12)
    )

    print()
    print("=" * 118)
    print("FINAL THREE-WAY SUMMARY")
    print("=" * 118)

    print(
        f"scikit-fem wall clock        = "
        f"{wall_time:.3f} s "
        f"({int(wall_time // 60)} min "
        f"{wall_time % 60:.1f} s)"
    )

    print()
    print("FULL-SPECTRUM ERROR")
    print("-" * 118)

    print_metric(
        "scikit-fem vs COMSOL R",
        m_sc_R,
    )
    print_metric(
        "scikit-fem vs COMSOL T",
        m_sc_T,
    )

    print_metric(
        "scikit-fem vs FDTD R",
        m_sf_R,
    )
    print_metric(
        "scikit-fem vs FDTD T",
        m_sf_T,
    )

    print_metric(
        "COMSOL vs FDTD R",
        m_cf_R,
    )
    print_metric(
        "COMSOL vs FDTD T",
        m_cf_T,
    )

    print()
    print("POWER CONSERVATION")
    print("-" * 118)

    print(
        f"scikit-fem max |R+T-1|      = "
        f"{np.max(np.abs(RT_fem - 1.0)):.6e}"
    )

    print(
        f"COMSOL max |R+T-1|          = "
        f"{np.max(np.abs(RT_comsol - 1.0)):.6e}"
    )

    print(
        f"FDTD max |R+T-1|            = "
        f"{np.max(np.abs(RT_fdtd - 1.0)):.6e}"
    )

    if np.any(stop):
        print()
        print("TARGET STOP BAND 1.4-1.8 um")
        print("-" * 118)

        print(
            f"COMSOL       mean R="
            f"{np.mean(R_comsol[stop]):.9f}  "
            f"mean T="
            f"{np.mean(T_comsol[stop]):.9e}"
        )

        print(
            f"scikit-fem   mean R="
            f"{np.mean(R_fem[stop]):.9f}  "
            f"mean T="
            f"{np.mean(T_fem[stop]):.9e}"
        )

        print(
            f"FDTD         mean R="
            f"{np.mean(R_fdtd[stop]):.9f}  "
            f"mean T="
            f"{np.mean(T_fdtd[stop]):.9e}"
        )

        print()
        print(
            f"Max |Rall-Rpm5| in band     = "
            f"{np.max(np.abs(R_fem[stop] - R5[stop])):.6e}"
        )

        print(
            f"Max |Tall-Tpm5| in band     = "
            f"{np.max(np.abs(T_fem[stop] - T5[stop])):.6e}"
        )

    print()
    print("=" * 118)
    print("SAVED")
    print("=" * 118)
    print(f"Aligned CSV                 = {csv_path}")
    print(f"Reflectance plot            = {r_plot}")
    print(f"Transmittance plot          = {t_plot}")
    print(f"+/-5 diagnostic            = {pm5_plot}")


if __name__ == "__main__":
    main()
