from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FILES
# ============================================================

ROOT = Path(".")

COMSOL_FILE = ROOT / "Disorderd1.csv"
FDTD_FILE = ROOT / "dataset_disordered_disks_fdtd.csv"
FEM_FILE = (
    ROOT
    / "results_disordered_improved_boundary"
    / "disordered_improved_boundary_full.csv"
)

OUT = ROOT / "results_final_3way"
OUT.mkdir(exist_ok=True)

# Measured directly from the COMSOL sweep shown in the solver log.
COMSOL_WALL_TIME_S = 163.0

# The improved scikit-fem sweep CSV stores per-wavelength timings,
# not the exact total parallel wall clock.  If you know the exact
# terminal wall-clock value from that run, put it here.
#
# Example:
# SKIFEM_EXACT_WALL_TIME_S = 120.3
#
# Leave as None to estimate from point_time_s using 2 workers.
SKIFEM_EXACT_WALL_TIME_S = None
SKIFEM_WORKERS = 2

# The supplied FDTD dataset does not contain a runtime.
FDTD_WALL_TIME_S = None

STOP_MIN = 1.4
STOP_MAX = 1.8


# ============================================================
# LOADERS
# ============================================================

def load_comsol(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"COMSOL export not found: {path}\n"
            "Put Disorderd1.csv in the project root."
        )

    # COMSOL export has metadata/header lines beginning with '%'.
    data = np.genfromtxt(
        path,
        delimiter=",",
        comments="%",
        dtype=float,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 numerical columns in {path}."
        )

    w = data[:, 0]
    R = data[:, 1]
    T = data[:, 2]

    if data.shape[1] >= 4:
        RT = data[:, 3]
    else:
        RT = R + T

    order = np.argsort(w)
    return w[order], R[order], T[order], RT[order]


def load_fdtd(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"FDTD file not found: {path}"
        )

    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            "FDTD file must contain wavelength, R, T."
        )

    w = data[:, 0]
    R = data[:, 1]
    T = data[:, 2]

    mask = (
        (w >= 1.0 - 1e-12)
        &
        (w <= 2.4 + 1e-12)
    )

    w = w[mask]
    R = R[mask]
    T = T[mask]

    order = np.argsort(w)
    return w[order], R[order], T[order]


def load_fem(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Improved scikit-fem file not found:\n{path}\n\n"
            "Expected output from run_disordered_sweep_improved_boundary.py"
        )

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )

    names = set(data.dtype.names or [])

    required = {
        "wavelength_um",
        "R_fem",
        "T_fem",
    }

    missing = required - names

    if missing:
        raise ValueError(
            f"Missing FEM CSV columns: {sorted(missing)}"
        )

    w = np.asarray(data["wavelength_um"], dtype=float)
    R = np.asarray(data["R_fem"], dtype=float)
    T = np.asarray(data["T_fem"], dtype=float)

    RT = (
        np.asarray(data["R_plus_T"], dtype=float)
        if "R_plus_T" in names
        else R + T
    )

    point_time = (
        np.asarray(data["point_time_s"], dtype=float)
        if "point_time_s" in names
        else None
    )

    order = np.argsort(w)

    return (
        w[order],
        R[order],
        T[order],
        RT[order],
        None if point_time is None else point_time[order],
    )


# ============================================================
# HELPERS
# ============================================================

def interp_to(x_target, x, y):
    return np.interp(x_target, x, y)


def metrics(a, b):
    d = np.asarray(a) - np.asarray(b)
    ad = np.abs(d)

    return {
        "mae": float(np.mean(ad)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max": float(np.max(ad)),
    }


def fmt_time(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "not provided"

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f} s"

    mins = int(seconds // 60)
    secs = seconds - 60 * mins

    return f"{seconds:.1f} s ({mins} min {secs:.0f} s)"


def save_spectrum_plot(
    path,
    ylabel,
    title,
    w,
    comsol_y,
    fem_y,
    fdtd_y,
    ylim,
):
    fig, ax = plt.subplots(figsize=(10.8, 6.4))

    ax.axvspan(
        STOP_MIN,
        STOP_MAX,
        alpha=0.10,
        label="Target stop band 1.4–1.8 µm",
    )

    ax.plot(
        w,
        fem_y,
        linewidth=2.15,
        label="scikit-fem Helmholtz",
    )

    ax.plot(
        w,
        comsol_y,
        linewidth=2.0,
        linestyle="--",
        label="COMSOL EWFD",
    )

    ax.plot(
        w,
        fdtd_y,
        linewidth=1.7,
        linestyle=":",
        label="FDTD reference",
    )

    ax.set_xlim(1.0, 2.4)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 96)
    print("DISORDERED MEDIA — COMSOL vs scikit-fem vs FDTD")
    print("=" * 96)

    wc, Rc, Tc, RTc = load_comsol(COMSOL_FILE)
    wf, Rf, Tf, RTf, fem_point_time = load_fem(FEM_FILE)
    wd, Rd, Td = load_fdtd(FDTD_FILE)

    # Use COMSOL's 97-point wavelength grid as the common comparison grid.
    w = wc

    Rf_i = interp_to(w, wf, Rf)
    Tf_i = interp_to(w, wf, Tf)
    RTf_i = interp_to(w, wf, RTf)

    Rd_i = interp_to(w, wd, Rd)
    Td_i = interp_to(w, wd, Td)

    # --------------------------
    # Timing
    # --------------------------
    if SKIFEM_EXACT_WALL_TIME_S is not None:
        fem_wall = float(SKIFEM_EXACT_WALL_TIME_S)
        fem_time_note = "measured wall time"
    elif fem_point_time is not None:
        # For a dynamic 2-worker pool this is a useful approximation,
        # but not a substitute for the measured perf_counter wall time.
        fem_wall = float(np.sum(fem_point_time) / SKIFEM_WORKERS)
        fem_time_note = (
            f"estimated from sum(point_time_s)/{SKIFEM_WORKERS}; "
            "worker setup/imbalance not included"
        )
    else:
        fem_wall = None
        fem_time_note = "not available"

    # --------------------------
    # Pairwise errors
    # --------------------------
    r_cf = metrics(Rc, Rf_i)
    t_cf = metrics(Tc, Tf_i)

    r_cd = metrics(Rc, Rd_i)
    t_cd = metrics(Tc, Td_i)

    r_fd = metrics(Rf_i, Rd_i)
    t_fd = metrics(Tf_i, Td_i)

    stop = (
        (w >= STOP_MIN - 1e-12)
        &
        (w <= STOP_MAX + 1e-12)
    )

    # --------------------------
    # Save separate R/T plots
    # --------------------------
    r_plot = OUT / "01_reflectance_COMSOL_scikitFEM_FDTD.png"
    t_plot = OUT / "02_transmittance_COMSOL_scikitFEM_FDTD.png"

    save_spectrum_plot(
        r_plot,
        ylabel="Reflectance R",
        title="Disordered medium — Reflectance comparison",
        w=w,
        comsol_y=Rc,
        fem_y=Rf_i,
        fdtd_y=Rd_i,
        ylim=(-0.04, 1.18),
    )

    save_spectrum_plot(
        t_plot,
        ylabel="Transmittance T",
        title="Disordered medium — Transmittance comparison",
        w=w,
        comsol_y=Tc,
        fem_y=Tf_i,
        fdtd_y=Td_i,
        ylim=(-0.04, 1.05),
    )

    # --------------------------
    # Runtime plot
    # --------------------------
    runtime_plot = OUT / "03_runtime_comparison.png"

    runtime_labels = []
    runtime_values = []

    if fem_wall is not None:
        runtime_labels.append("scikit-fem\n(2 workers)")
        runtime_values.append(fem_wall)

    runtime_labels.append("COMSOL")
    runtime_values.append(COMSOL_WALL_TIME_S)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    x = np.arange(len(runtime_labels))
    bars = ax.bar(x, runtime_values)

    ax.set_xticks(x)
    ax.set_xticklabels(runtime_labels)

    ax.set_ylabel("Sweep wall time [s]")
    ax.set_title("97-wavelength sweep runtime")

    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, runtime_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f} s",
            ha="center",
            va="bottom",
        )

    ax.text(
        0.5,
        -0.17,
        "FDTD runtime not provided with the reference dataset.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        runtime_plot,
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------
    # Combined aligned CSV
    # --------------------------
    aligned = np.column_stack(
        [
            w,
            Rc,
            Tc,
            Rc + Tc,
            Rf_i,
            Tf_i,
            RTf_i,
            Rd_i,
            Td_i,
            Rd_i + Td_i,
        ]
    )

    aligned_csv = OUT / "04_threeway_aligned_data.csv"

    np.savetxt(
        aligned_csv,
        aligned,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_comsol,T_comsol,RT_comsol,"
            "R_scikitfem,T_scikitfem,RT_scikitfem,"
            "R_fdtd,T_fdtd,RT_fdtd"
        ),
        comments="",
    )

    # --------------------------
    # Summary text
    # --------------------------
    summary = []

    summary.append(
        "DISORDERED MEDIA — THREE-WAY VALIDATION SUMMARY"
    )
    summary.append("=" * 72)
    summary.append("")
    summary.append(
        f"Common comparison wavelengths: {len(w)}"
    )
    summary.append(
        f"Range: {w.min():.6f}–{w.max():.6f} µm"
    )
    summary.append("")

    summary.append("RUNTIME")
    summary.append("-" * 72)
    summary.append(
        f"COMSOL sweep wall time:      {fmt_time(COMSOL_WALL_TIME_S)} "
        "(measured)"
    )
    summary.append(
        f"scikit-fem sweep time:       {fmt_time(fem_wall)} "
        f"({fem_time_note})"
    )
    summary.append(
        "FDTD reference runtime:      not provided"
    )

    known_times = [
        t
        for t in (
            COMSOL_WALL_TIME_S,
            fem_wall,
        )
        if t is not None
    ]

    if known_times:
        summary.append(
            f"Known COMSOL + FEM compute:   {fmt_time(sum(known_times))}"
        )

    summary.append("")
    summary.append("POWER CONSERVATION")
    summary.append("-" * 72)
    summary.append(
        f"COMSOL max |R+T-1|:          "
        f"{np.max(np.abs(RTc - 1.0)):.6e}"
    )
    summary.append(
        f"scikit-fem max |R+T-1|:      "
        f"{np.max(np.abs(RTf_i - 1.0)):.6e}"
    )
    summary.append(
        f"FDTD max |R+T-1|:            "
        f"{np.max(np.abs(Rd_i + Td_i - 1.0)):.6e}"
    )

    summary.append("")
    summary.append("PAIRWISE FULL-SPECTRUM ERROR")
    summary.append("-" * 72)

    summary.append(
        f"COMSOL vs scikit-fem R: MAE={r_cf['mae']:.6e}, "
        f"RMSE={r_cf['rmse']:.6e}, MAX={r_cf['max']:.6e}"
    )
    summary.append(
        f"COMSOL vs scikit-fem T: MAE={t_cf['mae']:.6e}, "
        f"RMSE={t_cf['rmse']:.6e}, MAX={t_cf['max']:.6e}"
    )

    summary.append(
        f"COMSOL vs FDTD R:       MAE={r_cd['mae']:.6e}, "
        f"RMSE={r_cd['rmse']:.6e}, MAX={r_cd['max']:.6e}"
    )
    summary.append(
        f"COMSOL vs FDTD T:       MAE={t_cd['mae']:.6e}, "
        f"RMSE={t_cd['rmse']:.6e}, MAX={t_cd['max']:.6e}"
    )

    summary.append(
        f"scikit-fem vs FDTD R:   MAE={r_fd['mae']:.6e}, "
        f"RMSE={r_fd['rmse']:.6e}, MAX={r_fd['max']:.6e}"
    )
    summary.append(
        f"scikit-fem vs FDTD T:   MAE={t_fd['mae']:.6e}, "
        f"RMSE={t_fd['rmse']:.6e}, MAX={t_fd['max']:.6e}"
    )

    summary.append("")
    summary.append("TARGET STOP BAND 1.4–1.8 µm")
    summary.append("-" * 72)

    for name, Rvals, Tvals in [
        ("COMSOL", Rc, Tc),
        ("scikit-fem", Rf_i, Tf_i),
        ("FDTD", Rd_i, Td_i),
    ]:
        summary.append(
            f"{name:<12} mean R={np.mean(Rvals[stop]):.9f}, "
            f"mean T={np.mean(Tvals[stop]):.6e}, "
            f"max T={np.max(Tvals[stop]):.6e}"
        )

    summary.append("")
    summary.append(
        "Note: runtime values are not a strict apples-to-apples benchmark "
        "because COMSOL and scikit-fem use different numerical formulations, "
        "meshes, solver implementations, and boundary treatments."
    )

    summary_text = "\n".join(summary)

    summary_path = OUT / "05_threeway_summary.txt"
    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(summary_text)
    print()
    print("=" * 96)
    print("SAVED")
    print("=" * 96)
    print(f"Reflectance plot    : {r_plot}")
    print(f"Transmittance plot  : {t_plot}")
    print(f"Runtime plot        : {runtime_plot}")
    print(f"Aligned data        : {aligned_csv}")
    print(f"Summary             : {summary_path}")


if __name__ == "__main__":
    main()
