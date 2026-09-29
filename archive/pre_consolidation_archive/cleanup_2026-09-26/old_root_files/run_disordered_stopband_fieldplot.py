from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import numpy as np

from photonics_fem.disordered_fast import (
    DisorderedConfig,
    diffraction_orders,
    solve_disordered,
)


MESH_FILE = Path(
    "meshes_disordered/disordered_196disks_pilot.msh"
)

POSITIONS_FILE = Path(
    "disks_positions_diameter=0.26_L=7.csv"
)

OUT = Path(
    "results_disordered_field"
)
OUT.mkdir(exist_ok=True)

WAVELENGTH = 1.500
DISK_RADIUS = 0.13


def make_cfg() -> DisorderedConfig:
    return DisorderedConfig(
        wavelength=WAVELENGTH,
        width=7.0,
        xmin=-3.5,
        xmax=3.5,
        scatter_ymin=-3.5,
        scatter_ymax=3.5,
        physical_ymin=-4.0,
        physical_ymax=4.0,
        pml_low=1.0,
        pml_high=1.0,
        pml_order=3,
        pml_sigma_max=8.0,
        n_background=1.0 + 0.0j,
        n_disk=3.0 + 0.0j,
    )


def evaluate_scattered_at_vertices(result):
    """
    Evaluate the P2 scattered field at the linear mesh vertices.

    Chunking avoids constructing an unnecessarily large interpolation
    matrix in one shot.
    """
    points = result.mesh.p
    n = points.shape[1]

    values = np.empty(
        n,
        dtype=np.complex128,
    )

    interp = result.basis.interpolator(
        result.u_scattered
    )

    chunk = 4000

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)

        values[start:stop] = np.asarray(
            interp(points[:, start:stop]),
            dtype=np.complex128,
        ).reshape(-1)

    return values


def main():
    if not MESH_FILE.exists():
        raise FileNotFoundError(
            f"Could not find mesh: {MESH_FILE}"
        )

    if not POSITIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find positions: {POSITIONS_FILE}"
        )

    cfg = make_cfg()

    print()
    print("=" * 92)
    print("STOP-BAND FIELD MAP")
    print("=" * 92)
    print(f"Wavelength = {cfg.wavelength:.4f} um")
    print(f"Mesh       = {MESH_FILE}")
    print("Plot       = total |Ez| in physical region, PML hidden")
    print("=" * 92)

    t0 = perf_counter()

    result = solve_disordered(
        cfg,
        MESH_FILE,
        intorder=8,
    )

    diff = diffraction_orders(
        result,
        npoints=1024,
    )

    print()
    print(f"R       = {diff['R']:.10f}")
    print(f"T       = {diff['T']:.10e}")
    print(f"R + T   = {diff['R_plus_T']:.10f}")
    print(
        f"|R+T-1| = {diff['energy_error']:.3e}"
    )

    u_sc = evaluate_scattered_at_vertices(
        result
    )

    x = result.mesh.p[0]
    y = result.mesh.p[1]

    # Background downward plane wave, exp(-i k y), e^{-i wt}.
    u_inc = np.exp(
        -1j
        * cfg.k0
        * cfg.n_background
        * y
    )

    u_total = (
        u_inc
        + u_sc
    )

    amplitude = np.abs(
        u_total
    )

    # Hide the PML from the field map.  It is an absorbing numerical
    # layer, not part of the physical structure we want to visualize.
    physical_triangle_mask = np.any(
        (
            y[result.mesh.t]
            < cfg.physical_ymin - 1e-12
        )
        |
        (
            y[result.mesh.t]
            > cfg.physical_ymax + 1e-12
        ),
        axis=0,
    )

    triangulation = mtri.Triangulation(
        x,
        y,
        result.mesh.t.T,
    )

    triangulation.set_mask(
        physical_triangle_mask
    )

    # Robust upper limit: prevent a tiny local hot spot from washing
    # out the entire figure.
    physical_vertices = (
        (y >= cfg.physical_ymin - 1e-12)
        &
        (y <= cfg.physical_ymax + 1e-12)
    )

    vmax = float(
        np.percentile(
            amplitude[physical_vertices],
            99.5,
        )
    )

    vmax = max(
        vmax,
        1.0,
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 9.0)
    )

    image = ax.tripcolor(
        triangulation,
        amplitude,
        shading="gouraud",
        vmin=0.0,
        vmax=vmax,
    )

    positions = np.loadtxt(
        POSITIONS_FILE
    )

    for px, py in positions:
        ax.add_patch(
            Circle(
                (float(px), float(py)),
                DISK_RADIUS,
                fill=False,
                linewidth=0.30,
                edgecolor="white",
                alpha=0.55,
            )
        )

        # Show wrapped periodic copy when a disk crosses x boundary.
        if px - DISK_RADIUS < cfg.xmin:
            ax.add_patch(
                Circle(
                    (float(px + cfg.width), float(py)),
                    DISK_RADIUS,
                    fill=False,
                    linewidth=0.30,
                    edgecolor="white",
                    alpha=0.55,
                )
            )

        if px + DISK_RADIUS > cfg.xmax:
            ax.add_patch(
                Circle(
                    (float(px - cfg.width), float(py)),
                    DISK_RADIUS,
                    fill=False,
                    linewidth=0.30,
                    edgecolor="white",
                    alpha=0.55,
                )
            )

    # Nominal 7x7 disordered region.
    ax.axhline(
        cfg.scatter_ymax,
        linewidth=0.8,
        linestyle="--",
        alpha=0.75,
    )

    ax.axhline(
        cfg.scatter_ymin,
        linewidth=0.8,
        linestyle="--",
        alpha=0.75,
    )

    # Monitor locations.
    ax.axhline(
        cfg.top_monitor_y,
        linewidth=0.7,
        linestyle=":",
        alpha=0.75,
    )

    ax.axhline(
        cfg.bottom_monitor_y,
        linewidth=0.7,
        linestyle=":",
        alpha=0.75,
    )

    ax.annotate(
        "incident wave",
        xy=(2.75, 3.25),
        xytext=(2.75, 3.88),
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.4,
        ),
    )

    ax.text(
        -3.32,
        3.60,
        "top air buffer",
        fontsize=9,
        va="center",
    )

    ax.text(
        -3.32,
        0.0,
        "disordered n = 3 disks",
        fontsize=9,
        va="center",
        rotation=90,
    )

    ax.text(
        -3.32,
        -3.60,
        "bottom air buffer",
        fontsize=9,
        va="center",
    )

    ax.set_xlim(
        cfg.xmin,
        cfg.xmax,
    )

    ax.set_ylim(
        cfg.physical_ymin,
        cfg.physical_ymax,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Disordered medium stop-band field\n"
        r"$\lambda = 1.50\,\mu m$"
        + f"   |   R = {diff['R']:.5f}, T = {diff['T']:.2e}"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.025,
    )

    colorbar.set_label(
        r"Total field magnitude $|E_z|$"
    )

    fig.tight_layout()

    out_png = (
        OUT
        / "Ez_total_stopband_1p50um.png"
    )

    fig.savefig(
        out_png,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Also save a compact numerical record.
    out_txt = (
        OUT
        / "Ez_total_stopband_1p50um_summary.txt"
    )

    out_txt.write_text(
        "\n".join(
            [
                f"wavelength_um={cfg.wavelength:.12g}",
                f"R={diff['R']:.12g}",
                f"T={diff['T']:.12g}",
                f"R_plus_T={diff['R_plus_T']:.12g}",
                f"energy_error={diff['energy_error']:.12g}",
                f"plot_vmax_99p5={vmax:.12g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Saved plot    : {out_png}")
    print(f"Saved summary : {out_txt}")
    print(
        f"Total runtime : "
        f"{perf_counter() - t0:.3f} s"
    )


if __name__ == "__main__":
    main()
