from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as mtri
import numpy as np

from photonics_fem.objective3_bodyfitted import (
    Objective3Config,
    build_bodyfitted_gmsh_mesh,
    diffraction_orders_bodyfitted,
    load_bodyfitted_mesh,
    solve_objective3_bodyfitted,
)


# ============================================================
# PATHS / SETTINGS
# ============================================================

OUT = Path(
    "results_objective3_redo"
)

OUT.mkdir(
    exist_ok=True
)

MESH_DIR = Path(
    "meshes"
)

MESH_DIR.mkdir(
    exist_ok=True
)

BASE_MESH = (
    MESH_DIR
    / "objective3_bodyfitted_fixed.msh"
)

FINE_MESH = (
    MESH_DIR
    / "objective3_bodyfitted_fine.msh"
)

FDTD_FILE = Path(
    "dataset_hollow_core_fdtd(1).csv"
)

FORCE_REMESH = False

# Original 26-point objective sweep.
WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    104,
)


def make_cfg(
    wavelength: float,
) -> Objective3Config:

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

        # Stronger PML than the first Objective-3 attempt.
        pml_top=1.00,
        pml_bottom=1.00,

        pml_order=3,
        pml_sigma_max=8.0,
    )


def build_meshes() -> None:

    reference_cfg = make_cfg(
        0.600
    )

    print()
    print(
        "Building/loading FIXED body-fitted meshes..."
    )

    build_bodyfitted_gmsh_mesh(
        BASE_MESH,
        reference_cfg,

        h_bulk=0.018,
        h_hole=0.008,

        refine_distance=0.20,

        force=FORCE_REMESH,
    )

    # One finer mesh is used only for the convergence test.
    build_bodyfitted_gmsh_mesh(
        FINE_MESH,
        reference_cfg,

        h_bulk=0.0135,
        h_hole=0.006,

        refine_distance=0.20,

        force=FORCE_REMESH,
    )


def mesh_diagnostics() -> None:

    cfg = make_cfg(
        0.600
    )

    mesh = load_bodyfitted_mesh(
        BASE_MESH,
        cfg,
    )

    print()
    print("=" * 84)
    print("FIXED BODY-FITTED MESH")
    print("=" * 84)

    print(
        f"vertices  = "
        f"{mesh.p.shape[1]}"
    )

    print(
        f"triangles = "
        f"{mesh.t.shape[1]}"
    )

    for name in (
        "top_pml",
        "top_air",
        "slab",
        "hole",
        "bottom_air",
        "bottom_pml",
    ):
        print(
            f"{name:12s} = "
            f"{len(mesh.subdomains[name])} elements"
        )

    # --------------------------------------------------------
    # Mesh visual: zoom around the actual circular boundary.
    # --------------------------------------------------------

    tri = mtri.Triangulation(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.triplot(
        tri,
        linewidth=0.35,
    )

    circle = patches.Circle(
        (
            cfg.hole_center_x,
            cfg.hole_center_y,
        ),
        cfg.hole_radius,
        fill=False,
        linewidth=2.0,
    )

    ax.add_patch(
        circle
    )

    ax.set_xlim(
        0.15,
        0.85,
    )

    ax.set_ylim(
        1.40,
        0.60,
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
        "Objective 3 body-fitted mesh: circular-hole zoom"
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "00_bodyfitted_mesh_hole_zoom.png",
        dpi=200,
    )

    plt.close()


def solve_one(
    wavelength: float,
    mesh_file: Path,
):

    cfg = make_cfg(
        wavelength
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

    return cfg, result, diff


def convergence_test() -> bool:

    print()
    print("=" * 84)
    print("MESH CONVERGENCE TEST @ 600 nm")
    print("=" * 84)

    cfg_b, result_b, base = solve_one(
        0.600,
        BASE_MESH,
    )

    cfg_f, result_f, fine = solve_one(
        0.600,
        FINE_MESH,
    )

    dR = abs(
        base["R"]
        - fine["R"]
    )

    dT = abs(
        base["T"]
        - fine["T"]
    )

    print()
    print("BASE MESH")

    print(
        f"DOFs = "
        f"{result_b.scattered.basis.N}"
    )

    print(
        f"R = {base['R']:.10f}"
    )

    print(
        f"T = {base['T']:.10f}"
    )

    print(
        f"R+T = {base['R_plus_T']:.10f}"
    )

    print(
        f"symmetry error = "
        f"{base['symmetry_abs_max']:.3e}"
    )

    print()
    print("FINE MESH")

    print(
        f"DOFs = "
        f"{result_f.scattered.basis.N}"
    )

    print(
        f"R = {fine['R']:.10f}"
    )

    print(
        f"T = {fine['T']:.10f}"
    )

    print(
        f"R+T = {fine['R_plus_T']:.10f}"
    )

    print(
        f"symmetry error = "
        f"{fine['symmetry_abs_max']:.3e}"
    )

    print()
    print(
        f"|R_base - R_fine| = "
        f"{dR:.6e}"
    )

    print(
        f"|T_base - T_fine| = "
        f"{dT:.6e}"
    )

    passed = (
        dR < 2e-3
        and dT < 2e-3
        and fine["energy_error"] < 2e-3
    )

    print()

    print(
        "CONVERGENCE:",
        "PASS"
        if passed
        else "CHECK",
    )

    return passed


def checkpoint_test() -> bool:

    wavelengths = [
        0.340,
        0.400,
        0.480,
        0.520,
        0.600,
        0.680,
        0.720,
        0.800,
    ]

    print()
    print("=" * 106)
    print("BODY-FITTED CHECKPOINTS")
    print("=" * 106)

    print(
        f"{'lambda':>8} "
        f"{'R':>12} "
        f"{'T':>12} "
        f"{'R+T':>12} "
        f"{'energy err':>12} "
        f"{'+m/-m err':>12} "
        f"{'cutoff':>14}"
    )

    all_pass = True

    for wavelength in wavelengths:

        _, _, diff = solve_one(
            wavelength,
            BASE_MESH,
        )

        cutoff_text = (
            str(
                diff["cutoff_orders"]
            )
        )

        passed = (
            diff["energy_error"] < 3e-3
            and diff["symmetry_abs_max"] < 3e-3
        )

        all_pass &= passed

        print(
            f"{wavelength:8.3f} "
            f"{diff['R']:12.8f} "
            f"{diff['T']:12.8f} "
            f"{diff['R_plus_T']:12.8f} "
            f"{diff['energy_error']:12.3e} "
            f"{diff['symmetry_abs_max']:12.3e} "
            f"{cutoff_text:>14}"
        )

    print()

    print(
        "CHECKPOINTS:",
        "PASS"
        if all_pass
        else "CHECK",
    )

    return all_pass


def field_plot_600nm() -> None:

    cfg, result, diff = solve_one(
        0.600,
        BASE_MESH,
    )

    sol = result.scattered

    rmesh, us = (
        sol.basis.refinterp(
            sol.u,
            nrefs=2,
        )
    )

    x = rmesh.p[0]
    y = rmesh.p[1]

    ui = np.exp(
        1j
        * cfg.k0
        * cfg.n_inc
        * y
    )

    total = (
        ui
        + us
    )

    triangle_y = (
        y[rmesh.t[0]]
        + y[rmesh.t[1]]
        + y[rmesh.t[2]]
    ) / 3.0

    mask = (
        (triangle_y < 0.0)
        |
        (
            triangle_y
            > cfg.total_height
        )
    )

    tri = mtri.Triangulation(
        x,
        y,
        rmesh.t.T,
    )

    tri.set_mask(
        mask
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5)
    )

    field = ax.tripcolor(
        tri,
        np.real(
            total
        ),
        shading="gouraud",
    )

    fig.colorbar(
        field,
        ax=ax,
        label="Re(E_total)",
    )

    circle = patches.Circle(
        (
            cfg.hole_center_x,
            cfg.hole_center_y,
        ),
        cfg.hole_radius,
        fill=False,
        linewidth=2.0,
    )

    ax.add_patch(
        circle
    )

    ax.axhline(
        cfg.slab_y0,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        cfg.slab_y1,
        linestyle="--",
        linewidth=1,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlim(
        0.0,
        cfg.width,
    )

    ax.set_ylim(
        cfg.total_height,
        0.0,
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Body-fitted Objective 3 field @ 600 nm"
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "01_bodyfitted_field_600nm.png",
        dpi=200,
    )

    plt.close()


def full_sweep():

    print()
    print("=" * 102)
    print("FULL OBJECTIVE 3 BODY-FITTED SWEEP")
    print("=" * 102)

    rows = []
    order_rows = []

    for i, wavelength in enumerate(
        WAVELENGTHS,
        start=1,
    ):

        cfg, result, diff = solve_one(
            float(wavelength),
            BASE_MESH,
        )

        propagating = [
            row["m"]
            for row in diff["orders"]
            if row["propagating"]
        ]

        cutoff = (
            diff["cutoff_orders"]
        )

        rows.append(
            [
                wavelength,
                diff["R"],
                diff["T"],
                diff["R_plus_T"],
                diff["energy_error"],
                diff["symmetry_abs_max"],
                result.scattered.basis.N,
                1.0 if cutoff else 0.0,
            ]
        )

        for row in diff["orders"]:

            if (
                row["propagating"]
                or row["near_cutoff"]
            ):
                order_rows.append(
                    [
                        wavelength,
                        row["m"],
                        row["R"],
                        row["T"],
                        1.0
                        if row["propagating"]
                        else 0.0,
                        1.0
                        if row["near_cutoff"]
                        else 0.0,
                    ]
                )

        print(
            f"[{i:02d}/{len(WAVELENGTHS)}] "
            f"lambda={wavelength * 1000:6.1f} nm  "
            f"R={diff['R']:.6f}  "
            f"T={diff['T']:.6f}  "
            f"R+T={diff['R_plus_T']:.6f}  "
            f"sym={diff['symmetry_abs_max']:.2e}  "
            f"orders={propagating}  "
            f"cutoff={cutoff}"
        )

    data = np.asarray(
        rows,
        dtype=float,
    )

    orders = np.asarray(
        order_rows,
        dtype=float,
    )

    np.savetxt(
        OUT
        / "objective3_bodyfitted_total_RT.csv",
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_total,"
            "T_total,"
            "R_plus_T,"
            "energy_error,"
            "symmetry_abs_max,"
            "dofs,"
            "cutoff_flag"
        ),
        comments="",
    )

    np.savetxt(
        OUT
        / "objective3_bodyfitted_orders.csv",
        orders,
        delimiter=",",
        header=(
            "wavelength_um,"
            "order_m,"
            "R_m,"
            "T_m,"
            "propagating,"
            "near_cutoff"
        ),
        comments="",
    )

    wavelength_nm = (
        data[:, 0]
        * 1000.0
    )

    # --------------------------------------------------------
    # Total R/T
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        wavelength_nm,
        data[:, 1],
        "o-",
        label="Body-fitted FEM R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 2],
        "o-",
        label="Body-fitted FEM T",
    )

    plt.plot(
        wavelength_nm,
        data[:, 3],
        "--",
        label="R + T",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Objective 3 redo: total R/T"
    )

    plt.ylim(
        -0.05,
        1.05,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "02_bodyfitted_total_RT.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Energy + symmetry errors.
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 4],
        "o-",
        label="|R + T - 1|",
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 5],
        "o-",
        label="max |+m - (-m)|",
    )

    cutoff_mask = (
        data[:, 7]
        > 0.5
    )

    if np.any(
        cutoff_mask
    ):
        plt.scatter(
            wavelength_nm[
                cutoff_mask
            ],
            data[
                cutoff_mask,
                4
            ],
            marker="x",
            s=80,
            label="exact Rayleigh cutoff",
        )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Absolute error"
    )

    plt.title(
        "Objective 3 numerical diagnostics"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "03_bodyfitted_diagnostics.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Order-resolved transmission.
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5.5)
    )

    unique_orders = np.unique(
        orders[:, 1]
    ).astype(int)

    for m in unique_orders:

        mask = (
            (orders[:, 1] == m)
            &
            (orders[:, 4] > 0.5)
        )

        if np.any(
            mask
        ):
            plt.plot(
                orders[
                    mask,
                    0
                ] * 1000.0,

                orders[
                    mask,
                    3
                ],

                "o-",

                label=f"T m={m}",
            )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Transmitted power"
    )

    plt.title(
        "Body-fitted transmitted diffraction orders"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "04_bodyfitted_transmission_orders.png",
        dpi=200,
    )

    plt.close()

    # Exclude exact cutoff points from the ordinary energy
    # metric; they are reported separately, not hidden.
    noncut = (
        data[:, 7]
        < 0.5
    )

    print()
    print(
        "Max non-cutoff |R+T-1| = "
        f"{np.max(data[noncut, 4]):.6e}"
    )

    print(
        "Max non-cutoff symmetry error = "
        f"{np.max(data[noncut, 5]):.6e}"
    )

    return data


def load_fdtd_reference():

    if not FDTD_FILE.exists():

        print()
        print(
            f"FDTD file not found at: "
            f"{FDTD_FILE}"
        )

        print(
            "Skipping FEM-vs-FDTD comparison."
        )

        return None

    raw = np.loadtxt(
        FDTD_FILE
    )

    if (
        raw.ndim != 2
        or raw.shape[1] < 3
    ):
        raise ValueError(
            "FDTD file must contain at least "
            "three whitespace-separated columns: "
            "wavelength R T."
        )

    fdtd = raw[
        :,
        :3
    ]

    order = np.argsort(
        fdtd[:, 0]
    )

    return fdtd[
        order
    ]


def compare_to_fdtd(
    fem_data,
) -> None:

    fdtd = load_fdtd_reference()

    if fdtd is None:
        return

    fem_w = fem_data[:, 0]
    fem_R = fem_data[:, 1]
    fem_T = fem_data[:, 2]

    fdtd_w = fdtd[:, 0]
    fdtd_R = fdtd[:, 1]
    fdtd_T = fdtd[:, 2]

    fdtd_R_at_fem = np.interp(
        fem_w,
        fdtd_w,
        fdtd_R,
    )

    fdtd_T_at_fem = np.interp(
        fem_w,
        fdtd_w,
        fdtd_T,
    )

    dR = (
        fem_R
        - fdtd_R_at_fem
    )

    dT = (
        fem_T
        - fdtd_T_at_fem
    )

    mae_R = float(
        np.mean(
            np.abs(dR)
        )
    )

    mae_T = float(
        np.mean(
            np.abs(dT)
        )
    )

    print()
    print("=" * 84)
    print("BODY-FITTED FEM vs SUPPLIED FDTD")
    print("=" * 84)

    print(
        f"MAE R = {mae_R:.6e}"
    )

    print(
        f"MAE T = {mae_T:.6e}"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        fdtd_w * 1000.0,
        fdtd_R,
        "-",
        label="FDTD R",
    )

    plt.plot(
        fdtd_w * 1000.0,
        fdtd_T,
        "-",
        label="FDTD T",
    )

    plt.plot(
        fem_w * 1000.0,
        fem_R,
        "o",
        label="Body-fitted FEM R",
    )

    plt.plot(
        fem_w * 1000.0,
        fem_T,
        "o",
        label="Body-fitted FEM T",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "Power fraction"
    )

    plt.title(
        "Objective 3: body-fitted FEM vs supplied FDTD"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "05_bodyfitted_vs_fdtd.png",
        dpi=200,
    )

    plt.close()

    comparison = np.column_stack(
        [
            fem_w,
            fem_R,
            fdtd_R_at_fem,
            dR,
            fem_T,
            fdtd_T_at_fem,
            dT,
        ]
    )

    np.savetxt(
        OUT
        / "objective3_bodyfitted_vs_fdtd.csv",
        comparison,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_fem,"
            "R_fdtd_interp,"
            "R_difference,"
            "T_fem,"
            "T_fdtd_interp,"
            "T_difference"
        ),
        comments="",
    )


def main() -> None:

    print()
    print("=" * 84)
    print("OBJECTIVE 3 COMPLETE REDO")
    print("BODY-FITTED CIRCLE + FIXED MESH + DIFFRACTION ORDERS")
    print("=" * 84)

    build_meshes()

    mesh_diagnostics()

    convergence_test()

    checkpoint_test()

    field_plot_600nm()

    fem_data = full_sweep()

    compare_to_fdtd(
        fem_data
    )

    print()
    print("=" * 84)
    print("OBJECTIVE 3 REDO COMPLETE")
    print("=" * 84)

    print()
    print(
        f"Results written to: {OUT}"
    )


if __name__ == "__main__":
    main()
