from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as mtri
import numpy as np

from photonics_fem.diffraction import (
    diffraction_orders,
)

from photonics_fem.periodic_hole import (
    PeriodicHoleConfig,
)

from photonics_fem.solver_periodic_hole import (
    solve_periodic_hole,
)


OUT = Path(
    "results_objective_3"
)

OUT.mkdir(
    exist_ok=True
)


WAVELENGTHS = np.linspace(
    0.300,
    0.800,
    26,
)


PPW = 12


def make_cfg(
    wavelength: float,
):

    return PeriodicHoleConfig(
        wavelength=wavelength,

        n_inc=1.0 + 0.0j,

        n_slab=1.5 + 0.0j,

        n_out=1.0 + 0.0j,

        slab_thickness=1.0,

        air_top=0.50,

        air_bottom=0.50,

        # Period = 1 um
        width=1.0,

        hole_diameter=0.50,

        pml_top=0.80,

        pml_bottom=0.80,

        pml_order=3,

        pml_sigma_max=6.0,
    )


def make_field_plot(
    result,
    cfg,
):

    sol = result.scattered

    rmesh, us = (
        sol.basis.refinterp(
            sol.u,
            nrefs=2,
        )
    )

    x = rmesh.p[0]
    y = rmesh.p[1]

    # Background incident field.
    ui = np.exp(
        1j
        * cfg.k0
        * y
    )

    utotal = (
        ui
        + us
    )

    triangle_y = (
        y[rmesh.t[0]]
        +
        y[rmesh.t[1]]
        +
        y[rmesh.t[2]]
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
        figsize=(8, 6)
    )

    field = ax.tripcolor(
        tri,
        np.real(
            utotal
        ),
        shading="gouraud",
    )

    fig.colorbar(
        field,
        ax=ax,
        label="Re(E_total)",
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

    circle = patches.Circle(
        (
            cfg.hole_center_x,
            cfg.hole_center_y,
        ),
        cfg.hole_radius,
        fill=False,
        linewidth=2,
    )

    ax.add_patch(
        circle
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Objective 3 periodic circular hole\n"
        f"lambda = {cfg.wavelength * 1000:.0f} nm"
    )

    ax.set_ylim(
        cfg.total_height,
        0.0,
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "objective3_field.png",
        dpi=180,
    )

    plt.close()


def main():

    print()
    print("=" * 84)
    print("OBJECTIVE 3")
    print("1 um PERIODIC n=1.5 SLAB WITH 0.5 um AIR HOLE")
    print("=" * 84)

    rows = []

    all_order_data = []

    for i, wavelength in enumerate(
        WAVELENGTHS,
        start=1,
    ):

        cfg = make_cfg(
            wavelength
        )

        h_target = (
            cfg.lambda_min_material
            / PPW
        )

        result = solve_periodic_hole(
            cfg,
            h_target=h_target,
        )

        diff = diffraction_orders(
            result,
            cfg,
            npoints=256,
        )

        energy_error = abs(
            diff["R_plus_T"]
            - 1.0
        )

        rows.append(
            [
                wavelength,

                diff["R"],

                diff["T"],

                diff["R_plus_T"],

                energy_error,

                result.scattered.basis.N,
            ]
        )

        # Store each propagating order.
        for order in diff["orders"]:

            if order["propagating"]:

                all_order_data.append(
                    [
                        wavelength,

                        order["m"],

                        order["R"],

                        order["T"],
                    ]
                )

        propagating = [
            order["m"]
            for order in diff["orders"]
            if order["propagating"]
        ]

        print(
            f"[{i:02d}/{len(WAVELENGTHS)}] "
            f"lambda="
            f"{wavelength * 1000:6.1f} nm  "
            f"R={diff['R']:.6f}  "
            f"T={diff['T']:.6f}  "
            f"R+T={diff['R_plus_T']:.6f}  "
            f"orders={propagating}"
        )

        # Make one representative field image.
        if abs(
            wavelength
            - 0.600
        ) < 1e-10:

            make_field_plot(
                result,
                cfg,
            )

    data = np.asarray(
        rows,
        dtype=float,
    )

    order_data = np.asarray(
        all_order_data,
        dtype=float,
    )

    np.savetxt(
        OUT / "objective3_total_RT.csv",
        data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "R_total,"
            "T_total,"
            "R_plus_T,"
            "energy_error,"
            "dofs"
        ),
        comments="",
    )

    np.savetxt(
        OUT / "objective3_orders.csv",
        order_data,
        delimiter=",",
        header=(
            "wavelength_um,"
            "order_m,"
            "R_m,"
            "T_m"
        ),
        comments="",
    )

    wavelength_nm = (
        data[:, 0]
        * 1000.0
    )

    # ========================================================
    # TOTAL R/T
    # ========================================================

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        wavelength_nm,
        data[:, 1],
        "o-",
        label="Total R",
    )

    plt.plot(
        wavelength_nm,
        data[:, 2],
        "o-",
        label="Total T",
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
        "Objective 3: periodic circular air hole"
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
        / "objective3_total_RT.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # ENERGY CONSERVATION
    # ========================================================

    plt.figure(
        figsize=(8.5, 5)
    )

    plt.semilogy(
        wavelength_nm,
        data[:, 4],
        "o-",
    )

    plt.xlabel(
        "Wavelength [nm]"
    )

    plt.ylabel(
        "|R + T - 1|"
    )

    plt.title(
        "Objective 3 energy conservation"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "objective3_energy_error.png",
        dpi=180,
    )

    plt.close()

    # ========================================================
    # ORDER-RESOLVED TRANSMISSION
    # ========================================================

    plt.figure(
        figsize=(9, 5.5)
    )

    unique_orders = np.unique(
        order_data[:, 1]
    ).astype(int)

    for m in unique_orders:

        mask = (
            order_data[:, 1]
            == m
        )

        plt.plot(
            order_data[
                mask,
                0
            ] * 1000.0,

            order_data[
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
        "Objective 3 transmitted diffraction orders"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "objective3_transmission_orders.png",
        dpi=180,
    )

    plt.close()

    print()
    print("=" * 84)

    print(
        "Maximum |R + T - 1| = "
        f"{np.max(data[:, 4]):.6e}"
    )

    print(
        "OBJECTIVE 3 SWEEP COMPLETE"
    )

    print("=" * 84)

    print()

    print(
        f"Results written to: "
        f"{OUT}"
    )


if __name__ == "__main__":
    main()