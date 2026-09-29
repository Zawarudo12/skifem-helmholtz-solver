from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# ============================================================
# DISORDERED-MEDIA GEOMETRY SETTINGS
# ============================================================

POSITIONS_FILE = Path(
    "disks_positions_diameter=0.26_L=7.csv"
)

OUT = Path(
    "results_disordered"
)

OUT.mkdir(
    exist_ok=True
)

L = 7.0                  # micrometres
HALF_L = L / 2.0

DISK_DIAMETER = 0.26     # micrometres
DISK_RADIUS = DISK_DIAMETER / 2.0


def load_positions() -> np.ndarray:

    if not POSITIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find: {POSITIONS_FILE}\n"
            "Put the supplied positions CSV in the project root."
        )

    # The supplied file is whitespace-separated despite the .csv suffix.
    positions = np.loadtxt(
        POSITIONS_FILE
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 2
    ):
        raise ValueError(
            "Expected exactly two columns: x_um y_um"
        )

    return positions


def periodic_x_geometry(
    positions: np.ndarray,
):
    """
    Return the original disk centers plus only the x-shifted periodic
    copies needed to represent disks that cross x = +/- L/2.

    These extra copies are important because x is periodic.
    """

    circles = []

    for x, y in positions:

        # Original disk.
        circles.append(
            (float(x), float(y), "original")
        )

        # Disk crosses the LEFT edge:
        # its continuation re-enters from the RIGHT edge.
        if x - DISK_RADIUS < -HALF_L:
            circles.append(
                (
                    float(x + L),
                    float(y),
                    "wrapped",
                )
            )

        # Disk crosses the RIGHT edge:
        # its continuation re-enters from the LEFT edge.
        if x + DISK_RADIUS > HALF_L:
            circles.append(
                (
                    float(x - L),
                    float(y),
                    "wrapped",
                )
            )

    return circles


def geometry_stats(
    positions: np.ndarray,
) -> None:

    print()
    print("=" * 78)
    print("DISORDERED GEOMETRY CHECK")
    print("=" * 78)

    print(
        f"Number of supplied disk centres = "
        f"{len(positions)}"
    )

    print(
        f"Cell size                       = "
        f"{L:.3f} x {L:.3f} um"
    )

    print(
        f"Disk diameter                   = "
        f"{DISK_DIAMETER:.3f} um"
    )

    print(
        f"Disk radius                     = "
        f"{DISK_RADIUS:.3f} um"
    )

    print()
    print(
        f"x-centre range                  = "
        f"[{positions[:, 0].min():.6f}, "
        f"{positions[:, 0].max():.6f}] um"
    )

    print(
        f"y-centre range                  = "
        f"[{positions[:, 1].min():.6f}, "
        f"{positions[:, 1].max():.6f}] um"
    )

    # Count disks whose full circular area crosses a cell boundary.
    crosses_left = np.sum(
        positions[:, 0] - DISK_RADIUS < -HALF_L
    )

    crosses_right = np.sum(
        positions[:, 0] + DISK_RADIUS > HALF_L
    )

    crosses_bottom = np.sum(
        positions[:, 1] - DISK_RADIUS < -HALF_L
    )

    crosses_top = np.sum(
        positions[:, 1] + DISK_RADIUS > HALF_L
    )

    print()
    print(
        f"Disks crossing left x edge      = "
        f"{crosses_left}"
    )

    print(
        f"Disks crossing right x edge     = "
        f"{crosses_right}"
    )

    print(
        f"Disks crossing bottom y edge    = "
        f"{crosses_bottom}"
    )

    print(
        f"Disks crossing top y edge       = "
        f"{crosses_top}"
    )

    # Minimum centre-centre spacing.
    minimum_distance = np.inf
    minimum_pair = None

    for i in range(
        len(positions)
    ):

        for j in range(
            i + 1,
            len(positions),
        ):

            dx = (
                positions[i, 0]
                - positions[j, 0]
            )

            dy = (
                positions[i, 1]
                - positions[j, 1]
            )

            distance = np.hypot(
                dx,
                dy,
            )

            if distance < minimum_distance:

                minimum_distance = distance
                minimum_pair = (
                    i,
                    j,
                )

    print()
    print(
        f"Minimum centre spacing          = "
        f"{minimum_distance:.6f} um"
    )

    print(
        f"Disk diameter                   = "
        f"{DISK_DIAMETER:.6f} um"
    )

    if minimum_distance > DISK_DIAMETER:

        print(
            "Ordinary-cell overlap check     = PASS"
        )

    else:

        print(
            "Ordinary-cell overlap check     = CHECK"
        )

    print()
    print(
        "NOTE: x-edge crossings are expected and will be wrapped because "
        "the x direction is periodic."
    )

    if (
        crosses_bottom > 0
        or crosses_top > 0
    ):
        print(
            "NOTE: some full disks extend beyond y = +/-3.5 um. "
            "The supplied centre coordinates are being preserved exactly; "
            "we are not clipping or moving them yet."
        )

    print("=" * 78)


def plot_raw_geometry(
    positions: np.ndarray,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # 7 x 7 supplied region.
    cell = Rectangle(
        (-HALF_L, -HALF_L),
        L,
        L,
        fill=False,
        linewidth=2.0,
    )

    ax.add_patch(
        cell
    )

    for x, y in positions:

        ax.add_patch(
            Circle(
                (x, y),
                DISK_RADIUS,
                fill=False,
                linewidth=0.8,
            )
        )

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=5,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    margin = 0.35

    ax.set_xlim(
        -HALF_L - margin,
        HALF_L + margin,
    )

    ax.set_ylim(
        -HALF_L - margin,
        HALF_L + margin,
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Supplied disordered-disk geometry"
    )

    ax.grid(
        alpha=0.15
    )

    plt.tight_layout()

    path = (
        OUT
        / "00_disordered_geometry_raw.png"
    )

    plt.savefig(
        path,
        dpi=220,
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def plot_periodic_cell(
    positions: np.ndarray,
) -> None:

    geometry = periodic_x_geometry(
        positions
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    cell = Rectangle(
        (-HALF_L, -HALF_L),
        L,
        L,
        fill=False,
        linewidth=2.0,
    )

    ax.add_patch(
        cell
    )

    # Original + only the wrapped copies required at x boundaries.
    for x, y, kind in geometry:

        # Wrapped circles only contribute where they re-enter the cell.
        circle = Circle(
            (x, y),
            DISK_RADIUS,
            fill=False,
            linewidth=0.9
            if kind == "original"
            else 1.4,
            linestyle="-"
            if kind == "original"
            else "--",
        )

        ax.add_patch(
            circle
        )

    ax.axvline(
        -HALF_L,
        linestyle=":",
        linewidth=1.0,
    )

    ax.axvline(
        HALF_L,
        linestyle=":",
        linewidth=1.0,
    )

    ax.axhline(
        -HALF_L,
        linestyle=":",
        linewidth=1.0,
    )

    ax.axhline(
        HALF_L,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlim(
        -HALF_L,
        HALF_L,
    )

    # A little y margin is intentional so we can see any disks that
    # extend beyond the nominal 7 um disorder region.
    ax.set_ylim(
        -HALF_L - 0.25,
        HALF_L + 0.25,
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Primary cell with x-periodic disk wrapping"
    )

    ax.grid(
        alpha=0.15
    )

    plt.tight_layout()

    path = (
        OUT
        / "01_disordered_geometry_periodic_x.png"
    )

    plt.savefig(
        path,
        dpi=220,
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def plot_three_periods(
    positions: np.ndarray,
) -> None:

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    for shift in (
        -L,
        0.0,
        L,
    ):

        for x, y in positions:

            ax.add_patch(
                Circle(
                    (
                        x + shift,
                        y,
                    ),
                    DISK_RADIUS,
                    fill=False,
                    linewidth=0.55,
                )
            )

    for boundary in (
        -1.5 * L,
        -0.5 * L,
        0.5 * L,
        1.5 * L,
    ):

        ax.axvline(
            boundary,
            linestyle=":",
            linewidth=1.0,
        )

    ax.axhline(
        -HALF_L,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        HALF_L,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlim(
        -1.5 * L,
        1.5 * L,
    )

    ax.set_ylim(
        -HALF_L - 0.35,
        HALF_L + 0.35,
    )

    ax.set_xlabel(
        "x [um]"
    )

    ax.set_ylabel(
        "y [um]"
    )

    ax.set_title(
        "Three repeated x-periodic cells"
    )

    ax.grid(
        alpha=0.12
    )

    plt.tight_layout()

    path = (
        OUT
        / "02_disordered_geometry_three_periods.png"
    )

    plt.savefig(
        path,
        dpi=220,
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def main() -> None:

    positions = load_positions()

    geometry_stats(
        positions
    )

    plot_raw_geometry(
        positions
    )

    plot_periodic_cell(
        positions
    )

    plot_three_periods(
        positions
    )

    print()
    print(
        "Geometry check complete."
    )


if __name__ == "__main__":
    main()
