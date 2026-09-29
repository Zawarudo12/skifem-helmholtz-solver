from __future__ import annotations

from pathlib import Path

import numpy as np

POSITIONS_FILE = Path(
    "geometry_data/Disordered/disks_positions_diameter=0.26_L=7.csv"
)

MESH_DIR = Path(
    "meshes_disordered"
)

MESH_DIR.mkdir(
    exist_ok=True
)

MESH_FILE = (
    MESH_DIR
    / "disordered_196disks_buffer2_pml3_materialfine.msh"
)

L = 7.0
XMIN = -L / 2.0
XMAX = +L / 2.0

Y_SCAT_MIN = -3.5
Y_SCAT_MAX = +3.5

AIR_BUFFER = 2.00
PML_THICKNESS = 3.00

Y_AIR_MIN = (
    Y_SCAT_MIN
    - AIR_BUFFER
)

Y_AIR_MAX = (
    Y_SCAT_MAX
    + AIR_BUFFER
)

YMIN = (
    Y_AIR_MIN
    - PML_THICKNESS
)

YMAX = (
    Y_AIR_MAX
    + PML_THICKNESS
)

DISK_DIAMETER = 0.26
DISK_RADIUS = (
    DISK_DIAMETER
    / 2.0
)

H_DISK = 0.020

H_SCATTER = 0.060

H_AIR = 0.100

H_PML = 0.140

REFINE_DISTANCE = 0.16

def load_positions() -> np.ndarray:

    if not POSITIONS_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {POSITIONS_FILE}"
        )

    positions = np.loadtxt(
        POSITIONS_FILE
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 2
    ):

        raise ValueError(
            "Expected two columns: x y"
        )

    return positions

def periodic_disk_centres(
    positions: np.ndarray,
):
    """
    Original disk centres plus ONLY the periodic copies needed
    when a disk crosses the left/right x boundary.

    The copies make the material geometry itself periodic.
    """

    centres = []

    for i, (
        x,
        y,
    ) in enumerate(
        positions
    ):

        centres.append(
            (
                i,
                float(x),
                float(y),
                "original",
            )
        )

        if (
            x
            - DISK_RADIUS
            < XMIN
        ):

            centres.append(
                (
                    i,
                    float(
                        x
                        + L
                    ),
                    float(y),
                    "wrapped",
                )
            )

        if (
            x
            + DISK_RADIUS
            > XMAX
        ):

            centres.append(
                (
                    i,
                    float(
                        x
                        - L
                    ),
                    float(y),
                    "wrapped",
                )
            )

    return centres

def curve_bbox(
    gmsh,
    tag: int,
):
    return gmsh.model.getBoundingBox(
        1,
        tag,
    )

def identify_vertical_boundary_curves(
    gmsh,
    x_target: float,
    tol: float = 5e-6,
):
    """
    Return all curve tags lying on x = x_target,
    sorted from bottom to top.

    Boolean fragmentation splits the side boundary wherever a
    disk/interface intersects it.  Sorting allows left/right
    segments to be paired periodically.
    """

    curves = []

    for _, tag in gmsh.model.getEntities(
        dim=1
    ):

        xmin, ymin, _, xmax, ymax, _ = (
            curve_bbox(
                gmsh,
                tag,
            )
        )

        x_mid = 0.5 * (xmin + xmax)
        x_width = xmax - xmin

        if (
            abs(x_mid - x_target) < tol
            and x_width < 2.0 * tol
        ):

            curves.append(
                (
                    0.5 * (ymin + ymax),
                    ymax - ymin,
                    tag,
                )
            )

    curves.sort(
        key=lambda row: (
            row[0],
            row[1],
        )
    )

    return curves

def identify_horizontal_boundary_curves(
    gmsh,
    y_target: float,
    tol: float = 5e-6,
):
    curves = []

    for _, tag in gmsh.model.getEntities(
        dim=1
    ):

        xmin, ymin, _, xmax, ymax, _ = (
            curve_bbox(
                gmsh,
                tag,
            )
        )

        y_mid = 0.5 * (ymin + ymax)
        y_height = ymax - ymin

        if (
            abs(y_mid - y_target) < tol
            and y_height < 2.0 * tol
        ):

            curves.append(
                (
                    0.5 * (xmin + xmax),
                    xmax - xmin,
                    tag,
                )
            )

    curves.sort(
        key=lambda row: (
            row[0],
            row[1],
        )
    )

    return curves

def add_named_physical_group(
    gmsh,
    dim: int,
    tags,
    name: str,
):
    tags = sorted(
        set(
            int(tag)
            for tag in tags
        )
    )

    if not tags:

        raise RuntimeError(
            f"No entities found for physical group '{name}'."
        )

    physical_tag = (
        gmsh.model.addPhysicalGroup(
            dim,
            tags,
        )
    )

    gmsh.model.setPhysicalName(
        dim,
        physical_tag,
        name,
    )

    return physical_tag

def build_mesh() -> None:

    import gmsh

    positions = load_positions()

    centres = periodic_disk_centres(
        positions
    )

    print()
    print("=" * 84)
    print("BUILDING DISORDERED MESH â€” AIR 2.0 / PML 3.0 / MATERIAL-FINE")
    print("=" * 84)

    print(
        f"Supplied disks       = "
        f"{len(positions)}"
    )

    print(
        f"Periodic disk copies = "
        f"{len(centres) - len(positions)}"
    )

    print(
        f"Domain x             = "
        f"[{XMIN:.3f}, {XMAX:.3f}] um"
    )

    print(
        f"Disorder centre box  = "
        f"y [{Y_SCAT_MIN:.3f}, {Y_SCAT_MAX:.3f}] um"
    )

    print(
        f"Air-buffer limits    = "
        f"y [{Y_AIR_MIN:.3f}, {Y_AIR_MAX:.3f}] um"
    )

    print(
        f"Total PML domain     = "
        f"y [{YMIN:.3f}, {YMAX:.3f}] um"
    )

    print(
        "Material-aware mesh  = "
        f"h_disk={H_DISK:.3f}, "
        f"h_scatter={H_SCATTER:.3f}, "
        f"h_air={H_AIR:.3f}, "
        f"h_pml={H_PML:.3f} um"
    )

    gmsh.initialize()

    try:

        gmsh.option.setNumber(
            "General.Terminal",
            1,
        )

        gmsh.model.add(
            "disordered_196disks"
        )

        occ = gmsh.model.occ

        region_specs = [
            (
                "bottom_pml",
                YMIN,
                PML_THICKNESS,
            ),
            (
                "bottom_air",
                Y_AIR_MIN,
                AIR_BUFFER,
            ),
            (
                "scatter_background",
                Y_SCAT_MIN,
                L,
            ),
            (
                "top_air",
                Y_SCAT_MAX,
                AIR_BUFFER,
            ),
            (
                "top_pml",
                Y_AIR_MAX,
                PML_THICKNESS,
            ),
        ]

        region_entities = []

        for (
            name,
            y0,
            height,
        ) in region_specs:

            tag = occ.addRectangle(
                XMIN,
                y0,
                0.0,
                L,
                height,
            )

            region_entities.append(
                (
                    name,
                    (
                        2,
                        tag,
                    ),
                )
            )

        disk_entities = []

        for (
            original_index,
            x,
            y,
            kind,
        ) in centres:

            tag = occ.addDisk(
                x,
                y,
                0.0,
                DISK_RADIUS,
                DISK_RADIUS,
            )

            disk_entities.append(
                (
                    original_index,
                    kind,
                    (
                        2,
                        tag,
                    ),
                )
            )

        occ.synchronize()

        region_dimtags = [
            entity
            for _, entity
            in region_entities
        ]

        disk_dimtags = [
            entity
            for _, _, entity
            in disk_entities
        ]

        (
            out_dimtags,
            out_map,
        ) = occ.fragment(
            region_dimtags,
            disk_dimtags,
            removeObject=True,
            removeTool=True,
        )

        occ.synchronize()

        n_regions = len(
            region_entities
        )

        region_surface_sets = {}

        for i, (
            name,
            _,
        ) in enumerate(
            region_entities
        ):

            region_surface_sets[
                name
            ] = {
                tag
                for dim, tag
                in out_map[i]
                if dim == 2
            }

        disk_surface_tags = set()

        for mapping in out_map[
            n_regions:
        ]:

            for dim, tag in mapping:

                if dim == 2:

                    disk_surface_tags.add(
                        tag
                    )

        all_region_surface_tags = set()

        for tags in (
            region_surface_sets.values()
        ):

            all_region_surface_tags.update(
                tags
            )

        disk_surface_tags &= (
            all_region_surface_tags
        )

        background_surface_sets = {}

        for (
            name,
            tags,
        ) in (
            region_surface_sets.items()
        ):

            background_surface_sets[
                name
            ] = (
                tags
                - disk_surface_tags
            )

        for (
            name,
            tags,
        ) in (
            background_surface_sets.items()
        ):

            add_named_physical_group(
                gmsh,
                2,
                tags,
                name,
            )

        add_named_physical_group(
            gmsh,
            2,
            disk_surface_tags,
            "disks",
        )

        left = (
            identify_vertical_boundary_curves(
                gmsh,
                XMIN,
            )
        )

        right = (
            identify_vertical_boundary_curves(
                gmsh,
                XMAX,
            )
        )

        bottom = (
            identify_horizontal_boundary_curves(
                gmsh,
                YMIN,
            )
        )

        top = (
            identify_horizontal_boundary_curves(
                gmsh,
                YMAX,
            )
        )

        print()
        print(
            f"Left boundary segments  = "
            f"{len(left)}"
        )

        print(
            f"Right boundary segments = "
            f"{len(right)}"
        )

        if not left or not right:
            print()
            print(
                "Boundary-detection diagnostic:"
            )

            candidate_rows = []

            for _, curve_tag in gmsh.model.getEntities(dim=1):
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(
                    1,
                    curve_tag,
                )

                xmid = 0.5 * (xmin + xmax)
                width = xmax - xmin

                distance_to_side = min(
                    abs(xmid - XMIN),
                    abs(xmid - XMAX),
                )

                candidate_rows.append(
                    (
                        distance_to_side,
                        curve_tag,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        width,
                    )
                )

            candidate_rows.sort(
                key=lambda row: row[0]
            )

            for row in candidate_rows[:12]:
                (
                    distance_to_side,
                    curve_tag,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    width,
                ) = row

                print(
                    f"  curve {curve_tag:4d}: "
                    f"x=[{xmin:.9f}, {xmax:.9f}]  "
                    f"y=[{ymin:.6f}, {ymax:.6f}]  "
                    f"xwidth={width:.3e}  "
                    f"side-distance={distance_to_side:.3e}"
                )

        if (
            len(left)
            != len(right)
        ):

            raise RuntimeError(
                "Left/right boundary segmentation does not match. "
                "Periodic geometry construction needs inspection."
            )

        for (
            left_row,
            right_row,
        ) in zip(
            left,
            right,
        ):

            y_left, len_left, _ = (
                left_row
            )

            y_right, len_right, _ = (
                right_row
            )

            if (
                abs(
                    y_left
                    - y_right
                )
                > 5e-6
                or
                abs(
                    len_left
                    - len_right
                )
                > 5e-6
            ):

                raise RuntimeError(
                    "Left/right periodic boundary segments "
                    "are not geometrically matched."
                )

        left_tags = [
            row[2]
            for row in left
        ]

        right_tags = [
            row[2]
            for row in right
        ]

        bottom_tags = [
            row[2]
            for row in bottom
        ]

        top_tags = [
            row[2]
            for row in top
        ]

        add_named_physical_group(
            gmsh,
            1,
            left_tags,
            "left",
        )

        add_named_physical_group(
            gmsh,
            1,
            right_tags,
            "right",
        )

        add_named_physical_group(
            gmsh,
            1,
            bottom_tags,
            "bottom",
        )

        add_named_physical_group(
            gmsh,
            1,
            top_tags,
            "top",
        )

        affine_left_to_right = [
            1.0, 0.0, 0.0, L,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

        gmsh.model.mesh.setPeriodic(
            1,
            right_tags,
            left_tags,
            affine_left_to_right,
        )

        disk_curves = set()

        for disk_surface in (
            disk_surface_tags
        ):

            boundary = (
                gmsh.model.getBoundary(
                    [
                        (
                            2,
                            disk_surface,
                        )
                    ],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            )

            for (
                dim,
                curve_tag,
            ) in boundary:

                if dim == 1:

                    disk_curves.add(
                        curve_tag
                    )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            H_DISK,
        )

        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            H_PML,
        )

        mesh_fields = []

        if disk_curves:

            distance_field = (
                gmsh.model.mesh.field.add(
                    "Distance"
                )
            )

            gmsh.model.mesh.field.setNumbers(
                distance_field,
                "CurvesList",
                sorted(
                    disk_curves
                ),
            )

            gmsh.model.mesh.field.setNumber(
                distance_field,
                "Sampling",
                150,
            )

            interface_field = (
                gmsh.model.mesh.field.add(
                    "Threshold"
                )
            )

            gmsh.model.mesh.field.setNumber(
                interface_field,
                "InField",
                distance_field,
            )

            gmsh.model.mesh.field.setNumber(
                interface_field,
                "SizeMin",
                H_DISK,
            )

            gmsh.model.mesh.field.setNumber(
                interface_field,
                "SizeMax",
                H_PML,
            )

            gmsh.model.mesh.field.setNumber(
                interface_field,
                "DistMin",
                0.0,
            )

            gmsh.model.mesh.field.setNumber(
                interface_field,
                "DistMax",
                REFINE_DISTANCE,
            )

            mesh_fields.append(
                interface_field
            )

        scatter_field = (
            gmsh.model.mesh.field.add(
                "Box"
            )
        )

        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "VIn",
            H_SCATTER,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "VOut",
            H_PML,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "XMin",
            XMIN - 1e-6,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "XMax",
            XMAX + 1e-6,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "YMin",
            Y_SCAT_MIN - 1e-6,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "YMax",
            Y_SCAT_MAX + 1e-6,
        )
        gmsh.model.mesh.field.setNumber(
            scatter_field,
            "Thickness",
            0.10,
        )

        mesh_fields.append(
            scatter_field
        )

        for y0, y1 in (
            (Y_AIR_MIN, Y_SCAT_MIN),
            (Y_SCAT_MAX, Y_AIR_MAX),
        ):
            air_field = (
                gmsh.model.mesh.field.add(
                    "Box"
                )
            )

            gmsh.model.mesh.field.setNumber(
                air_field,
                "VIn",
                H_AIR,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "VOut",
                H_PML,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "XMin",
                XMIN - 1e-6,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "XMax",
                XMAX + 1e-6,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "YMin",
                y0 - 1e-6,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "YMax",
                y1 + 1e-6,
            )
            gmsh.model.mesh.field.setNumber(
                air_field,
                "Thickness",
                0.10,
            )

            mesh_fields.append(
                air_field
            )

        min_field = (
            gmsh.model.mesh.field.add(
                "Min"
            )
        )

        gmsh.model.mesh.field.setNumbers(
            min_field,
            "FieldsList",
            mesh_fields,
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            min_field
        )

        gmsh.option.setNumber(
            "Mesh.Algorithm",
            6,
        )

        gmsh.option.setNumber(
            "Mesh.ElementOrder",
            1,
        )

        gmsh.option.setNumber(
            "Mesh.MshFileVersion",
            4.1,
        )

        print()
        print(
            "Generating 2D body-fitted mesh..."
        )

        gmsh.model.mesh.generate(
            2
        )

        (
            node_tags,
            _,
            _,
        ) = gmsh.model.mesh.getNodes()

        element_types, element_tags, _ = (
            gmsh.model.mesh.getElements(
                dim=2
            )
        )

        n_elements = sum(
            len(tags)
            for tags in element_tags
        )

        print()
        print(
            f"Mesh nodes       = "
            f"{len(node_tags)}"
        )

        print(
            f"2D elements      = "
            f"{n_elements}"
        )

        print(
            f"Disk surfaces    = "
            f"{len(disk_surface_tags)}"
        )

        print(
            f"Disk interfaces  = "
            f"{len(disk_curves)} curves"
        )

        gmsh.write(
            str(
                MESH_FILE
            )
        )

        print()
        print(
            f"Saved mesh: "
            f"{MESH_FILE}"
        )

        print()
        print(
            "MESH BUILD COMPLETE"
        )

    finally:

        gmsh.finalize()

if __name__ == "__main__":

    build_mesh()



