from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix

from skfem import Basis

@dataclass(frozen=True)
class BlochReduction:
    """Projection data for left/right Bloch-periodic DOF tying."""

    P: csr_matrix

    left_dofs: np.ndarray
    right_dofs: np.ndarray

    full_to_reduced: np.ndarray

    phase: complex

def build_bloch_projection(
    basis: Basis,
    period: float,
    kx: float,
    atol: float = 1e-10,
) -> BlochReduction:
    """Build P enforcing

        u(x + period, y)
            = exp(i kx period) u(x, y).

    The full solution is reconstructed using

        u_full = P @ u_reduced.
    """

    left = np.asarray(
        basis.get_dofs("left").flatten(),
        dtype=int,
    )

    right = np.asarray(
        basis.get_dofs("right").flatten(),
        dtype=int,
    )

    if left.size != right.size:
        raise ValueError(
            "Left/right boundary DOF counts do not match: "
            f"{left.size} != {right.size}"
        )

    y_left = basis.doflocs[1, left]
    y_right = basis.doflocs[1, right]

    left_order = np.argsort(y_left)
    right_order = np.argsort(y_right)

    left = left[left_order]
    right = right[right_order]

    y_left = basis.doflocs[1, left]
    y_right = basis.doflocs[1, right]

    max_pair_error = float(
        np.max(
            np.abs(y_left - y_right)
        )
    )

    if max_pair_error > atol:
        raise ValueError(
            "Periodic boundaries are not conforming. "
            f"Maximum paired y mismatch = {max_pair_error:.3e}"
        )

    phase = np.exp(
        1j * kx * period
    )

    n_full = basis.N

    is_right = np.zeros(
        n_full,
        dtype=bool,
    )

    is_right[right] = True

    masters = np.flatnonzero(
        ~is_right
    )

    n_reduced = masters.size

    master_to_reduced = np.full(
        n_full,
        -1,
        dtype=int,
    )

    master_to_reduced[masters] = np.arange(
        n_reduced,
        dtype=int,
    )

    full_to_reduced = master_to_reduced.copy()

    for l_dof, r_dof in zip(left, right):
        reduced_index = master_to_reduced[l_dof]

        if reduced_index < 0:
            raise RuntimeError(
                "Left periodic DOF was unexpectedly "
                "removed from the master set."
            )

        full_to_reduced[r_dof] = reduced_index

    if np.any(full_to_reduced < 0):
        raise RuntimeError(
            "Some full DOFs were not mapped to a "
            "reduced Bloch DOF."
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for full_dof in masters:
        rows.append(int(full_dof))
        cols.append(
            int(
                master_to_reduced[
                    full_dof
                ]
            )
        )
        data.append(1.0 + 0.0j)

    for l_dof, r_dof in zip(left, right):
        rows.append(int(r_dof))
        cols.append(
            int(
                master_to_reduced[
                    l_dof
                ]
            )
        )
        data.append(phase)

    P = coo_matrix(
        (
            np.asarray(
                data,
                dtype=np.complex128,
            ),
            (
                np.asarray(rows),
                np.asarray(cols),
            ),
        ),
        shape=(
            n_full,
            n_reduced,
        ),
    ).tocsr()

    return BlochReduction(
        P=P,
        left_dofs=left,
        right_dofs=right,
        full_to_reduced=full_to_reduced,
        phase=phase,
    )

def bloch_mismatch(
    u: np.ndarray,
    reduction: BlochReduction,
) -> float:
    """Relative mismatch in u_R = phase * u_L."""

    u_left = u[
        reduction.left_dofs
    ]

    u_right = u[
        reduction.right_dofs
    ]

    error = np.max(
        np.abs(
            u_right
            - reduction.phase * u_left
        )
    )

    scale = max(
        float(
            np.max(np.abs(u))
        ),
        1e-30,
    )

    return float(
        error / scale
    )
