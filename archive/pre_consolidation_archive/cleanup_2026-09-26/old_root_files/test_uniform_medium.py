from photonics_fem.config import SlabConfig
from photonics_fem.solver import solve_slab
from photonics_fem.postprocess import (
    fit_plane_waves,
    relative_l2_profile_error,
)
from photonics_fem.mesh import max_edge_length


def main() -> None:
    # Make the entire computational domain effectively air.
    cfg = SlabConfig(
        wavelength=0.633,
        n_inc=1.0 + 0.0j,
        n_slab=1.0 + 0.0j,
        n_out=1.0 + 0.0j,
        slab_thickness=0.50,
        air_top=0.50,
        air_bottom=0.50,
        width=0.50,
    )

    # Use P2 and about 18 elements per wavelength.
    ppw = 18
    h_target = cfg.lambda_min_material / ppw

    sol = solve_slab(
        cfg,
        h_target=h_target,
        order=2,
    )

    rt = fit_plane_waves(sol)
    l2_error = relative_l2_profile_error(sol)

    print("=== UNIFORM MEDIUM TEST ===")
    print(f"DOFs                  = {sol.basis.N}")
    print(f"hmax                  = {max_edge_length(sol.mesh):.6f} um")
    print()
    print(f"Relative L2 error     = {l2_error:.6e}")
    print()
    print(f"Incident amplitude    = {rt['a_inc']}")
    print(f"Reflected amplitude   = {rt['a_refl']}")
    print(f"Transmitted amplitude = {rt['a_trans']}")
    print()
    print(f"R                     = {rt['R']:.12e}")
    print(f"T                     = {rt['T']:.12f}")
    print(f"R + T                 = {rt['R_plus_T']:.12f}")
    print()
    print(
        f"Bottom incoming ratio = "
        f"{rt['bottom_incoming_ratio']:.6e}"
    )

    # Simple automatic validation
    passed = (
        rt["R"] < 1e-6
        and abs(rt["T"] - 1.0) < 1e-5
        and abs(rt["R_plus_T"] - 1.0) < 1e-5
    )

    print()
    print("RESULT:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()