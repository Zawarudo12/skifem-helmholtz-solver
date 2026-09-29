from photonics_fem.config import SlabConfig
from photonics_fem.solver import solve_slab
from photonics_fem.postprocess import (
    fit_plane_waves,
    relative_l2_profile_error,
)
from photonics_fem.analytics import slab_rt


def run_case(name: str, cfg: SlabConfig, ppw: int = 18) -> dict:
    """Solve one slab case and compare FEM against exact thin-film theory."""

    h_target = cfg.lambda_min_material / ppw

    sol = solve_slab(
        cfg,
        h_target=h_target,
        order=2,
    )

    fem = fit_plane_waves(sol)
    _, _, R_exact, T_exact = slab_rt(cfg)
    l2_error = relative_l2_profile_error(sol)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"DOFs               = {sol.basis.N}")
    print(f"Relative L2 error  = {l2_error:.6e}")
    print()

    print("FEM:")
    print(f"R                  = {fem['R']:.12f}")
    print(f"T                  = {fem['T']:.12f}")
    print(f"R + T              = {fem['R_plus_T']:.12f}")

    print()
    print("ANALYTIC:")
    print(f"R                  = {R_exact:.12f}")
    print(f"T                  = {T_exact:.12f}")
    print(f"R + T              = {R_exact + T_exact:.12f}")

    print()
    print("ERROR:")
    print(f"|R_FEM - R_exact|  = {abs(fem['R'] - R_exact):.6e}")
    print(f"|T_FEM - T_exact|  = {abs(fem['T'] - T_exact):.6e}")

    print()
    print(
        "bottom incoming    = "
        f"{fem['bottom_incoming_ratio']:.6e}"
    )

    return {
        "fem": fem,
        "R_exact": R_exact,
        "T_exact": T_exact,
        "l2": l2_error,
    }


def main() -> None:

    # ----------------------------------------------------------
    # TEST 1: Fabry-Perot resonance
    #
    # lambda_m = 2*n*d/m
    #
    # n = 1.7
    # d = 0.5 um
    # m = 2
    #
    # lambda = 0.85 um
    #
    # Expected:
    # R -> 0
    # T -> 1
    # ----------------------------------------------------------

    resonance_cfg = SlabConfig(
        wavelength=0.85,
        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.0j,
        n_out=1.0 + 0.0j,
        slab_thickness=0.50,
        air_top=0.50,
        air_bottom=0.50,
        width=0.50,
    )

    resonance = run_case(
        "TEST 1: FABRY-PEROT RESONANCE",
        resonance_cfg,
    )

    resonance_pass = (
        resonance["fem"]["R"] < 1e-5
        and abs(resonance["fem"]["T"] - 1.0) < 1e-4
        and resonance["l2"] < 1e-3
    )

    print()
    print(
        "RESONANCE RESULT:",
        "PASS" if resonance_pass else "FAIL"
    )

    # ----------------------------------------------------------
    # TEST 2: Lossy dielectric
    #
    # With exp(-i omega t):
    #
    # Im(n) > 0
    #
    # means attenuation / absorption.
    #
    # Therefore:
    #
    # R + T < 1
    #
    # ----------------------------------------------------------

    lossy_cfg = SlabConfig(
        wavelength=0.633,
        n_inc=1.0 + 0.0j,
        n_slab=1.7 + 0.05j,
        n_out=1.0 + 0.0j,
        slab_thickness=0.50,
        air_top=0.50,
        air_bottom=0.50,
        width=0.50,
    )

    lossy = run_case(
        "TEST 2: LOSSY DIELECTRIC",
        lossy_cfg,
    )

    absorption_fem = 1.0 - (
        lossy["fem"]["R"] + lossy["fem"]["T"]
    )

    absorption_exact = 1.0 - (
        lossy["R_exact"] + lossy["T_exact"]
    )

    print()
    print(f"FEM absorption A   = {absorption_fem:.12f}")
    print(f"Exact absorption A = {absorption_exact:.12f}")

    lossy_pass = (
        lossy["fem"]["R_plus_T"] < 1.0
        and abs(lossy["fem"]["R"] - lossy["R_exact"]) < 5e-4
        and abs(lossy["fem"]["T"] - lossy["T_exact"]) < 5e-4
        and lossy["l2"] < 1e-3
    )

    print()
    print(
        "LOSSY RESULT:",
        "PASS" if lossy_pass else "FAIL"
    )

    print()
    print("=" * 60)

    all_pass = resonance_pass and lossy_pass

    print(
        "OVERALL:",
        "ALL TESTS PASS" if all_pass else "SOME TESTS FAILED"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()