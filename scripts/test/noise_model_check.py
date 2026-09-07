"""Sanity check for `instruments.camera.noise` against the published PTC docs.

No hardware required -- this only imports `instruments.camera.noise`, which
has no vendor SDK dependency. Regenerates the "Quick-reference table" and
worked examples from `docs/photon_transfer.md` and
`docs/photon_transfer_pco.md` from the `NoiseModel` formulas and diffs them
against the published numbers, then checks the PRNU-floor behavior described
in the noise module's docstring, and the `sigma_from_frames` /
`error_from_frames` data-taking path (shape handling, dark subtraction,
quadrature combination across the frame axis).

Run: `python scripts/test/noise_model_check.py`
"""

from __future__ import annotations

import math

import numpy as np

from instruments.camera.noise import get_noise_model

# ---------------------------------------------------------------------- #
# Published quick-reference tables (docs/photon_transfer.md:186-193,
# docs/photon_transfer_pco.md:285-292): S, eps_int (1 frame, %),
# eps_T (1 frame, %), N for eps_T=0.5%, N for eps_T=0.1%.
# ---------------------------------------------------------------------- #

THORCAM_TABLE = [
    (100, 6.2, 6.1, 150, 3760),
    (250, 3.9, 3.9, 60, 1500),
    (500, 2.8, 2.7, 30, 750),
    (1000, 2.0, 1.9, 15, 380),
    (2000, 1.5, 1.4, 8, 190),
    (3600, 1.2, 1.0, 4, 104),
]

PCO_TABLE = [
    (1000, 5.74, 5.70, 130, 3249),
    (5000, 2.59, 2.55, 26, 650),
    (10000, 1.86, 1.80, 13, 325),
    (20000, 1.35, 1.27, 6, 162),
    (40000, 1.00, 0.90, 3, 81),
    (60000, 0.86, 0.74, 2, 54),
]

# Worked examples (docs/photon_transfer.md:195-204, photon_transfer_pco.md:294-303).
# The ThorCam example was computed with the full two-term formula
# (reference_signal_dn given explicitly, S_ref ~= 3600); the pco example
# turns out to have been computed with the single-term, reference-negligible
# approximation instead (reference_signal_dn omitted) -- verified by hand
# against both docs' own published numbers. That's a pre-existing
# inconsistency between the two docs' worked-example methodology, not a
# defect in NoiseModel; each row here uses whichever one the doc's own
# numbers actually match.
# (vendor, S_s, reference_signal_dn_or_None, N_for_0.5pct, N_for_0.1pct)
WORKED_EXAMPLES = [
    ("thorlabs", 3240, 3600, 9, 220),
    ("thorlabs", 180, 3600, 88, 2190),
    ("pco", 54000, None, 2, 60),
    ("pco", 3000, None, 43, 1083),
]

TOLERANCE_PCT = 15.0  # published tables are themselves rounded to ~2-3 sig figs
TOLERANCE_ABS_FRAMES = 2.0  # small N rounds coarsely; treat within +/-2 frames as a match too


def _within_tolerance(computed: float, published: float) -> bool:
    pct = 100 * abs(computed - published) / published
    return pct <= TOLERANCE_PCT or abs(computed - published) <= TOLERANCE_ABS_FRAMES


def check_table(vendor: str, table) -> bool:
    model = get_noise_model(vendor)
    print(f"\n--- {vendor} quick-reference table ---")
    print(f"{'S':>8} {'eps_int':>18} {'eps_T':>18} {'N(0.5%)':>16} {'N(0.1%)':>16}")
    ok = True
    for S, eps_int_pub, eps_T_pub, n05_pub, n01_pub in table:
        eps_int = model.relative_error_intensity(S)
        eps_T = model.relative_error_ratio(S, 1, 1e9, 1)  # reference ~negligible, matches doc's approx
        n05 = model.frames_for_error(S, 0.5)
        n01 = model.frames_for_error(S, 0.1)

        def fmt(computed, published):
            delta = 100 * abs(computed - published) / published
            match = _within_tolerance(computed, published)
            flag = "" if match else "  <-- MISMATCH"
            return f"{computed:.3g} (pub {published}, {delta:.1f}%{flag})", match

        s_int, ok_int = fmt(eps_int, eps_int_pub)
        s_T, ok_T = fmt(eps_T, eps_T_pub)
        s_n05, ok_n05 = fmt(n05, n05_pub)
        s_n01, ok_n01 = fmt(n01, n01_pub)
        ok = ok and ok_int and ok_T and ok_n05 and ok_n01
        print(f"{S:>8} {s_int:>18} {s_T:>18} {s_n05:>16} {s_n01:>16}")
    return ok


def check_worked_examples() -> bool:
    print("\n--- worked examples ---")
    ok = True
    for vendor, S_s, ref, n05_pub, n01_pub in WORKED_EXAMPLES:
        model = get_noise_model(vendor)
        kwargs = {} if ref is None else {"reference_signal_dn": ref}
        n05 = model.frames_for_error(S_s, 0.5, **kwargs)
        n01 = model.frames_for_error(S_s, 0.1, **kwargs)
        this_ok = _within_tolerance(n05, n05_pub) and _within_tolerance(n01, n01_pub)
        ok = ok and this_ok
        flag = "" if this_ok else "  <-- MISMATCH"
        print(f"{vendor:>10} S_s={S_s:>6} ref={ref}: N(0.5%)={n05} (pub {n05_pub}), "
              f"N(0.1%)={n01} (pub {n01_pub}){flag}")
    return ok


def check_prnu_floor() -> bool:
    print("\n--- PRNU averaging floor ---")
    ok = True
    for vendor in ("thorlabs", "pco"):
        model = get_noise_model(vendor)
        floor_pct = 100 * model.prnu_factor
        below = model.frames_for_error(1000, floor_pct - 0.05, kind="intensity")
        above = model.frames_for_error(1000, floor_pct + 0.05, kind="intensity")
        ratio_still_finite = model.frames_for_error(1000, floor_pct - 0.05, kind="ratio")
        this_ok = (below == math.inf) and math.isfinite(above) and math.isfinite(ratio_still_finite)
        ok = ok and this_ok
        flag = "" if this_ok else "  <-- MISMATCH"
        print(f"{vendor:>10}: floor={floor_pct:.3f}%, "
              f"intensity below floor -> {below}, above floor -> {above:.1f} frames, "
              f"ratio below floor -> {ratio_still_finite} frames{flag}")

        # sigma_total should converge to the FPN floor as n_frames -> inf
        S = 3000.0
        sigma_inf = model.sigma_total_dn(S, n_frames=10_000_000)
        expected = model.prnu_factor * S
        conv_ok = abs(sigma_inf - expected) / expected < 1e-3
        ok = ok and conv_ok
        flag = "" if conv_ok else "  <-- MISMATCH"
        print(f"{'':>10}  sigma_total(n=1e7) = {sigma_inf:.4f} DN, "
              f"P_N*S = {expected:.4f} DN{flag}")
    return ok


def check_array_shapes() -> bool:
    print("\n--- array shapes ---")
    model = get_noise_model("thorlabs")
    S = np.full((70, 70), 500.0)
    ok = True
    for name, fn in [
        ("sigma_shot_dn", lambda: model.sigma_shot_dn(S)),
        ("sigma_fpn_dn", lambda: model.sigma_fpn_dn(S)),
        ("sigma_temporal_dn", lambda: model.sigma_temporal_dn(S, n_frames=10)),
        ("sigma_total_dn", lambda: model.sigma_total_dn(S, n_frames=10)),
        ("snr", lambda: model.snr(S, n_frames=10)),
        ("relative_error_intensity", lambda: model.relative_error_intensity(S)),
        ("to_electrons", lambda: model.to_electrons(S)),
    ]:
        out = fn()
        this_ok = out.shape == S.shape
        ok = ok and this_ok
        flag = "" if this_ok else "  <-- MISMATCH"
        print(f"{name:>26}: {out.shape}{flag}")
    return ok


def check_frames_equivalence() -> bool:
    """For a stack of N identical frames at signal S (stationary case),
    sigma_from_frames must equal sqrt(N) * sigma_total_dn(S, n_frames=1) --
    the noise on a sum of N iid measurements -- and error_from_frames must
    equal 100 * that / (N*S)."""
    print("\n--- sigma_from_frames: quadrature-combine, stationary case ---")
    model = get_noise_model("thorlabs")
    N, S = 20, 500.0
    stack = np.full((N, 5, 5), S)

    sigma = model.sigma_from_frames(stack, frame_axis=0)
    expected_sigma = np.sqrt(N) * model.sigma_total_dn(S, n_frames=1)
    ok = np.allclose(sigma, expected_sigma)
    flag = "" if ok else "  <-- MISMATCH"
    print(f"sigma_from_frames = {sigma.flat[0]:.4f}, "
          f"sqrt(N)*sigma_total_dn(S,1) = {expected_sigma:.4f}{flag}")

    error = model.error_from_frames(stack, frame_axis=0)
    expected_error = 100.0 * expected_sigma / (N * S)
    err_ok = np.allclose(error, expected_error)
    ok = ok and err_ok
    flag = "" if err_ok else "  <-- MISMATCH"
    print(f"error_from_frames = {error.flat[0]:.4f}%, "
          f"expected = {expected_error:.4f}%{flag}")
    return ok


def check_frames_varying_signal() -> bool:
    """For a stack where each frame carries a different per-pixel signal
    (e.g. interference fringes stepping frame to frame), sigma_from_frames
    must match a manual per-frame quadrature sum -- not any signal average."""
    print("\n--- sigma_from_frames: per-frame-varying signal ---")
    model = get_noise_model("thorlabs")
    rng = np.random.default_rng(0)
    N, H, W = 10, 4, 4
    stack = rng.uniform(50, 3000, size=(N, H, W))  # different signal every frame

    sigma = model.sigma_from_frames(stack, frame_axis=0)
    manual = np.sqrt(sum(model.sigma_total_dn(stack[k], n_frames=1) ** 2 for k in range(N)))
    ok = np.allclose(sigma, manual)
    flag = "" if ok else "  <-- MISMATCH"
    print(f"matches manual per-frame quadrature sum: {ok}{flag}")

    error = model.error_from_frames(stack, frame_axis=0)
    manual_error = 100.0 * manual / stack.sum(axis=0)
    err_ok = np.allclose(error, manual_error)
    ok = ok and err_ok
    flag = "" if err_ok else "  <-- MISMATCH"
    print(f"error matches 100*sigma/sum(frames): {err_ok}{flag}")
    return ok


def check_frames_shapes() -> bool:
    """Shape handling: single image vs. stack, trailing C=1 squeeze, an
    arbitrary frame_axis position, and a multi-channel frame not misread
    as a stack."""
    print("\n--- sigma_from_frames: shape handling ---")
    model = get_noise_model("thorlabs")
    ok = True

    def check(name, arr, frame_axis, expected_shape):
        out = model.sigma_from_frames(arr, frame_axis=frame_axis)
        out = np.asarray(out)
        this_ok = out.shape == expected_shape
        flag = "" if this_ok else "  <-- MISMATCH"
        print(f"{name:>34}: shape={out.shape} (expected {expected_shape}){flag}")
        return this_ok

    ok &= check("(H,W)", np.full((5, 5), 200.0), None, (5, 5))
    ok &= check("(H,W,1) squeezed, single frame", np.full((5, 5, 1), 200.0), None, (5, 5))
    ok &= check("(n,H,W) frame_axis=0", np.full((10, 5, 5), 200.0), 0, (5, 5))
    ok &= check("(n,H,W,1) frame_axis=0", np.full((10, 5, 5, 1), 200.0), 0, (5, 5))
    ok &= check("(level,frame,H,W) frame_axis=1", np.full((38, 50, 236, 440), 200.0), 1,
                (38, 236, 440))
    ok &= check("(H,W,3) colour, single frame", np.full((5, 5, 3), 200.0), None, (5, 5, 3))
    return ok


def check_frames_dark_subtraction() -> bool:
    """sigma_from_frames(bright, dark=dark_stack, ...) must equal manually
    pre-subtracting the reduced master dark from every frame, and must
    accept the (level, frame, H, W) layout at frame_axis=1."""
    print("\n--- sigma_from_frames: dark subtraction ---")
    model = get_noise_model("thorlabs")
    rng = np.random.default_rng(2)
    ok = True

    bright = rng.uniform(50, 3000, size=(30, 5, 5))
    dark = rng.normal(50, 5, size=(30, 5, 5))
    from_frames = model.sigma_from_frames(bright, dark=dark, frame_axis=0)
    manual = np.sqrt(np.sum(
        model.sigma_total_dn(bright - dark.mean(axis=0), n_frames=1) ** 2, axis=0))
    this_ok = np.allclose(from_frames, manual)
    ok = ok and this_ok
    print(f"stack + dark, frame_axis=0, matches manual subtraction: {this_ok}"
          f"{'' if this_ok else '  <-- MISMATCH'}")

    sig4 = rng.uniform(50, 500, size=(5, 50, 20, 20))
    dark4 = rng.normal(50, 5, size=(5, 50, 20, 20))
    result = get_noise_model("pco").error_from_frames(sig4, dark=dark4, frame_axis=1)
    this_ok = result.shape == (5, 20, 20)
    ok = ok and this_ok
    print(f"(level,frame,H,W) layout, frame_axis=1, shape {result.shape}: {this_ok}"
          f"{'' if this_ok else '  <-- MISMATCH'}")
    return ok


def check_frames_finite() -> bool:
    """A dark-subtracted signal that goes negative on some pixels must
    still return finite numbers, not NaN."""
    print("\n--- sigma_from_frames: negative signal stays finite ---")
    model = get_noise_model("thorlabs")
    rng = np.random.default_rng(3)
    dim = rng.normal(50, 5, size=(30, 20, 20))
    dark = rng.normal(52, 5, size=(30, 20, 20))

    result = model.sigma_from_frames(dim, dark=dark, frame_axis=0)
    ok = bool(np.all(np.isfinite(result)))
    flag = "" if ok else "  <-- MISMATCH"
    print(f"all finite: {ok}{flag}")
    return ok


def main() -> int:
    results = [
        check_table("thorlabs", THORCAM_TABLE),
        check_table("pco", PCO_TABLE),
        check_worked_examples(),
        check_prnu_floor(),
        check_array_shapes(),
        check_frames_equivalence(),
        check_frames_varying_signal(),
        check_frames_shapes(),
        check_frames_dark_subtraction(),
        check_frames_finite(),
    ]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED -- see MISMATCH lines above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
