"""Photon transfer curve (difference method): measure K(g), e-/DN, per gain.

For a mean signal S (DN, raw/not dark-subtracted), shot-noise variance in DN is
S/K. Plotting temporal variance vs. mean signal is therefore a line of slope
1/K. The two-frame difference method isolates that temporal variance from
fixed-pattern noise: for a back-to-back pair (F1, F2) at identical
illumination, D = F1 - F2 cancels FPN/PRNU, and var(D)/2 is the single-frame
temporal variance.

A constant dark pedestal only shifts S horizontally, so it does not need to be
subtracted for K = 1/slope to come out right; the fit intercept then absorbs
the pedestal along with read noise.

Requires the operator to set up flat, defocused/diffuser illumination at one
fixed band before running (this script does not drive the monochromator).
"""

from datetime import datetime
from pathlib import Path

from instruments.config import load_equipment
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
from instruments.powermeter import PM400
import time
import contextlib
import numpy as np

_EQ = load_equipment()
SERIAL = _EQ.camera_serial
PM_SERIAL = _EQ.powermeter_serial
PM_WAVELENGTH_NM = 550.0   # set on PM400 to match the illumination band
BLACK_LEVEL = 0
GAIN_RANGE = [0, 100, 200]
ROI_FRACTION = 0.20        # central 20% x 20% sub-ROI (avoid vignetted edges)
N_LEVELS = 7               # signal levels per gain
FILL_TARGETS = np.linspace(0.10, 0.88, N_LEVELS)  # fraction of pixel_max_value
NUM_PAIRS = 5              # frame pairs per level
NUM_SETTLE_FRAMES = 2      # dropped after each exposure change
PROBE_EXPOSURE_US = 1000   # starting probe to find the DN<->exposure scale
PM_AVG_N = 10
MAX_EXPOSURE_US = 14_700_924  # CS126 hardware max
MIN_EXPOSURE_US = 28
SATURATION_FRACTION = 0.9  # points at/above this fraction of full well are excluded from the fit
BIT_DEPTH = np.uint16
OUT_BIT_DEPTH = np.float32


@contextlib.contextmanager
def _armed_camera(camera: ThorlabsCamera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()


def _central_roi(image: np.ndarray, frac: float) -> np.ndarray:
    """Return the central frac x frac sub-region of an (H, W, C) image."""
    h, w = image.shape[0], image.shape[1]
    cy, cx = h // 2, w // 2
    ry, rx = max(1, int(h * frac / 2)), max(1, int(w * frac / 2))
    return image[cy - ry: cy + ry, cx - rx: cx + rx]


def _probe_exposures(camera: ThorlabsCamera, pixel_max: float) -> np.ndarray:
    """Probe the DN<->exposure scale at the current gain, assuming DN is
    proportional to exposure, then return the N_LEVELS exposures (us) that
    should hit FILL_TARGETS fractions of full well."""
    probe_exposure = PROBE_EXPOSURE_US
    while True:
        camera.set_exposure_time_us(int(probe_exposure))
        frame = camera.get_image(num_frames_to_drop=NUM_SETTLE_FRAMES)
        probe_dn = float(_central_roi(frame, ROI_FRACTION).mean())
        if probe_dn <= SATURATION_FRACTION * pixel_max or probe_exposure <= MIN_EXPOSURE_US:
            break
        probe_exposure = max(probe_exposure / 2, MIN_EXPOSURE_US)

    if probe_dn <= 0:
        raise RuntimeError("Probe frame is fully dark; check illumination before running the PTC sweep.")

    exposures = probe_exposure * (FILL_TARGETS * pixel_max) / probe_dn
    exposures = np.clip(exposures, MIN_EXPOSURE_US, MAX_EXPOSURE_US).round().astype(int)
    return exposures


if __name__ == "__main__":
    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, SERIAL) as camera, PM400(serial=PM_SERIAL) as pm:
        camera.set_settings(
            exposure_time_us=PROBE_EXPOSURE_US,
            gain=GAIN_RANGE[0],
            black_level=BLACK_LEVEL,
            bit_depth=BIT_DEPTH,
            out_bit_depth=OUT_BIT_DEPTH,
        )
        pm.set_power_unit('W')
        pm.set_auto_range(True)
        pm.set_wavelength(PM_WAVELENGTH_NM)

        pixel_max = camera.pixel_max_value
        n_gains = len(GAIN_RANGE)
        signal_dn = np.zeros((n_gains, N_LEVELS, NUM_PAIRS), dtype=np.float64)
        var_half = np.zeros((n_gains, N_LEVELS, NUM_PAIRS), dtype=np.float64)
        power_w = np.zeros((n_gains, N_LEVELS, NUM_PAIRS), dtype=np.float64)
        exposures_us = np.zeros((n_gains, N_LEVELS), dtype=np.int64)
        pm_averaging_s = np.zeros((n_gains, N_LEVELS), dtype=np.float64)

        with _armed_camera(camera):
            for gi, gain in enumerate(GAIN_RANGE):
                camera.set_gain(gain)
                print(f"Gain {gain} ({gi + 1}/{n_gains})")

                exposures = _probe_exposures(camera, camera.pixel_max_value)
                exposures_us[gi] = exposures

                for li, exp in enumerate(exposures):
                    camera.set_exposure_time_us(int(exp))
                    # Match the PM400's averaging window to the camera exposure so
                    # lamp-power fluctuations are integrated over the same interval
                    # as the frame being measured.
                    pm.set_averaging_time(exp / 1e6)
                    pm_averaging_s[gi, li] = pm.get_averaging_time()
                    print(f"  level {li + 1}/{N_LEVELS}: exposure {exp} us "
                          f"(target fill {FILL_TARGETS[li]:.2f}, "
                          f"PM avg {pm_averaging_s[gi, li] * 1e6:.0f} us)")

                    for pi in range(NUM_PAIRS):
                        drop = NUM_SETTLE_FRAMES if pi == 0 else 0
                        pm_before = pm.get_power_avg(PM_AVG_N)
                        f1 = camera.get_image(num_frames_to_drop=drop)
                        f2 = camera.get_image()
                        pm_after = pm.get_power_avg(PM_AVG_N)

                        r1 = _central_roi(f1, ROI_FRACTION).astype(np.float64)
                        r2 = _central_roi(f2, ROI_FRACTION).astype(np.float64)

                        s = 0.5 * (r1.mean() + r2.mean())
                        d = r1 - r2
                        d -= d.mean()  # removes uniform lamp drift between the pair
                        v = d.var(ddof=1) / 2

                        signal_dn[gi, li, pi] = s
                        var_half[gi, li, pi] = v
                        power_w[gi, li, pi] = 0.5 * (pm_before + pm_after)

                        print(f"    pair {pi + 1}/{NUM_PAIRS}: S={s:.2f} DN, var/2={v:.4f} DN^2")

    # -------------------------------------------------------------------------
    # Fit K(g) = 1 / slope per gain: var_half = read_noise_var_dn2 + S / K
    # -------------------------------------------------------------------------
    K = np.full(n_gains, np.nan, dtype=np.float64)
    read_noise_var_dn2 = np.full(n_gains, np.nan, dtype=np.float64)
    read_noise_dn = np.full(n_gains, np.nan, dtype=np.float64)

    print("\nFit results:")
    for gi, gain in enumerate(GAIN_RANGE):
        s_flat = signal_dn[gi].ravel()
        v_flat = var_half[gi].ravel()
        mask = s_flat < SATURATION_FRACTION * pixel_max
        if mask.sum() < 2:
            print(f"  Gain {gain}: not enough unsaturated points to fit, skipping")
            continue

        slope, intercept = np.polyfit(s_flat[mask], v_flat[mask], 1)
        K[gi] = 1.0 / slope
        read_noise_var_dn2[gi] = intercept
        read_noise_dn[gi] = np.sqrt(max(intercept, 0.0))

        rn_e = read_noise_dn[gi] * K[gi]
        print(f"  Gain {gain}: K={K[gi]:.4f} e-/DN, read_noise={read_noise_dn[gi]:.3f} DN "
              f"({rn_e:.3f} e-), K*gain={K[gi] * gain:.4f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'thorcam_ptc_gain_{timestamp}.npz'

    description = (
        "Photon transfer curve, two-frame difference method. Per (gain, level, "
        "pair): signal_dn is the raw (not dark-subtracted) mean over a central "
        f"{ROI_FRACTION:.0%} ROI, averaged between the pair; var_half is "
        "var(F1-F2)/2 over that ROI, mean-subtracted to remove common-mode "
        "lamp drift within the pair. K = 1/slope from var_half = "
        "read_noise_var_dn2 + signal_dn/K per gain (points >= "
        f"{SATURATION_FRACTION:.0%} of full well excluded from the fit). "
        "signal_dn is raw, not dark-subtracted -- a constant dark pedestal "
        "only shifts K's intercept, not the slope."
    )

    np.savez(
        save_path,
        signal_dn=signal_dn,
        var_half=var_half,
        power_w=power_w,
        exposures_us=exposures_us,
        pm_averaging_s=pm_averaging_s,
        K=K,
        read_noise_var_dn2=read_noise_var_dn2,
        read_noise_dn=read_noise_dn,
        gain_range=np.array(GAIN_RANGE),
        roi_fraction=ROI_FRACTION,
        n_levels=N_LEVELS,
        num_pairs=NUM_PAIRS,
        pixel_max=pixel_max,
        pm_wavelength_nm=PM_WAVELENGTH_NM,
        black_level=BLACK_LEVEL,
        serial=SERIAL,
        pm_serial=PM_SERIAL,
        description=description,
    )

    print(f"Saved PTC gain results for {n_gains} gains -> {save_path}")
