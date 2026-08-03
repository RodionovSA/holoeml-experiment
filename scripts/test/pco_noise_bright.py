"""Bright/dark frame stacks for a photon-transfer / noise decomposition of the
PCO camera. Mirrors `thorcam_noise_bright.py`, adapted for the PCO's 16-bit
ADC, exposure-only control (no gain, no black level), and hardware exposure
range read from the device instead of hardcoded CS126 limits.
"""

from datetime import datetime
from pathlib import Path

from instruments.config import load_equipment
from instruments.pypcocam.pcocam import PcoCamera
from instruments.camera.utils import autoexposure
from instruments.filterwheel import FilterWheelControl
import time
import contextlib
import numpy as np

_EQ = load_equipment()
SERIAL = _EQ.camera_serial
ROI_FRACTION = 0.1        # central 10% x 10% of the sensor, restricted in hardware (see set_roi)
NUM_SETTLE_FRAMES = 2      # dropped after each exposure change, on top of the
                            # driver's own settle barrier (PcoCamera.EXPOSURE_SETTLE_FRAMES) --
                            # margin here, not the mechanism that guarantees freshness.
PROBE_EXPOSURE_US = 10_000    # starting probe to find the DN<->exposure scale
PROBE_SATURATION_FRACTION = 0.9  # guards only the fixed-exposure probe (level 0 seed) against
                                  # landing in the nonlinear near-saturation regime; does not
                                  # limit target_range, which is free to go up to 0.999
TARGET_RANGE = [0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]  # fraction of full well, log-spaced
NUM_FRAMES = 50
OUT_BIT_DEPTH = np.float32

AE_TOLERANCE_RELATIVE = 0.02   # autoexposure stop tolerance, fraction of the target DN
AE_TOLERANCE_FLOOR_DN = 1.0    # absolute floor so near-zero targets don't chase sub-DN precision
AE_INCREMENT = 0.1             # multiplicative exposure step per autoexposure iteration
AE_MAX_STEPS = 50

# Safety cap distinct from the hardware max: PcoCamera._get_single_frame's timeout now
# scales with exposure, so this is purely a runtime sanity check -- at 50 frames/level,
# a level exceeding this takes minutes on its own. If autoexposure wants more than this,
# raise the lamp power instead of raising this cap.
MAX_EXPOSURE_US = 2_000_000

FILTERWHEEL_ADDRESS = _EQ.filterwheel_address
FILTERWHEEL_OPEN = _EQ.filterwheel_empty_pos
FILTERWHEEL_CLOSED = _EQ.black_pos


@contextlib.contextmanager
def _armed_camera(camera: PcoCamera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()


def _probe_seed_exposure(camera: PcoCamera, target_dn: float, pixel_max: float) -> int:
    """Probe the DN<->exposure scale at PROBE_EXPOSURE_US, assuming DN is
    proportional to exposure, then return an exposure (us) expected to land
    near target_dn. Only used to seed the first autoexposure level -- later
    levels seed from the previous level's converged exposure instead."""
    camera.set_exposure_time_us(PROBE_EXPOSURE_US)
    frame = camera.get_image(num_frames_to_drop=NUM_SETTLE_FRAMES)
    probe_dn = float(frame.mean())

    if probe_dn <= 0:
        raise RuntimeError(
            "Probe frame is fully dark; check illumination before running the noise sweep."
        )
    if probe_dn >= PROBE_SATURATION_FRACTION * pixel_max:
        raise RuntimeError(
            f"Probe frame is saturated ({probe_dn:.0f} DN at {PROBE_EXPOSURE_US} us); "
            "lower PROBE_EXPOSURE_US or reduce illumination before running the noise sweep."
        )

    seed = PROBE_EXPOSURE_US * target_dn / probe_dn
    camera_min_us = int(camera.min_exposure_ms * 1000)
    camera_max_us = int(camera.max_exposure_ms * 1000)
    return int(np.clip(seed, camera_min_us, camera_max_us))


if __name__ == "__main__":
    with PcoCamera(SERIAL) as camera, FilterWheelControl(FILTERWHEEL_ADDRESS) as filterwheel:
        camera.set_exposure_time_us(PROBE_EXPOSURE_US)
        camera.set_out_bit_depth(OUT_BIT_DEPTH)

        pixel_max = camera.pixel_max_value
        min_exposure_us = camera.min_exposure_ms * 1000
        max_exposure_us = camera.max_exposure_ms * 1000
        camera_min_us = int(round(min_exposure_us))
        camera_max_us = int(round(max_exposure_us))
        print(f"Pixel max: {pixel_max} DN")
        print(f"Hardware exposure range: {min_exposure_us:.1f} - {max_exposure_us:.1f} us")

        N_LEVELS = len(TARGET_RANGE)

        # Cap autoexposure tolerance at half the distance (in fill fraction) to the
        # nearest neighboring level, so closely spaced targets can't fall inside
        # each other's acceptance window and collapse onto the same exposure.
        level_gap_frac = np.empty(N_LEVELS)
        for i in range(N_LEVELS):
            neighbor_gaps = []
            if i > 0:
                neighbor_gaps.append(abs(TARGET_RANGE[i] - TARGET_RANGE[i - 1]))
            if i < N_LEVELS - 1:
                neighbor_gaps.append(abs(TARGET_RANGE[i + 1] - TARGET_RANGE[i]))
            level_gap_frac[i] = min(neighbor_gaps)

        # Restrict the camera's *hardware* ROI to the analysis window instead of
        # cropping down in software after the fact -- by default the camera transfers
        # the entire sensor on every frame (multi-thousand-pixel, tens of MB at 16-bit),
        # which is both wasteful and was observed to intermittently blow the per-frame
        # timeout at multi-second exposures. Must happen before arming.
        sensor_h, sensor_w = camera.sensor_shape
        target_h = 2 * max(1, int(sensor_h * ROI_FRACTION / 2))
        target_w = 2 * max(1, int(sensor_w * ROI_FRACTION / 2))
        camera.set_roi(target_w, target_h)
        roi_h, roi_w = camera.image_shape  # actual size after hardware step-rounding

        array_bytes = 2 * N_LEVELS * NUM_FRAMES * roi_h * roi_w * np.dtype(np.float32).itemsize
        print(f"Sensor: {sensor_h} x {sensor_w} px; hardware ROI restricted to "
              f"{roi_h} x {roi_w} px; total signal+dark storage: {array_bytes / 1e6:.1f} MB")

        signal_dn = np.zeros((1, N_LEVELS, NUM_FRAMES, roi_h, roi_w), dtype=np.float32)
        dark_dn = np.zeros((1, N_LEVELS, NUM_FRAMES, roi_h, roi_w), dtype=np.float32)
        exposures_us = np.zeros((1, N_LEVELS), dtype=np.int32)
        # Measured ROI mean at each level, used both to seed the next level's exposure
        # (DN is ~proportional to exposure) and to record what was actually achieved,
        # since dim levels can be pinned at the hardware exposure floor (see below) and
        # never reach their nominal target_range fraction.
        achieved_dn = np.zeros((1, N_LEVELS), dtype=np.float64)

        with _armed_camera(camera):
            filterwheel.set_position(FILTERWHEEL_OPEN)

            for i in range(N_LEVELS):
                filterwheel.set_position(FILTERWHEEL_OPEN)
                target_dn = TARGET_RANGE[i] * pixel_max
                tolerance_dn = min(
                    max(AE_TOLERANCE_RELATIVE * target_dn, AE_TOLERANCE_FLOOR_DN),
                    0.5 * level_gap_frac[i] * pixel_max,
                )

                if i == 0:
                    initial_exposure = _probe_seed_exposure(camera, target_dn, pixel_max)
                else:
                    # Scale the previous level's exposure by the fill ratio (DN is
                    # ~proportional to exposure) instead of reusing it unchanged --
                    # lands autoexposure within a few percent so it converges in a
                    # couple of steps rather than ~10, which matters here since every
                    # exposure change restarts the recorder and autoexposure sleeps
                    # 0.5s per step.
                    initial_exposure = int(np.clip(
                        exposures_us[0, i - 1] * target_dn / max(achieved_dn[0, i - 1], 1.0),
                        camera_min_us, camera_max_us))

                # roi_fraction=None: the hardware ROI (set_roi, below) already restricts
                # every captured frame to the analysis window, so autoexposure's own
                # central-fraction crop is unnecessary here (and would double-crop).
                exp = autoexposure(camera,
                                   initial_exposure_time=initial_exposure,
                                   target_brightness=target_dn,
                                   tolerance=tolerance_dn,
                                   increment=AE_INCREMENT,
                                   max_number_of_steps=AE_MAX_STEPS,
                                   num_frames_to_drop=NUM_SETTLE_FRAMES,
                                   roi_fraction=None)

                # autoexposure returns what it *asked* the camera for; PcoCamera clamps
                # and quantizes exposure to the hardware grid internally, so read back
                # what was actually applied rather than trusting the request.
                exp = int(round(camera.exposure_ms * 1000))

                if exp > MAX_EXPOSURE_US:
                    raise RuntimeError(
                        f"Level {i + 1}/{N_LEVELS} (target fill {TARGET_RANGE[i]:.4f}) wants "
                        f"{exp} us, above the MAX_EXPOSURE_US safety cap ({MAX_EXPOSURE_US} us). "
                        "Raise lamp power, or raise MAX_EXPOSURE_US if you've confirmed the "
                        "longer per-level runtime is acceptable."
                    )
                if exp <= camera_min_us:
                    print(f"  WARNING: level {i + 1}/{N_LEVELS} pinned at the hardware exposure "
                          "minimum -- achieved fill will exceed the target (printed below after "
                          "capture); consider reducing lamp power if you need this level distinct.")

                exposures_us[0, i] = exp

                print(f"  level {i + 1}/{N_LEVELS}: exposure {exp} us "
                      f"(target fill {TARGET_RANGE[i]:.4f})")

                for pi in range(NUM_FRAMES):
                    drop = NUM_SETTLE_FRAMES if pi == 0 else 0
                    signal_dn[0, i, pi] = camera.get_image(num_frames_to_drop=drop)[..., 0]

                achieved_dn[0, i] = float(signal_dn[0, i].mean())
                print(f"    achieved fill: {achieved_dn[0, i] / pixel_max:.4f} "
                      f"({achieved_dn[0, i]:.1f} DN)")

                filterwheel.set_position(FILTERWHEEL_CLOSED)
                for pi in range(NUM_FRAMES):
                    drop = NUM_SETTLE_FRAMES if pi == 0 else 0
                    dark_dn[0, i, pi] = camera.get_image(num_frames_to_drop=drop)[..., 0]

                filterwheel.set_position(FILTERWHEEL_OPEN)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'pco_noise_bright_{timestamp}.npz'

    description = (
        f"Bright/dark frame stacks per target fill level, PCO camera (no gain control). For "
        f"each level: signal_dn is {NUM_FRAMES} raw frames captured with the filter wheel "
        f"open, dark_dn is {NUM_FRAMES} raw frames captured immediately after with the filter "
        f"wheel closed, both restricted to the central {ROI_FRACTION:.0%} of the sensor via the "
        "camera's hardware ROI. exposures_us is "
        "the exposure (from autoexposure, targeting the corresponding fraction of full well "
        "in target_range) used for both captures at that level. achieved_dn is the actual "
        "signal_dn ROI mean at that level -- dim levels can be pinned at the camera's hardware "
        "exposure minimum and never reach their nominal target_range fraction, so analysis "
        "should bin/label by achieved_dn, not target_range. The leading axis of "
        "signal_dn/dark_dn/exposures_us/achieved_dn is a singleton gain axis kept only for "
        "layout compatibility with thorcam_noise_bright.py's output -- this camera has no gain "
        "control (gain_range is a placeholder, not a real setting)."
    )

    np.savez(
        save_path,
        signal_dn=signal_dn,
        dark_dn=dark_dn,
        exposures_us=exposures_us,
        achieved_dn=achieved_dn,
        target_range=np.array(TARGET_RANGE),
        num_frames=NUM_FRAMES,
        gain_range=np.array([0]),  # placeholder -- PCO has no gain control
        roi_fraction=ROI_FRACTION,
        pixel_max=pixel_max,
        bit_resolution=int(np.log2(pixel_max + 1)),
        camera_vendor='pco',
        serial=SERIAL,
        description=description,
    )

    print(f"Saved bright/dark noise results -> {save_path}")
