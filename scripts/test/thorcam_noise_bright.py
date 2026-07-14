from datetime import datetime
from pathlib import Path

from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
from instruments.pythorcam.utils import autoexposure
from instruments.filterwheel import FilterWheelControl
import time
import contextlib
import numpy as np


SERIAL = "35596"
BLACK_LEVEL = 0
GAIN_RANGE = [0]
ROI_FRACTION = 0.10        # central 20% x 20% sub-ROI (avoid vignetted edges)
NUM_SETTLE_FRAMES = 2      # dropped after each exposure change
PROBE_EXPOSURE_US = 100_000   # starting probe to find the DN<->exposure scale
MAX_EXPOSURE_US = 14_700_924  # CS126 hardware max
MIN_EXPOSURE_US = 28
TARGET_RANGE = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
NUM_FRAMES = 50
BIT_DEPTH = np.uint16
OUT_BIT_DEPTH = np.float32

AE_TOLERANCE_RELATIVE = 0.02   # autoexposure stop tolerance, fraction of the target DN
AE_TOLERANCE_FLOOR_DN = 1.0    # absolute floor so near-zero targets don't chase sub-DN precision
AE_INCREMENT = 0.1             # multiplicative exposure step per autoexposure iteration
AE_MAX_STEPS = 50

FILTERWHEEL_ADDRESS = 'USB0::0x104D::0x1011::7A01020F12162064::RAW'
FILTERWHEEL_OPEN = 1
FILTERWHEEL_CLOSED = 4

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

if __name__ == "__main__":
    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, SERIAL) as camera, FilterWheelControl(FILTERWHEEL_ADDRESS) as filterwheel:
        camera.set_settings(
            exposure_time_us=PROBE_EXPOSURE_US,
            gain=GAIN_RANGE[0],
            black_level=BLACK_LEVEL,
            bit_depth=BIT_DEPTH,
            out_bit_depth=OUT_BIT_DEPTH,
        )

        pixel_max = camera.pixel_max_value
        n_gains = len(GAIN_RANGE)
        N_LEVELS = len(TARGET_RANGE)

        # Cap autoexposure tolerance at half the distance (in fill fraction) to the
        # nearest neighboring level, so closely spaced targets (e.g. the 0.95-0.999
        # cluster) can't fall inside each other's acceptance window and collapse
        # onto the same exposure.
        level_gap_frac = np.empty(N_LEVELS)
        for i in range(N_LEVELS):
            neighbor_gaps = []
            if i > 0:
                neighbor_gaps.append(abs(TARGET_RANGE[i] - TARGET_RANGE[i - 1]))
            if i < N_LEVELS - 1:
                neighbor_gaps.append(abs(TARGET_RANGE[i + 1] - TARGET_RANGE[i]))
            level_gap_frac[i] = min(neighbor_gaps)

        H, W = camera.image_shape
        roi_h = 2 * max(1, int(H * ROI_FRACTION / 2))
        roi_w = 2 * max(1, int(W * ROI_FRACTION / 2))

        signal_dn = np.zeros((n_gains, N_LEVELS, NUM_FRAMES, roi_h, roi_w), dtype=np.float32)
        dark_dn = np.zeros((n_gains, N_LEVELS, NUM_FRAMES, roi_h, roi_w), dtype=np.float32)
        exposures_us = np.zeros((n_gains, N_LEVELS), dtype=np.int32)

        with _armed_camera(camera):
            for gi, gain in enumerate(GAIN_RANGE):
                camera.set_gain(gain)
                print(f"Gain {gain} ({gi + 1}/{n_gains})")

                for i in range(N_LEVELS):
                    filterwheel.set_position(FILTERWHEEL_OPEN)
                    target_dn = TARGET_RANGE[i] * pixel_max
                    tolerance_dn = min(
                        max(AE_TOLERANCE_RELATIVE * target_dn, AE_TOLERANCE_FLOOR_DN),
                        0.5 * level_gap_frac[i] * pixel_max,
                    )
                    exp = autoexposure(camera,
                                       initial_exposure_time=int(exposures_us[gi, i-1]) if i>0 else PROBE_EXPOSURE_US,
                                       target_brightness=target_dn,
                                       tolerance=tolerance_dn,
                                       increment=AE_INCREMENT,
                                       max_number_of_steps=AE_MAX_STEPS,
                                       num_frames_to_drop=NUM_SETTLE_FRAMES,
                                       roi_fraction=ROI_FRACTION)
                    exposures_us[gi, i] = exp

                    print(f"  level {i + 1}/{N_LEVELS}: exposure {exp} us "
                          f"(target fill {TARGET_RANGE[i]:.3f})")

                    for pi in range(NUM_FRAMES):
                        drop = NUM_SETTLE_FRAMES if pi == 0 else 0
                        signal_dn[gi, i, pi] = _central_roi(
                            camera.get_image(num_frames_to_drop=drop), ROI_FRACTION)[..., 0]

                    filterwheel.set_position(FILTERWHEEL_CLOSED)
                    for pi in range(NUM_FRAMES):
                        drop = NUM_SETTLE_FRAMES if pi == 0 else 0
                        dark_dn[gi, i, pi] = _central_roi(
                            camera.get_image(num_frames_to_drop=drop), ROI_FRACTION)[..., 0]

                    filterwheel.set_position(FILTERWHEEL_OPEN)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'thorcam_noise_bright_{timestamp}.npz'

    description = (
        f"Bright/dark frame stacks per gain and target fill level. For each (gain, level): "
        f"signal_dn is {NUM_FRAMES} raw frames captured with the filter wheel open, dark_dn is "
        f"{NUM_FRAMES} raw frames captured immediately after with the filter wheel closed, "
        f"both cropped to the central {ROI_FRACTION:.0%} ROI. exposures_us is the exposure "
        "(from autoexposure, targeting the corresponding fraction of full well in "
        "target_range) used for both captures at that level."
    )

    np.savez(
        save_path,
        signal_dn=signal_dn,
        dark_dn=dark_dn,
        exposures_us=exposures_us,
        target_range=np.array(TARGET_RANGE),
        num_frames=NUM_FRAMES,
        gain_range=np.array(GAIN_RANGE),
        roi_fraction=ROI_FRACTION,
        pixel_max=pixel_max,
        black_level=BLACK_LEVEL,
        serial=SERIAL,
        description=description,
    )

    print(f"Saved bright/dark noise results for {n_gains} gains -> {save_path}")

