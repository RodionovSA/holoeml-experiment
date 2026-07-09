from datetime import datetime
from pathlib import Path

from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
import time
import contextlib
import numpy as np

SERIAL = "35596"
BLACK_LEVEL = 0
GAIN_RANGE = [0, 100, 200]
EXPOSURES_US = np.geomspace(28, 1_000_000, 8).round().astype(int)  # 8 pts, 28us -> 1s
NUM_FRAMES = 50
NUM_SETTLE_FRAMES = 2  # dropped after each exposure change, before accumulating
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

if __name__ == "__main__":
    # Setup camera. Cap the lens and let the temperature settle before running.
    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, SERIAL) as camera:
        camera.set_settings(
            exposure_time_us=int(EXPOSURES_US[0]),
            gain=GAIN_RANGE[0],
            black_level=BLACK_LEVEL,
            bit_depth=BIT_DEPTH,
            out_bit_depth=OUT_BIT_DEPTH,
        )

        n_gains = len(GAIN_RANGE)
        n_exp = len(EXPOSURES_US)
        median_variance = np.zeros((n_gains, n_exp), dtype=np.float64)
        mean_dn = np.zeros((n_gains, n_exp), dtype=np.float64)

        with _armed_camera(camera):
            for gi, gain in enumerate(GAIN_RANGE):
                camera.set_gain(gain)
                for ei, exp in enumerate(EXPOSURES_US):
                    camera.set_exposure_time_us(int(exp))
                    print(f"Gain {gain} ({gi + 1}/{n_gains}), exposure {exp} us ({ei + 1}/{n_exp})")

                    # Welford's online algorithm: per-pixel mean/variance over the
                    # 50-frame stack without ever holding the full stack in memory.
                    mean = np.zeros(camera.image_shape + (1 if camera.cam_type == 'MU' else 3,), dtype=np.float64)
                    m2 = np.zeros_like(mean)
                    for i in range(NUM_FRAMES):
                        # Drop a few settling frames right after the exposure change
                        # so a partially-exposed transition frame isn't included.
                        frame = camera.get_image(
                            num_frames_to_drop=NUM_SETTLE_FRAMES if i == 0 else 0
                        ).astype(np.float64)
                        delta = frame - mean
                        mean += delta / (i + 1)
                        m2 += delta * (frame - mean)

                    variance = m2 / (NUM_FRAMES - 1)
                    median_variance[gi, ei] = np.median(variance)
                    mean_dn[gi, ei] = mean.mean()
                    print(f"  var={median_variance[gi, ei]:.3f} DN^2, mean={mean_dn[gi, ei]:.2f} DN")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'thorcam_dark_current_{timestamp}.npz'

    np.savez(
        save_path,
        median_variance=median_variance,  # (n_gains, n_exp) median per-pixel temporal variance, DN^2
        mean_dn=mean_dn,                  # (n_gains, n_exp) spatial+temporal mean, DN
        gain_range=np.array(GAIN_RANGE),
        exposures_us=EXPOSURES_US,
        num_frames=NUM_FRAMES,
        black_level=BLACK_LEVEL,
        serial=SERIAL,
    )

    print(f"Saved dark-current stats for {n_gains} gains x {n_exp} exposures -> {save_path}")
