from datetime import datetime
from pathlib import Path

from instruments.config import load_equipment
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
import time
import contextlib
import numpy as np

SERIAL = load_equipment().camera_serial
BLACK_LEVEL = 0
GAIN_RANGE = [0, 30, 60, 100, 140, 180, 220, 250]
EXPOSURE = 28 # us
BIT_DEPTH = np.uint16
OUT_BIT_DEPTH = np.float32
NUM_FRAMES = 100

@contextlib.contextmanager
def _armed_camera(camera: ThorlabsCamera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()

if __name__ == "__main__":
    # Setup camera
    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, SERIAL) as camera:
        camera.set_settings(
            exposure_time_us=EXPOSURE,
            gain=GAIN_RANGE[0],
            black_level=BLACK_LEVEL,
            bit_depth=BIT_DEPTH,
            out_bit_depth=OUT_BIT_DEPTH,
        )

        H, W = camera.image_shape
        C = 1 if camera.cam_type == 'MU' else 3

        n_gains = len(GAIN_RANGE)
        read_noise = np.zeros(n_gains, dtype=np.float64)
        offset_level = np.zeros(n_gains, dtype=np.float64)
        std_maps = np.zeros((n_gains, H, W, C), dtype=OUT_BIT_DEPTH)
        mean_frames = np.zeros((n_gains, H, W, C), dtype=OUT_BIT_DEPTH)
        drift_traces = np.zeros((n_gains, NUM_FRAMES), dtype=np.float64)

        with _armed_camera(camera):
            for idx, gain in enumerate(GAIN_RANGE):
                camera.set_gain(gain)
                print(f"Gain {gain} ({idx + 1}/{n_gains})")

                # Welford's online algorithm: per-pixel mean/variance over the
                # 100-frame stack without ever holding the full stack in memory.
                mean = np.zeros((H, W, C), dtype=np.float64)
                m2 = np.zeros((H, W, C), dtype=np.float64)
                for i in range(NUM_FRAMES):
                    frame = camera.get_image().astype(np.float64)
                    drift_traces[idx, i] = frame.mean()
                    delta = frame - mean
                    mean += delta / (i + 1)
                    m2 += delta * (frame - mean)
                    if (i + 1) % 10 == 0 or (i + 1) == NUM_FRAMES:
                        print(f"  frame {i + 1}/{NUM_FRAMES}")

                std_map = np.sqrt(m2 / (NUM_FRAMES - 1))
                read_noise[idx] = np.median(std_map)
                offset_level[idx] = mean.mean()
                std_maps[idx] = std_map.astype(OUT_BIT_DEPTH)
                mean_frames[idx] = mean.astype(OUT_BIT_DEPTH)
                print(f"  read noise = {read_noise[idx]:.3f}, offset = {offset_level[idx]:.1f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'thorcam_read_noise_{timestamp}.npz'

    np.savez(
        save_path,
        read_noise=read_noise,          # (n_gains,) median per-pixel temporal std
        std_maps=std_maps,              # (n_gains, H, W, C) per-pixel temporal std
        mean_frames=mean_frames,        # (n_gains, H, W, C) per-pixel temporal mean (offset/DSNU)
        offset_level=offset_level,      # (n_gains,) spatial mean of mean_frames
        drift_traces=drift_traces,      # (n_gains, NUM_FRAMES) per-frame spatial mean
        gain_range=np.array(GAIN_RANGE),
        num_frames=NUM_FRAMES,
        exposure_us=EXPOSURE,
        black_level=BLACK_LEVEL,
        serial=SERIAL,
    )

    print(f"Saved read-noise stats for {n_gains} gains → {save_path}")

    