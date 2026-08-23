"""KIM101/PIAK10 step-calibration acquisition (open-loop inertial stepping).

Connects the camera and the KIM101 inertial piezo controller, records a baseline
sequence of stationary images (a stability check -- fringes should not move at
all), then drives NUM_CYCLES forward/backward sweeps of NUM_STEPS commanded
moves of STEP_SIZE steps each, recording one image after every move.

This script does NOT extract phase or fringe position -- it only saves raw
images plus the commanded-position metadata needed to do that afterwards. The
two quantities this data is meant to answer, by manually tracking fringe
position across the saved images:

  1. Fringe shift per commanded step (the effective step size), which for a
     slip-stick inertial actuator can vary with load, drive voltage, and
     direction.
  2. The reversal "dead zone": how many commanded steps are swallowed after a
     direction change before the actuator physically starts moving again.

IMPORTANT: KIM101Axis.get_position() is an open-loop *commanded* step counter,
not a position sensor -- it advances by exactly the requested step count
whether or not the actuator physically moved. It cannot reveal the dead zone
on its own. It is recorded here purely as the x-axis to plot fringe position
against; the dead zone shows up as the flat region of that plot immediately
after each direction reversal, not as anything visible in the counter itself.

Run from the repo root so `instruments` is importable:
    python scripts/phase/kim_step_calibration.py
"""

import contextlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.camera import Camera, open_camera
from instruments.config import load_equipment
from instruments.inertialpiezo import KIM101

# --------------------------------------------------------------------------- #
# Settings -- edit before running                                             #
# --------------------------------------------------------------------------- #
_EQ = load_equipment()
CAMERA_SERIAL = _EQ.camera_serial
KIM_SERIAL = _EQ.kim_serial
KIM_CHANNEL = _EQ.kim_channel

EXPOSURE_MS = 200.0
GAIN = 100
BLACK_LEVEL = 0
ROI = (1000, 1000, 2000, 2000)  # (x0, y0, w, h) px, applied as a crop after capture; None = full frame
FRAMES_TO_AVERAGE = 1
FRAMES_TO_DROP = 2
OUT_BIT_DEPTH = np.uint16  # np.float32 keeps sub-count precision when averaging (2x size)

# KIM101 drive parameters (20/100/125 mirrors values already hand-tried in
# scripts/test/test_kimpiezo.ipynb; factory defaults are 100/1000/125).
STEP_RATE = 50  # steps/s
STEP_ACCELERATION = 500  # steps/s^2
MAX_VOLTAGE = 125  # V

# Scan pattern
STEP_SIZE = 2  # commanded steps per move
NUM_STEPS = 30  # moves per half-cycle (per direction)
NUM_CYCLES = 2  # forward+backward cycles
SETTLE_S = 0.1  # pause after each move before capturing (mechanical settling)
NUM_BASELINE_FRAMES = 20  # stationary frames for the stability check
MOVE_TIMEOUT_S = 30.0
MAX_DATASET_GB = 4.0  # pre-flight abort threshold


@contextlib.contextmanager
def _armed_camera(camera: Camera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()


def _crop(img: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Apply ROI = (x0, y0, w, h) as a software crop. No-op when roi is None."""
    if roi is None:
        return img
    x0, y0, w, h = roi
    H, W = img.shape[0], img.shape[1]
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        raise ValueError(f"ROI {roi} does not fit inside frame of shape {(H, W)}")
    return img[y0:y0 + h, x0:x0 + w]


if __name__ == "__main__":
    t_start = time.monotonic()

    with contextlib.ExitStack() as stack:
        camera = open_camera(_EQ, stack)
        stack.enter_context(camera)
        kim = stack.enter_context(KIM101(KIM_SERIAL))
        axis = kim.axis(KIM_CHANNEL)

        camera.set_exposure_ms(EXPOSURE_MS)
        if camera.supports_gain:
            camera.set_gain(GAIN)
        if camera.supports_black_level:
            camera.set_black_level(BLACK_LEVEL)
        camera.set_raw_bit_depth(getattr(np, _EQ.camera_bit_depth))
        camera.set_out_bit_depth(OUT_BIT_DEPTH)

        axis.set_drive_parameters(
            step_rate=STEP_RATE,
            step_acceleration=STEP_ACCELERATION,
            max_voltage=MAX_VOLTAGE,
        )
        # The controller may clamp requested values -- record what was actually
        # applied, not what was requested.
        applied_drive_params = axis.get_drive_parameters()
        print(f"Drive parameters applied: {applied_drive_params}")

        axis.zero()

        H, W = camera.image_shape
        C = camera.num_channels
        if ROI is not None:
            _, _, crop_w, crop_h = ROI
            H, W = crop_h, crop_w

        # ------------------------------------------------------------------
        # Pre-flight size check -- a full-frame run at these defaults is
        # easily multi-GB, so abort before acquiring anything if the planned
        # dataset would exceed MAX_DATASET_GB.
        # ------------------------------------------------------------------
        n_scan = NUM_CYCLES * 2 * NUM_STEPS
        n_frames = NUM_BASELINE_FRAMES + n_scan
        bytes_per_frame = H * W * C * np.dtype(OUT_BIT_DEPTH).itemsize
        estimated_gb = n_frames * bytes_per_frame / 1e9
        print(f"Planned acquisition: {n_frames} frames of {H}x{W}x{C} "
              f"({OUT_BIT_DEPTH.__name__}) -> ~{estimated_gb:.2f} GB")
        if estimated_gb > MAX_DATASET_GB:
            raise RuntimeError(
                f"Planned dataset (~{estimated_gb:.2f} GB) exceeds MAX_DATASET_GB "
                f"({MAX_DATASET_GB} GB). Reduce NUM_STEPS/NUM_CYCLES/NUM_BASELINE_FRAMES, "
                f"set a smaller ROI, or raise MAX_DATASET_GB."
            )

        baseline_images = np.zeros((NUM_BASELINE_FRAMES, H, W, C), dtype=OUT_BIT_DEPTH)
        baseline_times = np.zeros(NUM_BASELINE_FRAMES, dtype=np.float64)

        scan_images = np.zeros((n_scan, H, W, C), dtype=OUT_BIT_DEPTH)
        scan_commanded = np.zeros(n_scan, dtype=np.int64)
        scan_counter = np.zeros(n_scan, dtype=np.int64)
        scan_direction = np.zeros(n_scan, dtype=np.int8)
        scan_cycle = np.zeros(n_scan, dtype=np.int64)
        scan_step_index = np.zeros(n_scan, dtype=np.int64)
        scan_times = np.zeros(n_scan, dtype=np.float64)

        # Track how many frames were actually captured so a failure partway through
        # still leaves us with an honest (correctly-trimmed), savable dataset instead
        # of losing the whole run.
        n_baseline_captured = 0
        idx = 0
        error: Exception | None = None

        try:
            with _armed_camera(camera):
                # ----------------------------------------------------------
                # 1. Baseline: hold position, record NUM_BASELINE_FRAMES
                #    stationary images as a stability check.
                # ----------------------------------------------------------
                print(f"Baseline: recording {NUM_BASELINE_FRAMES} stationary frames ...")
                for i in range(NUM_BASELINE_FRAMES):
                    img = camera.get_image(FRAMES_TO_AVERAGE, FRAMES_TO_DROP)
                    baseline_images[i] = _crop(img, ROI)
                    baseline_times[i] = time.monotonic() - t_start
                    n_baseline_captured = i + 1

                # ----------------------------------------------------------
                # 2. Step sweep: NUM_CYCLES forward+backward cycles, one
                #    image after every commanded move.
                # ----------------------------------------------------------
                commanded = 0
                for cycle in range(NUM_CYCLES):
                    print(f"Cycle {cycle + 1}/{NUM_CYCLES}")
                    for direction in (+1, -1):
                        label = "forward" if direction > 0 else "backward"
                        print(f"  {NUM_STEPS} {label} steps ...")
                        for s in range(NUM_STEPS):
                            axis.move_by(direction * STEP_SIZE, timeout=MOVE_TIMEOUT_S)
                            time.sleep(SETTLE_S)
                            commanded += direction * STEP_SIZE

                            img = camera.get_image(FRAMES_TO_AVERAGE, FRAMES_TO_DROP)
                            scan_images[idx] = _crop(img, ROI)
                            scan_commanded[idx] = commanded
                            scan_counter[idx] = axis.get_position()
                            scan_direction[idx] = direction
                            scan_cycle[idx] = cycle
                            scan_step_index[idx] = s
                            scan_times[idx] = time.monotonic() - t_start
                            idx += 1

                            if (s + 1) % 10 == 0 or (s + 1) == NUM_STEPS:
                                print(f"    step {s + 1}/{NUM_STEPS} -> "
                                      f"commanded {commanded}, counter {scan_counter[idx - 1]}")
        except Exception as exc:
            error = exc
            print(f"ERROR during acquisition: {exc!r}")
            print(f"Captured {n_baseline_captured} baseline / {idx} scan frames before the "
                  f"failure -- saving partial data.")
        finally:
            print("Returning to start position (commanded 0) ...")
            try:
                axis.move_to(0, timeout=MOVE_TIMEOUT_S)
            except Exception as move_exc:
                print(f"WARNING: failed to return to start position: {move_exc!r}")
            print(f"Final commanded position: {axis.get_position()}")

    # -------------------------------------------------------------------------
    # Save results -- trim every array to what was actually captured, so a
    # partial run (an exception mid-sweep, or during the return-to-start move)
    # still saves a correctly-shaped, honest dataset instead of losing everything
    # or saving zero-padded rows.
    # -------------------------------------------------------------------------
    baseline_images = baseline_images[:n_baseline_captured]
    baseline_times = baseline_times[:n_baseline_captured]
    scan_images = scan_images[:idx]
    scan_commanded = scan_commanded[:idx]
    scan_counter = scan_counter[:idx]
    scan_direction = scan_direction[:idx]
    scan_cycle = scan_cycle[:idx]
    scan_step_index = scan_step_index[:idx]
    scan_times = scan_times[:idx]

    description = (
        "KIM101/PIAK10 step-calibration data. No phase/fringe extraction was done "
        "here -- plot fringe position against 'scan_commanded' to find steps-per-fringe, "
        "and look for the flat region right after each 'scan_direction' flip to find the "
        "reversal dead zone. 'baseline_images' are stationary frames for a stability check. "
        "'scan_counter' is the open-loop commanded-step counter (KIM101Axis.get_position()), "
        "NOT a position sensor -- it cannot reveal the dead zone on its own."
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"kim_step_calibration_{timestamp}.npz"

    np.savez(
        save_path,
        baseline_images=baseline_images,      # (NUM_BASELINE_FRAMES, H, W, C)
        baseline_times=baseline_times,        # (NUM_BASELINE_FRAMES,) s since start
        scan_images=scan_images,              # (n_scan, H, W, C)
        scan_commanded=scan_commanded,        # (n_scan,) cumulative commanded steps
        scan_counter=scan_counter,            # (n_scan,) axis.get_position() readback
        scan_direction=scan_direction,        # (n_scan,) +1 forward / -1 backward
        scan_cycle=scan_cycle,                # (n_scan,)
        scan_step_index=scan_step_index,      # (n_scan,)
        scan_times=scan_times,                # (n_scan,) s since start
        description=description,
        timestamp=timestamp,
        exposure_ms=EXPOSURE_MS,
        gain=GAIN,
        black_level=BLACK_LEVEL,
        roi=np.array(ROI) if ROI is not None else np.array([]),
        frames_to_average=FRAMES_TO_AVERAGE,
        frames_to_drop=FRAMES_TO_DROP,
        out_bit_depth=OUT_BIT_DEPTH.__name__,
        step_size=STEP_SIZE,
        num_steps=NUM_STEPS,
        num_cycles=NUM_CYCLES,
        settle_s=SETTLE_S,
        num_baseline_frames=NUM_BASELINE_FRAMES,
        drive_step_rate=applied_drive_params["step_rate"],
        drive_step_acceleration=applied_drive_params["step_acceleration"],
        drive_max_voltage=applied_drive_params["max_voltage"],
        camera_serial=CAMERA_SERIAL,
        camera_vendor=_EQ.camera_vendor,
        kim_serial=KIM_SERIAL,
        kim_channel=KIM_CHANNEL,
    )

    size_mb = save_path.stat().st_size / 1e6
    print(f"Saved KIM101 step-calibration data ({size_mb:.1f} MB) -> {save_path}")

    if error is not None:
        raise error
