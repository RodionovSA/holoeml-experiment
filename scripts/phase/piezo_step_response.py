"""Piezo step-response acquisition (open-loop voltage drive).

Connects the ThorCam camera and the CT1P piezo, records a baseline sequence of
images at a fixed position, then for each step size in STEP_SIZES_UM:
  1. pre-offsets backward by PRE_OFFSET_STEPS * step_um (not imaged -- just
     repositioning so the block's real, imaged sweep is centered around
     wherever the block started, rather than only ever moving forward), then
  2. steps forward N_STEPS times, recording one image after each step.

Blocks are cumulative: each block's pre-offset/sweep starts from wherever the
previous block ended (net +PRE_OFFSET_STEPS*step_um relative to that block's
own start, since -10+20 steps nets +10). Results are saved to a timestamped
.npz next to this script.

The piezo is driven in OPEN LOOP via set_voltage(), not closed-loop
move_to()/move_by(): closed-loop position moves on this stage settle very
slowly (confirmed both in our own diagnostics and directly in the Kinesis
GUI), while voltage moves settle quickly. Desired um steps are converted to
volts using a linear calibration (position_um = slope*volts + intercept)
produced by piezo_voltage_calibration.py -- run that script first. Because
open-loop driving is feedforward (no position feedback correcting the
command), we still record the actual sensor-measured position
(get_position()) after every step as ground truth -- the position sensor
keeps reporting real position regardless of control mode.

Run from the repo root so `instruments` is importable:
    python scripts/phase/piezo_voltage_calibration.py   # first, if not already done
    python scripts/phase/piezo_step_response.py
"""

import contextlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.precisionpiezo import PrecisionPiezoCT1P
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk

# --------------------------------------------------------------------------- #
# Settings -- edit before running                                             #
# --------------------------------------------------------------------------- #
CAMERA_SERIAL = "35596"
PIEZO_SERIAL = "92100065"

EXPOSURE_US = 300000  # 300 ms
GAIN = 0
BLACK_LEVEL = 0

BASELINE_SECONDS = 20.0

N_STEPS = 20
PRE_OFFSET_STEPS = 10  # per block: move back this many steps before the imaged sweep
STEP_SIZES_UM = [0.2, 0.5, 1.0]

# Path to a calibration .npz from piezo_voltage_calibration.py. If None, the
# most recently modified piezo_voltage_calibration_*.npz next to this script
# is used automatically.
CALIBRATION_PATH = None

# Settle-detection after each voltage step (voltage moves settle fast).
POLL_INTERVAL_S = 0.02
N_STABLE = 10
STABLE_TOL_UM = 0.01
STABLE_TIMEOUT_S = 5.0


@contextlib.contextmanager
def _armed_camera(camera: ThorlabsCamera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()


def _latest_calibration() -> Path:
    candidates = sorted(Path(__file__).resolve().parent.glob("piezo_voltage_calibration_*.npz"))
    if not candidates:
        raise RuntimeError(
            "No piezo_voltage_calibration_*.npz found next to this script. "
            "Run scripts/phase/piezo_voltage_calibration.py first, or set "
            "CALIBRATION_PATH explicitly."
        )
    return candidates[-1]


def _wait_stable(piezo: PrecisionPiezoCT1P) -> tuple[float, bool]:
    """Poll get_position() until N_STABLE consecutive samples agree within
    STABLE_TOL_UM, or STABLE_TIMEOUT_S elapses. Returns (final_position, settled)."""
    deadline = time.monotonic() + STABLE_TIMEOUT_S
    window: list[float] = []
    pos = piezo.get_position()
    while time.monotonic() < deadline:
        pos = piezo.get_position()
        window.append(pos)
        if len(window) > N_STABLE:
            window.pop(0)
        if len(window) == N_STABLE and (max(window) - min(window)) <= STABLE_TOL_UM:
            return pos, True
        time.sleep(POLL_INTERVAL_S)
    return pos, False


def _simulate_trajectory(start_position_um: float, start_voltage: float, slope: float):
    """Simulate the full planned position/voltage trajectory (including every
    block's backward pre-offset and forward sweep) to find the worst-case
    min/max excursion, for the pre-flight safety check."""
    positions = [start_position_um]
    volts = [start_voltage]
    p, v = start_position_um, start_voltage
    for step_um in STEP_SIZES_UM:
        step_volts = step_um / slope
        for _ in range(PRE_OFFSET_STEPS):
            p -= step_um
            v -= step_volts
            positions.append(p)
            volts.append(v)
        for _ in range(N_STEPS):
            p += step_um
            v += step_volts
            positions.append(p)
            volts.append(v)
    return positions, volts


if __name__ == "__main__":
    t_start = time.monotonic()

    calibration_path = Path(CALIBRATION_PATH) if CALIBRATION_PATH else _latest_calibration()
    calib = np.load(calibration_path)
    slope = float(calib["slope"])
    intercept = float(calib["intercept"])
    print(f"Using calibration {calibration_path.name}: "
          f"position_um = {slope:.4f} * volts + {intercept:.4f}")

    camerasdk = create_camera_sdk()
    with ThorlabsCamera(camerasdk, CAMERA_SERIAL) as camera, \
            PrecisionPiezoCT1P(serial=PIEZO_SERIAL) as piezo:

        camera.set_settings(
            exposure_time_us=EXPOSURE_US,
            gain=GAIN,
            black_level=BLACK_LEVEL,
            bit_depth=np.uint16,
            out_bit_depth=np.uint16,
        )

        H, W = camera.image_shape
        C = 1 if camera.cam_type == 'MU' else 3

        piezo.set_open_loop()
        time.sleep(0.2)
        min_volts = piezo.get_min_voltage()
        max_volts = piezo.get_max_voltage()

        start_position_um = piezo.get_position()
        start_voltage = piezo.get_voltage()

        # ------------------------------------------------------------------
        # Pre-flight safety check: simulate the ENTIRE planned trajectory
        # (every block's backward pre-offset and forward sweep) and make sure
        # its worst-case min/max never leaves [0, max_travel_um] or
        # [min_volts, max_volts]. A simple "final position" check isn't
        # enough here since each block dips backward before climbing forward.
        # ------------------------------------------------------------------
        sim_positions, sim_volts = _simulate_trajectory(start_position_um, start_voltage, slope)
        pos_lo, pos_hi = min(sim_positions), max(sim_positions)
        volt_lo, volt_hi = min(sim_volts), max(sim_volts)
        print(f"Planned trajectory spans {pos_lo:.2f}-{pos_hi:.2f} um, "
              f"{volt_lo:.2f}-{volt_hi:.2f} V")
        if pos_lo < 0 or pos_hi > piezo.max_travel_um:
            raise RuntimeError(
                f"Planned trajectory would reach {pos_lo:.2f}-{pos_hi:.2f} um, outside "
                f"[0, {piezo.max_travel_um:.1f}] um. Move to a different starting position "
                f"before running this script."
            )
        if volt_lo < min_volts or volt_hi > max_volts:
            raise RuntimeError(
                f"Planned trajectory would drive the output voltage to "
                f"{volt_lo:.2f}-{volt_hi:.2f} V, outside [{min_volts:.2f}, {max_volts:.2f}] V. "
                f"Move to a different starting voltage before running this script."
            )

        with _armed_camera(camera):
            # ------------------------------------------------------------
            # 1. Baseline: hold voltage constant, record images for
            #    BASELINE_SECONDS.
            # ------------------------------------------------------------
            print(f"Baseline: recording for {BASELINE_SECONDS:.0f}s at "
                  f"{start_position_um:.2f} um ({start_voltage:.2f} V) ...")
            baseline_images = []
            baseline_positions = []
            baseline_times = []
            baseline_deadline = time.monotonic() + BASELINE_SECONDS
            while time.monotonic() < baseline_deadline:
                baseline_images.append(camera.get_image(num_frames_to_average=1))
                baseline_positions.append(piezo.get_position())
                baseline_times.append(time.monotonic() - t_start)
            baseline_images = np.stack(baseline_images, axis=0).astype(np.uint16)
            baseline_positions = np.array(baseline_positions, dtype=np.float64)
            baseline_times = np.array(baseline_times, dtype=np.float64)
            print(f"  captured {len(baseline_times)} frames")

            # ------------------------------------------------------------
            # 2. Step sweep: for each step size, first move PRE_OFFSET_STEPS
            #    backward (not imaged), then N_STEPS forward via voltage,
            #    recording one image after each forward step. Cumulative --
            #    each block starts wherever the previous block ended.
            # ------------------------------------------------------------
            n_blocks = len(STEP_SIZES_UM)
            step_images = np.zeros((n_blocks, N_STEPS, H, W, C), dtype=np.uint16)
            step_target_um = np.zeros((n_blocks, N_STEPS), dtype=np.float64)
            step_actual_um = np.zeros((n_blocks, N_STEPS), dtype=np.float64)
            step_voltages = np.zeros((n_blocks, N_STEPS), dtype=np.float64)
            step_times = np.zeros((n_blocks, N_STEPS), dtype=np.float64)
            step_settled = np.zeros((n_blocks, N_STEPS), dtype=bool)

            preoffset_target_um = np.zeros((n_blocks, PRE_OFFSET_STEPS), dtype=np.float64)
            preoffset_actual_um = np.zeros((n_blocks, PRE_OFFSET_STEPS), dtype=np.float64)
            preoffset_voltages = np.zeros((n_blocks, PRE_OFFSET_STEPS), dtype=np.float64)
            preoffset_settled = np.zeros((n_blocks, PRE_OFFSET_STEPS), dtype=bool)

            running_target_um = start_position_um
            running_voltage = start_voltage
            for b, step_um in enumerate(STEP_SIZES_UM):
                step_volts = step_um / slope
                print(f"Step block {b + 1}/{n_blocks}: step size {step_um} um "
                      f"({step_volts:+.4f} V)")

                print(f"  pre-offsetting back {PRE_OFFSET_STEPS} x {step_um} um ...")
                for k in range(PRE_OFFSET_STEPS):
                    running_target_um -= step_um
                    running_voltage -= step_volts
                    running_voltage = max(min_volts, min(max_volts, running_voltage))
                    piezo.set_voltage(running_voltage)
                    pos, settled = _wait_stable(piezo)
                    preoffset_target_um[b, k] = running_target_um
                    preoffset_actual_um[b, k] = pos
                    preoffset_voltages[b, k] = running_voltage
                    preoffset_settled[b, k] = settled

                print(f"  {N_STEPS} forward steps ...")
                for s in range(N_STEPS):
                    running_target_um += step_um
                    running_voltage += step_volts
                    running_voltage = max(min_volts, min(max_volts, running_voltage))
                    piezo.set_voltage(running_voltage)

                    pos, settled = _wait_stable(piezo)

                    step_target_um[b, s] = running_target_um
                    step_actual_um[b, s] = pos
                    step_voltages[b, s] = running_voltage
                    step_times[b, s] = time.monotonic() - t_start
                    step_settled[b, s] = settled
                    step_images[b, s] = camera.get_image(num_frames_to_average=1)

                    if (s + 1) % 5 == 0 or (s + 1) == N_STEPS:
                        flag = "" if settled else " (unsettled)"
                        print(f"    step {s + 1}/{N_STEPS} -> {pos:.3f} um{flag}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"piezo_step_response_{timestamp}.npz"

    np.savez(
        save_path,
        baseline_images=baseline_images,        # (N_base, H, W, C) uint16
        baseline_positions=baseline_positions,  # (N_base,) um
        baseline_times=baseline_times,          # (N_base,) s since start
        step_images=step_images,                # (n_blocks, N_STEPS, H, W, C) uint16
        step_sizes_um=np.array(STEP_SIZES_UM),  # (n_blocks,)
        step_target_um=step_target_um,          # (n_blocks, N_STEPS) voltage-implied target
        step_actual_um=step_actual_um,          # (n_blocks, N_STEPS) sensor-measured actual
        step_voltages=step_voltages,            # (n_blocks, N_STEPS) commanded volts
        step_times=step_times,                  # (n_blocks, N_STEPS) s since start
        step_settled=step_settled,              # (n_blocks, N_STEPS) bool
        preoffset_target_um=preoffset_target_um,    # (n_blocks, PRE_OFFSET_STEPS)
        preoffset_actual_um=preoffset_actual_um,    # (n_blocks, PRE_OFFSET_STEPS)
        preoffset_voltages=preoffset_voltages,      # (n_blocks, PRE_OFFSET_STEPS)
        preoffset_settled=preoffset_settled,        # (n_blocks, PRE_OFFSET_STEPS) bool
        pre_offset_steps=PRE_OFFSET_STEPS,
        start_position_um=start_position_um,
        start_voltage=start_voltage,
        calibration_slope=slope,
        calibration_intercept=intercept,
        calibration_path=str(calibration_path),
        n_steps=N_STEPS,
        baseline_seconds=BASELINE_SECONDS,
        exposure_us=EXPOSURE_US,
        gain=GAIN,
        black_level=BLACK_LEVEL,
        camera_serial=CAMERA_SERIAL,
        piezo_serial=PIEZO_SERIAL,
        max_travel_um=piezo.max_travel_um,
        min_voltage=min_volts,
        max_voltage=max_volts,
    )

    size_mb = save_path.stat().st_size / 1e6
    print(f"Saved piezo step-response data ({size_mb:.1f} MB) -> {save_path}")
