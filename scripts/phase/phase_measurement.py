""" A script for a full phase measurement at a fixed wavelength and polarization"""

import numpy as np
import argparse
from pathlib import Path
import contextlib
import concurrent.futures
import time
from typing import Tuple
from datetime import datetime
from dataclasses import asdict

import instruments.config
from instruments.camera import Camera, open_camera
from instruments.kinesismotor import KinesisMotor
from instruments.inertialpiezo import KIM101, KIM101Axis
from instruments.config import EquipmentConfig
from phase import aia, remove_carrier, subtract_reference
from phase.backend import asnumpy
from phase.combine import combine_acquisitions

NUM_PIEZO_STEPS = 20
STEP_SIZE = 5  
SETTLE_S = 0.1  # pause after each step before capturing (mechanical settling)
DRY_NUM = 10
DRY_STEP = 1

DRIVE_STEP_RATE = 200          # steps/s
DRIVE_STEP_ACCELERATION = 500 # steps/s^2
DRIVE_MAX_VOLTAGE = 125       # V

NUM_AVERAGES = 3
EXPOSURE_MS = 250
GAIN = 100
REFERENCE_X_BY = 0.0 # mm
REFERENCE_Y_BY = 0.8 # mm
IMAGE_SHAPE_X0 = 300 
IMAGE_SHAPE_Y0 = 400 
IMAGE_SHAPE_X1 = 3600 
IMAGE_SHAPE_Y1 = 2600 

CONFIG_ROOT = Path(instruments.config.__file__).resolve().parent

@contextlib.contextmanager
def _armed_camera(camera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()
        
def _acquire_repeat(camera: Camera,
                     piezo: KIM101Axis,
                     step_size: int,
                     num_piezo_steps: int) -> np.ndarray:
    """One phase-shift sweep's worth of hardware I/O for a single repeat:
    dry-run backlash compensation, the frame-capture loop, dry-run back, and
    return to the start position. Kept on the caller's thread -- the
    piezo/camera driver calls aren't meant to be used concurrently -- while
    `measure_phase` overlaps this with the (CPU/GPU-bound) `aia` solve of
    the *previous* repeat in a background thread instead.
    """

    images = []
    for step in range(num_piezo_steps):
        images.append(camera.get_image()[IMAGE_SHAPE_Y0:IMAGE_SHAPE_Y1,
                                         IMAGE_SHAPE_X0:IMAGE_SHAPE_X1, 0])
        if step < num_piezo_steps - 1:
            piezo.move_by(step_size)
            time.sleep(SETTLE_S)

    images = np.asarray(images)

    return images

def _process_repeat(images: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run AIA on one repeat's frame stack. Meant to be submitted to a
    background thread (see `measure_phase`) so it overlaps with the next
    repeat's hardware acquisition rather than leaving the piezo/camera idle
    for the whole solve."""
    res = aia(images, gain="auto", iters=60, device=device)
    print(f"Converged: {res.converged}")
    print(f"Kappa_p: {res.kappa_p}")
    print(f"Kappa_ps: {res.kappa_ps}")
    return res.a, res.b, res.phi

def measure_phase(camera: Camera,
                  piezo: KIM101Axis,
                  step_size: int,
                  num_piezo_steps: int,
                  num_averages: int,
                  device: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Acquire `num_averages` independent repeats and combine their AIA phase.

    Each repeat's `aia` solve (`_process_repeat`) runs in a background
    thread while the *next* repeat's dry-run + sweep (`_acquire_repeat`)
    proceeds on this thread, instead of the piezo/camera sitting idle for
    the whole solve. A single-worker pool is enough since only one solve is
    ever in flight; with `device="cuda"` that solve becomes fast enough to
    mostly hide behind the ~(2*DRY_NUM + num_piezo_steps)*SETTLE_S seconds
    of hardware time each repeat already takes.
    """
    init_position = piezo.get_position()
    
    # dry piezo run forward
    for i in range(DRY_NUM):
        piezo.move_by(DRY_STEP)
        time.sleep(SETTLE_S)
    
    phi_list, b_list = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for _ in range(num_averages):
            images = _acquire_repeat(camera, piezo, step_size, num_piezo_steps)
            if pending is not None:
                _, prev_b, prev_phi = pending.result()
                b_list.append(prev_b)
                phi_list.append(prev_phi)
            pending = pool.submit(_process_repeat, images, device)
        _, last_b, last_phi = pending.result()
        b_list.append(last_b)
        phi_list.append(last_phi)

    b = np.asarray([asnumpy(x) for x in b_list])
    phi = np.asarray([asnumpy(x) for x in phi_list])

    res = combine_acquisitions(phi, weights=b, device=device)
    if res.sign_flips:
        print(f"Sign flips among repeats: {res.sign_flips}")

    # dry piezo run backward
    for i in range(DRY_NUM):
        piezo.move_by(-DRY_STEP)
        time.sleep(SETTLE_S)

    piezo.move_to(init_position)

    return res.phi, res.mean_resultant
        
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Where to run the AIA/carrier/reference solves "
                             "(see phase.backend) -- 'auto' uses a GPU if one "
                             "is installed and available, else the CPU.")
    args = parser.parse_args()
    device = args.device

    cfg = EquipmentConfig.from_yaml(CONFIG_ROOT / "config.yaml")

    with contextlib.ExitStack() as stack:
        # Init devices
        camera = open_camera(cfg, stack)
        stack.enter_context(camera)
        kim = stack.enter_context(KIM101(cfg.kim_serial))
        piezo = kim.axis(cfg.kim_channel)
        stage_x = stack.enter_context(KinesisMotor(cfg.stage_x_serial))
        stage_y = stack.enter_context(KinesisMotor(cfg.stage_y_serial))

        # Set settings
        camera.set_exposure_ms(EXPOSURE_MS)
        if camera.supports_gain:
            camera.set_gain(GAIN)
        piezo.set_drive_parameters(
            step_rate=DRIVE_STEP_RATE,
            step_acceleration=DRIVE_STEP_ACCELERATION,
            max_voltage=DRIVE_MAX_VOLTAGE,
        )
        print(f"Drive parameters applied: {piezo.get_drive_parameters()}")
        stage_x.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)
        stage_y.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)

        init_x_pos = stage_x.get_position()
        init_y_pos = stage_y.get_position()
        start = time.time()
        with _armed_camera(camera):
            sample_phi, sample_mean_resultant = measure_phase(camera, piezo,
                                                            STEP_SIZE, NUM_PIEZO_STEPS,
                                                            NUM_AVERAGES, device=device)
            stage_x.move_by(REFERENCE_X_BY)
            stage_y.move_by(REFERENCE_Y_BY)

            reference_phi, reference_mean_resultant = measure_phase(camera, piezo,
                                            STEP_SIZE, NUM_PIEZO_STEPS,
                                            NUM_AVERAGES, device=device)

        stage_x.move_to(init_x_pos)
        stage_y.move_to(init_y_pos)

    # a pixel unreliable in *either* acquisition is unreliable in the
    # difference -- weight both the sign resolution and carrier fit by the
    # joint reliability rather than the sample's alone.
    weight = sample_mean_resultant * reference_mean_resultant

    diff = subtract_reference(sample_phi, reference_phi, weight, device=device)
    print(f"Sign branch: sign={diff.sign:+d} ambiguous={diff.ambiguous} "
          f"spread_same={diff.spread_same:.4f} spread_flipped={diff.spread_flipped:.4f}")
    carrier_res = remove_carrier(diff.phi, weight,
                                defocus=True, refine_iters=10, n_blocks=10, device=device)
    end = time.time()
    print(f"Phase measurement completed in {end-start:.2f} seconds")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"phase_measurement_{timestamp}.npz"
    # bring the one array-valued field back to the host explicitly -- np.savez
    # can't serialize a cupy array, and dataclasses.asdict won't do that
    # transfer on its own.
    out = asdict(carrier_res)
    out["phi"] = asnumpy(out["phi"])
    out["sign"] = diff.sign
    out["ambiguous"] = diff.ambiguous
    out["spread_same"] = diff.spread_same
    out["spread_flipped"] = diff.spread_flipped
    np.savez(save_path, **out)
    print(f"Saved measurement -> {save_path}")

if __name__ == "__main__":
    main()
        
    
    
    
    
    