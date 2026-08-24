""" A script for a full phase measurement at a fixed wavelength and polarization"""

import numpy as np
import argparse
from pathlib import Path
import contextlib
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
        
def get_phase(camera: Camera,
              piezo: KIM101Axis,
              step_size: int,
              num_piezo_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # dry piezo run forward
    for i in range(DRY_NUM):
        piezo.move_by(DRY_STEP)
        time.sleep(SETTLE_S)

    init_position = piezo.get_position()

    images = []
    for step in range(num_piezo_steps):
        images.append(camera.get_image()[IMAGE_SHAPE_Y0:IMAGE_SHAPE_Y1,
                                         IMAGE_SHAPE_X0:IMAGE_SHAPE_X1, 0])
        if step < num_piezo_steps - 1:
            piezo.move_by(step_size)
            time.sleep(SETTLE_S)

    images = np.asarray(images)

    # dry piezo run backward
    for i in range(DRY_NUM):
        piezo.move_by(-DRY_STEP)
        time.sleep(SETTLE_S)

    piezo.move_to(init_position)

    res = aia(images, gain="auto", iters=60)
    print(f"Converged: {res.converged}")
    print(f"Kappa_p: {res.kappa_p}")
    print(f"Kappa_ps: {res.kappa_ps}")
    
    return res.a, res.b, res.phi

def measure_phase(camera: Camera,
                  piezo: KIM101Axis,
                  step_size: int,
                  num_piezo_steps: int,
                  num_averages: int) -> Tuple[np.ndarray, np.ndarray]:
    phi = []
    b = []
    for num in range(num_averages):
        _, sample_b, sample_phi = get_phase(camera, piezo, step_size, num_piezo_steps)
        b.append(sample_b)
        phi.append(sample_phi)
        
    b = np.asarray(b)
    phi = np.asarray(phi)
    
    res = combine_acquisitions(phi, weights=b)
    
    return res.phi, res.mean_resultant
        
def main():
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
                                                            NUM_AVERAGES)
            stage_x.move_by(REFERENCE_X_BY)
            stage_y.move_by(REFERENCE_Y_BY)
            
            reference_phi, _ = measure_phase(camera, piezo, 
                                            STEP_SIZE, NUM_PIEZO_STEPS,
                                            NUM_AVERAGES)

        stage_x.move_to(init_x_pos)
        stage_y.move_to(init_y_pos)
        
    diff = subtract_reference(sample_phi, reference_phi, sample_mean_resultant)
    carrier_res = remove_carrier(diff.phi, sample_mean_resultant,
                                defocus=True, refine_iters=10, n_blocks=10)
    end = time.time()
    print(f"Phase measurement completed in {end-start:.2f} seconds")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"phase_measurement_{timestamp}.npz"
    np.savez(save_path, **asdict(carrier_res))
    print(f"Saved measurement -> {save_path}")

if __name__ == "__main__":
    main()
        
    
    
    
    
    