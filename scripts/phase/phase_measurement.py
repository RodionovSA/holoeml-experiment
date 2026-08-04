""" A script for a full phase measurement at a fixed wavelength and polarization"""

import numpy as np
import argparse
from pathlib import Path
import contextlib
import time
from typing import Tuple
import datetime
from dataclasses import asdict

import instruments.config
from instruments.pypcocam import PcoCamera
from instruments.camera import Camera, open_camera
from instruments.kinesismotor import KinesisMotor
from instruments.precisionpiezo import PrecisionPiezoCT1P
from instruments.config import EquipmentConfig
from phase import aia, remove_carrier, subtract_reference
from phase.combine import combine_acquisitions

NUM_PIEZO_STEPS = 15
PIEZO_STEP = 0.1 #V
NUM_AVERAGES = 5
EXPOSURE_MS = 200
REFERENCE_X_POS = 10.0 # mm
REFERENCE_Y_POS = 10.0 # mm

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
              piezo: PrecisionPiezoCT1P,
              piezo_step: float,
              num_piezo_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    init_voltage = piezo.get_voltage()
    
    images = []
    for step in num_piezo_steps:
        piezo.set_voltage(init_voltage + piezo_step*step)
        images.append(camera.get_image()[...,0])
        
    images = np.asarray(images)
    piezo.set_voltage(init_voltage)
    
    res = aia(images, gain="auto")
    print(f"Converged: {res.converged}")
    print(f"Kappa_p: {res.kappa_p}")
    print(f"Kappa_ps: {res.kappa_ps}")
    
    return res.a, res.b, res.phi

def measure_phase(camera: Camera, 
                  piezo: PrecisionPiezoCT1P,
                  piezo_step: float,
                  num_piezo_steps: int,
                  num_averages: int) -> Tuple[np.ndarray, np.ndarray]:
    phi = []
    b = []
    for num in num_averages:
        _, sample_b, sample_phi = get_phase(camera, piezo, piezo_step, num_piezo_steps)
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
        camera = stack.enter_context(PcoCamera(cfg.camera_serial))
        piezo = stack.enter_context(PrecisionPiezoCT1P(cfg.piezo_serial))
        stage_x = stack.enter_context(KinesisMotor(cfg.stage_x_serial))
        stage_y = stack.enter_context(KinesisMotor(cfg.stage_y_serial))

        # Set settings
        camera.set_exposure_ms(EXPOSURE_MS)
        piezo.set_open_loop()
        stage_x.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)
        stage_y.set_velocity(max_velocity=cfg.default_stage_max_velocity,
                            acceleration=cfg.default_stage_acceleration)

        init_x_pos = stage_x.get_position()
        init_y_pos = stage_y.get_position()
        with _armed_camera(camera):
            sample_phi, sample_mean_resultant = measure_phase(camera, piezo, 
                                                            PIEZO_STEP, NUM_PIEZO_STEPS, 
                                                            NUM_AVERAGES)
            stage_x.move_to(REFERENCE_X_POS)
            stage_y.move_to(REFERENCE_Y_POS)
            
            reference_phi, _ = measure_phase(camera, piezo, 
                                            PIEZO_STEP, NUM_PIEZO_STEPS, 
                                            NUM_AVERAGES)

        stage_x.move_to(init_x_pos)
        stage_y.move_to(init_y_pos)
        
    diff = subtract_reference(sample_phi, reference_phi, sample_mean_resultant)
    carrier_res = remove_carrier(diff.phi, sample_mean_resultant,
                                defocus=True, refine_iters=10, n_blocks=10)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"phase_measurement_{timestamp}.npz"
    np.savez(save_path, **asdict(carrier_res))
    print(f"Saved measurement -> {save_path}")

if __name__ == "__main__":
    main()
        
    
    
    
    
    