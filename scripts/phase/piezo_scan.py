import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from instruments.precisionpiezo import PrecisionPiezoCT1P
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk

CAMERA_SERIAL = "35596"
PIEZO_SERIAL = "92100065"

EXPOSURE_US = 300000  # 300 ms
GAIN = 0
BLACK_LEVEL = 0

NUM_STEPS = 20
STEP = 0.1 # Volts
FRAMES_TO_AVERAGE = 1

@contextlib.contextmanager
def _armed_camera(camera):
    camera.arm()
    time.sleep(0.1)
    try:
        yield
    finally:
        camera.disarm()

def piezo_scan(camera: ThorlabsCamera,
               piezo: PrecisionPiezoCT1P,
               num_steps: int = 10,
               step: float = 0.1,
               frames_to_average: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    piezo.set_open_loop()
    init_vol = piezo.get_voltage()
    
    images = []
    vol = []
    pos = []
    with _armed_camera(camera):
        for i in range(num_steps):
            piezo.set_voltage(init_vol + step*i)
            images.append(camera.get_image(frames_to_average))
            
            vol.append(piezo.get_voltage())
            pos.append(piezo.get_position())
            
    return np.asarray(images), np.asarray(vol), np.asarray(pos)

if __name__ == "__main__":
    camera_sdk = create_camera_sdk()
    try:
        with ThorlabsCamera(camera_sdk, CAMERA_SERIAL) as camera, \
             PrecisionPiezoCT1P(PIEZO_SERIAL) as piezo:
            camera.set_settings(EXPOSURE_US, GAIN, BLACK_LEVEL)

            piezo.set_open_loop()
            init_vol = piezo.get_voltage()
            print(f"Piezo init voltage: {init_vol} V, position: {piezo.get_position()} um")
            images, vol, pos = piezo_scan(camera, piezo, NUM_STEPS, STEP, FRAMES_TO_AVERAGE)
            print(f"Piezo final voltage: {piezo.get_voltage()} V, position: {piezo.get_position()} um")
            piezo.set_voltage(init_vol)
            print(f"Piezo returnal voltage: {piezo.get_voltage()} V, position: {piezo.get_position()} um")
    finally:
        camera_sdk.dispose()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(__file__).resolve().parent / f"piezo_scan_{timestamp}.npz"
    np.savez(save_path, images=images, volts=vol, positions=pos)
    print(f"Saved scan -> {save_path}")