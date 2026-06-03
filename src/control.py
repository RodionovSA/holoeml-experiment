import numpy as np

from pylablib.devices import Thorlabs
from src.pythorcam.thorcam import ThorlabsCamera
from src.monochromator.mono_class import MonochromatorControl

class Control:
    def __init__(self, 
                 camera: ThorlabsCamera, 
                 monochromator: MonochromatorControl,
                 focus: Thorlabs.KinesisMotor):
        
        self.camera = camera
        self.mono = monochromator
        self.focus = focus

        # camera is already connected in its __init__; only the mono needs initialization here.
        self.mono.initialize_arduino()
        # focus is initialized automatically

    def shutdown(self):
        self.camera.close()
        self.mono.disconnect()
        self.focus.close()

    def __del__(self):
        self.shutdown()

    def set_camera_settings(self,
                            exposure_time: int, 
                            gain: int, 
                            black_level: int, 
                            bit_depth=np.uint16, 
                            out_bit_depth=np.float32) -> None:
        """Exposure time in ms (converted to µs for the camera)."""
        self.camera.set_settings(exposure_time_us=exposure_time * 1000,
                                 gain=gain,
                                 black_level=black_level,
                                 bit_depth=bit_depth,
                                 out_bit_depth=out_bit_depth)
        
    



    