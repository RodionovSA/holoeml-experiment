from amplitude.config import Config
from amplitude.control import Control
import time

if __name__ == "__main__":
    # Initialize
    config = Config.from_yaml("amplitude\config\config.yaml")
    control = Control.from_config(config)
    
    # Calibration
    #control.brightness_calibration(override=True, use_current_as_initial=True, xpol=False)
    
    # Run measurement
    control.sample_measurement(xpol=False)
    control.sample_measurement(xpol=True)
    # control.reference_measurement(xpol=True)
    # control.reference_measurement(xpol=False)

    # control.go_to_default_state()
    
    # control.black_measurement(xpol=True)
    # control.black_measurement(xpol=False)
    # Return to the default state
    control.go_to_default_state()
    
