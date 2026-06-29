from src.config import Config
from src.control import Control

if __name__ == "__main__":
    # Initialize
    config = Config.from_yaml("src\config\config.yaml")
    control = Control.from_config(config)
    
    # Calibration
    #control.brightness_calibration(override=True, use_current_as_initial=True)
    
    # Run measurement
    #control.black_measurement()
    control.reference_measurement()
    #control.sample_measurement()
    
    # Return to the default state
    #control.go_to_default_state()
