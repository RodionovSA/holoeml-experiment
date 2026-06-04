import time

from src.config import Config
from src.control import Control

if __name__ == "__main__":
    # Initialize
    config = Config.from_yaml("src\config\config.yaml")
    control = Control.from_config(config)

    # Run calibration
    print('Started calibration')
    start = time.time()
    control.brightness_calibration(override=True, use_current_as_initial=True)
    end = time.time()
    print(f'Ended calibration in {end - start} s')