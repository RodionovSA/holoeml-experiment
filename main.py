import time

from src.config import Config
from src.control import Control

if __name__ == "__main__":
    # Initialize
    config = Config.from_yaml("src\config\config.yaml")
    control = Control.from_config(config)

    # Run reference measurement
    start = time.time()
    control.reference_measurement()
    end = time.time()

    print(end - start)