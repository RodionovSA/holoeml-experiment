"""CLI for running amplitude measurements and brightness calibration.

Loads a Config from YAML, builds a Control from it, optionally runs
brightness calibration, then runs the requested measurements (in order)
for each selected polarization. Returns the setup to its default state
after each measurement type and before exiting.

Examples
--------
Measure sample and reference for both polarizations (default --pol x y):
    python main.py --measure sample reference

Calibrate x-polarization only, then run all three measurements:
    python main.py --calibrate --pol x --measure sample reference black

Preview the sequence without touching hardware:
    python main.py --calibrate --measure sample reference --dry-run

Use a non-default config file:
    python main.py -c /path/to/config.yaml --measure sample
"""

import argparse
from pathlib import Path

import amplitude
from amplitude.config import Config
from amplitude.control import Control

PKG_ROOT = Path(amplitude.__file__).resolve().parent  # .../holoeml-experiment/amplitude

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path,
                   default=PKG_ROOT / "config/config.yaml")
    p.add_argument("-m", "--measure", nargs="+", default=[],
                   choices=["sample", "reference", "black"],
                   help="measurements to run, in order")
    p.add_argument("--pol", nargs="+", default=["x", "y"], choices=["x", "y"])
    p.add_argument('--calibrate', action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args=p.parse_args()
    
    config = Config.from_yaml(str(args.config))
    control = Control.from_config(config)
    xpols = [pol == "x" for pol in args.pol]
    
    if args.calibrate:
        for xpol in xpols:
            print(f"calibrating xpol={xpol}")
            if not args.dry_run:
                control.brightness_calibration(
                    override=True, use_current_as_initial=True, xpol=xpol)

    for name in args.measure:
        method = getattr(control, f"{name}_measurement")
        for xpol in xpols:
            print(f"{name} xpol={xpol}")
            if not args.dry_run:
                method(xpol=xpol)
        control.go_to_default_state()

    control.go_to_default_state()

if __name__ == "__main__":
    main()
