"""Monochromator band spectral acquisition script.

Sweeps the monochromator across a fixed grid of wavelength bands and records
a full spectrometer spectrum at each band, saving the result as a single
timestamped ``.npz`` (2-D ``spectra`` array: one spectrometer spectrum per
band). Serials, ports, and addresses come from the shared equipment config
(see ``instruments/config``); the sweep parameters below are specific to
this script.

Usage
-----
    python scripts/spectral.py
    python scripts/spectral.py -o /path/to/output_dir
    python scripts/spectral.py -c /path/to/equipment_config.yaml
"""

import argparse
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.config import load_equipment
from instruments.filterwheel import FilterWheelControl
from instruments.monochromator.mono import MonochromatorControl
from instruments.spectrometer import Spectrometer
from instruments.spectrometer.utils import autoexposure

# ── Sweep parameters ─────────────────────────────────────────────────────────
NUM_AVERAGE = 10        # spectrometer hardware scans averaged per band
EXPOSURE_TARGET = 0.6   # autoexposure target fill fraction (set once, before the sweep)

WVL_START = 400.0       # nm
WVL_END = 1000.0        # nm
NUM_WVL = 121            # number of monochromator bands (linspace)
FILTER_WVL = 600.0      # nm — long-pass filter inserted above this wavelength
DEFAULT_WVL = 550.0     # nm — resting wavelength before/after the sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, default=None,
                   help="path to the equipment config YAML "
                        "(defaults to instruments/config/config.yaml)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path.cwd(),
                   help="directory to save the output .npz into (default: cwd)")
    args = p.parse_args()

    eq = load_equipment(args.config)

    with contextlib.ExitStack() as stack:
        filterwheel = stack.enter_context(FilterWheelControl(eq.filterwheel_address))
        monochromator = stack.enter_context(MonochromatorControl(eq.monochromator_port))
        sp = stack.enter_context(Spectrometer(eq.spectrometer_serial))

        # Base setup
        filterwheel.set_position(eq.filterwheel_empty_pos)
        monochromator.initialize_arduino()
        monochromator.home_motor()
        monochromator.select_grating_mode(eq.monochromator_grating_mode)
        monochromator.set_wavelength(DEFAULT_WVL)

        integration_time_us = autoexposure(sp, target=EXPOSURE_TARGET)
        sp.set_scans_to_average(NUM_AVERAGE)
        device_info = sp.identify()

        setpoint_wavelengths = np.linspace(WVL_START, WVL_END, NUM_WVL)
        spectrometer_wavelengths = np.asarray(sp.get_wavelengths())
        spectra = np.empty((NUM_WVL, spectrometer_wavelengths.size), dtype=np.float64)

        print(f"Sweeping {NUM_WVL} bands from {WVL_START:.0f} to {WVL_END:.0f} nm "
              f"(integration time {integration_time_us} us, {NUM_AVERAGE} scans averaged)")

        try:
            for i, wvl in enumerate(setpoint_wavelengths):
                if wvl >= FILTER_WVL:
                    filterwheel.set_position(eq.longpass_pos)
                monochromator.set_wavelength(wvl)
                spectra[i] = sp.get_intensities()

                if (i + 1) % 10 == 0 or i + 1 == NUM_WVL:
                    print(f"  {i + 1:3d}/{NUM_WVL}  wavelength={wvl:.1f} nm")
        finally:
            # Always return the stage to rest, even if the sweep was interrupted.
            filterwheel.set_position(eq.filterwheel_empty_pos)
            monochromator.set_wavelength(DEFAULT_WVL)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / f'spectral_bands_{timestamp}.npz'

    description = (
        "Monochromator band spectral sweep. "
        "'spectra' is (n_bands, n_pixels) counts, one spectrometer spectrum per "
        "monochromator setpoint; 'setpoint_wavelengths' (nm) is the monochromator "
        "band axis indexing spectra's first dimension; 'spectrometer_wavelengths' "
        "(nm) is the spectrometer's fixed pixel axis indexing spectra's second "
        "dimension. Integration time is fixed for the whole sweep (set once by "
        "autoexposure beforehand; recorded in 'integration_time_us')."
    )

    np.savez(
        save_path,
        spectra=spectra,
        setpoint_wavelengths=setpoint_wavelengths,
        spectrometer_wavelengths=spectrometer_wavelengths,
        description=description,
        timestamp=timestamp,
        integration_time_us=integration_time_us,
        scans_averaged=NUM_AVERAGE,
        exposure_target=EXPOSURE_TARGET,
        filter_wvl=FILTER_WVL,
        device_info=str(device_info),
    )

    print(f"\nSaved {NUM_WVL} spectra -> {save_path}")


if __name__ == "__main__":
    main()
