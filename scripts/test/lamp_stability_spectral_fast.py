import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.config import load_equipment
from instruments.spectrometer import Spectrometer
from instruments.spectrometer.utils import autoexposure
from instruments.monochromator.mono import MonochromatorControl


if __name__ == "__main__":
    _eq = load_equipment()

    # Run monochromator
    mono = MonochromatorControl(port=_eq.monochromator_port)
    mono.initialize_arduino()
    mono.home_motor()
    mono.select_grating_mode(mode=_eq.monochromator_grating_mode)
    mono.goto_lamp_baseline()

    # Run spectrometer
    sp = Spectrometer(serial=_eq.spectrometer_serial)
    sp.set_scans_to_average(n=1)
    autoexposure(sp, target=0.5)

    # Capture fixed acquisition settings and wavelength axis before the loop.
    # Integration time is fixed after autoexposure — we want it constant so any
    # drift shows up in counts, not in exposure.
    wavelengths = np.array(sp.get_wavelengths(), dtype=np.float64)  # nm
    integration_time_us = sp.get_integration_time()                 # µs, fixed
    scans_averaged = sp.get_scans_to_average()
    device_info = sp.identify()

    # Measurement parameters
    duration_s = 5 * 60  # 5 minutes

    print(f"Starting fast lamp stability run: as fast as possible for {duration_s} s")
    print(f"  integration time : {integration_time_us} µs")
    print(f"  scans to average : {scans_averaged}")
    print(f"  device           : {device_info}")

    spectra_list = []
    elapsed_list = []

    t0 = time.monotonic()
    start_wall = datetime.now()
    last_reported_sec = -1

    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= duration_s:
                break

            counts = sp.get_intensities()   # list[float], current intensities
            elapsed = time.monotonic() - t0
            spectra_list.append(counts)
            elapsed_list.append(elapsed)

            # Progress roughly once per second
            sec = int(elapsed)
            if sec != last_reported_sec:
                last_reported_sec = sec
                print(f"  {len(spectra_list):5d} samples  t={elapsed:.1f}s")

    finally:
        mono.set_wavelength(550.0)
        sp.close()
        mono.disconnect()

    spectra = np.array(spectra_list, dtype=np.float64)   # (n_samples, n_pixels)
    elapsed_s = np.array(elapsed_list, dtype=np.float64)  # seconds since t0
    n_samples = spectra.shape[0]
    rate_hz = n_samples / elapsed_s[-1] if n_samples and elapsed_s[-1] > 0 else 0.0

    print(f"\nAcquired {n_samples} spectra in {elapsed_s[-1]:.2f}s "
          f"({rate_hz:.2f} Hz effective rate)")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = start_wall.strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'lamp_stability_spectral_fast_{timestamp}.npz'

    description = (
        "Lamp spectral fast-scale stability test. "
        "Monochromator: VIS grating positioned at lamp baseline. "
        "Spectrometer acquires spectra as fast as possible (no averaging, n=1) for "
        "5 minutes at a fixed integration time (set once by autoexposure targeting 0.5 "
        "fill fraction before the run; recorded in 'integration_time_us'). "
        "'spectra' is (n_samples, n_pixels) float64 counts; "
        "'wavelengths' is the nm axis; "
        "'elapsed_s' is seconds since run start for each sample."
    )

    np.savez(
        save_path,
        spectra=spectra,
        wavelengths=wavelengths,
        elapsed_s=elapsed_s,
        description=description,
        start_time=start_wall.isoformat(),
        duration_s=duration_s,
        n_samples=n_samples,
        integration_time_us=integration_time_us,
        scans_averaged=scans_averaged,
        device_info=str(device_info),
    )

    print(f"\nSaved {n_samples} spectra → {save_path}")
