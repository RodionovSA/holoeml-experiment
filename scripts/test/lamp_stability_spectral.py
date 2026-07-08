import time
from datetime import datetime
from pathlib import Path

import numpy as np

from instruments.spectrometer import Spectrometer
from instruments.spectrometer.utils import autoexposure
from instruments.monochromator.mono import MonochromatorControl


if __name__ == "__main__":
    # Run monochromator
    mono = MonochromatorControl(port='COM4')
    mono.initialize_arduino()
    mono.home_motor()
    mono.select_grating_mode(mode="VIS Grating")
    mono.goto_lamp_baseline()

    # Run spectrometer
    sp = Spectrometer(serial="SR600410")
    sp.set_scans_to_average(n=1)
    # autoexposure(sp, target=0.6)
    sp.set_integration_time(10230)
    sp.set_scans_to_average(n=10)

    # Capture fixed acquisition settings and wavelength axis before the loop.
    # Integration time is fixed after autoexposure — we want it constant so any
    # drift shows up in counts, not in exposure.
    wavelengths = np.array(sp.get_wavelengths(), dtype=np.float64)  # nm
    integration_time_us = sp.get_integration_time()                 # µs, fixed
    scans_averaged = sp.get_scans_to_average()
    device_info = sp.identify()

    # Measurement parameters
    interval_s = 1.0
    duration_s = 240 * 60                             # 240 minutes
    n_samples = int(duration_s / interval_s)         # 14400

    spectra = np.empty((n_samples, wavelengths.size), dtype=np.float64)
    elapsed = np.empty(n_samples, dtype=np.float64)  # seconds since t0

    print(f"Starting lamp stability run: {n_samples} spectra over {duration_s / 60:.0f} min")
    print(f"  integration time : {integration_time_us} µs")
    print(f"  scans to average : {scans_averaged}")
    print(f"  device           : {device_info}")

    t0 = time.monotonic()
    start_wall = datetime.now()

    try:
        for i in range(n_samples):
            # Drift-free scheduling: sleep until the absolute target time
            target = t0 + i * interval_s
            dt = target - time.monotonic()
            if dt > 0:
                time.sleep(dt)

            counts = sp.get_intensities()   # list[float], current intensities
            elapsed[i] = time.monotonic() - t0
            spectra[i] = counts

            # Warn if this sample fell outside its 1-second slot
            if elapsed[i] > (i + 1) * interval_s:
                print(f"[warn] sample {i:4d} late: t={elapsed[i]:.3f}s "
                      f"(expected ≤{(i + 1) * interval_s:.1f}s)")

            # Progress every 60 samples (~1 min)
            if (i + 1) % 60 == 0:
                pct = 100 * (i + 1) / n_samples
                print(f"  {i + 1:4d}/{n_samples} ({pct:.0f}%)  t={elapsed[i]:.1f}s")

    finally:
        mono.set_wavelength(550.0)
        sp.close()
        mono.disconnect()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    timestamp = start_wall.strftime('%Y%m%d_%H%M%S')
    save_path = Path.cwd() / f'lamp_stability_spectral_{timestamp}.npz'

    description = (
        "Lamp spectral stability test. "
        "Monochromator: VIS grating positioned at lamp baseline. "
        "Spectrometer acquires one spectrum per second for 4 hours at a fixed "
        "integration time (set once by autoexposure before the run; recorded in "
        "'integration_time_us'). "
        "'spectra' is (n_samples, n_pixels) float64 counts; "
        "'wavelengths' is the nm axis; "
        "'elapsed_s' is seconds since run start for each sample."
    )

    np.savez(
        save_path,
        spectra=spectra,
        wavelengths=wavelengths,
        elapsed_s=elapsed,
        description=description,
        start_time=start_wall.isoformat(),
        interval_s=interval_s,
        duration_s=duration_s,
        integration_time_us=integration_time_us,
        scans_averaged=scans_averaged,
        device_info=str(device_info),
    )

    print(f"\nSaved {n_samples} spectra → {save_path}")
