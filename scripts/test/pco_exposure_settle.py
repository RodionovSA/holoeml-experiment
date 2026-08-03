"""Regression check for the pco exposure-settle barrier (`PcoCamera.EXPOSURE_SETTLE_FRAMES`
/ `FLUSH_SETTLE_FRAMES` in `instruments/pypcocam/pcocam.py`).

Models the exposure-alternation experiment from `scripts/test/test_pco_camera.ipynb`
that first exposed the bug: with the pre-fix driver, alternating
`set_exposure_ms(200)` / `set_exposure_ms(1000)` on an armed camera and grabbing
immediately (`get_image()`, zero dropped frames) returned a frame from the
*previous* exposure setting every time -- a 200 ms request saturated (it was
really a completed 1000 ms frame) and a 1000 ms request came back under-exposed
(it was really a 200 ms frame).

This script reproduces that exact call pattern -- `set_exposure_ms` immediately
followed by `get_image(num_frames_to_drop=0)`, no manual sleep -- and asserts
every returned frame matches the exposure that was actually requested. The
settle barrier (not any sleep in this script) is what has to make that true.

Exit code is 0 if every grab matched its requested exposure, 1 otherwise.

Usage
-----
    python scripts/test/pco_exposure_settle.py                  # regression check
    python scripts/test/pco_exposure_settle.py --settle-frames  # calibrate EXPOSURE_SETTLE_FRAMES
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from instruments.config import load_equipment
from instruments.pypcocam.pcocam import PcoCamera

_EQ = load_equipment()
SERIAL = _EQ.camera_serial

ROI = 1000              # centered ROI, pixels (see PcoCamera.set_roi)
SHORT_MS = 200.0
LONG_MS = 1800.0
CYCLES = 4               # alternations per direction in --check mode
REFERENCE_NUM_FRAMES_TO_AVERAGE = 1

# A grab is accepted as matching a reference mean if within this fraction of it.
# Loose enough to tolerate lamp drift across the run; tight enough that a
# one-exposure-setting lag (which changes the mean by SHORT_MS/LONG_MS = 5x
# in this script) can't pass by accident.
MATCH_TOLERANCE_FRAC = 0.15


def _reference_mean(camera: PcoCamera, exposure_ms: float) -> tuple[float, float]:
    """Set *exposure_ms* and wait out several full exposure periods before
    measuring -- long enough to be trustworthy independent of the settle
    barrier under test, so the reference itself isn't circular."""
    camera.set_exposure_ms(exposure_ms)
    time.sleep(3 * exposure_ms / 1000 + 1.0)
    frame = camera.get_image(num_frames_to_average=REFERENCE_NUM_FRAMES_TO_AVERAGE)
    return float(frame.mean()), float(frame.std())


def _run_check(camera: PcoCamera, cycles: int, ref_short: float, ref_long: float,
                label: str = "") -> bool:
    """Alternate SHORT_MS/LONG_MS with immediate, undelayed grabs. Returns True
    iff every grab matched the reference mean for the exposure it requested."""
    ok = True
    print(f"\n{label}{'cycle':>6}  {'requested':>10}  {'mean':>10}  {'std':>8}  {'expected':>10}  result")
    for cycle in range(cycles):
        for exposure_ms, ref_mean in ((SHORT_MS, ref_short), (LONG_MS, ref_long)):
            camera.set_exposure_ms(exposure_ms)
            frame = camera.get_image(num_frames_to_drop=0)
            mean, std = float(frame.mean()), float(frame.std())
            passed = abs(mean - ref_mean) <= MATCH_TOLERANCE_FRAC * ref_mean
            ok &= passed
            print(f"{label}{cycle:>6}  {exposure_ms:>9.0f}ms  {mean:>10.1f}  {std:>8.1f}  "
                  f"{ref_mean:>10.1f}  {'PASS' if passed else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settle-frames", action="store_true",
                         help="Sweep PcoCamera.EXPOSURE_SETTLE_FRAMES over 0..3 and "
                              "report the smallest value that passes, instead of "
                              "running the regression check at the class default.")
    args = parser.parse_args()

    with PcoCamera(SERIAL) as camera:
        sensor_h, sensor_w = camera.sensor_shape
        x0 = (sensor_w - ROI) // 2 + 1     # 1-based, inclusive; see PcoCamera.set_roi
        y0 = (sensor_h - ROI) // 2 + 1
        camera.set_roi(x0, y0, x0 + ROI - 1, y0 + ROI - 1)
        camera.arm()
        time.sleep(0.1)

        print("Measuring reference means (generous settle, not using the barrier under test)...")
        ref_short, std_short = _reference_mean(camera, SHORT_MS)
        ref_long, std_long = _reference_mean(camera, LONG_MS)
        print(f"  reference short ({SHORT_MS:.0f}ms): mean={ref_short:.1f} std={std_short:.1f}")
        print(f"  reference long  ({LONG_MS:.0f}ms): mean={ref_long:.1f} std={std_long:.1f}")

        if ref_long <= (1 + MATCH_TOLERANCE_FRAC) * ref_short:
            print("ERROR: reference means for SHORT_MS/LONG_MS are not well separated "
                  "(check illumination / exposure range before trusting this script).")
            camera.disarm()
            return 1

        if not args.settle_frames:
            passed = _run_check(camera, CYCLES, ref_short, ref_long)
            camera.disarm()
            if passed:
                print("\nPASS: every grab matched the exposure that was actually requested.")
                return 0
            print("\nFAIL: at least one grab returned a stale frame from the previous "
                  "exposure setting -- the settle barrier did not hold.")
            return 1

        print("\nCalibrating PcoCamera.EXPOSURE_SETTLE_FRAMES (default "
              f"{PcoCamera.EXPOSURE_SETTLE_FRAMES})...")
        original = PcoCamera.EXPOSURE_SETTLE_FRAMES
        try:
            for k in range(4):
                PcoCamera.EXPOSURE_SETTLE_FRAMES = k
                passed = _run_check(camera, cycles=2, ref_short=ref_short, ref_long=ref_long,
                                     label=f"[K={k}] ")
                print(f"  K={k}: {'PASS' if passed else 'FAIL'}")
                if passed:
                    print(f"\nSmallest passing EXPOSURE_SETTLE_FRAMES = {k} "
                          f"(class default is {original}).")
                    camera.disarm()
                    return 0
            print(f"\nNo K in 0..3 passed reliably; keep the class default ({original}) "
                  "or investigate further.")
            camera.disarm()
            return 1
        finally:
            PcoCamera.EXPOSURE_SETTLE_FRAMES = original


if __name__ == "__main__":
    raise SystemExit(main())
