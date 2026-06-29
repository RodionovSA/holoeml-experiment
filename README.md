# holoeml-amp

A Python control system for wavelength-swept polarization-resolved optical spectroscopy. It orchestrates a monochromator, camera, motorized focus stage, polarizer rotation stage, and filter wheel to capture reference, dark, and sample image stacks across a configurable wavelength range.

> **Windows only.** The Thorlabs TSI SDK ships as native Windows DLLs and is not available on Linux or macOS.

---

## Hardware

| Device | Model | Interface |
|---|---|---|
| Monochrome camera | Thorlabs CS126MU / LP126MU | USB |
| Monochromator | Custom build — Arduino + stepper driver | USB serial (e.g. `COM4`) |
| Focus motor | Thorlabs KDC101 controller + Z-axis stage | USB (Kinesis) |
| Polarizer | Polarizer on Thorlabs K10CR1 cage rotation stage | USB (Kinesis) |
| Filter wheel | Newport USFW-100 (6-position) | USB (VISA) |

The monochromator firmware lives in `src/monochromator/monochromator_3modes/monochromator_3modes.ino` and must be flashed to the Arduino before first use.

---

## Prerequisites

- Windows 10/11 (64-bit)
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Arduino IDE — only needed for re-flashing the monochromator firmware

---

## Setup

### 1. Install ThorCam and extract the SDK

`thorlabs_tsi_sdk` is not on PyPI — it ships with the **ThorCam** desktop application.

1. Download and install **ThorCam** from the [Thorlabs Scientific Imaging software page](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam).
2. After installation, locate the SDK inside the ThorCam directory — typically:
   ```
   C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\Scientific Camera Interfaces\
   ```
3. Copy **only** the `Python Toolkit` and `Native Toolkit` sub-folders into the project root as `thorcam-sdk/`:
   ```
   thorcam-sdk/
     Python Toolkit/
       thorlabs_tsi_camera_python_sdk_package.zip
     Native Toolkit/
       dlls/
   ```

### 2. Install Python dependencies

```powershell
uv sync
```

This resolves `thorlabs-tsi-sdk` from the local path set in `pyproject.toml` — the SDK must exist at `thorcam-sdk/` before running this step.

### 3. DLL visibility

No manual step required. `src/pythorcam/windows_setup.py` automatically adds the `Native Toolkit/dlls/` directory to `PATH` at import time.

---

## Configuration

Edit `src/config/config.yaml` before first use. The key fields to set per machine:

| Field | Description |
|---|---|
| `monochromator_port` | COM port of the Arduino (e.g. `COM4`) — leave empty to be prompted at runtime |
| `camera_serial` | Serial number printed on the camera body |
| `focus_serial` | Serial number of the Kinesis focus-motor controller |
| `polarizer_serial` | Serial number of the Kinesis K10CR1 controller |
| `polarizer_x_position` | Polarizer angle (deg) for x-polarization; y-polarization uses `+90°` automatically |
| `filterwheel_address` | PyVISA USB address (e.g. `USB0::0x104D::0x1001::...::RAW`) |
| `wvl_start` / `wvl_stop` / `wvl_num` | Sweep range (nm) and number of steps |
| `save_dir` | Directory where measurement NPZ files are saved |
| `calib_roi_fraction` | Central image fraction (0–1) used for brightness averaging; `null` = full frame |

### Exposure settings files

Because light intensity differs between polarization states, three separate per-wavelength exposure settings files are maintained:

| File | Used when |
|---|---|
| `exposure_settings.json` | `xpol=None` — no polarizer movement |
| `exposure_settings_xpol.json` | `xpol=True` — x-polarization |
| `exposure_settings_ypol.json` | `xpol=False` — y-polarization |

All three are generated automatically by `brightness_calibration()` and must be produced before the corresponding measurement.

Per-wavelength focus settings are stored in `focus_settings.json` and generated during autofocus runs.

---

## Usage

For interactive use and development, open `test.ipynb` in Jupyter:

```powershell
uv run jupyter notebook test.ipynb
```

### Polarization-resolved measurement sequence

All measurement methods accept an `xpol` parameter (`True` / `False` / `None`) that selects the polarization state and the matching exposure settings:

```python
from src.config import Config
from src.control import Control

cfg = Config.from_yaml('src/config/config.yaml')
ctrl = Control.from_config(cfg)

# --- x-polarization ---
ctrl.brightness_calibration(xpol=True)   # calibrate and save exposure_settings_xpol.json
ctrl.reference_measurement(xpol=True)    # polarizer moves to polarizer_x_position
ctrl.black_measurement(xpol=True)        # polarizer not moved (beam blocked); uses xpol settings
ctrl.sample_measurement(xpol=True)       # polarizer moves to polarizer_x_position

# --- y-polarization ---
ctrl.brightness_calibration(xpol=False)  # calibrate and save exposure_settings_ypol.json
ctrl.reference_measurement(xpol=False)   # polarizer moves to polarizer_x_position + 90°
ctrl.black_measurement(xpol=False)
ctrl.sample_measurement(xpol=False)

# --- no polarizer (legacy / unpolarized) ---
ctrl.brightness_calibration(xpol=None)
ctrl.reference_measurement(xpol=None)
ctrl.black_measurement(xpol=None)
ctrl.sample_measurement(xpol=None)
```

Output filenames embed the polarization label:
- `reference_xpol_<timestamp>.npz` / `reference_ypol_<timestamp>.npz` / `reference_<timestamp>.npz`
- Same pattern for `black_` and `sample_`.

### ROI brightness calibration

If the illumination does not fill the full camera frame, set `calib_roi_fraction` in the YAML to restrict brightness averaging to the central sub-region:

```yaml
calib_roi_fraction: 0.6   # measure brightness over central 60%×60% of pixels
```

Leave it as `null` to use the full frame (default behaviour).

---

## Project structure

```
src/
  control.py            # Top-level orchestrator — camera, monochromator, focus motor, polarizer, filter wheel
  config/
    config.py           # Config, ExposureSettings, FocusSettings dataclasses
    config.yaml         # Per-machine hardware configuration
  pythorcam/
    thorcam.py          # ThorlabsCamera wrapper
    windows_setup.py    # Adds native DLLs to PATH at runtime
    utils.py            # Live view, autofocus, brightness calibration
  monochromator/
    mono.py             # MonochromatorControl — serial wrapper for Arduino
    monochromator_3modes/
      monochromator_3modes.ino  # Arduino firmware (AccelStepper)
  filterwheel/
    filterwheel.py      # Newport USFW-100 driver via PyVISA
```
