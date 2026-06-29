# holoeml-amp

A Python control system for wavelength-swept optical spectroscopy. It orchestrates a monochromator, camera, motorized focus stage, and filter wheel to capture reference, dark, and sample image stacks across a configurable wavelength range.

> **Windows only.** The Thorlabs TSI SDK ships as native Windows DLLs and is not available on Linux or macOS.

---

## Hardware

| Device | Model | Interface |
|---|---|---|
| Monochrome camera | Thorlabs CS126MU / LP126MU | USB |
| Monochromator | Custom build — Arduino + stepper driver | USB serial (e.g. `COM4`) |
| Focus motor | Thorlabs KDC101 controller + Z-axis stage | USB (Kinesis) |
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
| `mono_port` | COM port of the Arduino (e.g. `COM4`) — leave empty to be prompted at runtime |
| `camera_serial` | Serial number printed on the camera body |
| `focus_motor_serial` | Serial number of the Kinesis controller |
| `filter_wheel_address` | PyVISA USB address (e.g. `USB0::0x104D::0x1001::...::RAW`) |
| `wavelength_start/stop/num` | Sweep range and number of steps |
| `save_dir` | Directory where measurement NPZ files are saved |

Per-wavelength camera exposure and focus settings are stored as JSON files alongside the YAML and are generated automatically during calibration runs.

---

## Usage

Run a reference measurement:

```powershell
python main.py
```

This loads `src/config/config.yaml`, initializes all hardware, and captures a reference image stack across the configured wavelength sweep.

For interactive use and development, open `test.ipynb` in Jupyter:

```powershell
uv run jupyter notebook test.ipynb
```

---

## Project structure

```
src/
  control.py            # Top-level orchestrator (Control class)
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
