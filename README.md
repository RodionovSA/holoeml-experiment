# holoeml-experiment

Control system and measurement code for **all experiments and measurements in
the HoloEML project** — currently amplitude, with phase scaffolded to follow. Every experiment shares one set of hardware drivers (camera, monochromator, motorized stages, filter wheel, ...) and orchestrates them.

> The base package (analysis, config, non-hardware code) installs on any OS. **Driving the actual lab hardware requires Windows** — the Thorlabs TSI SDK and the other vendor DLLs (power meter, spectrometer) are Windows-only.

---

## Experiments

| Experiment | Status | Code | Notebooks/scripts | Docs |
|---|---|---|---|---|
| **Amplitude** | Implemented | `amplitude/` | `scripts/amplitude/` | [`docs/amplitude_measurement_protocol.md`](docs/amplitude_measurement_protocol.md) |
| **Phase** | Planned (scaffolded) | `phase/` | `scripts/phase/` | — |

Phase and characterization folders are currently reserved but empty — they
will reuse the same shared `instruments/` drivers as amplitude once built out.

---

## Hardware

| Device | Model | Interface |
|---|---|---|
| Monochrome camera | Thorlabs CS126MU / LP126MU | USB |
| Monochromator | Custom build — Arduino + stepper driver | USB serial (e.g. `COM4`) |
| Focus motor | Thorlabs KDC101 controller + Z-axis stage | USB (Kinesis) |
| Polarizer | Polarizer on Thorlabs K10CR1 cage rotation stage | USB (Kinesis) |
| Precision piezo z-stage | Thorlabs CT1P cage-compatible z-axis stage, integrated piezo driver | USB (Kinesis .NET) |
| Filter wheel | Newport USFW-100 (6-position) | USB (VISA) |
| Power meter | Thorlabs PM400 | USB (VISA / TLPM) |
| Spectrometer | Ocean Optics (OceanDirect-compatible) | USB |

These drivers live under `instruments/` and are shared by every experiment in this repo. The monochromator firmware lives in `instruments/monochromator/monochromator_3modes/monochromator_3modes.ino` and must be flashed to the Arduino before first use.

---

## Prerequisites

**All machines** (analysis, dev, no hardware attached):

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

**Hardware machines** additionally need:

- Windows 10/11 (64-bit)
- The three vendor SDKs described in [Vendor SDKs](#vendor-sdks-hardware-machines-only) below
- Arduino IDE — only needed for re-flashing the monochromator firmware

---

## Setup

Setup below is shared across all experiments — do it once per machine. Pick the
path that matches your machine.

### Option A — Analysis / no hardware attached (any OS)

Hardware-only packages (`pylablib`, `pyqt5-qt5`, `pyvisa`, `thorlabs-tsi-sdk`)
live behind the `hardware` extra in `pyproject.toml`, so you don't need any of
the vendor SDKs below. Install with:

```bash
uv sync --frozen
```

Use `--frozen` here, not plain `uv sync`. `thorlabs-tsi-sdk` is locked from a
local path (the camera SDK zip — see [Thorlabs camera](#thorlabs-camera--thorcam-sdk)
below), which only exists on a hardware machine. Plain `uv sync` re-validates
that path as part of checking the lockfile is up to date and will fail here
even though you're not installing that package. `--frozen` skips the
re-validation and installs straight from the committed `uv.lock`.

Trade-off: a machine set up this way won't notice if `pyproject.toml` changes
— you need to pull an updated `uv.lock` (regenerated on a hardware machine,
see below) to pick up dependency changes.

### Option B — Hardware machine (Windows)

1. Install the three vendor SDKs below **first** — `thorlabs-tsi-sdk` is
   resolved from a local file, so it must exist before syncing.
2. Install Python dependencies, including the hardware extra:
   ```powershell
   uv sync --extra hardware
   ```
3. If you've changed `pyproject.toml`, regenerate the lock here (this is the
   only kind of machine that can, since it has the camera SDK zip locally):
   ```powershell
   uv lock
   ```
   Commit the updated `uv.lock` so non-hardware machines pick up the change.

---

### Vendor SDKs (hardware machines only)

Four instruments need a vendor SDK installed separately — Python packaging
alone can't distribute them (native DLLs, or not published to PyPI).

#### Thorlabs camera — ThorCam SDK

`thorlabs-tsi-sdk` is not on PyPI — it ships with the **ThorCam** desktop
application.

1. Download and install **ThorCam** from the [Thorlabs Scientific Imaging software page](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam).
2. Locate the SDK inside the ThorCam install directory — typically:
   ```
   C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\Scientific Camera Interfaces\
   ```
3. Copy **only** the `Python Toolkit` and `Native Toolkit` sub-folders into the project as `thorcam-sdk/`:
   ```
   thorcam-sdk/
     Python Toolkit/
       thorlabs_tsi_camera_python_sdk_package.zip
     Native Toolkit/
       dlls/
   ```

   > **Note:** two different places in the code expect `thorcam-sdk/`:
   > `pyproject.toml` resolves the zip from
   > `instruments/pythorcam/thorcam-sdk/Python Toolkit/...zip`, while
   > `instruments/pythorcam/windows_setup.py` adds DLLs from a
   > **project-root** `thorcam-sdk/Native Toolkit/dlls/`. If one of the two
   > can't find its files, place (or symlink) `thorcam-sdk/` in both
   > locations.

Once the zip is in place, `uv sync --extra hardware` installs
`thorlabs-tsi-sdk` from it. DLL visibility needs no manual step —
`instruments/pythorcam/windows_setup.py` adds `Native Toolkit/dlls/` to `PATH`
automatically at import time.

#### Thorlabs PM400 power meter — Optical Power Monitor

1. Download and install the [Thorlabs Optical Power Monitor](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=OPM).
2. The power meter driver (`instruments/powermeter/powermeter.py`) imports
   `TLPM.py` from the install directory, by default:
   ```
   C:\Program Files (x86)\IVI Foundation\VISA\WinNT\TLPM\Examples\Python
   ```
   Override with the `TLPM_PYTHON` env var if installed elsewhere.
3. It then loads `TLPM_64.dll`, by default from:
   ```
   C:\Program Files\IVI Foundation\VISA\Win64\Bin
   ```
   Override with the `TLPM_BIN` env var, or the `dll_dir=` constructor
   argument.

No manual PATH edits are needed — both defaults match a standard OPM install.

#### Ocean Optics spectrometer — OceanDirect SDK

1. Download and install the [Ocean Optics OceanDirect SDK](https://www.oceanoptics.com/software/oceandirect/).
2. The spectrometer driver (`instruments/spectrometer/spectrometer.py`) loads
   `OceanDirect.dll` from the standard install path:
   ```
   C:\Program Files\Ocean Optics\OceanDirect SDK\Python
   ```
   `instruments.spectrometer.Spectrometer` adds this directory to both
   `sys.path` and the DLL search path automatically (it expects
   `oceandirect/lib/OceanDirect.dll` underneath it). Override with the
   `OCEANDIRECT_SDK` env var, or the `sdk_dir=` constructor argument, if
   installed elsewhere.

#### Thorlabs CT1P precision piezo — Kinesis

Unlike the focus/polarizer stages (`instruments/kinesismotor/`, driven through
pylablib), the CT1P's integrated piezo actuator is a Kinesis device type
(`IntegratedPrecisionPiezo`) that pylablib doesn't support. The driver
(`instruments/precisionpiezo/precisionpiezo.py`) instead talks to the Kinesis
.NET API directly via [`pythonnet`](https://pythonnet.github.io/) (installed
as part of the `hardware` extra).

1. Install **Kinesis** from the [Thorlabs Kinesis software page](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control&viewtab=0) (version 1.14.26 or later — earlier releases predate CT1P support).
2. The driver loads the Kinesis assemblies from the standard install path:
   ```
   C:\Program Files\Thorlabs\Kinesis
   ```
   Override with the `KINESIS_DIR` env var, or the `kinesis_dir=` constructor
   argument, if installed elsewhere.
3. Find the stage's serial number in the Kinesis GUI (or on the label) and
   pass it to `PrecisionPiezoCT1P(serial)`.

No manual PATH edits are needed — the driver calls `os.add_dll_directory` on
the Kinesis install folder automatically.

#### Monochromator firmware

The monochromator is a custom Arduino build, not a vendor SDK — flash
`instruments/monochromator/monochromator_3modes/monochromator_3modes.ino` to
the Arduino once, using the Arduino IDE, before first use.

---

## Amplitude experiment

Wavelength-swept, polarization-resolved image acquisition: for each polarization state (x, y, or none) it captures reference, dark, and sample stacks across the configured wavelength range.

- **Config**: `amplitude/config/config.yaml` (per-machine hardware config — serial numbers, ports, wavelength sweep range, save directory).
- **Exposure settings**: three per-wavelength JSON files under `amplitude/config/` (`exposure_settings.json`, `_xpol.json`, `_ypol.json`), generated by `brightness_calibration()`.
- **Run it**: `scripts/amplitude/main.py`, or interactively via `scripts/amplitude/check_results.ipynb`. Before driving real hardware, `scripts/amplitude/test_connection.py` runs a pre-flight check of every device (camera, monochromator, filter wheel, focus motor, polarizer) and prints a PASS/FAIL summary.

```python
from amplitude.config import Config
from amplitude.control import Control

cfg = Config.from_yaml('amplitude/config/config.yaml')
ctrl = Control.from_config(cfg)
ctrl.prepare_devices()  # homes stages, applies initial camera/motor settings
ctrl.go_to_default_state()

ctrl.brightness_calibration(xpol=True)
ctrl.reference_measurement(xpol=True)
ctrl.black_measurement(xpol=True)
ctrl.sample_measurement(xpol=True)
```

Full protocol and step-by-step instructions: [`docs/amplitude_measurement_protocol.md`](docs/amplitude_measurement_protocol.md).

---

## Phase experiment

Not yet implemented — `phase/` and `scripts/phase/` are reserved for the upcoming phase-measurement pipeline, which will build on the same shared `instruments/` drivers used by amplitude.

---

## Project structure

```
instruments/            # Shared hardware drivers, used by all experiments
  pythorcam/             # ThorlabsCamera wrapper, live view / autofocus / brightness calibration
  monochromator/         # MonochromatorControl — serial wrapper for Arduino
    monochromator_3modes/
      monochromator_3modes.ino  # Arduino firmware (AccelStepper)
  filterwheel/           # Newport USFW-100 driver via PyVISA
  kinesismotor/          # KinesisMotor — Kinesis stage control (focus + polarizer), via pylablib
  precisionpiezo/        # PrecisionPiezoCT1P — CT1P integrated piezo z-stage, via Kinesis .NET/pythonnet
  powermeter/            # PM400 — Thorlabs optical power meter (TLPM ctypes driver)
  spectrometer/          # Spectrometer — Ocean Optics spectrometer (OceanDirect SDK)

amplitude/               # Amplitude experiment (implemented)
  control.py             # Control — lifecycle (connect/prepare/close) + thin delegators
  devices.py             # Instruments — camera/monochromator/stages/filter wheel connection & lifecycle
  settings_store.py      # Polarization, ExposureStore, FocusStore — per-pol settings persistence
  measurements/          # One module per routine: default_state, brightness, reference, black, sample
    context.py           # MeasurementContext — shared device/settings access for routines
  config/
    config.py            # Config, ExposureSettings, FocusSettings dataclasses
    config.yaml          # Per-machine hardware configuration
    exposure_settings*.json
    focus_settings.json
  backup/

phase/                   # Phase experiment (planned, scaffolded)

scripts/
  amplitude/             # main.py entry point + interactive notebooks
  phase/                 # (planned, scaffolded)

docs/
  amplitude_measurement_protocol.md
```
