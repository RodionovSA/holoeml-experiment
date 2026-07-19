"""Pre-flight connection check for all hardware devices used by ``Control``.

Probes the camera, monochromator, filter wheel, focus motor, and polarizer
(the same devices ``amplitude.control.Control.from_config`` connects), opens
and immediately closes each one, and prints a PASS/FAIL summary. Unlike a
fail-fast check, every device is attempted regardless of earlier failures so
a single run tells you the status of the whole rig.

Usage::

    python scripts/amplitude/test_connection.py [path/to/config.yaml]
"""

import sys
from pathlib import Path

from instruments.filterwheel import FilterWheelControl
from instruments.kinesismotor import KinesisMotor
from instruments.monochromator.mono import MonochromatorControl
from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk

from amplitude.config import Config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "amplitude" / "config" / "config.yaml"


def check_camera(config: Config, sdk) -> None:
    with ThorlabsCamera(sdk, config.camera_serial) as cam:
        cam.image_shape  # touches the open handle to confirm it responds


def check_monochromator(config: Config) -> None:
    available = MonochromatorControl.list_available_ports()
    if config.monochromator_port not in available:
        raise RuntimeError(
            f"Port {config.monochromator_port!r} not found. Available: {available}"
        )
    with MonochromatorControl(port=config.monochromator_port) as mono:
        mono.initialize_arduino()
        if mono.get_position() is None:
            raise RuntimeError("No response from monochromator (get_position timed out).")


def check_filterwheel(config: Config) -> None:
    with FilterWheelControl(config.filterwheel_address) as fw:
        fw.get_position()


def check_focus(config: Config) -> None:
    with KinesisMotor(config.focus_serial, motor_type='stage') as motor:
        motor.get_position()


def check_polarizer(config: Config) -> None:
    with KinesisMotor(config.polarizer_serial, motor_type='K10CR1') as motor:
        motor.get_position()


def record(results: list, name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        results.append((name, False, str(exc)))
    else:
        print(f"[ OK ] {name}")
        results.append((name, True, ""))


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = Config.from_yaml(config_path)

    results: list[tuple[str, bool, str]] = []

    record(results, "Monochromator", lambda: check_monochromator(config))

    sdk = None
    try:
        sdk = create_camera_sdk()
        record(results, "Camera", lambda: check_camera(config, sdk))
    except Exception as exc:
        print(f"[FAIL] Camera: {exc}")
        results.append(("Camera", False, str(exc)))
    finally:
        if sdk is not None:
            sdk.dispose()

    record(results, "Filter wheel", lambda: check_filterwheel(config))
    record(results, "Focus motor", lambda: check_focus(config))
    record(results, "Polarizer", lambda: check_polarizer(config))

    print("\n--- Summary ---")
    failed = [name for name, ok, _ in results if not ok]
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = f" ({err})" if err else ""
        print(f"{status:4} {name}{suffix}")

    if failed:
        print(f"\n{len(failed)} device(s) failed to connect: {', '.join(failed)}")
        return 1

    print("\nAll devices connected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
