from __future__ import annotations

import os
import time
from ctypes import (
    byref,
    c_bool,
    c_double,
    c_int,
    c_int16,
    c_uint32,
    create_string_buffer,
)

from .TLPM import (
    TLPM,
    TLPM_ATTR_SET_VAL,
    TLPM_AUTORANGE_POWER_OFF,
    TLPM_AUTORANGE_POWER_ON,
    TLPM_POWER_UNIT_DBM,
    TLPM_POWER_UNIT_WATT,
)


def _decode(buf) -> str:
    """Decode a null-terminated ctypes string buffer to a plain str."""
    return buf.raw.split(b'\x00', 1)[0].decode()


class PM400:
    """Thin wrapper around Thorlabs' ``TLPM`` ctypes driver for the PM400
    optical power meter (and other TLPM-compatible power/energy meters).

    Discovers and opens a device by serial number (or the first one found),
    then exposes power readings and common settings (wavelength, averaging,
    range, unit) as plain Python floats/strings instead of raw ctypes calls.

    Windows only: the underlying driver loads ``TLPM_64.dll``/``TLPM_32.dll``
    at construction time. The DLL directory defaults to the ``TLPM_BIN``
    env var, falling back to the standard VISA install path.
    """

    _UNIT_TO_CODE = {'W': TLPM_POWER_UNIT_WATT, 'DBM': TLPM_POWER_UNIT_DBM}
    _CODE_TO_UNIT = {v: k for k, v in _UNIT_TO_CODE.items()}

    DEFAULT_DLL_DIR = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin"

    def __init__(self, serial: str | None = None, *, dll_dir: str | None = None,
                 reset: bool = False, id_query: bool = True):
        if os.name == 'nt':
            dll_dir = dll_dir or os.environ.get('TLPM_BIN', self.DEFAULT_DLL_DIR)
            os.add_dll_directory(dll_dir)

        self._closed = True
        self._tlpm = TLPM()

        resource_name, model, found_serial = self._find_resource(serial)

        self._tlpm.open(resource_name, c_bool(id_query), c_bool(reset))
        self._closed = False

        self.resource_name = _decode(resource_name)
        self.model = model
        self.serial = found_serial
        print(f"Connected to {self.model} (SN {self.serial})")

    def _find_resource(self, serial: str | None):
        """Discover devices and return (resourceName buffer, model, serial) for the match."""
        count = c_uint32()
        self._tlpm.findRsrc(byref(count))
        if count.value == 0:
            raise RuntimeError("No PM400-compatible power meter found.")

        for i in range(count.value):
            resource_name = create_string_buffer(1024)
            self._tlpm.getRsrcName(c_int(i), resource_name)

            model_buf = create_string_buffer(1024)
            serial_buf = create_string_buffer(1024)
            self._tlpm.getRsrcInfo(c_int(i), model_buf, serial_buf, None, None)
            found_serial = _decode(serial_buf)
            found_model = _decode(model_buf)

            if serial is None or serial in found_serial or serial in _decode(resource_name):
                return resource_name, found_model, found_serial

        raise RuntimeError(f"No power meter with serial matching {serial!r} found.")

    # ------------------------------------------------------------------ #
    # Measurement                                                          #
    # ------------------------------------------------------------------ #

    def get_power(self) -> float:
        """Return the current power reading, in whatever unit is set (see set_power_unit)."""
        power = c_double()
        self._tlpm.measPower(byref(power))
        return power.value

    def get_power_avg(self, n: int = 1, delay: float = 0.0) -> float:
        """Return the average of n successive power readings."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        readings = []
        for _ in range(n):
            readings.append(self.get_power())
            if delay:
                time.sleep(delay)
        return sum(readings) / len(readings)

    # ------------------------------------------------------------------ #
    # Settings                                                             #
    # ------------------------------------------------------------------ #

    def set_wavelength(self, nm: float):
        """Set the sensor's calibration wavelength in nanometers."""
        self._tlpm.setWavelength(c_double(nm))

    def get_wavelength(self) -> float:
        return self._get_double(self._tlpm.getWavelength)

    def set_averaging_time(self, seconds: float):
        """Set the averaging time in seconds (rounded internally to the sensor's sampling rate)."""
        self._tlpm.setAvgTime(c_double(seconds))

    def get_averaging_time(self) -> float:
        return self._get_double(self._tlpm.getAvgTime)

    def set_auto_range(self, on: bool):
        mode = TLPM_AUTORANGE_POWER_ON if on else TLPM_AUTORANGE_POWER_OFF
        self._tlpm.setPowerAutoRange(c_int16(mode))

    def get_auto_range(self) -> bool:
        mode = c_int16()
        self._tlpm.getPowerAutorange(byref(mode))
        return bool(mode.value)

    def set_power_range(self, watts: float):
        """Set the most positive signal level expected at the sensor input, in watts."""
        self._tlpm.setPowerRange(c_double(watts))

    def get_power_range(self) -> float:
        return self._get_double(self._tlpm.getPowerRange)

    def set_power_unit(self, unit: str):
        """Set the unit power readings are returned in: 'W' or 'dBm'."""
        code = self._UNIT_TO_CODE.get(unit.upper())
        if code is None:
            raise ValueError(f"unit must be one of {list(self._UNIT_TO_CODE)}, got {unit!r}")
        self._tlpm.setPowerUnit(c_int16(code))

    def get_power_unit(self) -> str:
        code = c_int16()
        self._tlpm.getPowerUnit(byref(code))
        return self._CODE_TO_UNIT[code.value]

    def set_beam_diameter(self, mm: float):
        """Set the beam diameter in millimeters, used to calculate power density."""
        self._tlpm.setBeamDia(c_double(mm))

    def get_beam_diameter(self) -> float:
        return self._get_double(self._tlpm.getBeamDia)

    def _get_double(self, getter, attribute=TLPM_ATTR_SET_VAL) -> float:
        """Call a TLPM getter of the form getter(attribute, c_double byref) and return the value."""
        value = c_double()
        getter(c_int16(attribute), byref(value))
        return value.value

    # ------------------------------------------------------------------ #
    # Zeroing                                                              #
    # ------------------------------------------------------------------ #

    def zero(self, timeout: float = 10.0):
        """Run the dark current/zero offset adjustment. Darken the input first.

        Blocks until the adjustment finishes or timeout (seconds) elapses.
        """
        self._tlpm.startDarkAdjust()
        state = c_int16()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._tlpm.getDarkAdjustState(byref(state))
            if state.value == 0:
                return
            time.sleep(0.1)
        self._tlpm.cancelDarkAdjust()
        raise TimeoutError(f"Dark adjustment did not finish within {timeout}s")

    # ------------------------------------------------------------------ #
    # Info                                                                 #
    # ------------------------------------------------------------------ #

    def identify(self) -> dict:
        """Return manufacturer, device name, serial number, and firmware revision."""
        manufacturer = create_string_buffer(256)
        device = create_string_buffer(256)
        serial = create_string_buffer(256)
        firmware = create_string_buffer(256)
        self._tlpm.identificationQuery(manufacturer, device, serial, firmware)
        return {
            'manufacturer': _decode(manufacturer),
            'device': _decode(device),
            'serial': _decode(serial),
            'firmware': _decode(firmware),
        }

    def get_sensor_info(self) -> dict:
        """Return sensor name, serial number, and calibration message."""
        name = create_string_buffer(256)
        serial = create_string_buffer(256)
        message = create_string_buffer(256)
        sensor_type = c_int16()
        sensor_subtype = c_int16()
        flags = c_int16()
        self._tlpm.getSensorInfo(name, serial, message, byref(sensor_type),
                                  byref(sensor_subtype), byref(flags))
        return {
            'name': _decode(name),
            'serial': _decode(serial),
            'message': _decode(message),
            'type': sensor_type.value,
            'subtype': sensor_subtype.value,
            'flags': flags.value,
        }

    # ------------------------------------------------------------------ #
    # Teardown                                                             #
    # ------------------------------------------------------------------ #

    def close(self):
        if not self._closed:
            self._tlpm.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
