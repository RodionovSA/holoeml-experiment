from __future__ import annotations

import os
import sys


class Spectrometer:
    """Thin wrapper around Ocean Optics' ``OceanDirectAPI`` for USB spectrometers.

    Discovers and opens a device by serial number (or the first one found),
    then exposes spectrum acquisition and common settings (integration time,
    averaging) as plain Python values instead of raw SDK calls.

    Windows only: the underlying driver loads ``OceanDirect.dll`` at construction
    time.  The SDK directory defaults to the ``OCEANDIRECT_SDK`` env var, falling
    back to the standard install path
    ``C:\\Program Files\\Ocean Optics\\OceanDirect SDK\\Python``.  Override via
    the ``sdk_dir=`` constructor argument or by setting ``OCEANDIRECT_SDK`` to
    the ``Python`` folder of your install.

    The ``Python`` folder must contain the ``oceandirect`` package sub-folder,
    which in turn must contain ``lib\\OceanDirect.dll``.  This matches the
    default install layout.

    Example::

        from instruments.spectrometer import Spectrometer

        with Spectrometer() as sp:
            sp.set_integration_time(100_000)   # 100 ms, units are microseconds
            wl, counts = sp.snapshot()
    """

    DEFAULT_SDK_DIR = r"C:\Program Files\Ocean Optics\OceanDirect SDK\Python"

    def __init__(
        self,
        serial: str | None = None,
        *,
        sdk_dir: str | None = None,
        integration_time_us: int | None = None,
        scans_to_average: int | None = None,
    ):
        # ------------------------------------------------------------------
        # 1. DLL / package path setup (Windows only, before any SDK import)
        # ------------------------------------------------------------------
        self._closed = True  # guard: set False only after a successful open

        if os.name == "nt":
            sdk_dir = sdk_dir or os.environ.get("OCEANDIRECT_SDK", self.DEFAULT_SDK_DIR)
            dll_dir = os.path.join(sdk_dir, "oceandirect", "lib")
            if sdk_dir not in sys.path:
                sys.path.insert(0, sdk_dir)
            os.add_dll_directory(dll_dir)

        # ------------------------------------------------------------------
        # 2. Deferred import — must come after the path additions above
        # ------------------------------------------------------------------
        from oceandirect.OceanDirectAPI import OceanDirectAPI  # noqa: PLC0415

        # ------------------------------------------------------------------
        # 3. Initialise the SDK (this loads OceanDirect.dll)
        # ------------------------------------------------------------------
        self._api = OceanDirectAPI()

        # ------------------------------------------------------------------
        # 4. Discover and open the device
        # ------------------------------------------------------------------
        n_found = self._api.find_usb_devices()
        if n_found == 0:
            raise RuntimeError("No USB OceanDirect spectrometer found.")

        device_ids = self._api.get_device_ids()

        # get_serial_number on the API-level object crashes on an unopened device
        # (null DLL handle). Open each device first, read serial from the device
        # object, close the wrong ones.
        self._device = None
        self._device_id = None
        found_serials: list[str] = []

        for did in device_ids:
            dev = self._api.open_device(did)
            sn = dev.get_serial_number()
            if serial is None or serial in sn:
                self._device_id = did
                self._device = dev
                break
            found_serials.append(sn)
            dev.close_device()

        if self._device is None:
            raise RuntimeError(
                f"No spectrometer with serial matching {serial!r} found. "
                f"Available: {found_serials}"
            )

        # ------------------------------------------------------------------
        # 5. Cache metadata and wavelength axis
        # ------------------------------------------------------------------
        self.serial = self._device.get_serial_number()
        self.model = self._device.get_model()
        self._wavelengths: list[float] = self._device.get_wavelengths()

        # ------------------------------------------------------------------
        # 6. Optional initial settings
        # ------------------------------------------------------------------
        if integration_time_us is not None:
            self._device.set_integration_time(integration_time_us)
        if scans_to_average is not None:
            self._device.set_scans_to_average(scans_to_average)

        self._closed = False
        print(f"Connected to {self.model} (SN {self.serial})")

    # ---------------------------------------------------------------------- #
    # Acquisition                                                              #
    # ---------------------------------------------------------------------- #

    def get_wavelengths(self) -> list[float]:
        """Return the wavelength axis in nanometers (cached at open time)."""
        return list(self._wavelengths)

    def get_intensities(self) -> list[float]:
        """Return the current dark- and non-linearity-corrected intensities (counts)."""
        return self._device.get_formatted_spectrum()

    def snapshot(self) -> tuple[list[float], list[float]]:
        """Return ``(wavelengths_nm, intensities_counts)`` for a single acquisition."""
        return self.get_wavelengths(), self.get_intensities()

    def get_max_intensity(self) -> float:
        """Return the detector's saturation value (maximum counts per pixel)."""
        return self._device.get_max_intensity()

    # ---------------------------------------------------------------------- #
    # Integration time                                                         #
    # ---------------------------------------------------------------------- #

    def set_integration_time(self, microseconds: int) -> None:
        """Set the integration time.

        Args:
            microseconds: Integration time in **microseconds** (µs).
                Use :meth:`get_integration_time_limits` to query the allowed range.
        """
        self._device.set_integration_time(microseconds)

    def get_integration_time(self) -> int:
        """Return the current integration time in microseconds."""
        return self._device.get_integration_time()

    def get_integration_time_limits(self) -> tuple[int, int]:
        """Return ``(min_us, max_us)`` — the integration time range in microseconds."""
        return (
            self._device.get_minimum_integration_time(),
            self._device.get_maximum_integration_time(),
        )

    # ---------------------------------------------------------------------- #
    # Averaging                                                                #
    # ---------------------------------------------------------------------- #

    def set_scans_to_average(self, n: int) -> None:
        """Set the number of hardware scans to average per :meth:`get_intensities` call."""
        self._device.set_scans_to_average(n)

    def get_scans_to_average(self) -> int:
        """Return the current number of scans averaged per acquisition."""
        return self._device.get_scans_to_average()

    # ---------------------------------------------------------------------- #
    # Info                                                                     #
    # ---------------------------------------------------------------------- #

    def identify(self) -> dict:
        """Return model, serial number, and device type."""
        return {
            "model": self.model,
            "serial": self.serial,
            "device_type": self._device.get_device_type(),
        }

    # ---------------------------------------------------------------------- #
    # Teardown                                                                 #
    # ---------------------------------------------------------------------- #

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._device.close_device()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
