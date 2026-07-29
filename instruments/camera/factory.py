"""Vendor dispatch: open the configured camera without callers needing to
know which vendor SDK is in play.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from instruments.camera.base import Camera

if TYPE_CHECKING:
    from amplitude.config import Config
    from instruments.config import EquipmentConfig


def open_camera(config: "Config | EquipmentConfig",
                stack: contextlib.ExitStack | None = None) -> Camera:
    """Connect to the camera described by *config* and return it.

    Dispatches on ``config.camera_vendor`` (``'thorlabs'`` or ``'pco'``,
    defaults to ``'thorlabs'`` for configs predating this field). Vendor
    modules are imported lazily so a machine with only one camera SDK
    installed doesn't need the other.

    The Thorlabs path additionally needs an SDK handle
    (:func:`instruments.pythorcam.thorcam.create_camera_sdk`) that outlives
    the camera object itself; when *stack* is given, that handle's disposal
    is registered on it so callers don't need vendor-specific teardown logic.
    pco is self-contained and ignores *stack*.
    """
    vendor = getattr(config, "camera_vendor", "thorlabs")
    if vendor == "thorlabs":
        from instruments.pythorcam.thorcam import ThorlabsCamera, create_camera_sdk
        sdk = create_camera_sdk()
        if stack is not None:
            stack.callback(sdk.dispose)
        return ThorlabsCamera(sdk, config.camera_serial)
    elif vendor == "pco":
        from instruments.pypcocam.pcocam import PcoCamera
        return PcoCamera(config.camera_serial)
    else:
        raise ValueError(f"Unknown camera_vendor {vendor!r}; expected 'thorlabs' or 'pco'")
