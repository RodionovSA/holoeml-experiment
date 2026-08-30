import numpy as np

from instruments.config import load_equipment
from instruments.inertialpiezo import KIM101, KIM101Axis


def crop(img: np.ndarray,
         x0: int = 0,
         y0: int = 0,
         x1: int | None = None,
         y1: int | None = None,
         channel: int = 0) -> np.ndarray:
    """Crop a captured frame to ``[y0:y1, x0:x1]`` and drop the channel axis.

    ``Camera.get_image`` returns shape ``(H, W, C)``
    (see ``instruments/camera/base.py``); phase acquisition works on a single
    channel, so *channel* selects one and the result is 2-D. ``None`` for
    *x1*/*y1* means "to the end", so a config with ``roi_x1``/``roi_y1``
    unset (:class:`phasectl.config.Config`) is a no-op crop.
    """
    return img[y0:y1, x0:x1, channel]
