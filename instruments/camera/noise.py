"""Vendor-keyed PTC noise models.

Constants below come from the photon-transfer-curve (PTC) analysis in
``docs/photon_transfer.md`` (ThorCam CS126) and ``docs/photon_transfer_pco.md``
(pco.edge 10 bi CLHS), following Janesick's two-frame-difference method. This
module is the single source of truth for the *numbers*; the docs are the
derivation and the acquisition/analysis record. If either camera is
re-calibrated, update both.

Every quantity here is in **DN of the dark-subtracted mean signal**, `S`,
unless the method name says otherwise (`*_e` = electrons). This matches the
convention used throughout both docs.

Two families of entry points:

- **Predictive** (`sigma_total_dn`, `relative_error_intensity`, `snr`, ...):
  take a signal `signal_dn` -- a scalar or an array with **no frame axis**
  -- plus a separate `n_frames` count. For planning an acquisition before
  you have data.
- **Data-taking** (`sigma_from_frames`, `error_from_frames`): take actual
  image data -- one frame or a stack along an explicit `frame_axis`. Each
  frame's noise is computed independently at its own local signal and
  combined in quadrature over the frame axis; the raw signal itself is
  never averaged.

Noise model (Janesick, Ch. 4)::

    sigma_total^2 = sigma_read^2 + sigma_shot^2 + sigma_fpn^2
    sigma_shot    = sqrt(S / K)          # K = conversion gain, e-/DN
    sigma_fpn     = P_N * S              # P_N = PRNU factor, fixed -- does not average down

``relative_error_ratio`` (used for transmission/ratio measurements, e.g. the
metasurface case in the docs) uses the full read+shot term, not the
shot-only approximation the docs' quick formula uses -- it reduces to the
same thing once the read term is negligible, and matters at low signal for
the pco (``read_noise_dn**2 = 21.29``). Small deviations from the docs'
quick-reference tables are expected for that reason, not a bug.

This module has no hardware dependency -- it can be imported and used
without either vendor SDK installed, for planning an acquisition offline.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """PTC-calibrated noise model for one camera vendor.

    All signal arguments named `signal_dn` / `S` are dark-subtracted mean
    signal in DN, and accept either a scalar or a numpy array (elementwise);
    they must carry **no frame axis** -- reduce a stack to a mean first
    (or use `sigma_from_frames` / `error_from_frames`, which do that for
    you and take real frame data instead).
    """

    # -- measured constants -------------------------------------------------
    conversion_gain_e_per_dn: float
    """K, e-/DN. For the ThorCam this was only calibrated at gain index 0
    (see `conditions`) -- K and read noise are gain-dependent and this value
    does not apply at other gain settings."""
    read_noise_dn: float
    """sigma_read, DN. Signal-independent noise floor, measured directly
    from dark frames (two-frame difference method)."""
    prnu_factor: float
    """P_N, fraction of signal (e.g. 0.0053 = 0.53%). Fixed-pattern noise
    term; does NOT average down with more frames (see module docstring)."""
    pixel_max: int
    """ADC full-scale value, DN (e.g. 4095 for 12-bit, 65535 for 16-bit)."""

    # -- provenance -----------------------------------------------------------
    camera_model: str
    calibrated_serial: str
    """Serial number of the physical unit this model was measured on. Keyed
    by vendor only, so any camera of this vendor uses these numbers --
    check this field if a different serial is in use."""
    conditions: str
    """Camera settings the calibration was measured under, e.g. 'gain index
    0' or 'fast scan mode, no gain control'. Applying these constants
    outside these conditions is not validated."""
    source: str
    """Path to the doc this model was transcribed from."""
    read_noise_is_upper_bound: bool = False
    """True if `read_noise_dn` could not be cleanly isolated (see source doc
    for the caveat) and may overstate the true read noise."""

    # ------------------------------------------------------------------ #
    # Derived constants                                                   #
    # ------------------------------------------------------------------ #

    @property
    def full_well_e(self) -> float:
        """Full well, electrons (`pixel_max * K`, ADC-clip bound)."""
        return self.pixel_max * self.conversion_gain_e_per_dn

    @property
    def read_noise_e(self) -> float:
        """Read noise, electrons."""
        return self.read_noise_dn * self.conversion_gain_e_per_dn

    @property
    def dynamic_range(self) -> float:
        """`pixel_max / read_noise_dn`, quantization-limited dynamic range."""
        return self.pixel_max / self.read_noise_dn

    # ------------------------------------------------------------------ #
    # Unit conversion                                                     #
    # ------------------------------------------------------------------ #

    def to_electrons(self, signal_dn):
        """Convert DN to electrons: `S * K`."""
        return np.asarray(signal_dn) * self.conversion_gain_e_per_dn

    def to_dn(self, signal_e):
        """Convert electrons to DN: `e / K`."""
        return np.asarray(signal_e) / self.conversion_gain_e_per_dn

    # ------------------------------------------------------------------ #
    # Noise terms (DN), single frame or averaged over n_frames            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clip_nonneg(S):
        """Clip signal at 0 for the shot term."""
        return np.maximum(S, 0.0)

    def sigma_shot_dn(self, signal_dn):
        """Photon shot noise, DN: `sqrt(S / K)`. `S` is clipped at 0."""
        S = self._clip_nonneg(np.asarray(signal_dn, dtype=float))
        return np.sqrt(S / self.conversion_gain_e_per_dn)

    def sigma_fpn_dn(self, signal_dn):
        """Fixed-pattern noise, DN: `P_N * S`. Does not shrink with averaging."""
        return self.prnu_factor * np.asarray(signal_dn, dtype=float)

    def sigma_temporal_dn(self, signal_dn, n_frames: int = 1):
        """Read + shot noise (temporal component only), DN, averaged over
        `n_frames`: `sqrt((sigma_read^2 + S/K) / n_frames)`. `S` is clipped
        at 0 for the shot term (see `sigma_shot_dn`)."""
        S = self._clip_nonneg(np.asarray(signal_dn, dtype=float))
        var = self.read_noise_dn ** 2 + S / self.conversion_gain_e_per_dn
        return np.sqrt(var / n_frames)

    def sigma_total_dn(self, signal_dn, n_frames: int = 1):
        """Full noise, DN: temporal term averaged over `n_frames`, in
        quadrature with the (unaveraged) FPN floor. `S` is clipped at 0 for
        the shot term (see `sigma_shot_dn`); the FPN term uses the
        unclipped signal."""
        S = np.asarray(signal_dn, dtype=float)
        temporal_var = (self.read_noise_dn ** 2 + self._clip_nonneg(S) / self.conversion_gain_e_per_dn) / n_frames
        fpn_var = (self.prnu_factor * S) ** 2
        return np.sqrt(temporal_var + fpn_var)

    def snr(self, signal_dn, n_frames: int = 1):
        """Signal-to-noise ratio: `S / sigma_total_dn`."""
        S = np.asarray(signal_dn, dtype=float)
        return S / self.sigma_total_dn(S, n_frames)

    # ------------------------------------------------------------------ #
    # Relative error estimators                                          #
    # ------------------------------------------------------------------ #

    def relative_error_intensity(self, signal_dn, n_frames: int = 1):
        """Per-pixel intensity error, %: `100 * sigma_total_dn / S`.

        Has an averaging floor at `100 * prnu_factor` percent -- the FPN
        term never shrinks (see module docstring). Use
        `relative_error_ratio` instead for a transmission/ratio measurement,
        where that floor cancels.
        """
        S = np.asarray(signal_dn, dtype=float)
        return 100.0 * self.sigma_total_dn(S, n_frames) / S

    def _ratio_variance(self, signal_dn, n_frames):
        """`(sigma_read^2 + S/K) / (n_frames * S^2)` -- the temporal-only
        relative variance used by `relative_error_ratio` (FPN cancels)."""
        S = np.asarray(signal_dn, dtype=float)
        return (self.read_noise_dn ** 2 + S / self.conversion_gain_e_per_dn) / (n_frames * S ** 2)

    def relative_error_ratio(self, signal_sample_dn, n_sample: int,
                              signal_ref_dn, n_ref: int):
        """Error, %, on a ratio `T = S_sample / S_ref` (e.g. transmission).

        The fixed per-pixel PRNU gain is the same multiplicative factor in
        numerator and denominator and cancels -- only the temporal
        (read+shot) term survives:

            eps_T = 100 * sqrt(v(S_sample, n_sample) + v(S_ref, n_ref))
            v(S, n) = (sigma_read^2 + S/K) / (n * S^2)

        Unlike `relative_error_intensity`, this has no averaging floor.
        """
        return 100.0 * np.sqrt(
            self._ratio_variance(signal_sample_dn, n_sample)
            + self._ratio_variance(signal_ref_dn, n_ref)
        )

    # ------------------------------------------------------------------ #
    # Data-taking entry points: real frames, not a pre-reduced signal     #
    # ------------------------------------------------------------------ #

    def _prepare_frames(self, frames, dark, frame_axis):
        """Squeeze a trailing size-1 channel axis and subtract a master
        dark, without touching the frame axis. Returns dark-subtracted
        `frames` and the resolved `axis` (or `None`)."""
        frames = np.asarray(frames, dtype=float)
        if frames.ndim >= 1 and frames.shape[-1] == 1:
            frames = frames[..., 0]

        axis = None
        if frame_axis is not None:
            ndim = frames.ndim
            axis = frame_axis if frame_axis >= 0 else frame_axis + ndim
            if not (0 <= axis < ndim):
                raise ValueError(
                    f"frame_axis={frame_axis} is out of range for an array of shape "
                    f"{frames.shape} (ndim={ndim})"
                )

        if dark is not None:
            dark = np.asarray(dark, dtype=float)
            if dark.ndim >= 1 and dark.shape[-1] == 1:
                dark = dark[..., 0]
            if axis is not None and dark.ndim == frames.ndim:
                dark = dark.mean(axis=axis, keepdims=True)
            frames = frames - dark

        return frames, axis

    def _sigma_from_prepared(self, frames, axis):
        sigma_per_frame = self.sigma_total_dn(frames, n_frames=1)
        if axis is None:
            return sigma_per_frame
        return np.sqrt(np.mean(sigma_per_frame ** 2, axis=axis))

    def sigma_from_frames(self, frames, *, dark=None, frame_axis: int | None = None):
        """Noise (DN), computed directly from real frame data.

        `frames` is one image, or a stack of images along `frame_axis`. A
        trailing channel axis of size 1 (as returned by `Camera.get_image`)
        is squeezed automatically. `dark`, if given, is subtracted (its own
        frame axis, if any, is averaged first into a master dark).

        For a single frame (`frame_axis=None`), this is `sigma_total_dn` of
        that frame. For a stack, each frame's noise is computed
        independently at its own local signal and the results are combined
        in quadrature over `frame_axis`.
        """
        frames, axis = self._prepare_frames(frames, dark, frame_axis)
        return self._sigma_from_prepared(frames, axis)

    def error_from_frames(self, frames, *, dark=None, frame_axis: int | None = None):
        """Relative error (%), computed directly from real frame data:
        `100 * sigma_from_frames(...) / sum(frames, axis=frame_axis)`.

        Same parameters as `sigma_from_frames`.
        """
        frames, axis = self._prepare_frames(frames, dark, frame_axis)
        sigma = self._sigma_from_prepared(frames, axis)
        S = frames if axis is None else frames.sum(axis=axis)
        return 100.0 * sigma / S
    
    def rms_from_frames(self, frames, *, dark=None, frame_axis: int | None = None):
        sigma = self.sigma_from_frames(frames, dark=dark, frame_axis=frame_axis)
        return np.sqrt((sigma**2).mean())
    
    def noise_floor_from_frames(self, frames, *, dark=None, frame_axis: int | None = None):
        rms = self.rms_from_frames(frames, dark=dark, frame_axis=frame_axis)
        frames, _ = self._prepare_frames(frames, dark, frame_axis)
        return rms / frames.std()

    # ------------------------------------------------------------------ #
    # Frame budget                                                       #
    # ------------------------------------------------------------------ #

    def frames_for_error(self, signal_dn, target_pct: float, *,
                          kind: Literal["ratio", "intensity"] = "ratio",
                          reference_signal_dn: float | None = None):
        """Number of frames needed to reach `target_pct` relative error.

        Parameters
        ----------
        signal_dn : float
            Mean signal, DN, of the (dimmer/limiting) quantity being
            measured.
        target_pct : float
            Target relative error, in percent.
        kind : "ratio" | "intensity"
            "ratio" (default): a transmission/ratio measurement
            (`relative_error_ratio`), where PRNU cancels and there is no
            averaging floor. Assumes the reference is averaged with the same
            `n_frames` and, unless `reference_signal_dn` is given, is much
            brighter than `signal_dn` so its contribution is negligible
            (matches the docs' single-term approximation).
            "intensity": a single-ROI intensity measurement
            (`relative_error_intensity`), which has a PRNU averaging floor
            at `100 * prnu_factor` percent.

        Returns
        -------
        int or float
            Number of frames (rounded up), or `math.inf` if `kind` is
            "intensity" and `target_pct` is at or below the PRNU floor --
            no amount of averaging reaches it (see module docstring).
        """
        S = float(signal_dn)
        if kind == "intensity":
            floor_pct = 100.0 * self.prnu_factor
            if target_pct <= floor_pct:
                return math.inf
            # (sigma_read^2 + S/K)/n + (P_N*S)^2 = (target_pct/100 * S)^2
            target_var = (target_pct / 100.0 * S) ** 2 - (self.prnu_factor * S) ** 2
            n = (self.read_noise_dn ** 2 + S / self.conversion_gain_e_per_dn) / target_var
        elif kind == "ratio":
            v_sample = self.read_noise_dn ** 2 + S / self.conversion_gain_e_per_dn
            if reference_signal_dn is not None:
                # Solve for n assuming n_sample == n_ref == n.
                v_ref = self.read_noise_dn ** 2 + float(reference_signal_dn) / self.conversion_gain_e_per_dn
                n = (v_sample / S ** 2 + v_ref / float(reference_signal_dn) ** 2) / (target_pct / 100.0) ** 2
            else:
                # Reference assumed much brighter/better-averaged: sample term dominates.
                n = v_sample / (S ** 2 * (target_pct / 100.0) ** 2)
        else:
            raise ValueError(f"kind must be 'ratio' or 'intensity', got {kind!r}")
        return math.ceil(n)

    def summary(self) -> str:
        lines = [
            f"{self.camera_model} ({self.calibrated_serial}), {self.conditions}",
            f"  K (conversion gain)  = {self.conversion_gain_e_per_dn:.4g} e-/DN",
            f"  read noise           = {self.read_noise_dn:.4g} DN "
            f"({self.read_noise_e:.4g} e-)"
            + (" [UPPER BOUND]" if self.read_noise_is_upper_bound else ""),
            f"  PRNU factor          = {100 * self.prnu_factor:.3g}% of signal",
            f"  full well            = {self.full_well_e:.4g} e-",
            f"  dynamic range        = {self.dynamic_range:.4g}:1",
            f"  source               = {self.source}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        bound = ", read_noise_is_upper_bound=True" if self.read_noise_is_upper_bound else ""
        return (f"NoiseModel({self.camera_model!r}, K={self.conversion_gain_e_per_dn:.4g} e-/DN, "
                f"read_noise={self.read_noise_dn:.4g} DN{bound})")


#: Registry of calibrated noise models, keyed by ``Camera.noise_vendor``
#: (``"thorlabs"`` / ``"pco"`` -- the same keys ``open_camera`` dispatches
#: on). Numbers transcribed from the "Derived constants" tables in
#: ``docs/photon_transfer.md`` and ``docs/photon_transfer_pco.md``.
NOISE_MODELS: dict[str, NoiseModel] = {
    "thorlabs": NoiseModel(
        conversion_gain_e_per_dn=2.6596,
        read_noise_dn=0.275,
        prnu_factor=0.0053,
        pixel_max=4095,
        camera_model="ThorCam CS126",
        calibrated_serial="35596",
        conditions="gain index 0",
        source="docs/photon_transfer.md",
    ),
    "pco": NoiseModel(
        conversion_gain_e_per_dn=0.3078,
        read_noise_dn=4.614,
        prnu_factor=0.0044,
        pixel_max=65535,
        camera_model="pco.edge 10 bi CLHS",
        calibrated_serial="22500190",
        conditions="fast scan mode, no gain control",
        source="docs/photon_transfer_pco.md",
        read_noise_is_upper_bound=True,
    ),
}


def get_noise_model(vendor: str) -> NoiseModel:
    """Look up the calibrated `NoiseModel` for `vendor` ("thorlabs" or "pco").

    Raises `KeyError` with the available vendors listed if `vendor` is not
    calibrated.
    """
    try:
        return NOISE_MODELS[vendor]
    except KeyError:
        raise KeyError(
            f"No calibrated noise model for vendor {vendor!r}; "
            f"available: {sorted(NOISE_MODELS)}"
        ) from None
