"""Piezo phase-shift sweep -> phase solve -> combine. Generic over PhaseConfig/method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phase import (CombinedResult, PhaseConfig, PhaseResult, PhaseSolver,
                    combine_acquisitions, remove_carrier)
from phase.backend import CUPY_AVAILABLE, asnumpy, get_array_module

from .context import MeasurementContext


def _release_gpu_memory() -> None:
    """Return CuPy's pooled (cached, not live) VRAM to the driver.

    Called after every repeat's solve (see `solve_stack`): once
    `PhaseSolver.fit` returns, only the small `PhaseResult` fields are still
    referenced, so the pool's cached blocks for everything else the solve
    allocated (the full frame stack, internal working arrays) can be freed
    immediately rather than sitting cached until the next allocation happens
    to reuse them. A no-op on a CPU-only install.
    """
    if CUPY_AVAILABLE:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def solve_stack(images: np.ndarray, phase_config: PhaseConfig | None = None,
                 device: str = "auto") -> PhaseResult:
    """Recover phase from one interferogram stack.

    Thin wrapper around `PhaseSolver.fit` -- generic over whichever method
    `phase_config` selects. Defaults to a plain `PhaseConfig()` (method="aia").
    """
    result = PhaseSolver(phase_config or PhaseConfig(), device=device).fit(images).result_
    _release_gpu_memory()
    return result


def _single_acquisition(phi: np.ndarray, weight: np.ndarray, device: str) -> CombinedResult:
    """Standardize a lone repeat into the same shape `combine_acquisitions` returns
    for n >= 2: carrier/defocus/piston removed (the same defaults
    `combine_acquisitions` applies internally), trivial `scatter=0`/`mean_resultant=1`
    -- there's nothing to disagree with at n=1.
    """
    p = remove_carrier(phi, weight=weight, defocus=True, refine_iters=10,
                        n_blocks=10, device=device).phi
    xp = get_array_module(p)
    return CombinedResult(phi=p, scatter=xp.zeros_like(p), mean_resultant=xp.ones_like(p),
                           n=1, sign_flips=[])


@dataclass
class SweepResult:
    """`measure_phase`'s result for one sample- or reference-side sweep.

    `combined` is the merged phase map (see `measure_phase`). Everything
    else is per-repeat, first axis length `num_averages`, preserved from
    each repeat's `PhaseResult` rather than discarded once combined -- so
    convergence/fit-quality/drift across repeats stays inspectable.
    `method_param` is kept as a plain list (not stacked): it's opaque and
    method-specific (e.g. an aia-only `AIAParam` for `method="aia"`), not a
    fixed-shape array, and is **not** host-converted -- it may still be
    device-resident; convert its fields yourself if you need to save it.
    Every other field (including `combined`'s) is always a plain host numpy
    array regardless of `device` -- safe to hand straight to `ctx.save_npz`
    -- and is the full per-repeat data, not reduced, so memory scales with
    `num_averages`.
    """
    combined: CombinedResult
    a: np.ndarray
    b: np.ndarray
    delta: np.ndarray
    g: np.ndarray
    alpha: np.ndarray
    reconstruction_error: np.ndarray
    method_param: list[Any]


def measure_phase(ctx: MeasurementContext,
                   phase_config: PhaseConfig | None = None,
                   num_averages: int | None = None,
                   device: str = "auto") -> SweepResult:
    """Acquire `num_averages` forward repeats, solve each, and combine them.

    One leading `dry_run(+1)` covers the whole forward run (repeats keep walking
    further forward, same as within one `acquire_stack` call); afterward a
    `dry_run(-1)` + `return_sweep` covering `num_averages * num_piezo_return_steps`
    moves brings the piezo back. Each repeat is solved independently via
    `solve_stack`; their phase/modulation maps are merged with
    `combine_acquisitions`.

    Returns a `SweepResult`: the combined phase map alongside every repeat's
    full `PhaseResult` fields, so nothing from the solve is thrown away.
    Always the same shape regardless of `num_averages` -- `1` goes through
    the same standardized path as `n >= 2` (see `_single_acquisition`), just
    with every per-repeat field having length 1.
    """
    cfg = ctx.config
    n = num_averages if num_averages is not None else cfg.num_averages

    ctx.dry_run(+1)
    results = [solve_stack(ctx.acquire_stack(direction=+1), phase_config, device)
               for _ in range(n)]

    ctx.dry_run(-1)
    ctx.return_sweep(cfg.num_piezo_return_steps * n)

    xp = get_array_module(results[0].phi)
    phi = xp.stack([r.phi for r in results])
    b = xp.stack([r.b for r in results])

    if n == 1:
        combined = _single_acquisition(phi[0], b[0], device)
    else:
        combined = combine_acquisitions(phi, weights=b, device=device)
        if combined.sign_flips:
            print(f"Sign flips among repeats: {combined.sign_flips}")

    # Host conversion happens here, at measure_phase's boundary -- not inside
    # solve_stack/_single_acquisition, which stay device-consistent like the
    # library's own functions. Everything else in phasectl (ctx.save_npz
    # included) assumes plain numpy, so this is where hardware/ctx-driven
    # acquisition needs to cross back into that world. Doing this *before*
    # the GPU-memory release (matching the old script's own asnumpy-then-
    # release order) also means the release can reclaim these result fields
    # too, not just the combine/carrier-removal step's own transients.
    result = SweepResult(
        combined=CombinedResult(phi=asnumpy(combined.phi), scatter=asnumpy(combined.scatter),
                                 mean_resultant=asnumpy(combined.mean_resultant),
                                 n=combined.n, sign_flips=combined.sign_flips),
        a=asnumpy(xp.stack([r.a for r in results])),
        b=asnumpy(b),
        delta=asnumpy(xp.stack([r.delta for r in results])),
        g=asnumpy(xp.stack([r.g for r in results])),
        alpha=asnumpy(xp.stack([r.alpha for r in results])),
        reconstruction_error=np.asarray([r.reconstruction_error for r in results]),
        method_param=[r.method_param for r in results],
    )
    _release_gpu_memory()
    return result
