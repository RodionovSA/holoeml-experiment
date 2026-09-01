"""Piezo phase-shift sweep -> phase solve -> combine. Generic over PhaseConfig/method."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, replace

import numpy as np

from phase import (CarrierResult, CombinedResult, DifferenceResult, PhaseConfig,
                    PhaseResult, PhaseSolver, combine_acquisitions, remove_carrier,
                    subtract_reference)
from phase.backend import CUPY_AVAILABLE, asnumpy, get_array_module

from .context import MeasurementContext


def _release_gpu_memory() -> None:
    """Return CuPy's pooled (cached, not live) VRAM to the driver. No-op without CuPy."""
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
    result.print_summary()
    return result


def _solve_and_release(images: np.ndarray, phase_config: PhaseConfig, device: str) -> PhaseResult:
    """One repeat's full post-acquisition work: solve, bring to host, release GPU memory.

    The unit submitted to the background thread in `measure_phase`'s
    acquire/solve pipelining.
    """
    result = solve_stack(images, phase_config, device).to_device("cpu")
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

    `combined` is the merged phase map, left on `device`. `results` is
    every repeat's full `PhaseResult`, already moved to host. `phase_config`
    is what produced them.
    """
    combined: CombinedResult
    results: list[PhaseResult]
    phase_config: PhaseConfig


def measure_phase(ctx: MeasurementContext,
                   phase_config: PhaseConfig | None = None,
                   num_averages: int | None = None,
                   device: str = "auto") -> SweepResult:
    """Acquire `num_averages` forward repeats, solve each, and combine them.

    One leading `dry_run(+1)` covers the whole forward run; afterward a
    `dry_run(-1)` + `return_sweep` covering `num_averages * num_piezo_return_steps`
    moves brings the piezo back. Each repeat's solve runs in a background
    thread while the next repeat's acquisition proceeds on the main thread,
    so the piezo/camera aren't idle during the solve; each repeat is then
    moved to host and released before its result is collected, so peak VRAM
    stays bounded regardless of `num_averages`.

    Returns a `SweepResult` -- nothing pre-extracted, take what you need.
    """
    cfg = ctx.config
    n = num_averages if num_averages is not None else cfg.num_averages
    phase_config = phase_config or PhaseConfig()

    ctx.dry_run(+1)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for _ in range(n):
            images = ctx.acquire_stack(direction=+1)
            if pending is not None:
                results.append(pending.result())
            pending = pool.submit(_solve_and_release, images, phase_config, device)
        results.append(pending.result())

    ctx.dry_run(-1)
    ctx.return_sweep(cfg.num_piezo_return_steps * n)

    if n == 1:
        combined = _single_acquisition(results[0].phi, results[0].b, device)
    else:
        phi = [r.phi for r in results]
        b = [r.b for r in results]
        combined = combine_acquisitions(phi, weights=b, device=device)
        if combined.sign_flips:
            print(f"Sign flips among repeats: {combined.sign_flips}")
    _release_gpu_memory()

    return SweepResult(combined=combined, results=results, phase_config=phase_config)


def _combined_to_host(c: CombinedResult) -> CombinedResult:
    return CombinedResult(phi=asnumpy(c.phi), scatter=asnumpy(c.scatter),
                           mean_resultant=asnumpy(c.mean_resultant), n=c.n, sign_flips=c.sign_flips)


@dataclass
class PhaseMeasurement:
    """One full sample+reference phase measurement.

    `diff` is the sign-resolved sample-minus-reference phase, before carrier
    removal; `carrier` is the final object phase with carrier/defocus/piston
    removed. `weight` is the joint sample*reference reliability map used for
    both steps. `sample`/`reference` are the two `SweepResult`s the phase
    came from. Every array here, including `sample`/`reference`'s
    `combined`, is host (numpy) regardless of `device`.
    """
    diff: DifferenceResult
    carrier: CarrierResult
    weight: np.ndarray
    sample: SweepResult
    reference: SweepResult


def full_phase_measurement(ctx: MeasurementContext,
                            phase_config: PhaseConfig | None = None,
                            num_averages: int | None = None,
                            device: str = "auto") -> PhaseMeasurement:
    """Measure the sample, measure the reference, subtract, remove the carrier.

    Arms the camera, runs `measure_phase` for the sample, moves to the
    reference position (`ctx.stage_reference`, restored even on failure) and
    runs `measure_phase` again for the reference, then resolves the sign
    branch and removes the residual carrier. Returns a `PhaseMeasurement`
    with every array brought back to host.
    """
    with ctx.armed():
        sample = measure_phase(ctx, phase_config, num_averages, device)
        with ctx.stage_reference():
            reference = measure_phase(ctx, phase_config, num_averages, device)

    weight = sample.combined.mean_resultant * reference.combined.mean_resultant
    diff = subtract_reference(sample.combined.phi, reference.combined.phi, weight, device=device)
    carrier = remove_carrier(diff.phi, weight, defocus=True, refine_iters=10,
                              n_blocks=10, device=device)

    result = PhaseMeasurement(
        diff=replace(diff, phi=asnumpy(diff.phi)),
        carrier=replace(carrier, phi=asnumpy(carrier.phi)),
        weight=asnumpy(weight),
        sample=replace(sample, combined=_combined_to_host(sample.combined)),
        reference=replace(reference, combined=_combined_to_host(reference.combined)),
    )
    _release_gpu_memory()
    return result
