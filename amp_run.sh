#!/usr/bin/env bash
#
# Run amplitude brightness calibration and/or measurements.
# Thin wrapper around scripts/amplitude/main.py — all arguments are
# forwarded as-is, so see that file's docstring/--help for options.
#
# Examples:
#   ./amp_run.sh --measure sample reference
#   ./amp_run.sh --calibrate --pol x --measure sample reference black
#   ./amp_run.sh --calibrate --measure sample --dry-run
#   ./amp_run.sh -c /path/to/config.yaml --measure sample

uv run python -m scripts.amplitude.main "$@"