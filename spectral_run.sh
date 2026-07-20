#!/usr/bin/env bash
#
# Run the monochromator band spectral acquisition script.
# Thin wrapper around scripts/spectral.py — all arguments are forwarded
# as-is, so see that file's docstring/--help for options.
#
# Examples:
#   ./spectral_run.sh
#   ./spectral_run.sh -o /path/to/output_dir
#   ./spectral_run.sh -c /path/to/equipment_config.yaml

uv run python -m scripts.spectral "$@"
