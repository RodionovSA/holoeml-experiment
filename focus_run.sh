#!/usr/bin/env bash
#
# Run the real-time focusing app.
# Thin wrapper around scripts/focus.py — all arguments are forwarded
# as-is, so see that file's docstring/--help for options.
#
# Examples:
#   ./focus_run.sh
#   ./focus_run.sh --display-width 1200
#   ./focus_run.sh -c /path/to/equipment_config.yaml
#   ./focus_run.sh --profile

uv run python -m scripts.focus "$@"
