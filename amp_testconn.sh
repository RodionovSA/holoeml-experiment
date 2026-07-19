#!/usr/bin/env bash
#
# Pre-flight check: probes camera, monochromator, filter wheel, focus
# motor, and polarizer, then prints a PASS/FAIL summary per device.
# Thin wrapper around scripts/amplitude/test_connection.py.
#
# Examples:
#   ./amp_testconn.sh                      

uv run python -m scripts.amplitude.test_connection