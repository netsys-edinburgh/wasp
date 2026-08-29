#!/usr/bin/env bash
# End-to-end Wasp run on emulated devices (OPT-125M, tiny config).
# Requires the Wasp Docker image (see the repo Dockerfile).
set -euo pipefail
STEPS="${STEPS:-3}"
COORDS="${COORDS:-2}"
DEVICES_PER_COORD="${DEVICES_PER_COORD:-1}"

wasp run \
  --model-name facebook/opt-125m \
  --coords "${COORDS}" \
  --devices-per-coord "${DEVICES_PER_COORD}" \
  --steps "${STEPS}" \
  --tiny
