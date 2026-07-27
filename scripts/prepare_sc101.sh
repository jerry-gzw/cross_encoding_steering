#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_SC101_DATA_CONFIG:-${REPO_ROOT}/configs/sc101_data.example.json}"
OUTPUT_DIR="${NDD_SC101_PREPARED_DIR:-${NDD_PROJECT_ROOT}/outputs/prepared/sc101}"

log_step "Build SC101 social-norm action-only pairs"
artifact_cli prepare-sc101 \
  --config "${CONFIG}" \
  --project-root "${NDD_PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
