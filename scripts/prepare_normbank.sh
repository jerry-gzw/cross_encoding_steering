#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_DATASET_CONFIG:-${REPO_ROOT}/configs/datasets.example.json}"
OUTPUT_DIR="${NDD_NORMBANK_PREPARED_DIR:-${NDD_PROJECT_ROOT}/outputs/prepared/normbank}"

log_step "Check dataset availability"
artifact_cli check-data --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}"

log_step "Build deterministic NormBank pairs and ranked endpoints"
artifact_cli prepare-normbank \
  --config "${CONFIG}" \
  --project-root "${NDD_PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
