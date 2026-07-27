#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_STRICT_DATA_CONFIG:-${REPO_ROOT}/configs/normbank_group_disjoint_data.example.json}"
OUTPUT_DIR="${NDD_STRICT_PREPARED_DIR:-${NDD_PROJECT_ROOT}/outputs/prepared/normbank_group_disjoint}"

log_step "Build group- and endpoint-disjoint NormBank pairs"
artifact_cli prepare-normbank \
  --config "${CONFIG}" \
  --project-root "${NDD_PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
