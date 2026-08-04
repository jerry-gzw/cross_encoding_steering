#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_LAYER_CONFIG:-${REPO_ROOT}/configs/normbank_layer_attribution_audit.example.json}"
PHASE="${NDD_LAYER_PHASE:-all}"
log_step "Run layer-wise attribution audit (${PHASE})"
artifact_cli run-layer-attribution --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" --phase "${PHASE}" "${model_args[@]}"
