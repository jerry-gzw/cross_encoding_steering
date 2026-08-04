#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_READOUT_TRANSFER_CONFIG:-${REPO_ROOT}/configs/normbank_readout_vocabulary_transfer.example.json}"
PHASE="${NDD_READOUT_TRANSFER_PHASE:-all}"
log_step "Run cross-vocabulary readout transfer (${PHASE})"
artifact_cli run-readout-vocabulary-transfer --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" --phase "${PHASE}" "${model_args[@]}"
