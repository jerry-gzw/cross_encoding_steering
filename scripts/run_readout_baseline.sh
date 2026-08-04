#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_READOUT_BASELINE_CONFIG:-${REPO_ROOT}/configs/normbank_readout_baseline_audit.example.json}"
PHASE="${NDD_READOUT_BASELINE_PHASE:-all}"
log_step "Run direct readout baselines (${PHASE})"
artifact_cli run-readout-baseline --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" --phase "${PHASE}" "${model_args[@]}"
