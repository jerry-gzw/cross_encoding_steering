#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_READOUT_GEOMETRY_CONFIG:-${REPO_ROOT}/configs/normbank_readout_geometry_audit.example.json}"
PHASE="${NDD_READOUT_GEOMETRY_PHASE:-all}"
log_step "Run local readout-geometry audit (${PHASE})"
artifact_cli run-readout-geometry --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" --phase "${PHASE}" "${model_args[@]}"
