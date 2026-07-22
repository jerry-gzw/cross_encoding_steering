#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_VALIDITY_CONFIG:-${REPO_ROOT}/configs/evaluation_validity_controls.example.json}"
PHASE="${NDD_VALIDITY_PHASE:-all}"

log_step "Evaluation validity controls: phase=${PHASE}"
artifact_cli run-validity-controls \
  --config "${CONFIG}" \
  --project-root "${NDD_PROJECT_ROOT}" \
  --phase "${PHASE}" "${model_args[@]}"
