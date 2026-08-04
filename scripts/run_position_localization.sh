#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_POSITION_CONFIG:-${REPO_ROOT}/configs/normbank_group_disjoint_position_audit.example.json}"
log_step "Run extraction-position localization"
artifact_cli run-position-audit --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" "${model_args[@]}"
