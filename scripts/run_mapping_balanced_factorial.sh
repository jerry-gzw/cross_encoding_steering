#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_MAPPING_BALANCED_CONFIG:-${REPO_ROOT}/configs/normbank_group_disjoint_mapping_balanced_factorial.example.json}"
PHASE="${NDD_MAPPING_BALANCED_PHASE:-all}"
log_step "Run in-family mapping-balanced factorial audit (${PHASE})"
artifact_cli run-interface-factorial --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}" --phase "${PHASE}" "${model_args[@]}"
