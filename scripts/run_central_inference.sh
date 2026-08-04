#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_CENTRAL_INFERENCE_CONFIG:-${REPO_ROOT}/configs/central_group_cluster_inference.example.json}"
log_step "Run setting-behavior group-cluster inference (CPU)"
CUDA_VISIBLE_DEVICES="" artifact_cli run-central-inference --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}"
