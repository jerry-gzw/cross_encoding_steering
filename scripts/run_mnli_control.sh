#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

PHASE="${NDD_MNLI_PHASE:-all}"
CONFIG="${NDD_MNLI_CONFIG:-${REPO_ROOT}/configs/mnli_non_norm_control.example.json}"

if [[ "${PHASE}" == "run" || "${PHASE}" == "all" ]]; then
  log_step "MNLI: matched non-norm cross-encoding control"
  artifact_cli run-mnli-control \
    --config "${CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}" "${model_args[@]}"
fi

if [[ "${PHASE}" == "statistics" || "${PHASE}" == "all" ]]; then
  log_step "MNLI: paired bootstrap statistics"
  CUDA_VISIBLE_DEVICES="" artifact_cli summarize-mnli \
    --config "${CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}"
fi
