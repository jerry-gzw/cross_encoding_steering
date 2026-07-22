#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_CAA_CONFIG:-${REPO_ROOT}/configs/published_caa_protocol_audit.example.json}"
PHASE="${NDD_CAA_PHASE:-all}"
args=(
  run-published-caa
  --config "${CONFIG}"
  --project-root "${NDD_PROJECT_ROOT}"
  --phase "${PHASE}"
)
if [[ -n "${NDD_CAA_MODELS:-}" ]]; then
  args+=(--models "${NDD_CAA_MODELS}")
fi
if [[ -n "${NDD_CAA_BEHAVIORS:-}" ]]; then
  args+=(--behaviors "${NDD_CAA_BEHAVIORS}")
fi
if [[ -n "${NDD_CAA_JUDGES:-}" ]]; then
  args+=(--judges "${NDD_CAA_JUDGES}")
fi
if [[ -n "${NDD_MODEL_SOURCE_OVERRIDES:-}" ]]; then
  args+=(--model-source-overrides "${NDD_MODEL_SOURCE_OVERRIDES}")
fi

log_step "Published CAA audit: phase=${PHASE}"
artifact_cli "${args[@]}"
