#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

CONFIG="${NDD_SC101_CONFIG:-${REPO_ROOT}/configs/sc101_letter_permutation.example.json}"

log_step "SC101 action-only exhaustive letter-mapping scope audit"
artifact_cli run-letter-permutations \
  --config "${CONFIG}" \
  --project-root "${NDD_PROJECT_ROOT}" \
  --phase "${NDD_SC101_PHASE:-all}" "${model_args[@]}"
