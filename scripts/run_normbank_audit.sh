#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

TASK="${NDD_NORMBANK_TASK:-all}"

run_cross_interface() {
  log_step "NormBank: frozen-direction cross-interface audit"
  artifact_cli run-cross-interface \
    --config "${REPO_ROOT}/configs/normbank_cross_interface_steering.example.json" \
    --project-root "${NDD_PROJECT_ROOT}" "${model_args[@]}"
}

run_nuisance() {
  log_step "NormBank: mapping-balanced, label-cue, and nuisance baselines"
  artifact_cli run-cross-interface \
    --config "${REPO_ROOT}/configs/normbank_interface_nuisance_baselines.example.json" \
    --project-root "${NDD_PROJECT_ROOT}" "${model_args[@]}"
}

run_permutations() {
  log_step "NormBank: exhaustive A/B/C permutation audit"
  artifact_cli run-letter-permutations \
    --config "${REPO_ROOT}/configs/normbank_letter_permutation_audit.example.json" \
    --project-root "${NDD_PROJECT_ROOT}" \
    --phase "${NDD_LETTER_PHASE:-all}" "${model_args[@]}"
}

run_context() {
  log_step "NormBank: matched-context selectivity"
  CUDA_VISIBLE_DEVICES="" artifact_cli run-context-selectivity \
    --config "${REPO_ROOT}/configs/normbank_context_discrimination.example.json" \
    --project-root "${NDD_PROJECT_ROOT}"
}

run_statistics() {
  log_step "NormBank: paired statistical synthesis"
  CUDA_VISIBLE_DEVICES="" artifact_cli summarize-interface-controls \
    --config "${REPO_ROOT}/configs/interface_statistical_synthesis.example.json" \
    --project-root "${NDD_PROJECT_ROOT}"
}

case "${TASK}" in
  cross-interface) run_cross_interface ;;
  nuisance) run_nuisance ;;
  permutations) run_permutations ;;
  context) run_context ;;
  statistics) run_statistics ;;
  all)
    run_cross_interface
    run_nuisance
    run_permutations
    run_context
    run_statistics
    ;;
  *)
    echo "Unknown NDD_NORMBANK_TASK=${TASK}" >&2
    echo "Use: cross-interface, nuisance, permutations, context, statistics, or all" >&2
    exit 2
    ;;
esac
