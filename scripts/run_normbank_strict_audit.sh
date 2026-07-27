#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

TASK="${NDD_STRICT_TASK:-all}"
CAA_CONFIG="${NDD_STRICT_CAA_CONFIG:-${REPO_ROOT}/configs/normbank_group_disjoint_letter_permutation.example.json}"
ITI_CONFIG="${NDD_ITI_CONFIG:-${REPO_ROOT}/configs/normbank_group_disjoint_iti_probe.example.json}"
FACTORIAL_CONFIG="${NDD_FACTORIAL_CONFIG:-${REPO_ROOT}/configs/normbank_interface_factorial.example.json}"

run_caa() {
  log_step "Strict NormBank: exhaustive CAA letter mappings"
  artifact_cli run-letter-permutations \
    --config "${CAA_CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}" \
    --phase "${NDD_STRICT_CAA_PHASE:-all}" "${model_args[@]}"
}

run_iti() {
  log_step "Strict NormBank: ITI-style head-probe intervention"
  artifact_cli run-iti-probe \
    --config "${ITI_CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}" \
    --phase "${NDD_ITI_PHASE:-all}" "${model_args[@]}"
}

run_factorial() {
  log_step "Strict NormBank: identifier-position-semantics factorial"
  artifact_cli run-interface-factorial \
    --config "${FACTORIAL_CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}" \
    --phase "${NDD_FACTORIAL_PHASE:-all}" "${model_args[@]}"
}

summarize_factorial() {
  log_step "Strict NormBank: factorial CPU statistics"
  CUDA_VISIBLE_DEVICES="" artifact_cli summarize-interface-factorial \
    --config "${FACTORIAL_CONFIG}" \
    --project-root "${NDD_PROJECT_ROOT}"
}

case "${TASK}" in
  caa) run_caa ;;
  iti) run_iti ;;
  factorial) run_factorial ;;
  factorial-statistics) summarize_factorial ;;
  all)
    run_caa
    run_iti
    run_factorial
    summarize_factorial
    ;;
  *)
    echo "Unknown NDD_STRICT_TASK=${TASK}" >&2
    echo "Use: caa, iti, factorial, factorial-statistics, or all" >&2
    exit 2
    ;;
esac
