#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

INPUT="${NDD_MIC_INPUT:-${NDD_PROJECT_ROOT}/datasets/MIC.csv}"
PREPARED="${NDD_PROJECT_ROOT}/outputs/prepared/mic"
POSITION="${NDD_MIC_POSITION:-both}"

if [[ "${NDD_MIC_PREPARE:-1}" == "1" ]]; then
  log_step "Prepare MIC rule-conditioned pairs"
  artifact_cli prepare-mic --input "${INPUT}" --output-dir "${PREPARED}"
fi

run_position() {
  local name="$1"
  local config="${REPO_ROOT}/configs/mic_${name}_fixed_mapping.example.json"
  log_step "Run MIC ${name//_/ } fixed-direction sensitivity"
  artifact_cli run-mapping-audit --config "${config}" --project-root "${NDD_PROJECT_ROOT}" "${model_args[@]}"
}

case "${POSITION}" in
  pre_answer) run_position pre_answer ;;
  scenario_end) run_position scenario_end ;;
  both) run_position pre_answer; run_position scenario_end ;;
  *) echo "Unknown NDD_MIC_POSITION: ${POSITION}" >&2; exit 2 ;;
esac
