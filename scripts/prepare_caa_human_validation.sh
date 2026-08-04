#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

CONFIG="${NDD_HUMAN_CONFIG:-${REPO_ROOT}/configs/caa_human_annotation_180.example.json}"
ACTION="${NDD_HUMAN_ACTION:-prepare}"

case "${ACTION}" in
  prepare) COMMAND="prepare-caa-human-annotation" ;;
  summarize) COMMAND="summarize-caa-human-annotation" ;;
  prepare-adjudication) COMMAND="prepare-caa-human-adjudication" ;;
  summarize-adjudication) COMMAND="summarize-caa-human-adjudication" ;;
  *) echo "Unknown NDD_HUMAN_ACTION: ${ACTION}" >&2; exit 2 ;;
esac

log_step "CAA human-validation action: ${ACTION}"
artifact_cli "${COMMAND}" --config "${CONFIG}" --project-root "${NDD_PROJECT_ROOT}"
