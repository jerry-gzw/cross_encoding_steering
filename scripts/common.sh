#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

if [[ -z "${NDD_PROJECT_ROOT:-}" ]]; then
  NDD_PROJECT_ROOT="${REPO_ROOT}"
fi
export NDD_PROJECT_ROOT

log_step() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

artifact_cli() {
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${NDD_PYTHON_BIN:-python}" -m cross_encoding_steering.cli "$@"
}

model_args=()
if [[ -n "${NDD_MODELS:-}" ]]; then
  model_args+=(--models "${NDD_MODELS}")
fi
if [[ -n "${NDD_MODEL_SOURCE_OVERRIDES:-}" ]]; then
  model_args+=(--model-source-overrides "${NDD_MODEL_SOURCE_OVERRIDES}")
fi
