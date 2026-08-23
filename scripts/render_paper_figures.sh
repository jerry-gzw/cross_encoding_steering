#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${NDD_PYTHON_BIN:-python}" \
  -m cross_encoding_steering.render_paper_figures \
  --results-dir "${REPO_ROOT}/paper_results" \
  --output-dir "${REPO_ROOT}/paper_results/figures"
