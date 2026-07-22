#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

TARGET="${NDD_CAA_REFERENCE_DIR:-${NDD_PROJECT_ROOT}/datasets/CAA}"
if [[ -d "${TARGET}/.git" ]]; then
  echo "CAA reference repository already exists: ${TARGET}"
  exit 0
fi
if [[ -e "${TARGET}" ]]; then
  echo "Target exists but is not a git repository: ${TARGET}" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${TARGET}")"
git clone https://github.com/nrimsky/CAA.git "${TARGET}"
