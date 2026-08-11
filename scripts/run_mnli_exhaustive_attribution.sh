#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

PHASE="${NDD_MNLI_ATTRIBUTION_PHASE:-all}"
CONFIG="${NDD_MNLI_ATTRIBUTION_CONFIG:-${REPO_ROOT}/configs/mnli_exhaustive_attribution.example.json}"
GPU_0="${NDD_MNLI_ATTRIBUTION_GPU_0:-0}"
GPU_1="${NDD_MNLI_ATTRIBUTION_GPU_1:-1}"
GPU_0_MODELS="${NDD_MNLI_ATTRIBUTION_GPU_0_MODELS:-qwen2_5_7b_instruct,gemma2_9b_it}"
GPU_1_MODELS="${NDD_MNLI_ATTRIBUTION_GPU_1_MODELS:-llama3_1_8b_instruct,mistral7b_instruct_v03}"

run_cli() {
  local phase="$1"
  local models="${2:-}"
  local -a args=(
    run-mnli-control
    --config "${CONFIG}"
    --project-root "${NDD_PROJECT_ROOT}"
    --phase "${phase}"
  )
  if [[ -n "${models}" ]]; then
    args+=(--models "${models}")
  elif [[ -n "${NDD_MODELS:-}" ]]; then
    args+=(--models "${NDD_MODELS}")
  fi
  if [[ -n "${NDD_MODEL_SOURCE_OVERRIDES:-}" ]]; then
    args+=(--model-source-overrides "${NDD_MODEL_SOURCE_OVERRIDES}")
  fi
  artifact_cli "${args[@]}"
}

preflight() {
  log_step "Preflight exhaustive MNLI attribution design"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${NDD_PYTHON_BIN:-python}" - "${CONFIG}" "${NDD_PROJECT_ROOT}" <<'PY'
import itertools
import sys

from cross_encoding_steering.mnli_control import MnliControlConfig

config = MnliControlConfig.from_json(sys.argv[1], project_root=sys.argv[2])
orders = [interface.option_order for interface in config.interfaces if interface.kind == "letter_mcq"]
expected = set(itertools.permutations(("entailment", "neutral", "contradiction")))
if set(orders) != expected:
    raise ValueError("The MNLI attribution config must contain every A/B/C label assignment exactly once")
if not config.attribution_analysis:
    raise ValueError("attribution_analysis must be true")
if len(config.random_seeds) != 5:
    raise ValueError("The paper-ready audit requires five fixed random seeds")
if config.include_inverse_control:
    raise ValueError("The exhaustive attribution run should not include the legacy inverse control")
if not config.input_path.exists():
    raise FileNotFoundError(config.input_path)
print(f"input={config.input_path}")
print(f"output={config.output_dir}")
print(f"models={len(config.models)}; contrasts={len(config.contrasts)}; templates={len(config.templates)}")
print(f"letter mappings={len(orders)}; random controls={len(config.random_seeds)}")
print("direction, layer, position, and dose remain frozen across all six mappings")
PY
}

case "${PHASE}" in
  preflight)
    preflight
    ;;
  run)
    log_step "Run exhaustive MNLI attribution on current CUDA_VISIBLE_DEVICES"
    run_cli run "${NDD_MNLI_ATTRIBUTION_MODELS:-}"
    ;;
  parallel)
    preflight
    log_step "Run MNLI attribution workers on GPUs ${GPU_0} and ${GPU_1}"
    (export CUDA_VISIBLE_DEVICES="${GPU_0}"; run_cli run "${GPU_0_MODELS}") &
    pid_0=$!
    (export CUDA_VISIBLE_DEVICES="${GPU_1}"; run_cli run "${GPU_1_MODELS}") &
    pid_1=$!
    status=0
    wait "${pid_0}" || status=$?
    wait "${pid_1}" || status=$?
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi
    ;;
  aggregate)
    log_step "Aggregate completed MNLI model outputs (CPU only)"
    CUDA_VISIBLE_DEVICES="" run_cli aggregate
    ;;
  statistics)
    log_step "Build premise-cluster attribution intervals (CPU only)"
    CUDA_VISIBLE_DEVICES="" artifact_cli summarize-mnli \
      --config "${CONFIG}" \
      --project-root "${NDD_PROJECT_ROOT}"
    ;;
  all)
    NDD_MNLI_ATTRIBUTION_PHASE=parallel bash "${BASH_SOURCE[0]}"
    NDD_MNLI_ATTRIBUTION_PHASE=aggregate bash "${BASH_SOURCE[0]}"
    NDD_MNLI_ATTRIBUTION_PHASE=statistics bash "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "Unknown NDD_MNLI_ATTRIBUTION_PHASE=${PHASE}; use preflight, run, parallel, aggregate, statistics, or all" >&2
    exit 2
    ;;
esac
