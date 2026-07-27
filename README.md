# What Does Activation Steering Control?

Code for the paper **What Does Activation Steering Control? Cross-Interface Evaluation for Alignment**.

The central experiment extracts an activation-steering direction once, freezes
it, and changes how the same answer meanings are represented. The released
pipeline tests whether the resulting movement follows the current label,
the answer identifier used during extraction, or the row in which that
identifier originally appeared.

## Evidence included

- **Strict NormBank identification:** all six A/B/C mappings on
  setting-behavior-group- and endpoint-disjoint splits.
- **Factorial attribution:** independently varies semantic mapping, answer
  identifier vocabulary, and displayed row order.
- **Cross-method replication:** CAA-style residual-stream addition and an
  ITI-style head-probe intervention.
- **SC101 scope replication:** action-only, weakly matched Social Chemistry 101
  pairs provide a contrasting task-level interface profile.
- **Matched-context selectivity:** asks whether one frozen intervention
  distinguishes contexts that share a setting and behavior but differ in
  constraints.
- **Non-norm control:** applies the same interface protocol to same-premise
  MultiNLI pairs.
- **Published CAA case study:** compares the original multiple-choice and
  open-ended evaluation branches for hallucination, refusal, and sycophancy.

## Layout

```text
configs/                    Reproduction configurations
datasets/                   Place raw datasets here
scripts/                    Resumable experiment runners
src/cross_interface_steering/
                             Pairing, interventions, scoring, and statistics
tests/                      CPU unit tests
paper_results/              Compact machine-readable reported summaries
outputs/                    Generated at runtime and ignored by Git
```

Raw datasets, model weights, API credentials, and generated outputs are not
redistributed.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu,dev]"
```

Open-ended automated judging additionally requires:

```bash
pip install -e ".[judges]"
```

The reported environment used Ubuntu 22.04.5 LTS, CUDA 12.2, Python 3.11.7,
PyTorch 2.7.0, Transformers 4.52.4, two NVIDIA A100-PCIE-40GB GPUs, an AMD
EPYC 7702P CPU, and 503 GB RAM.

Scripts infer the repository root. `NDD_PROJECT_ROOT` is optional and is only
needed when datasets and outputs live elsewhere.

## Data

Place the raw files at:

```text
datasets/NormBank.csv
datasets/social-chem-101.v1.0.tsv
datasets/multinli_1.0_train.jsonl
datasets/CAA/
```

Alternative paths accepted by the configs are documented in
[DATASETS.md](DATASETS.md). Fetch the public CAA reference repository with:

```bash
bash scripts/fetch_caa_reference.sh
```

## Models

The main NormBank experiments use approximately 75%-depth residual-stream
blocks and `alpha=0.8`:

| Alias | Model | Zero-based block |
|---|---|---:|
| `qwen2_5_7b_instruct` | Qwen2.5-7B-Instruct | 20 |
| `llama3_1_8b_instruct` | Llama-3.1-8B-Instruct | 23 |
| `mistral7b_instruct_v03` | Mistral-7B-Instruct-v0.3 | 23 |
| `gemma2_9b_it` | Gemma-2-9B-IT | 31 |

SC101 uses the middle-layer settings recorded in
`configs/sc101_letter_permutation.example.json`. The ITI-style audit selects
heads and intervention strength on validation data. The published CAA case
study follows that protocol's own layers and multipliers.

Local model paths can be supplied without editing tracked configs:

```bash
export NDD_MODEL_SOURCE_OVERRIDES='{
  "mistral7b_instruct_v03": "/path/to/Mistral-7B-Instruct-v0.3"
}'
```

Run a subset of models with:

```bash
export NDD_MODELS=qwen2_5_7b_instruct,llama3_1_8b_instruct
```

The caller selects GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 ...
```

## Reproduce the main evidence

### 1. Strict NormBank split

```bash
bash scripts/prepare_normbank_strict.sh
```

This creates group- and endpoint-disjoint train, validation, and test splits in
`outputs/prepared/normbank_group_disjoint/` and writes a split-isolation audit.

### 2. Exhaustive CAA mapping audit

```bash
CUDA_VISIBLE_DEVICES=0 \
NDD_STRICT_TASK=caa \
bash scripts/run_normbank_strict_audit.sh
```

To distribute models across GPUs, invoke the same command in separate shells
with different `CUDA_VISIBLE_DEVICES` and `NDD_MODELS`, then aggregate:

```bash
NDD_STRICT_TASK=caa NDD_STRICT_CAA_PHASE=summarize \
bash scripts/run_normbank_strict_audit.sh
```

### 3. Identifier-position-semantics factorial

```bash
CUDA_VISIBLE_DEVICES=0 \
NDD_STRICT_TASK=factorial \
NDD_FACTORIAL_PHASE=run \
bash scripts/run_normbank_strict_audit.sh

NDD_STRICT_TASK=factorial \
NDD_FACTORIAL_PHASE=aggregate \
bash scripts/run_normbank_strict_audit.sh

NDD_STRICT_TASK=factorial-statistics \
bash scripts/run_normbank_strict_audit.sh
```

### 4. ITI-style replication

```bash
CUDA_VISIBLE_DEVICES=0 \
NDD_STRICT_TASK=iti \
NDD_ITI_PHASE=run \
bash scripts/run_normbank_strict_audit.sh

NDD_STRICT_TASK=iti NDD_ITI_PHASE=summarize \
bash scripts/run_normbank_strict_audit.sh
```

The primary ITI aggregate contains only models that pass the prespecified
source-interface validation gate; the competence table reports every model.

### 5. SC101 scope replication

```bash
bash scripts/prepare_sc101.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sc101_audit.sh
```

Use `NDD_SC101_PHASE=run`, `aggregate`, or `summarize` to resume individual
phases.

## Supporting evidence

The original pair-ID split remains useful for direct-label and opaque-codeword
interfaces, nuisance directions, and matched-context diagnostics:

```bash
bash scripts/prepare_normbank.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_normbank_audit.sh
```

Each component can be resumed with `NDD_NORMBANK_TASK`:
`cross-interface`, `nuisance`, `permutations`, `context`, or `statistics`.

Run the same-premise MultiNLI control with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mnli_control.sh
```

Run the published CAA case study with:

```bash
NDD_CAA_PHASE=prepare bash scripts/run_published_caa_audit.sh
CUDA_VISIBLE_DEVICES=0 NDD_CAA_PHASE=run \
  NDD_CAA_MODELS=llama2_7b_chat bash scripts/run_published_caa_audit.sh
CUDA_VISIBLE_DEVICES=1 NDD_CAA_PHASE=run \
  NDD_CAA_MODELS=llama2_13b_chat bash scripts/run_published_caa_audit.sh
NDD_CAA_PHASE=aggregate bash scripts/run_published_caa_audit.sh
```

API judging is disabled by default. Enable only the desired entries in a local
copy of `configs/published_caa_protocol_audit.example.json`, set the provider
API key, and run `NDD_CAA_PHASE=judge`.

## Principal outputs

| Evidence | Output directory |
|---|---|
| Strict CAA mappings | `outputs/normbank/group_disjoint_letter_permutations/` |
| Factorial attribution | `outputs/normbank/interface_factorial/` |
| ITI replication | `outputs/normbank/iti_probe_intervention/` |
| SC101 scope | `outputs/sc101/letter_permutations/` |
| Supporting NormBank audit | `outputs/normbank/` |
| MultiNLI control | `outputs/mnli_control/` |
| Published CAA case study | `outputs/published_caa/` |

`paper_results/` contains compact copies of the reported summaries and an
evidence-to-file map. The experiment commands above regenerate their upstream
outputs from raw datasets and public model checkpoints.

## Tests

```bash
pytest -q
```

The CPU tests cover split isolation, prompt and interface construction,
factorial attribution, ITI probe selection and controls, permutation
statistics, published CAA aggregation, and last-token indexing.

## License

MIT. Dataset and model licenses remain with their original publishers.
