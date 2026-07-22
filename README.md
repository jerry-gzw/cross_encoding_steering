# Cross-Interface Evaluation of Activation Steering

Code for the paper **Do Activation-Steering Gains Survive the Answer Interface?**

The repository evaluates a contrastive activation direction after freezing it
and changing how the same answer labels are represented. It contains the code
for the paper's three questions:

1. Does a frozen steering effect remain detectable across answer interfaces?
2. Does steering improve discrimination between matched contexts with
   different constraints?
3. Can the original multiple-choice and open-ended evaluations of a published
   CAA protocol yield different verdicts?

## Included experiments

- **NormBank cross-interface audit:** original A/B/C, all five non-source A/B/C
  mappings, direct labels, and opaque codewords.
- **Direction controls:** mapping-balanced, mapping-sensitive, label-cue,
  slot-layout, wrong-direction, and five norm-matched Gaussian controls.
- **Matched-context selectivity:** pair-level interaction and ranking diagnostics
  on setting-behavior-matched NormBank pairs.
- **MNLI control:** the same interface protocol on a matched non-norm
  three-class task.
- **Published CAA case study:** original and swapped A/B, direct text, opaque
  codewords, key competence, and open-ended generation for hallucination,
  refusal, and sycophancy.

## Repository layout

```text
configs/                    Reproduction configurations
datasets/                   Place raw datasets here; data are not redistributed
scripts/                    Composable experiment runners
src/cross_interface_steering/
                             Pairing, steering, scoring, controls, and statistics
tests/                      CPU unit tests for the released analysis code
outputs/                    Generated at runtime and ignored by Git
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu]"
```

CPU-only data preparation and statistical aggregation require only the base
dependencies:

```bash
pip install -e .
```

The scripts infer the repository root automatically. Set `NDD_PROJECT_ROOT`
only when `datasets/` and `outputs/` should live under another directory.

## Data

Place the following files under `datasets/`:

```text
datasets/NormBank.csv
datasets/multinli_1.0_train.jsonl
datasets/CAA/                       cloned public CAA repository
```

See [DATASETS.md](DATASETS.md) for sources, schemas, and exactly which fields
are used. To fetch the public CAA reference repository:

```bash
bash scripts/fetch_caa_reference.sh
```

## Models and locked layers

The controlled NormBank and MNLI experiments use a common 75%-depth rule:

| Alias | Model | Zero-based block | Decoder blocks |
|---|---|---:|---:|
| `qwen2_5_7b_instruct` | Qwen2.5-7B-Instruct | 20 | 28 |
| `llama3_1_8b_instruct` | Llama-3.1-8B-Instruct | 23 | 32 |
| `mistral7b_instruct_v03` | Mistral-7B-Instruct-v0.3 | 23 | 32 |
| `gemma2_9b_it` | Gemma-2-9B-IT | 31 | 42 |

The intervention strength is `alpha=0.8`. The published CAA replication uses
the layers and multipliers specified by that protocol; see
`configs/published_caa_protocol_audit.example.json`.

Local model paths can be supplied without editing a config:

```bash
export NDD_MODEL_SOURCE_OVERRIDES='{
  "mistral7b_instruct_v03": "/path/to/Mistral-7B-Instruct-v0.3"
}'
```

Use `NDD_MODELS` to run a subset of the four controlled-audit models:

```bash
export NDD_MODELS=qwen2_5_7b_instruct,llama3_1_8b_instruct
```

## Reproduction

### 1. Prepare NormBank

```bash
bash scripts/prepare_normbank.sh
```

This writes deterministic pair-ID splits, pair diagnostics, and ranked
endpoints to `outputs/prepared/normbank/`.

### 2. Run the NormBank audit

Run the full sequence on the GPU selected by the caller:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_normbank_audit.sh
```

Individual tasks can be resumed without rerunning earlier outputs:

```bash
CUDA_VISIBLE_DEVICES=0 NDD_NORMBANK_TASK=cross-interface \
  bash scripts/run_normbank_audit.sh
CUDA_VISIBLE_DEVICES=0 NDD_NORMBANK_TASK=nuisance \
  bash scripts/run_normbank_audit.sh
CUDA_VISIBLE_DEVICES=0 NDD_NORMBANK_TASK=permutations \
  bash scripts/run_normbank_audit.sh
NDD_NORMBANK_TASK=context bash scripts/run_normbank_audit.sh
NDD_NORMBANK_TASK=statistics bash scripts/run_normbank_audit.sh
```

### 3. Run the MNLI control

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mnli_control.sh
```

Set `NDD_MNLI_PHASE=run` or `NDD_MNLI_PHASE=statistics` to execute only one
phase.

### 4. Run the published CAA case study

First validate the downloaded CAA files:

```bash
NDD_CAA_PHASE=prepare bash scripts/run_published_caa_audit.sh
```

The two Llama-2 models can be run independently:

```bash
CUDA_VISIBLE_DEVICES=0 NDD_CAA_PHASE=run \
  NDD_CAA_MODELS=llama2_7b_chat bash scripts/run_published_caa_audit.sh
CUDA_VISIBLE_DEVICES=1 NDD_CAA_PHASE=run \
  NDD_CAA_MODELS=llama2_13b_chat bash scripts/run_published_caa_audit.sh
NDD_CAA_PHASE=aggregate bash scripts/run_published_caa_audit.sh
```

Open-ended judging is disabled in the example config. Enable its `judge`
section in a local config, set the configured API-key environment variable,
and run:

```bash
NDD_CAA_CONFIG=/path/to/local_caa_config.json \
NDD_CAA_PHASE=judge bash scripts/run_published_caa_audit.sh
NDD_CAA_PHASE=aggregate bash scripts/run_published_caa_audit.sh
```

### 5. Run validity controls

```bash
CUDA_VISIBLE_DEVICES=0 NDD_VALIDITY_PHASE=random \
  bash scripts/run_validity_controls.sh
NDD_VALIDITY_PHASE=summarize bash scripts/run_validity_controls.sh
```

## Principal outputs

| Evidence | Output directory |
|---|---|
| Cross-interface effects | `outputs/normbank/cross_interface/` |
| Nuisance directions | `outputs/normbank/nuisance_baselines/` |
| Five non-source mappings | `outputs/normbank/letter_permutations/` |
| Context selectivity | `outputs/normbank/context_selectivity/` |
| Control-adjusted statistics | `outputs/interface_statistics/` |
| MNLI control | `outputs/mnli_control/` |
| Published CAA case study | `outputs/published_caa/` |
| Random/key/judge validity checks | `outputs/validity_controls/` |

All GPU runners are resumable at the model directory level unless
`force_rerun` is enabled in the corresponding configuration.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests exercise prompt/interface construction, current-label versus
source-slot accounting, permutation statistics, CAA aggregation, validity
controls, and last-token indexing without downloading a model.
