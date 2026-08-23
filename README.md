# What Does Activation Steering Control?

Code and compact result tables for **What Does Activation Steering Control? Attribution Across Answer Encodings and Output-Sensitive Subspaces**.

The core audit extracts one steering direction, freezes it, and changes the
answer encoding while keeping the item, model, direction, layer, position, and
intervention dose fixed. It tests whether score changes follow the semantic
label under the test encoding, the identifier index defined during extraction,
or the extraction-time displayed row. We call the second quantity the
*extraction index*: index 1 is instantiated by A, X, or 1; index 2 by B, Y, or
2; and index 3 by C, Z, or 3. The repository also contains the paper's depth and
position localization, output-sensitive-subspace, matched-context, cross-task,
cross-method, and open-generation checks.

## Included evidence

- strict group- and endpoint-disjoint NormBank splits;
- all six A/B/C semantic mappings;
- the 108-condition semantic-mapping by identifier-vocabulary by row-order factorial;
- CAA and ITI-style interventions;
- layer and extraction-position localization;
- direct output-sensitive baselines and local Jacobian projection/residual interventions;
- cross-vocabulary readout transfer;
- in-family mapping-balanced factorial evaluation;
- SC101, MultiNLI, matched-context, random-direction, and competence controls;
- the published CAA MCQ/open-generation case study, multi-judge scoring, and the 180-response human validation.

Exploratory datasets and legacy stage pipelines are outside the scope of this
artifact.

## Repository layout

```text
configs/                    Paper experiment configurations
datasets/                   Raw-data placement guide (data are not redistributed)
scripts/                    Resumable GPU and CPU entry points
src/cross_encoding_steering/
                            Pairing, interventions, attribution, and inference
tests/                      CPU tests for the released evidence chain
paper_results/              Reported tables and figures in machine-readable form
outputs/                    Generated locally; ignored by Git
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu,dev]"
```

For automated open-generation judges:

```bash
pip install -e ".[judges]"
```

The reported environment used Ubuntu 22.04.5 LTS, CUDA 12.2, Python 3.11.7,
PyTorch 2.7.0, Transformers 4.52.4, two NVIDIA A100-PCIE-40GB GPUs, an AMD
EPYC 7702P CPU, and 503 GB RAM.

Scripts infer the repository root. Set `NDD_PROJECT_ROOT` only when data and
outputs live elsewhere. Model checkpoints can be redirected without changing
tracked configs:

```bash
export NDD_MODEL_SOURCE_OVERRIDES='{
  "mistral7b_instruct_v03": "/path/to/Mistral-7B-Instruct-v0.3"
}'
export NDD_MODELS=qwen2_5_7b_instruct,llama3_1_8b_instruct
```

The caller chooses the GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/run_layer_attribution.sh
```

## Data

Place the source files under `datasets/`:

```text
datasets/NormBank.csv
datasets/social-chem-101.v1.0.tsv
datasets/multinli_1.0_train.jsonl
datasets/MIC.csv
datasets/CAA/
```

See [DATASETS.md](DATASETS.md) for schemas, sources, and preprocessing scope.
The CAA reference repository can be fetched with:

```bash
bash scripts/fetch_caa_reference.sh
```

## Main reproduction path

Prepare the strict NormBank split once:

```bash
bash scripts/prepare_normbank_strict.sh
```

Run or resume each GPU component independently. Use `NDD_MODELS` to divide
models across workers and set the component-specific phase to `run` or
`aggregate` where needed.

```bash
# Exhaustive letter encodings, factorial attribution, and ITI replication
NDD_STRICT_TASK=caa NDD_STRICT_CAA_PHASE=all bash scripts/run_normbank_strict_audit.sh
NDD_STRICT_TASK=factorial NDD_FACTORIAL_PHASE=all bash scripts/run_normbank_strict_audit.sh
NDD_STRICT_TASK=iti NDD_ITI_PHASE=all bash scripts/run_normbank_strict_audit.sh

# Localization and readout analyses
bash scripts/run_layer_attribution.sh
bash scripts/run_position_localization.sh
bash scripts/run_readout_geometry.sh
bash scripts/run_readout_baseline.sh
bash scripts/run_readout_vocabulary_transfer.sh

# In-family mapping-balanced factorial
bash scripts/run_mapping_balanced_factorial.sh
```

The default phases are `all`. For multi-GPU execution, run separate model
subsets with phase `run`, then invoke phase `aggregate` without a GPU. Relevant
variables are documented at the top of each script.

After the GPU outputs exist, run the central setting-behavior group-cluster
inference:

```bash
bash scripts/run_central_inference.sh
```

## Scope and boundary checks

```bash
# SC101 task-level scope
bash scripts/prepare_sc101.sh
bash scripts/run_sc101_audit.sh

# Pair-ID NormBank encodings, matched-context selectivity, and controls
bash scripts/prepare_normbank.sh
bash scripts/run_normbank_audit.sh
bash scripts/run_validity_controls.sh

# Full six-mapping same-premise MNLI attribution reported in the paper
bash scripts/run_mnli_exhaustive_attribution.sh

# Additional direct-label and opaque-codeword MNLI comparison
bash scripts/run_mnli_control.sh

# Binary MIC extraction-position sensitivity (supplement)
bash scripts/run_mic_position_sensitivity.sh
```

The published CAA case study is phase-controlled:

```bash
NDD_CAA_PHASE=prepare bash scripts/run_published_caa_audit.sh
NDD_CAA_PHASE=run bash scripts/run_published_caa_audit.sh
NDD_CAA_PHASE=aggregate bash scripts/run_published_caa_audit.sh
```

Automated judging requires a local config with the desired providers enabled
and the corresponding API keys. Human-validation forms and summaries use:

```bash
NDD_HUMAN_ACTION=prepare bash scripts/prepare_caa_human_validation.sh
NDD_HUMAN_ACTION=summarize bash scripts/prepare_caa_human_validation.sh
NDD_HUMAN_ACTION=prepare-adjudication bash scripts/prepare_caa_human_validation.sh
NDD_HUMAN_ACTION=summarize-adjudication bash scripts/prepare_caa_human_validation.sh
```

The blinded scoring rubric and adjudication protocol are documented in
[`docs/caa_open_ended_human_annotation_guideline.md`](docs/caa_open_ended_human_annotation_guideline.md).

## Models and locked layers

The main NormBank intervention uses `alpha=0.8` at approximately 75% model
depth. The layer audit additionally evaluates 50%, 62.5%, 75%, and 87.5%, with
each model--contrast direction rescaled to its norm at 75% depth.

| Alias | Model | Main zero-based block |
|---|---|---:|
| `qwen2_5_7b_instruct` | Qwen2.5-7B-Instruct | 20/28 |
| `llama3_1_8b_instruct` | Llama-3.1-8B-Instruct | 23/32 |
| `mistral7b_instruct_v03` | Mistral-7B-Instruct-v0.3 | 23/32 |
| `gemma2_9b_it` | Gemma-2-9B-IT | 31/42 |

SC101 and published CAA use the settings recorded in their respective configs.
ITI heads and strength are selected on validation data.

## Reported assets

`paper_results/` contains compact CSV copies of the reported summaries and the
paper figures. Its README maps claims to files. Full pair-level outputs are
regenerated under `outputs/` and are not committed.

The four quantitative figures used by the current paper and supplement can be
regenerated directly from these compact CSVs:

```bash
pip install -e ".[figures]"
bash scripts/render_paper_figures.sh
```

PDF figure export also requires Ghostscript (`gs`). The renderer converts
figure text to vector outlines so the exported PDFs do not embed Type 3 or CID
fonts.

## Validation

```bash
pytest -q
python -m cross_encoding_steering.cli --help
```

## License

MIT. Dataset and model licenses remain with their original publishers.
