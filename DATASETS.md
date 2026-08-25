# Dataset guide

Raw data are not included. Download each dataset under its original terms and
place it below `datasets/`.

## NormBank

- Source: <https://huggingface.co/datasets/SALT-NLP/NormBank>
- Local path: `datasets/NormBank.csv`
- Required fields: `setting`, `behavior`, `constraints`, `norm`; `split` is
  optional.
- Labels: `taboo`, `normal`, `expected`.

Pairs share the same exact `setting + behavior` and differ in constraints and
label. The strict main split hashes the full setting-behavior group, does not
include pair type in the split key, and rejects any endpoint identity appearing
in more than one split. The supporting pipeline retains the older pair-ID split
for analyses that require its previously reported outputs.

## Social Chemistry 101

- Source: <https://github.com/mbforbes/social-chemistry-101>
- Local paths accepted:
  - `datasets/social-chem-101.v1.0.tsv`
  - `datasets/social-chem-101/social-chem-101.v1.0.tsv`
- Required fields: `action`, `action-moral-judgment`; the adapter also uses
  `split`, `rot-bad`, `rot-categorization`, `situation`, `area`, and `rot-id`
  when present.

The released scope experiment keeps valid social-norm rows, collapses the
five-point action judgment into `bad`, `ok`, and `good`, and uses action-only
prompts. Pairs are weakly matched within the available situation or area
grouping; they are not treated as strict counterfactual contexts.

## MultiNLI

- Source: <https://cims.nyu.edu/~sbowman/multinli/>
- Local path: `datasets/multinli_1.0_train.jsonl`
- Required fields: `sentence1`, `sentence2`, `gold_label`.
- Labels: `contradiction`, `neutral`, `entailment`.

The audit matches examples sharing the same premise and carrying different
relation labels, assigns complete premise groups to disjoint splits, extracts
one direction under the E/N/C mapping, and evaluates all six A/B/C mappings
without re-extraction. It is a non-norm comparison for the cross-encoding
evaluation, not a general NLI benchmark claim.

## Public CAA repository

- Source: <https://github.com/nrimsky/CAA>
- Local directory: `datasets/CAA/`
- Fetch command: `bash scripts/fetch_caa_reference.sh`

The case study uses the released contrastive, multiple-choice, and open-ended
data for hallucination, refusal, and sycophancy, plus the public behavior data
needed by the original normalization procedure.

## Scope

The release deliberately excludes exploratory adapters and datasets that do
not contribute to the final AAAI evidence chain. Dataset preprocessing choices
used by every included experiment are encoded in the tracked JSON configs.
