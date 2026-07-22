# Datasets used by the AAAI 2027 artifact

Raw data are not included. Download each dataset under its original terms and
place it below `datasets/` using the paths in this document.

## NormBank

- Source: <https://huggingface.co/datasets/SALT-NLP/NormBank>
- Local path: `datasets/NormBank.csv`
- Required fields: `setting`, `behavior`, `constraints`, `norm`, and optionally
  `split`.
- Labels used: `taboo`, `normal`, `expected`.

Pairs are built within the same exact `setting + behavior` group. Lower- and
higher-label rows therefore retain the setting and behavior while changing the
constraints and annotated label. The three ordered contrasts are
`taboo-normal`, `taboo-expected`, and `normal-expected`. Complete pairs, rather
than individual endpoints, are assigned to train, validation, or test by a
deterministic pair-ID hash.

The paper uses up to 2,048 training pairs and 512 pair-ID-held-out test pairs
per contrast. Pair-ID separation does not imply that every setting-behavior
group or endpoint is unseen across contrasts; the paper reports this boundary.

## MultiNLI

- Source: <https://cims.nyu.edu/~sbowman/multinli/>
- Local path: `datasets/multinli_1.0_train.jsonl`
- Required fields: `sentence1`, `sentence2`, `gold_label`.
- Labels used: `contradiction`, `neutral`, `entailment`.

The control matches examples sharing the same premise and carrying different
relation labels, then applies the same three-label interface audit used for
NormBank. It is a non-norm counterexample, not a broad benchmark comparison.

## Public CAA repository

- Source: <https://github.com/nrimsky/CAA>
- Local directory: `datasets/CAA/`
- Fetch command: `bash scripts/fetch_caa_reference.sh`

The artifact uses the released contrastive data, multiple-choice questions,
and open-ended prompts for hallucination, refusal, and sycophancy. It also
loads the seven public behavior datasets needed by the original direction
normalization procedure. The CAA files remain governed by their upstream
license and are not copied into this repository.

## Not used

MIC, Social Chemistry 101, NormDial, Moral Stories, ETHICS, Scruples,
MoralExceptQA, CultureBank, CulturalBench, NormAd, and ProsocialDialog were part
of earlier exploratory work but do not support the final AAAI 2027 evidence
chain and are deliberately absent from this release.
