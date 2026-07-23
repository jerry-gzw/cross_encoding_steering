# Machine-readable paper results

These compact CSV files are the final summaries used to check the paper's
tables, figures, and direct supplementary controls. They are not substitute
datasets and do not contain raw prompts, model generations, or API keys.

## Main paper

| Evidence | Files |
|---|---|
| Interface-specific detectability | `main_interface_calibration_global.csv`, `main_interface_calibration_per_model.csv` |
| Five non-source letter mappings | `main_letter_permutation_by_model.csv` |
| Matched-context selectivity | `main_context_discrimination_statistics.csv` |
| Published CAA MCQ/open-ended comparison | `main_published_caa_mcq_open_ended_comparison.csv` |

## Supplementary controls

| Evidence | Files |
|---|---|
| Mapping-average norm retention | `appendix_mapping_balance_norm_retention_cells.csv`, `appendix_mapping_balance_norm_retention_overall.csv` |
| Letter-mapping uncertainty and contrast breakdown | `appendix_letter_permutation_bootstrap_ci.csv`, `appendix_letter_permutation_by_contrast.csv`, `appendix_letter_permutation_swap_exclusion_decision.csv` |
| Context-pair filtering and cluster bootstrap | `appendix_context_pair_filter_audit.csv`, `appendix_context_cluster_bootstrap_sensitivity.csv` |
| Five-random-control sensitivity | `multi_random_standardized_gain_summary.csv`, `multi_random_context_summary.csv` |
| Opaque-key competence | `normbank_key_competence_summary.csv`, `appendix_published_caa_opaque_competence.csv` |
| Multi-judge CAA robustness | `appendix_published_caa_multi_judge_agreement.csv`, `appendix_published_caa_multi_judge_completeness.csv`, `appendix_published_caa_multi_judge_quality.csv`, `appendix_published_caa_multi_judge_statistics.csv`, `appendix_published_caa_open_ended_paired_statistics.csv` |

The raw and intermediate outputs are regenerated under `outputs/` by the
commands in the repository README and are intentionally not committed.
