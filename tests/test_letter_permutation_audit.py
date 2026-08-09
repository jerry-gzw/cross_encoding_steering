from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from cross_encoding_steering.letter_permutation_audit import (
    LetterPermutationStatisticsConfig,
    _bootstrap_statistics,
    _decision_table,
    _exclude_exact_swap_contrasts,
    _mapping_inventory,
    _pair_adjusted_effects,
    _stratum_statistics,
)


def _config() -> LetterPermutationStatisticsConfig:
    path = Path(__file__).parents[1] / "configs" / "normbank_letter_permutation_audit.example.json"
    return replace(
        LetterPermutationStatisticsConfig.from_json(path, project_root=Path("/tmp/ndd-test")),
        n_boot=100,
    )


def _synthetic_rows(config: LetterPermutationStatisticsConfig) -> pd.DataFrame:
    rows = []
    interfaces = [item.name for item in config.audit.interfaces]
    for model_alias in ["model_a", "model_b"]:
        for pair_type in ["taboo_vs_normal", "taboo_vs_expected", "normal_vs_expected"]:
            for pair_index in range(4):
                for interface_index, interface in enumerate(interfaces):
                    semantic = 1.0 if interface_index != 5 else 0.1
                    original = 1.0 if interface_index == 0 else 0.2
                    for mode in ["raw_pre_answer", *[
                        f"random_direction_control_seed_{seed}"
                        for seed in config.audit.random_control_seeds
                    ]]:
                        random_offset = 0.05 * (pair_index - 1.5) if mode != "raw_pre_answer" else 0.0
                        semantic_gain = semantic if mode == "raw_pre_answer" else random_offset
                        original_gain = original if mode == "raw_pre_answer" else random_offset
                        for endpoint in ["low", "high"]:
                            for template in ["template_a", "template_b"]:
                                rows.append(
                                    {
                                        "model_alias": model_alias,
                                        "model_name": model_alias,
                                        "pair_id": f"{pair_type}_{pair_index}",
                                        "pair_type": pair_type,
                                        "mode": mode,
                                        "interface": interface,
                                        "evaluation_policy": "counterfactual",
                                        "template": template,
                                        "endpoint": endpoint,
                                        "target_margin_gain": semantic_gain,
                                        "original_slot_margin_gain": original_gain,
                                    }
                                )
    return pd.DataFrame(rows)


def test_mapping_inventory_exhausts_six_permutations() -> None:
    inventory = _mapping_inventory(_config())

    assert len(inventory) == 6
    assert inventory["option_order"].nunique() == 6
    assert inventory["is_source_mapping"].sum() == 1
    assert set(inventory["permutation_class"]) == {"identity", "transposition", "three_cycle"}


def test_letter_permutation_statistics_keep_semantic_and_original_slot_separate() -> None:
    config = _config()
    effects = _pair_adjusted_effects(_synthetic_rows(config), config)
    strata, per_seed, summary = _stratum_statistics(effects)
    bootstrap = _bootstrap_statistics(effects, config)
    decision = _decision_table(bootstrap, _mapping_inventory(config))

    assert set(effects["metric"]) == {
        "semantic_target_margin",
        "original_slot_margin",
        "semantic_minus_original_slot",
    }
    assert len(strata) > 0 and len(per_seed) > 0 and len(summary) > 0
    canonical = decision.loc[decision["is_source_mapping"]].iloc[0]
    assert canonical["semantic_minus_original_slot__mean_adjusted_gain"] == 0.0
    non_source = decision.loc[~decision["is_source_mapping"]]
    assert non_source["semantic_target_margin__mean_adjusted_gain"].mean() > 0.0


def test_exact_swap_sensitivity_excludes_one_contrast_per_transposition() -> None:
    config = _config()
    inventory = _mapping_inventory(config)
    effects = _pair_adjusted_effects(_synthetic_rows(config), config)

    filtered, audit = _exclude_exact_swap_contrasts(effects, inventory)

    assert audit["is_exact_source_target_swap"].sum() == 3
    included = audit.groupby("permutation_class")["sensitivity_includes_contrast"].sum()
    assert included["three_cycle"] == 6
    assert included["transposition"] == 6
    observed = filtered[["interface", "pair_type"]].drop_duplicates().groupby("interface").size()
    for item in inventory.loc[~inventory["is_source_mapping"]].itertuples(index=False):
        expected = 3 if item.permutation_class == "three_cycle" else 2
        assert observed[item.interface] == expected
