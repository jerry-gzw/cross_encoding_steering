from math import isclose
from pathlib import Path

import pandas as pd

from cross_interface_steering.interface_statistics import (
    InterfaceStatisticsConfig,
    build_interface_statistics,
)


def _synthetic_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    nuisance_rows = []
    context_rows = []
    interfaces = {
        "letter_canonical": 2.0,
        "letter_reversed": 0.2,
        "direct_label_completion": 1.0,
        "opaque_completion": 0.5,
    }
    random_effects = {
        "letter_canonical": 0.1,
        "letter_reversed": -0.1,
        "direct_label_completion": 0.0,
        "opaque_completion": 0.0,
    }
    for model_index, model in enumerate(("model_a", "model_b")):
        for contrast_index, contrast in enumerate(("low_mid", "mid_high")):
            for pair_index in range(4):
                pair_id = f"{model}-{contrast}-{pair_index}"
                offset = 0.1 * pair_index
                for template in ("one", "two"):
                    for mode in ("raw_pre_answer", "random_direction_control"):
                        for interface, raw_effect in interfaces.items():
                            effect = (
                                raw_effect + offset
                                if mode == "raw_pre_answer"
                                else random_effects[interface]
                            )
                            common = {
                                "model_alias": model,
                                "model_name": model,
                                "pair_id": pair_id,
                                "pair_type": contrast,
                                "mode": mode,
                                "interface": interface,
                                "template": template,
                            }
                            nuisance_rows.append({**common, "target_margin_gain": effect})
                            interaction = 0.6 + offset if mode == "raw_pre_answer" else 0.0
                            context_rows.append({
                                **common,
                                "context_interaction": interaction,
                                "discrimination_gap_change": (
                                    -0.4 - offset if mode == "raw_pre_answer" else 0.0
                                ),
                                "patched_rank_correct": float(mode == "raw_pre_answer"),
                                "rank_sign_preserved": 1.0,
                            })
    return pd.DataFrame(nuisance_rows), pd.DataFrame(context_rows)


def test_build_interface_statistics_pairs_modes_and_interfaces(tmp_path: Path) -> None:
    nuisance, context = _synthetic_rows()
    config = InterfaceStatisticsConfig(
        project_root=tmp_path,
        cross_interface_pair_effects=tmp_path / "cross_interface.csv",
        nuisance_pair_effects=tmp_path / "nuisance.csv",
        context_pair_rows=tmp_path / "context.csv",
        output_dir=tmp_path / "output",
        n_boot=200,
        confidence=0.95,
        bootstrap_chunk_size=50,
        seed=13,
    )
    tables = build_interface_statistics(nuisance, context, config)

    nuisance_global = tables["nuisance_mode_global_bootstrap_ci"]
    raw_canonical = nuisance_global[
        nuisance_global["comparison"].eq("raw_minus_random")
        & nuisance_global["interface"].eq("letter_canonical")
    ].iloc[0]
    assert isclose(raw_canonical["estimate"], 2.05)
    assert bool(raw_canonical["ci_excludes_zero"])

    interface_global = tables["nuisance_interface_global_bootstrap_ci"]
    canonical_reversed = interface_global[
        interface_global["mode"].eq("raw_pre_answer")
        & interface_global["comparison"].eq("canonical_minus_reversed")
    ].iloc[0]
    assert isclose(canonical_reversed["estimate"], 1.8)

    mode_effect = tables["nuisance_mode_effect_global_bootstrap_ci"]
    raw_effect = mode_effect[
        mode_effect["mode"].eq("raw_pre_answer")
        & mode_effect["interface"].eq("letter_canonical")
    ].iloc[0]
    assert isclose(raw_effect["estimate"], 2.15)

    adjusted = tables["nuisance_control_adjusted_interface_global_bootstrap_ci"]
    adjusted_gap = adjusted[
        adjusted["comparison"].eq("raw_minus_random")
        & adjusted["interface_comparison"].eq("canonical_minus_reversed")
    ].iloc[0]
    assert isclose(adjusted_gap["estimate"], 1.6)

    context_global = tables["context_mode_global_bootstrap_ci"]
    interaction = context_global[
        context_global["comparison"].eq("raw_minus_random")
        & context_global["metric"].eq("context_interaction")
        & context_global["interface"].eq("letter_canonical")
    ].iloc[0]
    assert isclose(interaction["estimate"], 0.75)
    assert bool(interaction["ci_excludes_zero"])

    ranking_margin = context_global[
        context_global["comparison"].eq("raw_minus_random")
        & context_global["metric"].eq("discrimination_gap_change")
        & context_global["interface"].eq("letter_canonical")
    ].iloc[0]
    assert isclose(ranking_margin["estimate"], -0.55)
    assert bool(ranking_margin["ci_excludes_zero"])

    inventory = tables["interface_statistics_inventory"]
    assert "skipped_missing_mode" in set(inventory["status"])

    calibration = tables["interface_calibration_global"]
    raw_calibration = calibration[
        calibration["comparison"].eq("raw_minus_random")
        & calibration["interface"].eq("letter_canonical")
    ].iloc[0]
    assert raw_calibration["mean_paired_standardized_gain"] > 1.0
    assert isclose(raw_calibration["mean_positive_pair_rate"], 1.0)

    agreement = tables["interface_sign_agreement_global"]
    reversed_agreement = agreement[
        agreement["comparison"].eq("raw_minus_random")
        & agreement["interface"].eq("letter_reversed")
    ].iloc[0]
    assert isclose(reversed_agreement["mean_sign_agreement_rate"], 1.0)
    assert isclose(reversed_agreement["mean_joint_positive_rate"], 1.0)
